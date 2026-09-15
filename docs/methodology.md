# Methodology

How the numbers in `docs/results.md` are produced, and what they can and cannot
support.

## 1. Data

A seeded simulator, not a download. The procedure and its limitations are in
`docs/data-sources.md`. Default scale: 5,000 customers, 10,000 terminals, 183
days, about 1.8 million transactions at a 0.81% fraud rate.

Everything is determined by `run.seed`. The same configuration produces the same
transactions, the same split, the same models and the same metrics on any
machine.

## 2. Features

Fifteen inputs in three families.

**Transaction** (3): `tx_amount`, `tx_during_weekend`, `tx_during_night`.

**Customer** (6): trailing transaction count and average amount over 1, 7 and 30
days. No delay — these use no labels, and the transaction being scored is
included in its own window because its amount is known at scoring time.

**Terminal** (6): trailing transaction count and fraud rate over 1, 7 and 30
days, **each ending seven days before the transaction being scored**.

### The delay period

A fraud label does not exist when the transaction happens. It exists when a
customer disputes the charge or an investigator confirms it, typically days
later. A terminal risk feature computed from labels up to the moment of scoring
uses information the production system could not have had.

So terminal risk over a window covers `[t - delay - window, t - delay]`.
Mechanically it is the difference of two trailing sums — one over
`delay + window` days, one over `delay` days — which leaves exactly the window
ending `delay` before now.

This is asserted, not assumed. `tests/test_features.py::test_a_label_cannot_influence_features_before_the_delay`
marks a single transaction as fraudulent and checks that no feature computed
before `D + delay` changes, at delays of 0, 3, 7 and 14 days.

### Correctness

Grouped time-window aggregates are easy to get wrong in a way that does not
raise. `groupby(...).rolling(...)` returns values ordered by *(group, time)*,
not by input row; re-attaching them to a time-ordered frame assigns each row
another row's history. The result has correct dtypes, a plausible range and no
nulls.

An early version of this pipeline did exactly that. The only symptom was
PR-AUC 0.18 instead of 0.73.

`tests/test_features.py::test_features_match_a_brute_force_reference` pins every
feature against an obviously-correct O(n²) implementation that scans every row
against every other row.

### Batch and online must agree

The batch pipeline trains the model; the online store serves it. If they
disagree, the model degrades in production in a way that looks like data drift.

`tests/test_features.py::test_online_features_match_batch_exactly` replays a
transaction stream through both and asserts every feature agrees.

Replayed over the whole test window at full scale — 556,547 events, in strict
event-time order — the online store reproduces the batch table to float32
rounding: worst deviation 2.7e-05, on the customer average-amount features. 49
rows in 556,547 (0.009%) differ by one on a terminal transaction count, where an
event falls exactly on a window boundary. Scored with the champion, the two
paths give identical metrics: PR-AUC 0.7312 either way.

## 3. Splitting

Chronological, with a gap:

```
|<--------- train --------->|<- delay ->|<------- test ------->|
                 |<- val ->|
    days 0-97      98-111     112-118        119-174
```

A random split lets a model learn from transactions that happen *after* the ones
it is scored on. Every number from a randomly split fraud model is unreachable
in production.

Validation is carved from the **end** of training rather than sampled from it,
so operating points are chosen on the most recent data the model is allowed to
see — the closest available proxy for what it will meet.

At the default configuration: 973,244 training rows (7,597 frauds), 138,739
validation (1,181), 556,547 test (4,640).

## 4. Metrics

| Metric | Definition | Why |
| --- | --- | --- |
| `pr_auc` | Average precision — the step-wise summary of the precision-recall curve | The headline. Preferred over trapezoidal AUC, which interpolates between operating points no threshold produces |
| `roc_auc` | Standard ROC AUC | Reported for comparability, not for ranking. Optimistic under imbalance |
| `precision`, `recall`, `f1` | At the selected threshold | What an operations team experiences |
| Confusion counts | Raw TP/FP/FN/TN and alert volume | "Precision 0.87" and "426 false positives" are the same fact; only the second says whether the queue is workable |
| `card_precision_at_k` | Mean daily precision over the *k* highest-risk **cards** | The operational metric. A team investigates a card, not a swipe, and has fixed daily capacity |
| `brier` | Brier score | Calibration. Scores feed a threshold, so drifting probabilities invalidate the operating point even when ranking improves |
| latency p50/p95/p99 | Single-row scoring time | The streaming consumer scores one event at a time; batch throughput flatters it |

### Why not ROC-AUC

Measured, on the same scores at full scale: random forest has the best ROC-AUC
(0.9012) and the third-best PR-AUC (0.7053). Ranking on ROC-AUC picks a model
that is worse at the task and 81× slower to serve.

### Threshold selection

A probability model is not a decision until a threshold makes it one, and the
threshold is a business choice — investigator time against undetected fraud.

Thresholds are selected on **validation**, never on test. Selecting on the
evaluation window is the most common way a fraud model's reported precision
turns out to be unreachable. Three objectives are available: maximise F1, or
maximise recall subject to a precision floor, or maximise precision subject to a
recall floor.

### Uncertainty

Every headline metric carries a 95% percentile bootstrap interval, stratified by
class. An unstratified bootstrap of a 0.8% positive rate occasionally draws a
sample with no positives at all.

The test window holds 4,640 frauds, so intervals are tight — about ±0.012 on
PR-AUC. This is the opposite situation from a small evaluation set, and it means
the differences reported between models are real rather than noise.

## 5. Reproducibility

- One seed governs Python's RNG, NumPy, every model's solver, the split and the
  bootstrap.
- No data is downloaded at any point.
- The whole pipeline runs in CI on every change at the reduced configuration —
  generate, train, stream, drift — so a break cannot reach `main` unnoticed.
- CI asserts the model registry actually recorded a champion, because a silent
  registry failure leaves the serving bundle and the registry disagreeing about
  what is deployed.

```bash
make install
make data      # about 58 seconds
make train     # about 19 minutes at the default configuration
make stream
make drift
```

## 6. Threats to validity

| Threat | Effect | Mitigation, or why not |
| --- | --- | --- |
| Simulated data | Contains only the patterns it was told to contain | None available. Stated plainly in `docs/data-sources.md` and `docs/results.md` §11 |
| No adversarial drift | Real fraud changes *because* it is detected | Not reproducible by any generator |
| Single simulation seed | Results could be a favourable draw | The bootstrap covers sampling within the run, not across seeds. Section 12 of the results lists a multi-seed run as a follow-up |
| Hyperparameters not tuned | Model ranking might change under tuning | Deliberate: all four families use comparable, untuned settings so the comparison is like-for-like. A tuned comparison is a separate experiment |
| Fraud rate varies with scale | Scenarios inject fixed counts per day, so a smaller population has a higher rate | Both configurations are reported with their own base rate; metrics are never compared across scales |
| Latency measured on shared hardware | Contention inflates it | Measured in a single process with nothing else running. An earlier contaminated measurement reported XGBoost at 51.8ms against a true 0.72ms — a 70× error, which is why this is called out |
