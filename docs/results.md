# Results

Every number here was produced by running the code in this repository. Nothing
is estimated, projected, or carried over from a paper. Where a measurement is
missing, it says so rather than being filled in.

Reproduce with:

```bash
make install
make data      # generate transactions and the feature table
make train     # train every family, gate a promotion
make stream    # replay the test window through the broker
make drift     # compare two windows
```

---

## 1. Environment

| | |
| --- | --- |
| Platform | Linux 6.18.44 x86_64, glibc 2.39 |
| CPU | Intel Xeon @ 2.80 GHz, 4 cores |
| Memory | 15 GB |
| Python | 3.11.15 |
| NumPy / pandas | 2.4.6 / 3.0.5 |
| scikit-learn | 1.9.1 |
| XGBoost / LightGBM | 3.2.0 / 4.7.0 |
| MLflow | 3.16.0 |
| Measured | 2026-09-15 |

The full training run — five families, bootstrap intervals and latency
measurement — took **1,125 s (18.7 minutes)**.

CPU contention distorts single-row latency badly, so this run was executed with
nothing else on the machine. An earlier run measured under load reported XGBoost
at 51.8 ms p95 against a true 0.72 ms — a 70× error. Every latency figure below
comes from the uncontended run.

---

## 2. Dataset

Produced by `rtml.data.simulator` from the seed in `configs/default.yaml`.
Nothing is downloaded. The procedure and its limitations are in
[`data-sources.md`](data-sources.md).

| | |
| --- | --- |
| Configuration | `configs/default.yaml` |
| Seed | 20260101 |
| Customers | 5,000 |
| Terminals | 10,000 |
| Period | 183 days from 2025-01-01 |
| Transactions | 1,817,275 |
| Frauds | 14,755 (0.812%) |

Fraud by scenario over the whole generated period:

| Scenario | Pattern | Frauds | Share |
| --- | --- | ---: | ---: |
| 1 | Amount above 220 | 1,103 | 7.5% |
| 2 | Compromised terminal, 28 days | 9,086 | 61.6% |
| 3 | Stolen credentials, 14 days, amounts ×5 | 4,566 | 30.9% |

Scenario 2 is the majority, and it is invisible at transaction level — the
amounts look entirely ordinary. Only terminal history detects it. That is why
§5 reports recall per scenario rather than an aggregate alone.

Fifteen features: three transaction-level, six customer aggregates over 1/7/30
days, six terminal aggregates over the same windows, each terminal window ending
seven days before the transaction being scored.

---

## 3. Split

Chronological, with a seven-day delay period between train and test standing in
for the time fraud labels take to arrive.

| Part | Days | Rows | Frauds | Rate |
| --- | --- | ---: | ---: | ---: |
| Train | 0–111 | 973,244 | 7,597 | 0.781% |
| ↳ validation (carved from the end) | 98–111 | 138,739 | 1,181 | 0.851% |
| Delay | 112–118 | — | — | — |
| Test | 119–174 | 556,547 | 4,640 | 0.834% |

The delay period is not cosmetic. Without it, terminal risk features at the
start of the test window would be computed from labels that, in production,
would not yet have arrived.

---

## 4. Model comparison

Five families on the same split, comparable untuned settings so the comparison
is like-for-like. Ranked by PR-AUC on the test window. Thresholds were selected
on validation, never on test.

| Model | PR-AUC | 95% CI | ROC-AUC | Brier | Precision | Recall | F1 | Card P@100 | Fit (s) |
| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **lightgbm** | **0.7312** | 0.7187 – 0.7431 | 0.9005 | 0.0265 | 0.871 | 0.621 | 0.725 | 0.561 | 7.2 |
| xgboost | 0.7284 | 0.7155 – 0.7408 | 0.8985 | 0.0249 | 0.883 | 0.612 | 0.723 | 0.559 | 11.5 |
| random_forest | 0.7053 | 0.6920 – 0.7187 | **0.9012** | 0.0437 | 0.857 | 0.650 | **0.739** | 0.548 | 161.8 |
| logistic_regression | 0.6122 | 0.5980 – 0.6267 | 0.8926 | 0.0657 | 0.732 | 0.553 | 0.630 | 0.515 | 2.6 |
| isolation_forest | 0.2847 | 0.2719 – 0.2989 | 0.8442 | 0.0526 | 0.466 | 0.436 | 0.451 | 0.396 | 0.8 |

Intervals are 95% percentile bootstrap, stratified by class, over 1,000
resamples. Base rate in the test window is 0.834%, so the champion's PR-AUC of
0.731 is **88× the no-skill baseline**.

### What this says

**ROC-AUC would have picked the wrong model.** Random forest has the best
ROC-AUC (0.9012) and only the third-best PR-AUC (0.7053). The spread across
supervised families is 0.0086 on ROC-AUC and 0.1190 on PR-AUC — fourteen times
wider. At a 0.83% base rate, ROC-AUC is dominated by the 551,907 negatives, so
hundreds of extra false positives barely move it.

**XGBoost and LightGBM are not distinguishable on accuracy here.** Their
intervals overlap across almost their entire width (0.7187–0.7431 against
0.7155–0.7408). Claiming one is better on this evidence would be unsupported.
LightGBM was promoted because it fits in 62% of the time and the two are within
0.13 ms of each other on p95 latency — a tie broken on cost, not on a metric
difference the data cannot support.

**The linear baseline earns its place.** Logistic regression reaches PR-AUC
0.6122 — 84% of the champion's — on the same features. The 0.119 gap is what
interactions between the fifteen features are worth, and it is large enough to
justify the trees while small enough to say the features are doing most of the
work.

**Unsupervised detection is not competitive, but it is not nothing.** Isolation
forest never sees a label and still reaches PR-AUC 0.2847 — 34× the base rate,
and 39% of the supervised champion. That is a useful floor for a portfolio with
no labels yet, and a clear answer to whether labels are worth collecting.

**Random forest is best calibrated in the wrong direction.** Its Brier score
(0.0437) is the worst of the four supervised families despite the best ROC-AUC,
because bagged vote fractions are not probabilities. Since the deployed
threshold is chosen on those scores, calibration is a promotion gate here rather
than a footnote.

---

## 5. The champion in operational terms

LightGBM, threshold 0.9848 (F1-optimal on validation), over the 56-day test
window:

| | |
| --- | --- |
| Alerts raised | 3,307 (0.594% of transactions, ~59/day) |
| True positives | 2,881 |
| False positives | 426 |
| Missed frauds | 1,759 |
| Precision | 0.871 |
| Recall | 0.621 |
| Card precision@100/day | 0.561 (sd 0.066 over 56 days) |

"Precision 0.871" and "426 false positives in eight weeks" are the same fact;
only the second says whether the queue is workable. At roughly 59 alerts a day,
about 8 of them wrong, this is a workload a small team can absorb.

Card precision@100 is the operational metric: of the 100 highest-risk *cards*
each day, 56 were actually compromised. It is lower than transaction precision
because it forces the model to fill a fixed daily budget whether or not that
much fraud exists.

### Recall by fraud scenario

Aggregate recall averages three different problems together. Split apart, at the
deployed threshold and at 0.5:

| Scenario | Test frauds | Recall @ 0.9848 | Recall @ 0.5 |
| --- | ---: | ---: | ---: |
| 1 — amount above 220 | 316 | 0.313 | **0.997** |
| 2 — compromised terminal | 2,973 | 0.634 | 0.698 |
| 3 — stolen credentials | 1,351 | 0.663 | 0.905 |
| **All** | **4,640** | **0.621** | **0.778** |

| | Threshold 0.9848 | Threshold 0.5 |
| --- | ---: | ---: |
| Precision | 0.871 | 0.373 |
| Recall | 0.621 | 0.778 |
| Alerts | 3,307 | 9,671 |
| False positives | 426 | 6,059 |

This is the most useful measurement in the report, and it is not about the
model. Scenario 1 is a deterministic rule — in the test window every one of the
953 transactions above 220 is fraudulent and no legitimate transaction exceeds
it — and the model learns it almost perfectly: 99.7% of scenario-1 frauds score
above 0.5. Yet at the deployed threshold only 31% are caught, because the
F1-optimal cut-off (0.9848) falls above the *median* score of that scenario
(0.9805).

The operating point, not the model, loses two thirds of the easiest fraud class.
A single global threshold chosen to maximise F1 silently trades away an entire
category the model has already solved. Dropping to 0.5 recovers it — at 14× the
false positives, which is exactly the trade-off F1 assumes away by treating a
missed fraud and a wasted investigation as equally costly. Cost-sensitive
thresholds (§12) are the fix; per-scenario reporting is what makes the problem
visible in the first place.

---

## 6. Serving latency

Single-row scoring, which is what the streaming consumer actually does. Batch
throughput flatters every one of these numbers, so both are given. 200 sampled
rows per model, measured with the machine otherwise idle.

| Model | p50 (ms) | p95 (ms) | p99 (ms) | Batch (rows/s) |
| --- | ---: | ---: | ---: | ---: |
| logistic_regression | 0.426 | 0.560 | 0.622 | 449,439 |
| xgboost | 0.492 | 0.723 | 1.233 | 209,337 |
| **lightgbm** | 0.751 | 0.849 | 1.321 | 133,044 |
| isolation_forest | 10.587 | 11.465 | 12.378 | 14,110 |
| random_forest | 58.078 | 68.451 | 72.379 | 2,579 |

**Random forest is not deployable on this path.** At 68 ms p95 it is **81×
slower** than the champion, for 0.026 less PR-AUC. Ranking on ROC-AUC alone
would have selected exactly this model.

The gap between single-row and batch is the reason both are reported: LightGBM
scores 133,000 rows a second in a batch and about 1,260 a second one at a time
(mean 0.79 ms). A throughput figure taken from a batch benchmark would overstate
a streaming service's capacity by a hundredfold.

---

## 7. Streaming

The whole test window replayed through the producer, the in-process broker and
the scoring consumer, with the online feature store warmed from history first.
No external service; reproducible from a clean checkout with `make stream`.

| | |
| --- | --- |
| Backend | in-process (Kafka-compatible interface) |
| Partitions | 4, keyed by customer |
| Warm-up | 1,181,268 rows in 16.2 s |
| Produced | 556,547 events at 65,435/s |
| Scored | 556,547 events at **1,076/s** |
| Alerts | 3,307 |
| Scoring errors | 0 |
| Labels past the reordering allowance | 0 |

Per-event scoring latency, which includes online feature computation, the model
call and the state update:

| | ms |
| --- | ---: |
| mean | 0.871 |
| p50 | 0.828 |
| p95 | 0.998 |
| p99 | 1.289 |
| max | 1280.586 |

One event took 1.28 s — three orders of magnitude above p99, once in 556,547
events. It is not the online store's array compaction: that triggers only once an
entity accumulates more than 1,024 evicted entries, and at this configuration a
customer holds about 240 events across the whole warm-up and a terminal about 49,
so it never fires. That leaves a garbage-collection or scheduler pause, which
this run does not distinguish between. It is reported rather than trimmed,
because a single 1.3-second stall is exactly what a p99 hides.

Throughput of 1,076 events/s against a mean latency of 0.871 ms means roughly
94% of wall-clock time is inside `score_event`; the remainder is broker polling
and publishing. Scaling this path means more partitions, not a faster loop.

### Streamed and batch agree exactly

| | Batch evaluation | Streamed |
| --- | ---: | ---: |
| PR-AUC | 0.731191 | 0.731191 |
| ROC-AUC | 0.900456 | 0.900456 |
| Precision | 0.871 | 0.871 |
| Recall | 0.621 | 0.621 |
| Alerts | 3,307 | 3,307 |
| True / false positives | 2,881 / 426 | 2,881 / 426 |

This is the guarantee the whole design exists to provide, and it is not free.
Getting here required finding two bugs that produced entirely plausible output.

### The partition drain bug

**Symptom.** Streamed PR-AUC 0.4940 against a batch 0.7312 — the same model, the
same data, the same feature code. Recall 0.394 against 0.621. No errors, no
warnings, no obviously wrong value anywhere.

**Ruling things out first.** Replaying the identical test window through the
online store in strict event-time order reproduced the batch table to float32
rounding and scored **0.7312 — identical to batch**. That eliminated the feature
implementation entirely and put the loss in the consumer path.

**Cause.** `InProcessBroker.poll` filled each batch from the first partition
before touching the next, despite a docstring promising round-robin. The
consumer therefore saw partition 0's entire 56-day history, then restarted at
day 119 for partition 1. Measured over the stream: 75% of events were scored
behind the consumer's own clock, by a mean of **28 days**, with three backward
jumps of 56 days each.

Customer features survived this, because partitioning by customer keeps a
customer's events together and in order. **Terminal** state is shared across all
four partitions, and terminal risk is the strongest feature family here — so it
was computed against a history the store had already advanced past and evicted.
Retained terminal events at the end of the run: 108,010, against the 377,344
that should have been there.

**Fix.** A genuine multi-pass round-robin: every partition contributes an equal
share of each batch, and passes repeat until the batch is full or nothing
remains. Mean lag fell from 28 days to 0.95, worst case from 56 days to 2.88.

### Labels released in event time, not arrival order

Fair polling reduces the skew; it does not remove it. Partitions hold unequal
numbers of events — 135,441 to 142,438 here, a 5% spread — so the smallest drains
first and the rest finish up to 2.88 days behind.

That residual matters because the online history is a prefix-sum index over a
sorted array. One out-of-order append invalidates both binary searches, and every
later window query for that entity returns a wrong answer with a plausible value
and no error.

**Fix.** Pending labels sit in a min-heap and are released against a watermark:
nothing reaches the feature store until the stream has certainly moved past it.
The seven-day label delay doubles as the reordering allowance, and 2.88 days of
measured skew sits comfortably inside it. Over the full run, **0** labels
exceeded the allowance; 487,009 were applied in order and 69,538 remained pending
at the end, which is exactly the seven-day tail.

Beyond the allowance a label is counted and dropped rather than applied out of
order, because corrupting the index is worse than losing one observation. And
`_EntityHistory.append` now rejects a backward timestamp outright — one
comparison that turns this entire class of silent corruption into a stack trace.

### A cold consumer is measurably worse

The same stream with no warm-up, so every window starts empty:

| | Cold | Warmed |
| --- | ---: | ---: |
| PR-AUC | 0.6245 | **0.7312** |
| ROC-AUC | 0.8500 | 0.9005 |
| Precision | 0.816 | 0.871 |
| Recall | 0.484 | 0.621 |
| True positives | 2,247 | 2,881 |
| Alerts | 2,753 | 3,307 |

A cold start costs 0.107 PR-AUC and **634 missed frauds** over eight weeks. This
is not a simulator artefact: any stateful streaming model pays it after a
deployment, a restart or a partition reassignment. The warm-up path exists for
that reason and belongs in the startup probe, before an instance takes traffic.

Note that the cold run scores *faster* — 1,091 events/s against 1,076 — because
empty windows are cheaper to query. Throughput and correctness move in opposite
directions here, which is worth knowing before tuning for throughput.

---

## 8. Drift monitoring

Two windows of the generated stream compared feature by feature: days 162–175 as
reference (138,791 rows) against days 176–182 (69,610 rows).

| | |
| --- | --- |
| Features monitored | 15 |
| Drifted (PSI ≥ 0.10) | 0 |
| Significant (PSI ≥ 0.25 **and** KS p < 0.01) | 0 |
| Largest PSI | 0.0009 (`customer_nb_tx_30d`) |
| Alert raised | no |
| Retraining recommended | no |

A negative result, and the expected one: the simulator is stationary, so there
is no drift to find. A detector that reported drift here would be reporting
noise.

### Why significance alone is not a trigger

The interesting measurement is what happens when drift is injected. Scaling
`tx_amount` in the comparison window by a fixed factor:

| Shift | PSI | KS p-value | Severity |
| ---: | ---: | ---: | --- |
| 1.00× | 0.0002 | 0.89 | stable |
| 1.05× | 0.0039 | 2.6e-22 | stable |
| 1.10× | 0.0146 | 2.3e-82 | stable |
| 1.25× | 0.0775 | < 1e-300 | stable |
| 1.50× | 0.2401 | < 1e-300 | moderate |
| 1.60× | 0.3150 | < 1e-300 | **significant** |
| 2.00× | 0.6238 | < 1e-300 | **significant** |

A 5% shift in average transaction amount produces a KS p-value of 2.6e-22. On
significance alone that is an alert, and it would be wrong: at 69,610
observations the test detects shifts far smaller than anything worth waking
someone for. PSI puts the same shift at 0.0039 — indistinguishable from noise.

This is why `classify()` requires **both** a large effect and a significant one.
Alerting on the p-value alone produces a monitor nobody trusts within a week,
which is the same as having no monitor. The measured firing point on this data
is a 1.6× shift in mean amount.

The ordering of the signals matters too. Feature and prediction drift move
within minutes of a change; a precision regression is only visible once
investigations close, days later. The leading indicators are alerted on
*because* the lagging one cannot be measured yet — and a drift alert starts a
retraining run, never a deployment. Rules are in
[`deployment/prometheus/alerts.yml`](../deployment/prometheus/alerts.yml).

---

## 9. Promotion

The champion was promoted. This was the first run, so there was no incumbent to
compare against and only the two gates that do not require one were evaluated:

| Gate | Result | Detail |
| --- | --- | --- |
| `absolute_pr_auc` | pass | PR-AUC 0.7312 against floor 0.3000 |
| `evaluation_sample_size` | pass | 4,640 frauds in the evaluation window, minimum 50 |
| `improvement_margin` | not evaluated | no incumbent |
| `calibration` | not evaluated | no incumbent |
| `latency` | not evaluated | no incumbent |

MLflow registered `fraud-scorer` version 1 and moved the `champion` alias to it.
The three comparative gates activate on the second run; `tests/test_mlops.py`
covers each of them against a synthetic incumbent, including the case where a
challenger with better PR-AUC is refused for a calibration regression.

Retraining and promotion are deliberately separate decisions. `should_retrain()`
fires liberally — on a drift alert, a measured performance drop, or model age —
because retraining is cheap. Promotion is gated hard, because deploying a worse
model is not.

---

## 10. Failure analyses

Five bugs in this pipeline produced plausible-looking output while being badly
wrong. Three are below; the two streaming ones are in §7, next to the
measurements that exposed them. They are recorded because the debugging is the
useful part, and because every one of them would have shipped.

### The feature alignment bug

**Symptom.** PR-AUC 0.18, ROC-AUC 0.58 — barely above chance — while every
feature column had correct dtypes, a plausible range and no nulls.

**Cause.** Trailing window aggregates were computed with
`frame.groupby(key).rolling(window)`. That returns its values ordered by
*(group, timestamp)*, not by input row. Assigning them back to a time-ordered
frame gives each row a *different row's* history — the first transaction of one
customer receives another customer's count, and so on down the frame.

**How it was found.** By hand, on a four-row frame. Transaction 1 must have a
one-day count of 1.0, because nothing precedes it. It had 2.0.

**Fix.** Scatter positionally instead of by index, using the integer positions
`groupby(...).indices` already provides:

```python
for positions in frame.groupby(group_column, sort=False).indices.values():
    window = pd.Series(values[positions], index=timestamps[positions])
    out[positions] = getattr(window.rolling(offset), how)().to_numpy()
```

**Result.** PR-AUC 0.18 → 0.73.

**Guard.** `test_features_match_a_brute_force_reference` pins every feature
against an O(n²) implementation that scans every row against every other row.
It is slow and obviously correct, which is what a reference should be.

### The streaming throughput collapse

**Symptom.** The consumer scored 59 events a second. At that rate the test window
would have taken over two and a half hours.

**Two candidate causes**, both linear scans, each growing with a different
quantity: `_release_labels` rescanned the whole pending-label list on every
transaction, and the window aggregate walked backwards through an entity's
history until it left the window. Both were replaced. Which one actually
mattered is a question of measurement, and the intuitive answer is wrong.

Benchmarked separately, at the sizes this configuration actually reaches:

| Window aggregate, per event (3 windows) | Backward scan | Prefix sum | |
| ---: | ---: | ---: | ---: |
| 40 events held | 0.0052 ms | 0.0014 ms | 4× |
| 400 | 0.0414 ms | 0.0018 ms | 24× |
| 4,000 | 0.4682 ms | 0.0023 ms | 200× |
| 40,000 | 5.7015 ms | 0.0038 ms | 1,515× |

| Pending-label release, per event | Full rescan | Heap/deque drain | |
| ---: | ---: | ---: | ---: |
| 1,000 backlog | 0.0216 ms | 0.0001 ms | 158× |
| 10,000 | 0.3654 ms | 0.0001 ms | 2,551× |
| **70,000** | **6.6237 ms** | 0.0002 ms | 42,113× |
| 140,000 | 25.4950 ms | 0.0003 ms | 73,701× |

**The label backlog was the collapse.** A seven-day delay over ~9,900
transactions a day holds about 70,000 pending labels, and rescanning that list
per event costs 6.6 ms on its own — which accounts for the observed rate. After
eviction each entity holds only around 40 events, so the window scan cost 0.005
ms and was never the bottleneck.

**The prefix-sum change was still right**, for a different reason than the one it
was made for: it stops the window query degrading as retention grows. At 40,000
events held it is 1,515× faster. It just was not what fixed the throughput.

The general point: two plausible O(n) suspects, and the one that looked more
algorithmically interesting was not the one costing anything. Profiling the
components separately took a few minutes and moved the claim from wrong to
right.

### The empty model registry

**Symptom.** Training completed, MLflow recorded every run and metric, and the
model registry held nothing. Promotion appeared to succeed.

**Cause.** Two independent faults. `create_model_version` needs the `run_id` of
the run that logged the model, which was not being passed. And MLflow 3 defaults
the sklearn flavour to `skops` serialisation, which refuses to write LightGBM and
XGBoost boosters because it cannot vouch for their types — so the champion,
always one of those two, failed to log at all.

**Fix.** Dispatch to the flavour matching the model's library, pin the sklearn
path to cloudpickle so a pipeline with a non-sklearn step still serialises, and
thread `run_id` through registration. A registration failure is now logged at
error level rather than swallowed, because a silent failure leaves the serving
bundle and the registry disagreeing about what is deployed.

**A third fault, in the debugging itself.** The first repair silently did
nothing: the edit searched for `registered_name` where the code said
`registered_model_name`, and the run was declared fixed on the strength of a
printed success message rather than an assertion. A verification that cannot
fail is not a verification.

---

## 11. Limitations

What these numbers do not support:

- **They are not a production result.** Nothing here has served real traffic.
  The data is generated, so it contains only the three patterns it was told to
  contain.
- **Absolute metrics do not transfer.** PR-AUC 0.731 on this simulation says the
  pipeline extracts the signal that was injected. It says nothing about what any
  model would achieve on a real portfolio, where fraud is adversarial and changes
  *because* it is detected.
- **One seed.** The bootstrap covers sampling variation within this run, not
  variation across simulations. A multi-seed study is the first item in §12.
- **Untuned hyperparameters.** Deliberate, so the family comparison is
  like-for-like — but the ranking could change under tuning, particularly between
  XGBoost and LightGBM, which are already inseparable here.
- **Latency is single-process, single-machine.** It is a floor for what a
  deployment would see, not a service level.
- **The Kafka path was not run against a real cluster.** It is exercised against
  an in-process broker implementing the same interface and semantics, plus tests
  marked `kafka` that are deselected when no broker is reachable.
- **The Docker stack was not executed.** The images build from the Dockerfile in
  this repository and the compose file is complete, but no Docker daemon was
  available in the environment used for these measurements.
- **Drift detection is univariate.** Correlated multivariate drift that leaves
  every marginal distribution unchanged will not be caught.
- **Card precision@k is a proxy** for a fixed investigator budget, not a monetary
  loss model.

---

## 12. Next experiments

In the order they would be worth running:

1. **Cost-sensitive thresholds.** The single most valuable change, and §5 says
   why: an F1-optimal global threshold discards two thirds of a fraud class the
   model has already solved. Replacing F1 with an explicit loss — investigator
   cost against expected fraud value — and selecting the operating point on that
   would recover it. Per-scenario thresholds are the stronger version of the
   same idea.
2. **Multi-seed replication.** Ten simulation seeds, full pipeline each,
   reporting the between-seed standard deviation of PR-AUC. That is the honest
   error bar on every number above and it is currently unknown; the bootstrap
   intervals in §4 are narrower than the truth because they hold the simulation
   fixed.
3. **Feature ablation.** Retrain with each feature family removed. §5 predicts
   that dropping terminal features collapses scenario 2 — 64% of all fraud —
   while the aggregate metric moves far less. Measuring it would confirm the
   feature design rather than assuming it.
4. **Hyperparameter search.** A bounded random search per family, on validation
   only, to see whether the XGBoost/LightGBM tie survives tuning.
5. **Drift-triggered against scheduled retraining.** Compare the two policies on
   the same stream. The promotion gates already exist to make the comparison
   safe; what is missing is a long enough stream for drift to accumulate.
6. **Partition rebalancing.** Snapshot and rehydrate the online feature store so
   a consumer can take over a partition without paying the cold-start penalty
   measured in §7. This is the largest untested claim in the scaling discussion.
7. **Terminal-keyed state.** §7 shows terminal features are the ones exposed by
   cross-partition skew, because partitioning by customer gives locality to only
   one entity. A second consumer group keyed by terminal, feeding a shared risk
   store, would remove the reordering requirement entirely — at the cost of a
   second state store and a consistency question between them. Worth measuring
   against the watermark approach rather than assuming either is better.

