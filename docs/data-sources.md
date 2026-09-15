# Data source

## The generation procedure

Transactions are produced by a simulator implementing the procedure described
in:

> Le Borgne, Y.-A., Siblini, W., Lebichot, B. and Bontempi, G. (2022).
> *Reproducible Machine Learning for Credit Card Fraud Detection — Practical
> Handbook.* Université Libre de Bruxelles.
> https://fraud-detection-handbook.github.io/fraud-detection-handbook/

| | |
| --- | --- |
| Licence | The handbook is published openly for research and teaching |
| What is used | The generation procedure and the feature methodology, reimplemented |
| What is not used | No code or data file is copied or redistributed |
| Network access | None. Nothing is downloaded at any point |

The reference implementation loops over customers and days in Python. This one
draws the same distributions with the same parameters vectorised over NumPy,
which makes million-transaction runs practical: **1,817,275 transactions in 3.1
seconds** measured, with the feature table taking a further 55 seconds. No
comparison against the reference implementation's runtime is given here because
it was not run; only the absolute figure is measured.

## The procedure

**Customers** get a uniform position on a 100×100 grid, a mean spend drawn
uniformly from [5, 100] with standard deviation half the mean, and a mean
transaction rate drawn uniformly from [0, 4] per day.

**Terminals** get a uniform position on the same grid. A customer may transact
at any terminal within radius 5, which produces realistic spatial locality —
most customers use a handful of terminals, a few use many.

**Transactions** per customer per day are Poisson distributed at that customer's
rate. Times are Gaussian about midday, resampled if they fall outside the day.
Amounts are Gaussian about the customer's mean, resampled uniformly when
negative.

**Fraud** is injected through three scenarios, each modelling a different
compromise and each requiring different features to detect:

| Scenario | Pattern | What detects it |
| --- | --- | --- |
| 1 | Any amount above 220 is fraudulent | The transaction alone |
| 2 | Two terminals compromised per day, every transaction on them fraudulent for 28 days | Terminal history |
| 3 | Three customers per day have credentials stolen; for 14 days a third of their transactions are inflated 5× and fraudulent | Customer spending history |

This is why all three matter: a pipeline that only computes transaction-level
features catches scenario 1 and nothing else, and its aggregate score still
looks respectable. `docs/results.md` reports per-scenario recall so that failure
cannot hide.

## What a simulator can and cannot tell you

**Can**: whether the pipeline is correct end to end; whether features are
computed without leakage; whether online and batch agree; whether the streaming
path keeps up; how models compare on a controlled, reproducible problem.

**Cannot**: how any of this performs on real fraud. Real fraud is adversarial
and non-stationary — patterns change specifically because they are being
detected — and no generator reproduces that. Concept drift here is whatever the
scenarios happen to produce, not what an adversary would do.

Treat every number in this repository as a measurement of the *pipeline*, not a
forecast of production performance.

## Alternatives considered

| Dataset | Why not |
| --- | --- |
| ULB Credit Card Fraud (Kaggle) | Features are anonymised PCA components. There is no customer or terminal identity, so no entity-level aggregate features are possible — the entire feature methodology here would be impossible |
| IEEE-CIS Fraud Detection | Licence restricts redistribution; a single static extract with limited time structure |
| PaySim | Simulates mobile money transfers, a different transaction shape, and has no terminal concept |

## Handling rules

1. Generated data lands in `data/processed/`, which is git-ignored.
2. Model artifacts land in `artifacts/` and `mlartifacts/`, also ignored.
3. MLflow's SQLite database is ignored.
4. CI fails if any tracked file exceeds 2MB or matches a generated-artifact
   pattern.
5. Credentials are read from the environment; `.env.example` carries
   placeholders only and `.env` is never committed.
