# Data directory

## Layout

```
data/
├── raw/          ignored - generator output, if written separately
├── interim/      ignored - cleaned intermediates
├── processed/    ignored - transactions.parquet, features.parquet, the SQLite database
└── README.md     this file
```

**Nothing here is committed.** Unlike a project built around a downloaded
dataset, this one has nothing to commit: the data is produced from a seed, so
regenerating it is faster than cloning it would be.

```bash
make data          # configs/default.yaml  - 1,817,275 transactions, ~58 seconds
make data-fast     # configs/fast.yaml     -   271,427 transactions, ~15 seconds (fraud rate 2.36%)
```

Almost all of that is the feature table, not the transactions: at the default
configuration the simulator itself takes 3.1 seconds and the trailing time-window
aggregates take 55.

CI fails the build if any tracked file exceeds 2MB, or if a `.parquet`,
`.joblib`, `.sqlite3` or `mlflow.db` is ever added to the index.

## What gets generated

At the default configuration:

| | |
| --- | --- |
| Transactions | ~1,817,000 |
| Customers | 5,000 |
| Terminals | 10,000 |
| Period | 183 days from 2025-01-01 |
| Fraud rate | ~0.81% |
| Files | `transactions.parquet` (~50MB), `features.parquet` (~64MB) |

## Schema

`transactions.parquet`:

| column | type | meaning |
| --- | --- | --- |
| `transaction_id` | int64 | Unique, contiguous from 0 |
| `customer_id` | int64 | Cardholder |
| `terminal_id` | int64 | Point of sale |
| `tx_datetime` | datetime64 | When it happened |
| `tx_day` | int32 | Days since the simulation start, used for temporal splitting |
| `tx_time_seconds` | int64 | Seconds into the day |
| `tx_amount` | float64 | Amount |
| `is_fraud` | int8 | The label |
| `fraud_scenario` | int8 | 0 legitimate, 1-3 which pattern produced it |

`features.parquet` adds the 15 model inputs listed in `docs/methodology.md`.

`fraud_scenario` exists so results can be broken down by pattern. A model that
scores well overall while missing every scenario-2 case has a different problem
from one that misses scenario 3, and the aggregate hides which.

## Why a simulator

The generation procedure follows *Reproducible Machine Learning for Credit Card
Fraud Detection* (Le Borgne, Siblini, Lebichot and Bontempi, Université Libre de
Bruxelles, 2022) — see `docs/data-sources.md` for the full citation and the
procedure.

Real card transaction data is not publicly redistributable. Every public
alternative is either heavily anonymised into PCA components, which destroys the
entity structure this platform's features are built on, or is a small static
extract with no time dimension worth splitting.

A generator gives what neither can: reproducibility from a seed, arbitrary
scale, and known ground truth about *which* pattern produced each fraud.

**The cost, stated plainly.** The simulator produces the fraud patterns it was
told to produce. A model's score here measures whether the pipeline can recover
known injected structure — not whether it would catch real fraud, which is
adversarial, non-stationary, and does not announce its scenario. Section 11 of
`docs/results.md` sets out what that does and does not license you to conclude.
