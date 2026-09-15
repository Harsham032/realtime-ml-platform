# realtime-ml-platform

A streaming machine learning platform for real-time transaction risk scoring. It
covers the whole path a fraud model actually has to survive: generating a
labelled transaction stream, building leakage-free time-window features,
training and comparing model families on a temporal split, tracking every run,
gating promotion behind measurable criteria, replaying the stream through a
partitioned broker, serving scores over HTTP, and watching the feature
distribution for drift.

The dataset is generated, not collected. Every number in this repository comes
from a run of the code in it; nothing is projected, rounded up, or borrowed from
a paper. See [`docs/results.md`](docs/results.md) for the measurements and
[`docs/methodology.md`](docs/methodology.md) for how they were produced.

---

## Why this problem is harder than it looks

Card fraud detection is the canonical example of a task where the obvious
approach quietly fails:

- **The positive class is rare.** Fraud runs below 1% of transactions, so
  accuracy is meaningless and ROC-AUC is optimistic. The primary metric here is
  **PR-AUC**, reported with a bootstrap confidence interval.
- **Labels arrive late.** A chargeback is not known the moment a transaction is
  scored. Training on data that includes labels a deployed model could not have
  had inflates every metric. The split therefore inserts a **delay period**
  between train and test.
- **The strongest features are aggregates over time.** Customer spend over the
  last 7 days, terminal fraud rate over the last 30 — computed carelessly, these
  leak the future or, worse, attach one entity's history to another entity's
  row. That second failure is silent: correct dtypes, correct ranges, no nulls,
  and a destroyed signal. It cost this project a 0.55 drop in PR-AUC before it
  was caught (see [`docs/results.md`](docs/results.md#the-feature-alignment-bug)).
- **Batch and online must agree.** The features computed over a dataframe during
  training and the features computed incrementally per event at serving time are
  two independent implementations of the same definition. If they disagree, the
  model is served inputs it was never trained on. Here they are tested against
  each other directly.

---

## What is in the box

| Layer | Module | What it does |
|---|---|---|
| Data | `rtml.data.simulator` | Vectorised transaction generator with three injected fraud scenarios |
| Features (batch) | `rtml.features.engineering` | Trailing time-window aggregates over a dataframe, in original row order |
| Features (online) | `rtml.features.online` | Incremental per-entity state with prefix-sum window queries |
| Splitting | `rtml.training.splitting` | Temporal split with a label-availability delay |
| Models | `rtml.models.estimators` | Logistic regression, random forest, XGBoost, LightGBM, isolation forest |
| Evaluation | `rtml.evaluation` | PR-AUC, ROC-AUC, precision@k per day, Brier score, bootstrap CIs, threshold selection |
| Tracking | `rtml.models.registry` | MLflow experiment tracking and model registry, with a champion alias |
| Promotion | `rtml.training.promotion` | Five explicit gates; a challenger is held unless all pass |
| Streaming | `rtml.streaming` | Partitioned broker abstraction (in-process or Kafka), producer, scoring consumer |
| Serving | `rtml.services.api` | FastAPI scoring service with Prometheus metrics |
| Monitoring | `rtml.monitoring.drift` | PSI and Kolmogorov–Smirnov drift detection per feature |
| Persistence | `rtml.data.store` | SQLAlchemy store for predictions, labels and deployments |

---

## Quickstart

Requires Python 3.11 or newer.

```bash
git clone https://github.com/Harsham032/realtime-ml-platform.git
cd realtime-ml-platform
make install
```

Run the whole pipeline on the reduced configuration (a few minutes, no external
services required):

```bash
make data-fast
make train-fast
```

Or the full pipeline, which is what `docs/results.md` reports:

```bash
make pipeline        # data -> train -> stream -> drift
```

Individual stages:

```bash
make data            # generate transactions and the feature table
make train           # train every family, track runs, gate promotion
make stream          # replay the test period through the broker
make drift           # compare two windows and write a drift report
make serve           # start the scoring API on :8000
make mlflow-ui       # browse tracked runs at :5000
```

Quality gates:

```bash
make check           # ruff + black --check + mypy + pytest
make coverage        # tests with a coverage report
make secrets-scan    # look for credential-shaped strings in tracked files
```

`make help` lists every target.

---

## Results

Measured on the full configuration (`configs/default.yaml`): 1,817,275
generated transactions over 183 days, of which 973,244 form the training set and
556,547 the test set, separated by a 7-day label-availability delay. Test-set
fraud rate 0.83% (4,640 frauds).

| Model | PR-AUC | 95% CI | ROC-AUC | Precision | Recall | p95 latency |
| --- | ---: | :---: | ---: | ---: | ---: | ---: |
| **lightgbm** (promoted) | **0.7312** | 0.7187 – 0.7431 | 0.9005 | 0.871 | 0.621 | 0.85 ms |
| xgboost | 0.7284 | 0.7155 – 0.7408 | 0.8985 | 0.883 | 0.612 | 0.72 ms |
| random_forest | 0.7053 | 0.6920 – 0.7187 | **0.9012** | 0.857 | 0.650 | 68.45 ms |
| logistic_regression | 0.6122 | 0.5980 – 0.6267 | 0.8926 | 0.732 | 0.553 | 0.56 ms |
| isolation_forest (unsupervised) | 0.2847 | 0.2719 – 0.2989 | 0.8442 | 0.466 | 0.436 | 11.47 ms |

PR-AUC 0.731 against a 0.834% base rate is 88× the no-skill baseline. Three
things in that table are worth more than the headline number:

- **Ranking on ROC-AUC picks the wrong model.** Random forest wins on ROC-AUC,
  places third on PR-AUC, and is 81× slower to serve.
- **XGBoost and LightGBM are not separable here.** Their confidence intervals
  overlap almost entirely; LightGBM was promoted on fit time and latency, not on
  a metric difference this data can support.
- **The threshold costs more than the model choice.** At the deployed operating
  point the champion catches 31% of the simplest fraud scenario — one it scores
  above 0.5 in 99.7% of cases. A single global F1-optimal threshold throws away
  a class the model has already solved. `docs/results.md` §5 has the numbers.

Streamed through the broker and consumer, the same model reproduces the batch
evaluation exactly — PR-AUC 0.731191 either way, 3,307 alerts either way, at
1,076 events/s and 0.998 ms p95. Reaching that took finding two bugs that
produced entirely plausible output and cost 0.24 PR-AUC between them; both are
written up in `docs/results.md` §7.

Full numbers, confidence intervals, environment, and the failure analyses are in
[`docs/results.md`](docs/results.md).

---

## Architecture

```
                 ┌────────────────────────┐
                 │  simulator / raw feed  │
                 └───────────┬────────────┘
                             │ transactions
            ┌────────────────┴─────────────────┐
            │                                  │
   ┌────────▼─────────┐              ┌─────────▼──────────┐
   │ batch features   │              │ producer           │
   │ (dataframe)      │              │ partition by       │
   └────────┬─────────┘              │ customer_id        │
            │                        └─────────┬──────────┘
   ┌────────▼─────────┐                        │
   │ temporal split   │              ┌─────────▼──────────┐
   │ train/val/test   │              │ broker             │
   └────────┬─────────┘              │ in-process | Kafka │
            │                        └─────────┬──────────┘
   ┌────────▼─────────┐                        │
   │ train_all        │              ┌─────────▼──────────┐
   │ 5 families       │              │ scoring consumer   │
   └────────┬─────────┘              │ online features    │
            │                        │ + champion model   │
   ┌────────▼─────────┐              └─────────┬──────────┘
   │ MLflow tracking  │◄───────────────────────┤
   │ + registry       │   champion alias       │
   └────────┬─────────┘                        │
            │                        ┌─────────▼──────────┐
   ┌────────▼─────────┐              │ prediction store   │
   │ promotion gates  │              │ + label feedback   │
   │ 5 criteria       │              └─────────┬──────────┘
   └────────┬─────────┘                        │
            │ promote                ┌─────────▼──────────┐
   ┌────────▼─────────┐              │ drift monitor      │
   │ serving bundle   │              │ PSI + KS per feat. │
   │ + FastAPI        │              └────────────────────┘
   └──────────────────┘
```

The same feature definitions drive both paths. `tests/test_features.py` asserts
that the online store reproduces the batch table row for row.

See [`docs/architecture.md`](docs/architecture.md) for the component-level
detail and the design decisions behind each boundary.

---

## The scoring API

```bash
make serve
```

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness, model loaded, online store size |
| `POST` | `/score` | Score one transaction |
| `POST` | `/score/batch` | Score a batch in one call |
| `POST` | `/labels` | Submit an outcome label for a scored transaction |
| `GET` | `/model` | Champion name, version, threshold, feature order |
| `GET` | `/drift` | Current drift status per feature |
| `GET` | `/metrics` | Prometheus exposition |

```bash
curl -s localhost:8000/score -H 'content-type: application/json' -d '{
  "transaction_id": "tx-1",
  "customer_id": 42,
  "terminal_id": 900,
  "tx_amount": 87.5,
  "tx_datetime": "2025-06-01T23:14:00"
}'
```

The response carries the score, the decision at the deployed threshold, the
feature vector actually used, and the model version that produced it — enough to
reconstruct any decision after the fact.

Interactive documentation is at `/docs` when the service is running.

---

## Running with Docker

```bash
docker compose up --build
```

Brings up PostgreSQL, Kafka, the scoring API, Prometheus and Grafana. Set
`RTML_STREAM_BACKEND=kafka` to route the stream through the broker rather than
the in-process implementation. Configuration is in
[`docker-compose.yml`](docker-compose.yml); alert rules are in
[`deployment/prometheus/alerts.yml`](deployment/prometheus/alerts.yml).

> The container images in this repository build and the compose file is complete,
> but no Docker daemon was available in the environment used to produce
> `docs/results.md`, so the stack has not been executed end to end. The
> measurements there come from the local pipeline, not from the compose stack.

---

## Configuration

Two kinds of setting, deliberately kept apart.

**Pipeline configuration** — what an experiment does — lives in YAML.
`configs/default.yaml` is the full run; `configs/fast.yaml` is the reduced run
used by CI and by the examples. Override any key on the command line:

```bash
python scripts/train.py --config configs/default.yaml --set models.enabled=lightgbm
```

**Deployment settings** — where things connect — come from `RTML_*` environment
variables, so a container needs no file edits:

```bash
export RTML_STREAM_BACKEND=kafka
export RTML_KAFKA_BOOTSTRAP_SERVERS=broker:9092
```

`RTML_DATABASE_URL` takes a SQLAlchemy URL — `sqlite:///...` locally,
`postgresql+psycopg://...` in deployment. Keep it in a secret store, not in a
shell history: the full list of variables and their defaults is in
[`.env.example`](.env.example).

Secrets are never read from YAML. Copy [`.env.example`](.env.example) to `.env`
and fill in the placeholders; `.env` is gitignored and no credential is
committed anywhere in this repository.

---

## Repository layout

```
configs/         run configurations (full and reduced)
data/            generated datasets (gitignored; see data/README.md)
deployment/      Prometheus scrape config, alert rules, Grafana datasource
docs/            architecture, methodology, data sources, results
examples/        a runnable end-to-end walkthrough
notebooks/       exploratory analysis of the generated data
scripts/         data generation, training, streaming, drift, secrets scan
src/rtml/        the package
tests/           unit and integration tests
```

---

## Development

```bash
make install     # venv + dependencies + editable install
make fmt         # black + ruff --fix
make check       # lint + typecheck + tests
```

Continuous integration runs the test suite and the quality gates on every push
and pull request; see [`.github/workflows`](.github/workflows).

Conventions: PEP 8 via black (line length 100), ruff for linting and import
order, mypy with `disallow_untyped_defs`, structured logging via `structlog`,
deterministic seeds everywhere, and conventional commit messages.

---

## Limitations

Stated plainly, because they bound what the numbers mean:

- **The data is simulated.** The generator follows the scenario design of the
  *Reproducible Machine Learning for Credit Card Fraud Detection* handbook. Real
  fraud is adversarial and non-stationary in ways no generator reproduces, so
  absolute metrics here transfer to real portfolios only as a lower bound on
  engineering correctness, not as an expected production score.
- **No production deployment exists.** Nothing in this repository has served real
  traffic. Latency figures are single-process measurements on one machine, and
  the environment is recorded in `docs/results.md`.
- **The Kafka path is tested against an in-process broker** implementing the same
  interface, plus tests marked `kafka` that are deselected unless a broker is
  reachable. The abstraction is exercised; a real cluster was not available.
- **Drift detection is univariate.** PSI and KS operate per feature; correlated
  multivariate drift that leaves every marginal unchanged will not be caught.
- **Cost-sensitive evaluation is approximate.** Precision@k per day is a proxy
  for a fixed investigator budget, not a monetary loss model.

---

## Further reading

- [`docs/methodology.md`](docs/methodology.md) — split design, feature
  definitions, metric choices, promotion gates
- [`docs/results.md`](docs/results.md) — measured results, environment,
  interpretation, failure analyses, next experiments
- [`docs/architecture.md`](docs/architecture.md) — components and data flow
- [`docs/data-sources.md`](docs/data-sources.md) — where the data comes from and
  why it is generated
- [`examples/`](examples/) — a runnable walkthrough of the whole loop

---

## License

[MIT](LICENSE)
