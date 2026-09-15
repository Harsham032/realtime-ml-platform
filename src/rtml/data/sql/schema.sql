-- PostgreSQL schema for transactions, predictions, drift and model metadata.
-- The SQLAlchemy Core definitions in store.py create an equivalent schema on
-- SQLite so tests and the local demo need no running service.

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id   BIGINT       PRIMARY KEY,
    customer_id      BIGINT       NOT NULL,
    terminal_id      BIGINT       NOT NULL,
    tx_datetime      TIMESTAMPTZ  NOT NULL,
    tx_day           INTEGER      NOT NULL,
    tx_amount        DOUBLE PRECISION NOT NULL,
    is_fraud         SMALLINT     NOT NULL DEFAULT 0,
    fraud_scenario   SMALLINT     NOT NULL DEFAULT 0
);

-- Scoring is keyed by customer and ordered by time, so these two indexes carry
-- the online feature lookups.
CREATE INDEX IF NOT EXISTS transactions_customer_time_idx ON transactions (customer_id, tx_datetime DESC);
CREATE INDEX IF NOT EXISTS transactions_terminal_time_idx ON transactions (terminal_id, tx_datetime DESC);
CREATE INDEX IF NOT EXISTS transactions_day_idx           ON transactions (tx_day);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id    BIGSERIAL    PRIMARY KEY,
    transaction_id   BIGINT       NOT NULL,
    customer_id      BIGINT       NOT NULL,
    terminal_id      BIGINT       NOT NULL,
    tx_datetime      TIMESTAMPTZ  NOT NULL,
    tx_day           INTEGER      NOT NULL,
    tx_amount        DOUBLE PRECISION NOT NULL,
    score            DOUBLE PRECISION NOT NULL,
    threshold        DOUBLE PRECISION NOT NULL,
    is_alert         BOOLEAN      NOT NULL,
    model_name       TEXT         NOT NULL DEFAULT '',
    model_version    TEXT         NOT NULL DEFAULT '',
    latency_ms       DOUBLE PRECISION NOT NULL DEFAULT 0,
    -- Nullable on purpose: the label does not exist when the prediction is
    -- written, and is backfilled once an investigation closes.
    is_fraud         SMALLINT,
    scored_at        TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS predictions_tx_idx     ON predictions (transaction_id);
CREATE INDEX IF NOT EXISTS predictions_day_idx    ON predictions (tx_day);
CREATE INDEX IF NOT EXISTS predictions_alert_idx  ON predictions (is_alert, tx_day);

-- Drift observations, retained so a performance regression can be correlated
-- against what the inputs were doing at the time.
CREATE TABLE IF NOT EXISTS drift_observations (
    observation_id   BIGSERIAL    PRIMARY KEY,
    computed_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    feature          TEXT         NOT NULL,
    psi              DOUBLE PRECISION NOT NULL,
    ks_statistic     DOUBLE PRECISION NOT NULL,
    ks_pvalue        DOUBLE PRECISION NOT NULL,
    severity         TEXT         NOT NULL,
    reference_start  INTEGER      NOT NULL,
    reference_end    INTEGER      NOT NULL,
    comparison_start INTEGER      NOT NULL,
    comparison_end   INTEGER      NOT NULL
);

CREATE INDEX IF NOT EXISTS drift_feature_idx ON drift_observations (feature, computed_at DESC);

-- What was deployed when, so a metric can be attributed to a model version.
CREATE TABLE IF NOT EXISTS model_deployments (
    deployment_id    BIGSERIAL    PRIMARY KEY,
    model_name       TEXT         NOT NULL,
    model_version    TEXT         NOT NULL,
    alias            TEXT         NOT NULL,
    threshold        DOUBLE PRECISION NOT NULL,
    pr_auc           DOUBLE PRECISION,
    deployed_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    promotion_reason TEXT         NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS model_deployments_idx ON model_deployments (model_name, deployed_at DESC);
