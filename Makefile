PYTHON ?= python3.11
VENV   ?= .venv
BIN    := $(VENV)/bin
CONFIG ?= configs/default.yaml
FAST   := configs/fast.yaml

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

.PHONY: install
install: $(BIN)/python ## Install runtime and development dependencies
	$(BIN)/pip install -r requirements-dev.txt
	$(BIN)/pip install -e .

.PHONY: fmt
fmt: ## Apply formatting and import ordering
	$(BIN)/black src tests scripts examples
	$(BIN)/ruff check --fix src tests scripts examples

.PHONY: lint
lint: ## Run linters and formatting checks
	$(BIN)/ruff check src tests scripts examples
	$(BIN)/black --check src tests scripts examples

.PHONY: typecheck
typecheck: ## Run static type analysis
	$(BIN)/mypy

.PHONY: test
test: ## Run the test suite
	$(BIN)/pytest

.PHONY: coverage
coverage: ## Run tests with a coverage report
	$(BIN)/pytest --cov=rtml --cov-report=term-missing

.PHONY: check
check: lint typecheck test ## Run every quality gate

.PHONY: data
data: ## Generate the dataset and feature table
	$(BIN)/python scripts/generate_data.py --config $(CONFIG)

.PHONY: data-fast
data-fast: ## Generate the reduced dataset used by CI
	$(BIN)/python scripts/generate_data.py --config $(FAST)

.PHONY: train
train: ## Train every model, track runs and gate a promotion
	$(BIN)/python scripts/train.py --config $(CONFIG)

.PHONY: train-fast
train-fast: ## Train on the reduced dataset
	$(BIN)/python scripts/train.py --config $(FAST)

.PHONY: stream
stream: ## Run the streaming path end to end
	$(BIN)/python scripts/run_stream.py --config $(CONFIG) --write-database

.PHONY: drift
drift: ## Produce a drift report
	$(BIN)/python scripts/drift_report.py --config $(CONFIG)

.PHONY: pipeline
pipeline: data train stream drift ## Run the whole pipeline in order

.PHONY: serve
serve: ## Start the scoring API locally
	$(BIN)/uvicorn rtml.services.api:app --reload --host 0.0.0.0 --port 8000

.PHONY: mlflow-ui
mlflow-ui: ## Browse tracked experiments
	$(BIN)/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

.PHONY: docker-build
docker-build: ## Build the service container image
	docker build -t rtml:local .

.PHONY: up
up: ## Start PostgreSQL, Kafka and the API
	docker compose up --build

.PHONY: down
down: ## Stop and remove the local stack
	docker compose down -v

.PHONY: secrets-scan
secrets-scan: ## Look for credential-shaped strings in tracked files
	$(BIN)/python scripts/secrets_scan.py

.PHONY: clean
clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
