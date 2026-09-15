"""Model registry and experiment tracking.

Wraps MLflow so the rest of the codebase never imports it directly, which keeps
the tracking backend swappable and makes the pipeline testable without a
tracking server.

On the backend: MLflow's filesystem store is deprecated and recent versions
refuse to start on it. A SQLite backend needs no server, works offline, and
unlike the file store supports the model registry - so a champion/challenger
workflow is available on a laptop, not only against a hosted tracking server.

Stage names follow the registry's alias convention rather than the deprecated
stage transitions: ``champion`` is what serves traffic, ``challenger`` is what
is being evaluated against it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..errors import RegistryError
from ..logging_utils import get_logger

logger = get_logger(__name__)

CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"


class ExperimentTracker:
    """Records runs, metrics, parameters and models."""

    def __init__(
        self,
        tracking_uri: str = "sqlite:///mlflow.db",
        experiment: str = "transaction-risk",
        artifact_root: str | None = None,
    ) -> None:
        try:
            import mlflow
        except ImportError as exc:  # pragma: no cover - mlflow is a hard dependency
            raise RegistryError("mlflow is not installed") from exc

        self._mlflow = mlflow
        self.tracking_uri = tracking_uri
        self.experiment = experiment

        if tracking_uri.startswith("sqlite:///"):
            # SQLite will not create intermediate directories for its file.
            db_path = Path(tracking_uri.removeprefix("sqlite:///"))
            if db_path.parent != Path():
                db_path.parent.mkdir(parents=True, exist_ok=True)

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)
        try:
            if artifact_root:
                Path(artifact_root).mkdir(parents=True, exist_ok=True)
                existing = mlflow.get_experiment_by_name(experiment)
                if existing is None:
                    mlflow.create_experiment(experiment, artifact_location=artifact_root)
            mlflow.set_experiment(experiment)
        except Exception as exc:
            raise RegistryError(f"could not initialise tracking at {tracking_uri}: {exc}") from exc

    @contextmanager
    def run(self, name: str, tags: dict[str, str] | None = None) -> Iterator[Any]:
        """Open a tracked run."""
        with self._mlflow.start_run(run_name=name, tags=tags or {}) as active:
            yield active

    def log_params(self, params: dict[str, Any]) -> None:
        # MLflow stores parameters as strings; nested values are serialised so a
        # dict of hyperparameters survives a round trip legibly.
        flat = {
            key: json.dumps(value) if isinstance(value, (dict, list)) else value
            for key, value in params.items()
        }
        self._mlflow.log_params(flat)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        numeric = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if numeric:
            self._mlflow.log_metrics(numeric, step=step)

    def log_dict(self, payload: dict[str, Any], filename: str) -> None:
        self._mlflow.log_dict(payload, filename)

    def log_model(self, model: Any, name: str, registered_name: str | None = None) -> str | None:
        """Log a fitted estimator under the flavour matching its library.

        Dispatch matters rather than being tidiness. MLflow 3 defaults the
        sklearn flavour to ``skops`` serialisation, which refuses to write
        LightGBM and XGBoost boosters because it cannot vouch for their types -
        so logging a gradient-boosted model through it fails outright and the
        registry silently ends up empty. Each library's own flavour uses that
        library's native format, and the sklearn path is pinned to cloudpickle
        so a pipeline carrying a non-sklearn step still serialises.
        """
        try:
            module = type(model).__module__
            if module.startswith("lightgbm"):
                import mlflow.lightgbm

                info = mlflow.lightgbm.log_model(
                    model, name=name, registered_model_name=registered_name
                )
            elif module.startswith("xgboost"):
                import mlflow.xgboost

                info = mlflow.xgboost.log_model(
                    model, name=name, registered_model_name=registered_name
                )
            else:
                import mlflow.sklearn

                info = mlflow.sklearn.log_model(
                    model,
                    name=name,
                    registered_model_name=registered_name,
                    serialization_format="cloudpickle",
                )
            return getattr(info, "model_uri", None)
        except Exception as exc:
            # A failure to serialise a model must not lose the run's metrics,
            # which are usually the reason the run was made.
            logger.warning("model_logging_failed", model=name, error=str(exc))
            return None

    def search_runs(self, max_results: int = 100) -> Any:
        return self._mlflow.search_runs(experiment_names=[self.experiment], max_results=max_results)


class ModelRegistry:
    """Champion/challenger aliases over the MLflow model registry."""

    def __init__(self, tracking_uri: str = "sqlite:///mlflow.db") -> None:
        try:
            from mlflow.tracking import MlflowClient
        except ImportError as exc:  # pragma: no cover - mlflow is a hard dependency
            raise RegistryError("mlflow is not installed") from exc
        self._client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)

    def ensure_model(self, name: str) -> None:
        try:
            self._client.get_registered_model(name)
        except Exception:
            self._client.create_registered_model(name)

    def register(
        self, model_uri: str, name: str, *, run_id: str | None = None, description: str = ""
    ) -> str:
        """Register a new version and return its version number.

        ``run_id`` links the version back to the run that produced it, which is
        how a deployed model is traced to its metrics and parameters. Recent
        MLflow versions also need the link to resolve a ``models:/`` URI, so
        omitting it fails rather than merely losing the lineage.
        """
        self.ensure_model(name)
        try:
            version = self._client.create_model_version(
                name=name, source=model_uri, run_id=run_id, description=description
            )
        except Exception as exc:
            raise RegistryError(f"could not register {model_uri} as {name}: {exc}") from exc
        return str(version.version)

    def set_alias(self, name: str, alias: str, version: str) -> None:
        try:
            self._client.set_registered_model_alias(name, alias, version)
        except Exception as exc:
            raise RegistryError(f"could not set alias {alias} on {name} v{version}: {exc}") from exc
        logger.info("registry_alias_set", model=name, alias=alias, version=version)

    def get_alias_version(self, name: str, alias: str) -> str | None:
        try:
            return str(self._client.get_model_version_by_alias(name, alias).version)
        except Exception:
            return None

    def promote(self, name: str, version: str) -> dict[str, str | None]:
        """Make ``version`` champion, demoting the incumbent to challenger.

        The previous champion keeps an alias rather than being deleted, which is
        what makes a rollback a single alias move instead of a retrain.
        """
        previous = self.get_alias_version(name, CHAMPION_ALIAS)
        self.set_alias(name, CHAMPION_ALIAS, version)
        if previous and previous != version:
            self.set_alias(name, CHALLENGER_ALIAS, previous)
        return {"champion": version, "previous_champion": previous}

    def rollback(self, name: str) -> dict[str, str | None]:
        """Swap champion and challenger, for when a promotion goes wrong."""
        champion = self.get_alias_version(name, CHAMPION_ALIAS)
        challenger = self.get_alias_version(name, CHALLENGER_ALIAS)
        if challenger is None:
            raise RegistryError(f"{name} has no challenger to roll back to")
        self.set_alias(name, CHAMPION_ALIAS, challenger)
        if champion:
            self.set_alias(name, CHALLENGER_ALIAS, champion)
        return {"champion": challenger, "demoted": champion}

    def versions(self, name: str) -> list[dict[str, Any]]:
        """Every registered version, with the aliases actually pointing at it.

        ``search_model_versions`` leaves ``aliases`` empty on the SQL store, so
        reading it from the version rows reports no version deployed even when
        one is. Aliases live on the registered model, so they are resolved from
        there and attached here.
        """
        try:
            found = self._client.search_model_versions(f"name='{name}'")
        except Exception:
            return []
        by_version: dict[str, list[str]] = {}
        try:
            for alias, version in (self._client.get_registered_model(name).aliases or {}).items():
                by_version.setdefault(str(version), []).append(str(alias))
        except Exception:
            # An alias lookup failure must not hide the versions themselves.
            logger.warning("alias_lookup_failed", model=name)
        return [
            {
                "version": str(v.version),
                "run_id": v.run_id,
                "status": v.status,
                "aliases": sorted(by_version.get(str(v.version), [])),
            }
            for v in found
        ]
