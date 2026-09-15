"""Configuration loading, validation and overrides."""

from __future__ import annotations

import pytest

from rtml.config import PipelineConfig, Settings
from rtml.errors import ConfigurationError


def test_shipped_configs_load(repo_root) -> None:
    for name in ("default.yaml", "fast.yaml"):
        config = PipelineConfig.from_yaml(repo_root / "configs" / name)
        assert config.split.total_days <= config.simulation.n_days


def test_missing_file_raises() -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        PipelineConfig.from_yaml("configs/nope.yaml")


def test_overrides_parse_scalars_and_lists(repo_root) -> None:
    config = PipelineConfig.from_yaml(repo_root / "configs" / "fast.yaml")
    updated = config.with_overrides({"models.enabled": "[lightgbm]", "run.seed": "7"})
    assert updated.models.enabled == ["lightgbm"]
    assert updated.run.seed == 7
    # The original is untouched, so a sweep cannot corrupt its base.
    assert config.run.seed != 7


def test_unknown_override_key_raises(repo_root) -> None:
    config = PipelineConfig.from_yaml(repo_root / "configs" / "fast.yaml")
    with pytest.raises(ConfigurationError, match="unknown configuration key"):
        config.with_overrides({"models.nonexistent": "1"})


def test_a_split_longer_than_the_simulation_is_rejected(repo_root) -> None:
    """Catches the mistake at config time rather than after generating data."""
    config = PipelineConfig.from_yaml(repo_root / "configs" / "fast.yaml")
    with pytest.raises(ConfigurationError):
        config.with_overrides({"split.train_days": "900"})


def test_validation_must_fit_inside_training(repo_root) -> None:
    config = PipelineConfig.from_yaml(repo_root / "configs" / "fast.yaml")
    with pytest.raises(ConfigurationError):
        config.with_overrides({"split.validation_days": "999"})


def test_drift_bands_must_be_ordered(repo_root) -> None:
    config = PipelineConfig.from_yaml(repo_root / "configs" / "fast.yaml")
    with pytest.raises(ConfigurationError):
        config.with_overrides({"drift.psi_warn": "0.9", "drift.psi_alert": "0.1"})


def test_settings_read_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RTML_API_PORT", "9100")
    monkeypatch.setenv("RTML_STREAM_BACKEND", "kafka")
    settings = Settings(_env_file=None)
    assert settings.api_port == 9100
    assert settings.stream_backend == "kafka"


def test_settings_default_to_an_offline_stack() -> None:
    """Defaults must need no external service, or nothing is reproducible."""
    settings = Settings(_env_file=None)
    assert settings.stream_backend == "inprocess"
    assert settings.database_url.startswith("sqlite:///")
    assert settings.mlflow_tracking_uri.startswith("sqlite:///")


def test_logging_level_can_change_after_first_use(capsys: pytest.CaptureFixture[str]) -> None:
    """A script's --log-level must take effect even though modules bind at import."""
    from rtml.logging_utils import configure_logging, get_logger

    configure_logging("INFO")
    logger = get_logger("level-test")
    logger.info("visible_at_info")
    assert "visible_at_info" in capsys.readouterr().out

    configure_logging("WARNING")
    logger.info("hidden_at_warning")
    logger.warning("visible_at_warning")
    captured = capsys.readouterr().out
    assert "hidden_at_warning" not in captured
    assert "visible_at_warning" in captured
    configure_logging("INFO")
