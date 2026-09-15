"""Structured logging.

Key/value output in development, JSON everywhere else, so logs can be shipped to
an aggregator without reparsing.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_CONFIGURED = False
_LEVEL: int | None = None


def configure_logging(level: str = "INFO", *, json_output: bool = False) -> None:
    """Configure the standard library and ``structlog``.

    Reconfigures when the requested level differs from the active one. Modules
    bind loggers at import time with the default level, so an entry point that
    later asks for a different level - a script's ``--log-level`` flag, or the
    service reading ``RTML_LOG_LEVEL`` - would otherwise be silently ignored.
    """
    global _CONFIGURED, _LEVEL

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    if _CONFIGURED and numeric_level == _LEVEL:
        return

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=numeric_level, force=True)

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )
    _CONFIGURED = True
    _LEVEL = numeric_level


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger, configuring logging on first use."""
    configure_logging()
    return structlog.get_logger(name)  # type: ignore[no-any-return]
