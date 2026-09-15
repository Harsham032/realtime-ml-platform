"""Exception hierarchy.

Every failure raised by library code derives from :class:`RtmlError`, so callers
can tell an expected domain failure from a genuine bug.
"""

from __future__ import annotations


class RtmlError(Exception):
    """Base class for every error raised by this package."""


class ConfigurationError(RtmlError):
    """Configuration is missing, malformed or internally inconsistent."""


class DataGenerationError(RtmlError):
    """The transaction simulator was given parameters it cannot satisfy."""


class FeatureError(RtmlError):
    """Feature computation failed or was given inconsistent inputs."""


class ModelError(RtmlError):
    """A model could not be trained, loaded or scored."""


class RegistryError(RtmlError):
    """A model registry operation failed."""


class StreamError(RtmlError):
    """A streaming produce or consume operation failed."""


class ValidationGateError(RtmlError):
    """A candidate model failed a promotion gate."""


class EvaluationError(RtmlError):
    """An evaluation run was given inconsistent inputs."""
