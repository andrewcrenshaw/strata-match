"""Structured errors for strata-match.

All public errors inherit from :class:`StrataMatchError`. Where appropriate,
types also inherit from built-in exceptions (:class:`ValueError`,
:class:`ImportError`) so existing ``except ValueError`` / ``except ImportError``
handlers remain valid.
"""

from __future__ import annotations


class StrataMatchError(Exception):
    """Base class for all strata-match errors."""


class ConfigurationError(StrataMatchError, ValueError):
    """Invalid configuration (e.g. unknown provider name, bad parameters)."""


class ProviderError(StrataMatchError, ImportError):
    """Provider initialization failed (e.g. missing optional dependency)."""


class EmbeddingError(StrataMatchError):
    """Embedding API call failed (network, auth, rate limits, etc.)."""


class ScoringError(StrataMatchError):
    """LLM chat-completion / scoring call failed."""
