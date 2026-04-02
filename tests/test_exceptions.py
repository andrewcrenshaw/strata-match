"""Structured exception hierarchy (StrataMatchError and subclasses)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from strata_match.exceptions import (
    ConfigurationError,
    EmbeddingError,
    ProviderError,
    ScoringError,
    StrataMatchError,
)
from strata_match.llm_providers import OpenAILLMProvider, create_llm_provider
from strata_match.providers import OpenAIEmbeddingProvider, create_embedding_provider


@pytest.mark.verification
class TestExceptionHierarchy:
    def test_all_subclasses_inherit_strata_match_error(self) -> None:
        assert issubclass(ConfigurationError, StrataMatchError)
        assert issubclass(EmbeddingError, StrataMatchError)
        assert issubclass(ProviderError, StrataMatchError)
        assert issubclass(ScoringError, StrataMatchError)

    def test_backward_compat_value_error(self) -> None:
        err = ConfigurationError("bad")
        assert isinstance(err, ValueError)
        assert isinstance(err, Exception)

    def test_backward_compat_import_error(self) -> None:
        err = ProviderError("missing")
        assert isinstance(err, ImportError)
        assert isinstance(err, Exception)

    def test_unknown_embedding_provider_is_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError, match="Unknown embedding provider"):
            create_embedding_provider("not-a-real-provider")

    def test_unknown_llm_provider_is_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError, match="Unknown LLM provider"):
            create_llm_provider("not-a-real-provider")

    @pytest.mark.asyncio
    async def test_embedding_api_failure_wraps_embedding_error(self) -> None:
        client = MagicMock()
        client.embeddings.create = AsyncMock(side_effect=RuntimeError("API down"))
        provider = OpenAIEmbeddingProvider(client=client)
        with pytest.raises(EmbeddingError, match="OpenAI embedding request failed"):
            await provider.embed("hello")

    @pytest.mark.asyncio
    async def test_llm_completion_failure_wraps_scoring_error(self) -> None:
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=RuntimeError("timeout"))
        provider = OpenAILLMProvider(client=client)
        with pytest.raises(ScoringError, match="OpenAI chat completion failed"):
            await provider.complete([{"role": "user", "content": "hi"}])


@pytest.mark.verification
class TestPublicExportsExceptions:
    def test_exceptions_exported_from_package_root(self) -> None:
        import strata_match

        for name in (
            "StrataMatchError",
            "ConfigurationError",
            "EmbeddingError",
            "ProviderError",
            "ScoringError",
        ):
            assert hasattr(strata_match, name)
            assert name in strata_match.__all__
