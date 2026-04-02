"""Tests for concrete embedding providers and the provider factory."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from strata_match.embeddings import EmbeddingProvider, EmbeddingProviderType
from strata_match.exceptions import ConfigurationError
from strata_match.providers import (
    GeminiEmbeddingProvider,
    OllamaEmbeddingProvider,
    OpenAIEmbeddingProvider,
    create_embedding_provider,
)


def _make_embedding(dim: int, value: float = 0.01) -> list[float]:
    return [value] * dim


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------


def _mock_openai_client(embeddings: list[list[float]]) -> MagicMock:
    client = MagicMock()
    resp = MagicMock()
    resp.data = [MagicMock(embedding=e) for e in embeddings]
    client.embeddings.create = AsyncMock(return_value=resp)
    return client


@pytest.mark.verification
class TestOpenAIEmbeddingProvider:
    def test_is_embedding_provider(self) -> None:
        client = _mock_openai_client([_make_embedding(1536)])
        provider = OpenAIEmbeddingProvider(client=client)
        assert isinstance(provider, EmbeddingProvider)

    def test_dimension_default(self) -> None:
        provider = OpenAIEmbeddingProvider(client=MagicMock())
        assert provider.dimension == 1536

    def test_custom_dimension(self) -> None:
        provider = OpenAIEmbeddingProvider(client=MagicMock(), dimension=512)
        assert provider.dimension == 512

    @pytest.mark.asyncio
    async def test_embed_returns_correct_shape(self) -> None:
        client = _mock_openai_client([_make_embedding(1536)])
        provider = OpenAIEmbeddingProvider(client=client)
        vec = await provider.embed("test text")
        assert vec.shape == (1536,)
        assert vec.dtype == np.float32

    @pytest.mark.asyncio
    async def test_embed_calls_api_with_correct_params(self) -> None:
        client = _mock_openai_client([_make_embedding(1536)])
        provider = OpenAIEmbeddingProvider(
            client=client, model="text-embedding-3-small", dimension=1536
        )
        await provider.embed("hello world")
        client.embeddings.create.assert_awaited_once_with(
            model="text-embedding-3-small",
            input="hello world",
            dimensions=1536,
        )

    @pytest.mark.asyncio
    async def test_embed_batch_returns_multiple_vectors(self) -> None:
        embeddings = [_make_embedding(1536, v) for v in [0.01, 0.02, 0.03]]
        client = _mock_openai_client(embeddings)
        provider = OpenAIEmbeddingProvider(client=client)
        vecs = await provider.embed_batch(["a", "b", "c"])
        assert len(vecs) == 3
        for v in vecs:
            assert v.shape == (1536,)
            assert v.dtype == np.float32

    @pytest.mark.asyncio
    async def test_embed_batch_sends_list_input(self) -> None:
        client = _mock_openai_client([_make_embedding(1536)] * 2)
        provider = OpenAIEmbeddingProvider(client=client)
        await provider.embed_batch(["a", "b"])
        client.embeddings.create.assert_awaited_once_with(
            model="text-embedding-3-small",
            input=["a", "b"],
            dimensions=1536,
        )

    @pytest.mark.asyncio
    async def test_embed_batch_empty_returns_empty(self) -> None:
        provider = OpenAIEmbeddingProvider(client=MagicMock())
        vecs = await provider.embed_batch([])
        assert vecs == []


# ---------------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------------


def _mock_gemini_client(embeddings: list[list[float]]) -> MagicMock:
    client = MagicMock()
    resp = MagicMock()
    resp.embeddings = [MagicMock(values=e) for e in embeddings]
    client.aio.models.embed_content = AsyncMock(return_value=resp)
    return client


@pytest.mark.verification
class TestGeminiEmbeddingProvider:
    def test_is_embedding_provider(self) -> None:
        provider = GeminiEmbeddingProvider(client=MagicMock())
        assert isinstance(provider, EmbeddingProvider)

    def test_dimension_default(self) -> None:
        provider = GeminiEmbeddingProvider(client=MagicMock())
        assert provider.dimension == 768

    @pytest.mark.asyncio
    async def test_embed_returns_correct_shape(self) -> None:
        client = _mock_gemini_client([_make_embedding(768)])
        provider = GeminiEmbeddingProvider(client=client)
        vec = await provider.embed("test text")
        assert vec.shape == (768,)
        assert vec.dtype == np.float32

    @pytest.mark.asyncio
    async def test_embed_calls_api_correctly(self) -> None:
        client = _mock_gemini_client([_make_embedding(768)])
        provider = GeminiEmbeddingProvider(client=client, model="text-embedding-004")
        await provider.embed("hello world")
        client.aio.models.embed_content.assert_awaited_once_with(
            model="text-embedding-004",
            contents="hello world",
        )

    @pytest.mark.asyncio
    async def test_embed_batch_returns_multiple_vectors(self) -> None:
        embeddings = [_make_embedding(768, v) for v in [0.01, 0.02, 0.03]]
        client = _mock_gemini_client(embeddings)
        provider = GeminiEmbeddingProvider(client=client)
        vecs = await provider.embed_batch(["a", "b", "c"])
        assert len(vecs) == 3
        for v in vecs:
            assert v.shape == (768,)

    @pytest.mark.asyncio
    async def test_embed_batch_calls_api_with_list(self) -> None:
        client = _mock_gemini_client([_make_embedding(768)] * 2)
        provider = GeminiEmbeddingProvider(client=client)
        await provider.embed_batch(["a", "b"])
        client.aio.models.embed_content.assert_awaited_once_with(
            model="text-embedding-004",
            contents=["a", "b"],
        )

    @pytest.mark.asyncio
    async def test_embed_batch_empty_returns_empty(self) -> None:
        provider = GeminiEmbeddingProvider(client=MagicMock())
        vecs = await provider.embed_batch([])
        assert vecs == []


# ---------------------------------------------------------------------------
# Ollama provider
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestOllamaEmbeddingProvider:
    def test_is_embedding_provider(self) -> None:
        provider = OllamaEmbeddingProvider()
        assert isinstance(provider, EmbeddingProvider)

    def test_dimension_default(self) -> None:
        provider = OllamaEmbeddingProvider()
        assert provider.dimension == 768

    def test_custom_config(self) -> None:
        provider = OllamaEmbeddingProvider(
            model="mxbai-embed-large", base_url="http://gpu-box:11434", dimension=1024
        )
        assert provider.dimension == 1024

    @pytest.mark.asyncio
    async def test_embed_returns_correct_shape(self) -> None:
        provider = OllamaEmbeddingProvider(dimension=768)
        provider._request = AsyncMock(  # type: ignore[assignment]
            return_value={"embeddings": [_make_embedding(768)]}
        )
        vec = await provider.embed("test text")
        assert vec.shape == (768,)
        assert vec.dtype == np.float32

    @pytest.mark.asyncio
    async def test_embed_sends_correct_payload(self) -> None:
        provider = OllamaEmbeddingProvider(model="nomic-embed-text")
        provider._request = AsyncMock(  # type: ignore[assignment]
            return_value={"embeddings": [_make_embedding(768)]}
        )
        await provider.embed("hello world")
        provider._request.assert_awaited_once_with(
            {"model": "nomic-embed-text", "input": "hello world"}
        )

    @pytest.mark.asyncio
    async def test_embed_batch(self) -> None:
        provider = OllamaEmbeddingProvider(dimension=768)
        provider._request = AsyncMock(  # type: ignore[assignment]
            return_value={"embeddings": [_make_embedding(768)] * 3}
        )
        vecs = await provider.embed_batch(["a", "b", "c"])
        assert len(vecs) == 3
        for v in vecs:
            assert v.shape == (768,)

    @pytest.mark.asyncio
    async def test_embed_batch_sends_list_input(self) -> None:
        provider = OllamaEmbeddingProvider(model="nomic-embed-text")
        provider._request = AsyncMock(  # type: ignore[assignment]
            return_value={"embeddings": [_make_embedding(768)] * 2}
        )
        await provider.embed_batch(["a", "b"])
        provider._request.assert_awaited_once_with(
            {"model": "nomic-embed-text", "input": ["a", "b"]}
        )

    @pytest.mark.asyncio
    async def test_embed_batch_empty_returns_empty(self) -> None:
        provider = OllamaEmbeddingProvider()
        vecs = await provider.embed_batch([])
        assert vecs == []


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestCreateEmbeddingProvider:
    def test_openai_by_name(self) -> None:
        mock_client = _mock_openai_client([_make_embedding(1536)])
        provider = create_embedding_provider("openai", _client=mock_client)
        assert isinstance(provider, OpenAIEmbeddingProvider)
        assert provider.dimension == 1536

    def test_openai_by_enum(self) -> None:
        mock_client = _mock_openai_client([_make_embedding(1536)])
        provider = create_embedding_provider(EmbeddingProviderType.OPENAI, _client=mock_client)
        assert isinstance(provider, OpenAIEmbeddingProvider)

    def test_gemini_by_name(self) -> None:
        mock_client = _mock_gemini_client([_make_embedding(768)])
        provider = create_embedding_provider("gemini", _client=mock_client)
        assert isinstance(provider, GeminiEmbeddingProvider)
        assert provider.dimension == 768

    def test_ollama_by_name(self) -> None:
        provider = create_embedding_provider("ollama")
        assert isinstance(provider, OllamaEmbeddingProvider)
        assert provider.dimension == 768

    def test_custom_model_and_dimension(self) -> None:
        mock_client = _mock_openai_client([_make_embedding(512)])
        provider = create_embedding_provider(
            "openai", model="text-embedding-3-large", dimension=512, _client=mock_client
        )
        assert isinstance(provider, OpenAIEmbeddingProvider)
        assert provider.dimension == 512

    def test_ollama_custom_base_url(self) -> None:
        provider = create_embedding_provider(
            "ollama", base_url="http://gpu-box:11434", dimension=1024
        )
        assert isinstance(provider, OllamaEmbeddingProvider)
        assert provider.dimension == 1024

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="Unknown embedding provider"):
            create_embedding_provider("unknown_provider")

    def test_case_insensitive_name(self) -> None:
        provider = create_embedding_provider("OLLAMA")
        assert isinstance(provider, OllamaEmbeddingProvider)


# ---------------------------------------------------------------------------
# ImportError handling — providers raise when packages not installed
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestProviderImportErrors:
    def test_openai_embedding_raises_import_error_without_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When openai package is missing and no client provided, raise ImportError."""
        monkeypatch.setitem(__import__("sys").modules, "openai", None)  # type: ignore[arg-type]
        with pytest.raises((ImportError, ModuleNotFoundError)):
            OpenAIEmbeddingProvider()

    def test_gemini_embedding_raises_import_error_without_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When google-genai package is missing and no client provided, raise ImportError."""
        monkeypatch.setattr(
            "strata_match.providers.GeminiEmbeddingProvider.__init__",
            lambda self, **kw: (_ for _ in ()).throw(ImportError("google-genai missing")),
        )
        with pytest.raises((ImportError, ModuleNotFoundError)):
            GeminiEmbeddingProvider()

    @pytest.mark.asyncio
    async def test_ollama_request_raises_import_error_without_aiohttp(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When aiohttp is missing, Ollama _request raises ImportError."""
        monkeypatch.setitem(__import__("sys").modules, "aiohttp", None)  # type: ignore[arg-type]
        provider = OllamaEmbeddingProvider()
        with pytest.raises((ImportError, ModuleNotFoundError)):
            await provider.embed("test text")


# ---------------------------------------------------------------------------
# Integration with create_matcher (string-based provider resolution)
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestCreateMatcherWithProviderString:
    def test_create_matcher_resolves_string_provider(self) -> None:
        from strata_match.matcher import create_matcher

        mock_client = _mock_openai_client([_make_embedding(1536)])
        matcher = create_matcher("openai", _provider_client=mock_client)
        assert matcher.vector_scorer.provider.dimension == 1536

    def test_create_matcher_still_accepts_provider_instance(self) -> None:
        from strata_match.matcher import create_matcher
        from tests.conftest import FakeEmbeddingProvider

        fake = FakeEmbeddingProvider(dimension=16)
        matcher = create_matcher(fake)
        assert matcher.vector_scorer.provider.dimension == 16
