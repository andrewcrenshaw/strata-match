"""Concrete embedding providers and provider factory.

Each provider wraps a third-party embedding API behind the
:class:`~strata_match.embeddings.EmbeddingProvider` interface.  External
dependencies are imported lazily so that only the chosen provider's package
needs to be installed.

Usage::

    from strata_match.providers import create_embedding_provider

    provider = create_embedding_provider("openai", api_key="sk-...")
    vec = await provider.embed("some text")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from strata_match.embeddings import EmbeddingProvider

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by the OpenAI embeddings API.

    Requires the ``openai`` package (``pip install strata-match[openai]``).
    Pass a pre-built ``AsyncOpenAI`` client via *client* to skip auto-creation
    (useful for testing or custom configuration).
    """

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        dimension: int = 1536,
        client: Any = None,
    ) -> None:
        self._model = model
        self._dimension = dimension
        if client is not None:
            self._client = client
        else:
            try:
                import openai
            except ImportError as exc:
                raise ImportError(
                    "OpenAI provider requires the 'openai' package. "
                    "Install with: pip install strata-match[openai]"
                ) from exc
            self._client = openai.AsyncOpenAI(api_key=api_key)

    async def embed(self, text: str) -> NDArray[np.float32]:
        resp = await self._client.embeddings.create(
            model=self._model,
            input=text,
            dimensions=self._dimension,
        )
        return np.array(resp.data[0].embedding, dtype=np.float32)

    async def embed_batch(self, texts: list[str]) -> list[NDArray[np.float32]]:
        if not texts:
            return []
        resp = await self._client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=self._dimension,
        )
        return [np.array(d.embedding, dtype=np.float32) for d in resp.data]

    @property
    def dimension(self) -> int:
        return self._dimension


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by the Google Gemini (GenAI) API.

    Requires the ``google-genai`` package
    (``pip install strata-match[gemini]``).
    """

    def __init__(
        self,
        *,
        model: str = "text-embedding-004",
        api_key: str | None = None,
        dimension: int = 768,
        client: Any = None,
    ) -> None:
        self._model = model
        self._dimension = dimension
        if client is not None:
            self._client = client
        else:
            try:
                from google import genai
            except ImportError as exc:
                raise ImportError(
                    "Gemini provider requires the 'google-genai' package. "
                    "Install with: pip install strata-match[gemini]"
                ) from exc
            self._client = genai.Client(api_key=api_key)

    async def embed(self, text: str) -> NDArray[np.float32]:
        resp = await self._client.aio.models.embed_content(
            model=self._model,
            contents=text,
        )
        return np.array(resp.embeddings[0].values, dtype=np.float32)

    async def embed_batch(self, texts: list[str]) -> list[NDArray[np.float32]]:
        if not texts:
            return []
        resp = await self._client.aio.models.embed_content(
            model=self._model,
            contents=texts,
        )
        return [np.array(e.values, dtype=np.float32) for e in resp.embeddings]

    @property
    def dimension(self) -> int:
        return self._dimension


# ---------------------------------------------------------------------------
# Ollama (local / offline)
# ---------------------------------------------------------------------------


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by a local Ollama instance.

    Communicates with Ollama's ``/api/embed`` REST endpoint.
    Requires the ``aiohttp`` package (``pip install strata-match[ollama]``).
    """

    def __init__(
        self,
        *,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        dimension: int = 768,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._dimension = dimension

    async def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST to the Ollama embed API."""
        try:
            import aiohttp
        except ImportError as exc:
            raise ImportError(
                "Ollama provider requires the 'aiohttp' package. "
                "Install with: pip install strata-match[ollama]"
            ) from exc

        async with (
            aiohttp.ClientSession() as session,
            session.post(f"{self._base_url}/api/embed", json=payload) as resp,
        ):
            resp.raise_for_status()
            data: dict[str, Any] = await resp.json()
            return data

    async def embed(self, text: str) -> NDArray[np.float32]:
        data = await self._request({"model": self._model, "input": text})
        return np.array(data["embeddings"][0], dtype=np.float32)

    async def embed_batch(self, texts: list[str]) -> list[NDArray[np.float32]]:
        if not texts:
            return []
        data = await self._request({"model": self._model, "input": texts})
        return [np.array(e, dtype=np.float32) for e in data["embeddings"]]

    @property
    def dimension(self) -> int:
        return self._dimension


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "openai": {"model": "text-embedding-3-small", "dimension": 1536},
    "gemini": {"model": "text-embedding-004", "dimension": 768},
    "ollama": {"model": "nomic-embed-text", "dimension": 768},
}


def create_embedding_provider(
    name: str,
    *,
    model: str | None = None,
    dimension: int | None = None,
    _client: Any = None,
    **config: Any,
) -> EmbeddingProvider:
    """Create an embedding provider by name.

    Args:
        name: Provider identifier — ``"openai"``, ``"gemini"``, or ``"ollama"``
              (case-insensitive).  Also accepts :class:`EmbeddingProviderType` values.
        model: Model identifier override.  Falls back to the provider default.
        dimension: Embedding dimensionality override.
        _client: Pre-built API client (for testing).
        **config: Additional keyword arguments forwarded to the provider constructor
                  (e.g. ``api_key``, ``base_url``).

    Returns:
        A configured :class:`EmbeddingProvider` instance.

    Raises:
        ValueError: If *name* does not match a known provider.
    """
    key = str(name).lower()

    defaults = _PROVIDER_DEFAULTS.get(key)
    if defaults is None:
        raise ValueError(
            f"Unknown embedding provider '{name}'. "
            f"Choose from: {', '.join(_PROVIDER_DEFAULTS)}"
        )

    resolved_model = model or defaults["model"]
    resolved_dim = dimension or defaults["dimension"]

    if key == "openai":
        return OpenAIEmbeddingProvider(
            model=resolved_model,
            dimension=resolved_dim,
            client=_client,
            **config,
        )

    if key == "gemini":
        return GeminiEmbeddingProvider(
            model=resolved_model,
            dimension=resolved_dim,
            client=_client,
            **config,
        )

    # ollama — no client injection needed (uses HTTP)
    return OllamaEmbeddingProvider(
        model=resolved_model,
        dimension=resolved_dim,
        **config,
    )
