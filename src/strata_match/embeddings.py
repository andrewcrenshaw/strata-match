"""Embedding provider abstraction for vector similarity scoring."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from strata_match.exceptions import EmbeddingError

if TYPE_CHECKING:
    from numpy.typing import NDArray


class EmbeddingProviderType(StrEnum):
    """Supported embedding providers."""

    OPENAI = "openai"
    GEMINI = "gemini"
    OLLAMA = "ollama"


class EmbeddingProvider(ABC):
    """Abstract base for embedding providers."""

    @abstractmethod
    async def embed(self, text: str) -> NDArray[np.float32]:
        """Generate an embedding vector for the given text."""

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[NDArray[np.float32]]:
        """Generate embedding vectors for multiple texts."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors produced by this provider."""


def cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Compute cosine similarity between two vectors. Returns value in [-1, 1]."""
    if a.shape != b.shape:
        raise EmbeddingError(f"Dimension mismatch: {a.shape} vs {b.shape}")
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
