"""Tests for embedding provider abstraction and cosine similarity."""

import numpy as np
import pytest

from strata_match.embeddings import cosine_similarity

from tests.conftest import FakeEmbeddingProvider


@pytest.mark.verification
class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self) -> None:
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, b) == 0.0


@pytest.mark.verification
class TestFakeEmbeddingProvider:
    @pytest.mark.asyncio
    async def test_embed_returns_correct_dimension(self) -> None:
        provider = FakeEmbeddingProvider(dimension=16)
        vec = await provider.embed("test text")
        assert vec.shape == (16,)
        assert vec.dtype == np.float32

    @pytest.mark.asyncio
    async def test_embed_is_deterministic(self) -> None:
        provider = FakeEmbeddingProvider(dimension=8)
        v1 = await provider.embed("hello")
        v2 = await provider.embed("hello")
        np.testing.assert_array_equal(v1, v2)

    @pytest.mark.asyncio
    async def test_embed_different_texts_differ(self) -> None:
        provider = FakeEmbeddingProvider(dimension=8)
        v1 = await provider.embed("hello")
        v2 = await provider.embed("world")
        assert not np.array_equal(v1, v2)

    @pytest.mark.asyncio
    async def test_embed_batch(self) -> None:
        provider = FakeEmbeddingProvider(dimension=8)
        vecs = await provider.embed_batch(["a", "b", "c"])
        assert len(vecs) == 3
        for v in vecs:
            assert v.shape == (8,)

    @pytest.mark.asyncio
    async def test_embed_produces_unit_vectors(self) -> None:
        provider = FakeEmbeddingProvider(dimension=8)
        vec = await provider.embed("normalize me")
        assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-5)

    def test_dimension_property(self) -> None:
        provider = FakeEmbeddingProvider(dimension=32)
        assert provider.dimension == 32
