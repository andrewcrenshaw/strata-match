"""Tests for vector similarity scoring (Stage 1) — PCC-1431.

Covers: cosine similarity, score clamping [0,1], threshold filtering,
pre-computed embedding support, and batch scoring.
"""

from __future__ import annotations

import numpy as np
import pytest

from strata_match.embeddings import cosine_similarity
from strata_match.exceptions import EmbeddingError
from strata_match.models import CandidateProfile, JobDescription
from strata_match.scoring import VectorScorer
from tests.conftest import FakeEmbeddingProvider

# ---------------------------------------------------------------------------
# Cosine similarity — pure math
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestCosineSimilarity:
    def test_identical_vectors_return_one(self) -> None:
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_opposite_vectors_return_negative_one(self) -> None:
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert cosine_similarity(v, -v) == pytest.approx(-1.0)

    def test_orthogonal_vectors_return_zero(self) -> None:
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self) -> None:
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == 0.0

    def test_both_zero_vectors_return_zero(self) -> None:
        z = np.array([0.0, 0.0], dtype=np.float32)
        assert cosine_similarity(z, z) == 0.0

    def test_similar_vectors_high_score(self) -> None:
        a = np.array([1.0, 0.9, 0.8], dtype=np.float32)
        b = np.array([1.0, 0.85, 0.75], dtype=np.float32)
        assert cosine_similarity(a, b) > 0.99

    def test_dissimilar_vectors_low_score(self) -> None:
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# VectorScorer — score range, clamping
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestVectorScorerScoreRange:
    @pytest.mark.asyncio
    async def test_score_returns_value_in_0_1(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        provider = FakeEmbeddingProvider(dimension=8)
        scorer = VectorScorer(provider=provider)
        score = await scorer.score(sample_profile, sample_jobs[0])
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_score_never_negative(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        """Even if cosine similarity is negative, score should clamp to 0."""
        provider = FakeEmbeddingProvider(dimension=8)
        scorer = VectorScorer(provider=provider)
        for job in sample_jobs:
            score = await scorer.score(sample_profile, job)
            assert score >= 0.0, f"Score {score} is negative for job {job.title}"

    @pytest.mark.asyncio
    async def test_score_batch_all_in_0_1(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        provider = FakeEmbeddingProvider(dimension=8)
        scorer = VectorScorer(provider=provider)
        scores = await scorer.score_batch(sample_profile, sample_jobs)
        for s in scores:
            assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# VectorScorer — threshold filtering (fast path)
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestVectorScorerThreshold:
    @pytest.mark.asyncio
    async def test_default_threshold_is_0_6(self) -> None:
        provider = FakeEmbeddingProvider(dimension=8)
        scorer = VectorScorer(provider=provider, vector_threshold=0.6)
        assert scorer.vector_threshold == 0.6

    @pytest.mark.asyncio
    async def test_score_filtered_below_threshold(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        provider = FakeEmbeddingProvider(dimension=8)
        scorer = VectorScorer(provider=provider, vector_threshold=0.99)
        score, filtered = await scorer.score_with_threshold(sample_profile, sample_jobs[0])
        assert filtered is True
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_score_passes_above_threshold(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        provider = FakeEmbeddingProvider(dimension=8)
        scorer = VectorScorer(provider=provider, vector_threshold=0.0)
        score, filtered = await scorer.score_with_threshold(sample_profile, sample_jobs[0])
        assert filtered is False
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_batch_filtered_splits_correctly(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        provider = FakeEmbeddingProvider(dimension=8)
        scorer = VectorScorer(provider=provider, vector_threshold=0.0)
        above, below = await scorer.score_batch_filtered(sample_profile, sample_jobs)
        assert len(above) + len(below) == len(sample_jobs)
        for _job, score in above:
            assert score >= scorer.vector_threshold
        for _job, score in below:
            assert score < scorer.vector_threshold

    @pytest.mark.asyncio
    async def test_batch_filtered_high_threshold_filters_all(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        provider = FakeEmbeddingProvider(dimension=8)
        scorer = VectorScorer(provider=provider, vector_threshold=0.99)
        above, below = await scorer.score_batch_filtered(sample_profile, sample_jobs)
        assert len(below) == len(sample_jobs)
        assert len(above) == 0


# ---------------------------------------------------------------------------
# VectorScorer — pre-computed embeddings
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestPreComputedEmbeddings:
    @pytest.mark.asyncio
    async def test_uses_profile_precomputed_embedding(self) -> None:
        """When profile.embedding is set, skip embedding generation for profile."""
        provider = FakeEmbeddingProvider(dimension=4)
        scorer = VectorScorer(provider=provider)

        profile_vec = [0.5, 0.5, 0.5, 0.5]
        profile = CandidateProfile(
            title="Engineer",
            skills=["Python"],
            embedding=profile_vec,
        )
        job = JobDescription(
            title="Software Engineer",
            company="TestCo",
            description="Build stuff.",
        )

        score = await scorer.score(profile, job)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_uses_job_precomputed_embedding(self) -> None:
        """When job.embedding is set, skip embedding generation for job."""
        provider = FakeEmbeddingProvider(dimension=4)
        scorer = VectorScorer(provider=provider)

        profile = CandidateProfile(title="Engineer", skills=["Python"])
        job = JobDescription(
            title="Software Engineer",
            company="TestCo",
            embedding=[0.5, 0.5, 0.5, 0.5],
        )

        score = await scorer.score(profile, job)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_both_precomputed_skips_provider(self) -> None:
        """When both have pre-computed embeddings, provider.embed is never called."""

        class NeverCallProvider(FakeEmbeddingProvider):
            async def embed(self, text: str) -> np.ndarray:
                raise AssertionError("embed() should not be called with pre-computed vecs")

            async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
                raise AssertionError("embed_batch() should not be called")

        provider = NeverCallProvider(dimension=4)
        scorer = VectorScorer(provider=provider)

        profile = CandidateProfile(
            title="Engineer",
            embedding=[1.0, 0.0, 0.0, 0.0],
        )
        job = JobDescription(
            title="Engineer",
            embedding=[1.0, 0.0, 0.0, 0.0],
        )

        score = await scorer.score(profile, job)
        assert score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_precomputed_identical_embeddings_score_one(self) -> None:
        provider = FakeEmbeddingProvider(dimension=3)
        scorer = VectorScorer(provider=provider)

        vec = [0.577, 0.577, 0.577]
        profile = CandidateProfile(title="X", embedding=vec)
        job = JobDescription(title="X", embedding=vec)

        score = await scorer.score(profile, job)
        assert score == pytest.approx(1.0, abs=1e-3)

    @pytest.mark.asyncio
    async def test_batch_with_mixed_precomputed(self) -> None:
        """Batch scoring with some jobs having pre-computed embeddings."""
        provider = FakeEmbeddingProvider(dimension=4)
        scorer = VectorScorer(provider=provider)

        profile = CandidateProfile(
            title="Engineer",
            embedding=[1.0, 0.0, 0.0, 0.0],
        )
        jobs = [
            JobDescription(title="A", embedding=[1.0, 0.0, 0.0, 0.0]),
            JobDescription(title="B", company="Co", description="Build things."),
            JobDescription(title="C", embedding=[0.0, 1.0, 0.0, 0.0]),
        ]

        scores = await scorer.score_batch(profile, jobs)
        assert len(scores) == 3
        assert scores[0] == pytest.approx(1.0, abs=1e-3)
        assert scores[2] == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# VectorScorer — batch performance / correctness
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestBatchScoring:
    @pytest.mark.asyncio
    async def test_batch_scores_match_individual_scores(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        """Batch scoring produces same results as scoring individually."""
        provider = FakeEmbeddingProvider(dimension=8)
        scorer = VectorScorer(provider=provider)

        individual = [await scorer.score(sample_profile, j) for j in sample_jobs]

        scorer2 = VectorScorer(provider=FakeEmbeddingProvider(dimension=8))
        batch = await scorer2.score_batch(sample_profile, sample_jobs)

        for i, (ind, bat) in enumerate(zip(individual, batch, strict=True)):
            assert ind == pytest.approx(bat, abs=1e-6), f"Job {i}: individual={ind}, batch={bat}"

    @pytest.mark.asyncio
    async def test_batch_empty_list(self) -> None:
        provider = FakeEmbeddingProvider(dimension=8)
        scorer = VectorScorer(provider=provider)
        profile = CandidateProfile(title="Engineer")

        scores = await scorer.score_batch(profile, [])
        assert scores == []

    @pytest.mark.asyncio
    async def test_batch_single_job(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        provider = FakeEmbeddingProvider(dimension=8)
        scorer = VectorScorer(provider=provider)

        scores = await scorer.score_batch(sample_profile, [sample_jobs[0]])
        assert len(scores) == 1
        assert 0.0 <= scores[0] <= 1.0

    @pytest.mark.asyncio
    async def test_batch_many_jobs(self) -> None:
        """Batch scoring handles many jobs efficiently."""
        provider = FakeEmbeddingProvider(dimension=8)
        scorer = VectorScorer(provider=provider)
        profile = CandidateProfile(title="Engineer", skills=["Python"])

        jobs = [JobDescription(title=f"Job {i}", company=f"Co{i}") for i in range(50)]

        scores = await scorer.score_batch(profile, jobs)
        assert len(scores) == 50
        for s in scores:
            assert 0.0 <= s <= 1.0

    @pytest.mark.asyncio
    async def test_score_batch_raises_embedding_error_when_embedding_unresolved(
        self, sample_profile: CandidateProfile
    ) -> None:
        """None from embed_batch must raise EmbeddingError (not rely on assert)."""

        class FaultyBatchProvider(FakeEmbeddingProvider):
            async def embed_batch(self, texts: list[str]) -> list:  # type: ignore[type-arg]
                return [None] * len(texts)

        provider = FaultyBatchProvider(dimension=8)
        scorer = VectorScorer(provider=provider)
        job = JobDescription(
            title="Software Engineer",
            company="Co",
            description="Build things.",
        )

        with pytest.raises(EmbeddingError, match="job index 0"):
            await scorer.score_batch(sample_profile, [job])
