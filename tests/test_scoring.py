"""Tests for scoring logic — vector scoring and confidence classification."""

import pytest

from strata_match.models import CandidateProfile, ConfidenceTier, JobDescription
from strata_match.scoring import (
    DEFAULT_HIGH_VECTOR,
    DEFAULT_MEDIUM_VECTOR,
    DEFAULT_MEDIUM_VECTOR_NO_LLM,
    DEFAULT_VERY_HIGH_VECTOR,
    VectorScorer,
    build_match_result,
    classify_confidence,
)
from tests.conftest import FakeEmbeddingProvider


@pytest.mark.verification
class TestClassifyConfidence:
    def test_high_confidence(self) -> None:
        tier = classify_confidence(0.8, llm_confirmed=True, llm_score=80.0)
        assert tier == ConfidenceTier.HIGH

    def test_high_vector_no_llm(self) -> None:
        tier = classify_confidence(0.8, llm_confirmed=False)
        assert tier == ConfidenceTier.MEDIUM

    def test_medium_vector_with_llm(self) -> None:
        tier = classify_confidence(0.6, llm_confirmed=True)
        assert tier == ConfidenceTier.MEDIUM

    def test_low_vector_no_llm(self) -> None:
        tier = classify_confidence(0.2, llm_confirmed=False)
        assert tier == ConfidenceTier.LOW

    def test_custom_thresholds(self) -> None:
        tier = classify_confidence(
            0.6,
            llm_confirmed=True,
            llm_score=80.0,
            high_threshold=0.5,
            medium_threshold=0.3,
        )
        assert tier == ConfidenceTier.HIGH

    # ------------------------------------------------------------------
    # nomic-embed-text calibration — verify new defaults are reachable
    # Observed production data: strong matches (LLM 85-92) produce
    # vector cosine scores in 0.60-0.72 range; max ever seen: 0.716.
    # ------------------------------------------------------------------

    def test_very_high_reachable_with_nomic_scores(self) -> None:
        """VERY_HIGH is reachable for top-tier nomic-embed-text scores."""
        # vector=0.68 is within the observed 0.60-0.72 strong-match range
        tier = classify_confidence(0.68, llm_confirmed=True, llm_score=92.0)
        assert tier == ConfidenceTier.VERY_HIGH

    def test_very_high_boundary_just_above(self) -> None:
        """Exactly at DEFAULT_VERY_HIGH_VECTOR with sufficient LLM score → VERY_HIGH."""
        tier = classify_confidence(DEFAULT_VERY_HIGH_VECTOR, llm_confirmed=True, llm_score=85.0)
        assert tier == ConfidenceTier.VERY_HIGH

    def test_very_high_boundary_just_below(self) -> None:
        """Just below DEFAULT_VERY_HIGH_VECTOR falls back to HIGH (not VERY_HIGH)."""
        vector = DEFAULT_VERY_HIGH_VECTOR - 0.01
        tier = classify_confidence(vector, llm_confirmed=True, llm_score=92.0)
        assert tier == ConfidenceTier.HIGH

    def test_high_reachable_with_nomic_scores(self) -> None:
        """HIGH is reachable for mid-range nomic-embed-text strong matches."""
        # vector=0.62 is squarely in the observed 0.60-0.72 range
        tier = classify_confidence(0.62, llm_confirmed=True, llm_score=85.0)
        assert tier == ConfidenceTier.HIGH

    def test_high_boundary_just_above(self) -> None:
        """Exactly at DEFAULT_HIGH_VECTOR with LLM confirmed → HIGH."""
        tier = classify_confidence(DEFAULT_HIGH_VECTOR, llm_confirmed=True, llm_score=70.0)
        assert tier == ConfidenceTier.HIGH

    def test_medium_no_llm_reachable_at_high_vector(self) -> None:
        """Vector-only path reaches MEDIUM at DEFAULT_MEDIUM_VECTOR_NO_LLM."""
        tier = classify_confidence(DEFAULT_MEDIUM_VECTOR_NO_LLM, llm_confirmed=False)
        assert tier == ConfidenceTier.MEDIUM

    def test_nomic_defaults_ordering(self) -> None:
        """Sanity check: threshold ordering is consistent."""
        assert DEFAULT_VERY_HIGH_VECTOR > DEFAULT_HIGH_VECTOR
        assert DEFAULT_HIGH_VECTOR >= DEFAULT_MEDIUM_VECTOR_NO_LLM
        assert DEFAULT_MEDIUM_VECTOR_NO_LLM > DEFAULT_MEDIUM_VECTOR


@pytest.mark.verification
class TestBuildMatchResult:
    def test_builds_result(self) -> None:
        job = JobDescription(title="Engineer", company="Acme")
        result = build_match_result(
            job,
            score=75.0,
            vector_score=72.0,
            rationale="Good fit.",
            strengths=["Python"],
            gaps=["Leadership"],
            confidence_tier=ConfidenceTier.MEDIUM,
            llm_scored=True,
        )
        assert result.job_title == "Engineer"
        assert result.job_company == "Acme"
        assert result.score == 75.0
        assert result.vector_score == 72.0
        assert result.rationale == "Good fit."
        assert result.strengths == ["Python"]
        assert result.gaps == ["Leadership"]
        assert result.llm_scored is True

    def test_builds_result_with_new_fields(self) -> None:
        job = JobDescription(title="Staff Engineer", company="BigCo")
        result = build_match_result(
            job,
            score=88.0,
            vector_score=75.0,
            salary_match=True,
            culture_signals=["remote-friendly", "engineering-led"],
            tokens_used=2000,
            confidence_tier=ConfidenceTier.HIGH,
            llm_scored=True,
        )
        assert result.salary_match is True
        assert result.culture_signals == ["remote-friendly", "engineering-led"]
        assert result.tokens_used == 2000

    def test_builds_result_defaults(self) -> None:
        job = JobDescription(title="Engineer")
        result = build_match_result(job, score=50.0)
        assert result.salary_match is None
        assert result.culture_signals == []
        assert result.tokens_used == 0
        assert result.rationale == ""
        assert result.strengths == []
        assert result.gaps == []


@pytest.mark.verification
class TestVectorScorer:
    @pytest.mark.asyncio
    async def test_score_returns_float(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        provider = FakeEmbeddingProvider(dimension=8)
        scorer = VectorScorer(provider=provider)
        score = await scorer.score(sample_profile, sample_jobs[0])
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_score_batch_returns_list(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        provider = FakeEmbeddingProvider(dimension=8)
        scorer = VectorScorer(provider=provider)
        scores = await scorer.score_batch(sample_profile, sample_jobs)
        assert len(scores) == len(sample_jobs)
        for s in scores:
            assert isinstance(s, float)

    @pytest.mark.asyncio
    async def test_score_is_deterministic(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        provider = FakeEmbeddingProvider(dimension=8)
        scorer = VectorScorer(provider=provider)
        s1 = await scorer.score(sample_profile, sample_jobs[0])
        s2 = await scorer.score(sample_profile, sample_jobs[0])
        assert s1 == pytest.approx(s2)

    @pytest.mark.asyncio
    async def test_profile_embedding_cached(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        provider = FakeEmbeddingProvider(dimension=8)
        scorer = VectorScorer(provider=provider)
        await scorer.score(sample_profile, sample_jobs[0])
        await scorer.score(sample_profile, sample_jobs[1])
        assert len(scorer._profile_cache) == 1
