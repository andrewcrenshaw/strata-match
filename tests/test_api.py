"""Integration tests for the public API surface.

Tests the full match pipeline: create_matcher → match_job → match_batch
with mock embedding and LLM providers, verifying configurable providers,
thresholds, and token tracking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from strata_match.embeddings import EmbeddingProvider
from strata_match.llm import LLMProvider, LLMResponse, LLMScorer
from strata_match.matcher import Matcher, create_matcher, match_batch, match_job
from strata_match.models import (
    BatchMatchResult,
    CandidateProfile,
    ConfidenceTier,
    JobDescription,
    MatchResult,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Mock providers
# ---------------------------------------------------------------------------


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider that returns unit vectors based on text hash."""

    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension
        self.embed_calls: list[str] = []
        self.embed_batch_calls: list[list[str]] = []

    async def embed(self, text: str) -> NDArray[np.float32]:
        self.embed_calls.append(text)
        rng = np.random.default_rng(seed=hash(text) % (2**31))
        vec = rng.standard_normal(self._dimension).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    async def embed_batch(self, texts: list[str]) -> list[NDArray[np.float32]]:
        self.embed_batch_calls.append(texts)
        return [await self.embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dimension


@dataclass
class MockLLMProvider(LLMProvider):
    """Deterministic LLM provider that returns configurable JSON responses."""

    response_score: float = 82.0
    response_rationale: str = "Strong fit for the role."
    response_strengths: list[str] | None = None
    response_gaps: list[str] | None = None
    input_tokens: int = 500
    output_tokens: int = 200
    model: str = "mock-llm"
    call_count: int = 0

    def __post_init__(self) -> None:
        if self.response_strengths is None:
            self.response_strengths = ["Python expertise", "System design"]
        if self.response_gaps is None:
            self.response_gaps = ["No leadership experience"]

    async def complete(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> LLMResponse:
        self.call_count += 1
        content = json.dumps({
            "score": self.response_score,
            "rationale": self.response_rationale,
            "strengths": self.response_strengths,
            "gaps": self.response_gaps,
            "salary_match": True,
            "culture_signals": ["remote-friendly"],
        })
        return LLMResponse(
            content=content,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )

    @property
    def model_name(self) -> str:
        return self.model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_embedding() -> MockEmbeddingProvider:
    return MockEmbeddingProvider(dimension=8)


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    return MockLLMProvider()


@pytest.fixture
def profile() -> CandidateProfile:
    return CandidateProfile(
        title="Senior Software Engineer",
        skills=["Python", "FastAPI", "PostgreSQL", "AWS"],
        years_of_experience=8,
        experience_summary="Full-stack engineer focused on distributed systems.",
        industries=["SaaS", "FinTech"],
    )


@pytest.fixture
def jobs() -> list[JobDescription]:
    return [
        JobDescription(
            title="Staff Engineer — Backend Platform",
            company="Acme Corp",
            description="Lead our backend platform team.",
            requirements=["Python", "System Design", "PostgreSQL"],
            salary_range="$180k-$220k",
        ),
        JobDescription(
            title="Junior Frontend Developer",
            company="WebCo",
            description="Build React components.",
            requirements=["React", "CSS", "JavaScript"],
        ),
        JobDescription(
            title="Data Scientist",
            company="DataLabs",
            description="Build ML models for churn prediction.",
            requirements=["Python", "scikit-learn", "SQL"],
        ),
    ]


# ---------------------------------------------------------------------------
# create_matcher — factory tests
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestCreateMatcherFactory:
    def test_accepts_embedding_model_param(self) -> None:
        """create_matcher should accept embedding_model as a named parameter."""
        mock_client = MockEmbeddingProvider(dimension=8)
        matcher = create_matcher(mock_client, embedding_model="text-embedding-3-small")
        assert isinstance(matcher, Matcher)

    def test_accepts_scoring_provider_and_model(
        self, mock_embedding: MockEmbeddingProvider
    ) -> None:
        """create_matcher should accept scoring_provider to auto-build LLM scorer."""
        mock_llm = MockLLMProvider()
        matcher = create_matcher(
            mock_embedding,
            scoring_provider=mock_llm,
            scoring_model="gpt-4o-mini",
        )
        assert isinstance(matcher, Matcher)
        assert matcher.llm_scorer is not None

    def test_scoring_provider_string_creates_llm_scorer(self) -> None:
        """Passing scoring_provider as string should auto-resolve the LLM provider."""
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        matcher = create_matcher(
            "openai",
            scoring_provider="openai",
            scoring_model="gpt-4o-mini",
            _provider_client=mock_client,
            _scoring_client=mock_client,
        )
        assert isinstance(matcher, Matcher)
        assert matcher.llm_scorer is not None

    def test_no_scoring_provider_means_no_llm_scorer(
        self, mock_embedding: MockEmbeddingProvider
    ) -> None:
        """When scoring_provider is None, Matcher has no LLM scorer."""
        matcher = create_matcher(mock_embedding)
        assert matcher.llm_scorer is None

    def test_vector_threshold_passed_through(
        self, mock_embedding: MockEmbeddingProvider
    ) -> None:
        matcher = create_matcher(mock_embedding, vector_threshold=0.5)
        assert matcher.vector_threshold == 0.5

    def test_backward_compat_llm_scorer_param(
        self, mock_embedding: MockEmbeddingProvider, mock_llm: MockLLMProvider
    ) -> None:
        """Existing code passing llm_scorer= directly should still work."""
        scorer = LLMScorer(provider=mock_llm)
        matcher = create_matcher(mock_embedding, llm_scorer=scorer)
        assert matcher.llm_scorer is scorer


# ---------------------------------------------------------------------------
# match_job — convenience function integration tests
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestMatchJobIntegration:
    @pytest.mark.asyncio
    async def test_match_job_returns_match_result(
        self,
        mock_embedding: MockEmbeddingProvider,
        profile: CandidateProfile,
        jobs: list[JobDescription],
    ) -> None:
        matcher = create_matcher(mock_embedding, vector_threshold=0.0)
        result = await match_job(matcher, profile, jobs[0])
        assert isinstance(result, MatchResult)
        assert result.job_title == "Staff Engineer — Backend Platform"
        assert 0.0 <= result.score <= 100.0

    @pytest.mark.asyncio
    async def test_match_job_with_llm_scoring(
        self,
        mock_embedding: MockEmbeddingProvider,
        mock_llm: MockLLMProvider,
        profile: CandidateProfile,
        jobs: list[JobDescription],
    ) -> None:
        """Full two-stage pipeline: vector → LLM scoring."""
        matcher = create_matcher(
            mock_embedding,
            scoring_provider=mock_llm,
            vector_threshold=0.0,
        )
        result = await match_job(matcher, profile, jobs[0])
        assert result.llm_scored is True
        assert result.score == 82.0
        assert result.tokens_used == 700
        assert "Python expertise" in result.strengths
        assert result.rationale == "Strong fit for the role."

    @pytest.mark.asyncio
    async def test_match_job_below_threshold_skips_llm(
        self,
        mock_embedding: MockEmbeddingProvider,
        mock_llm: MockLLMProvider,
        profile: CandidateProfile,
        jobs: list[JobDescription],
    ) -> None:
        """Jobs below vector threshold should not invoke the LLM."""
        matcher = create_matcher(
            mock_embedding,
            scoring_provider=mock_llm,
            vector_threshold=0.99,
        )
        result = await match_job(matcher, profile, jobs[0])
        assert result.llm_scored is False
        assert mock_llm.call_count == 0
        assert result.confidence_tier == ConfidenceTier.LOW

    @pytest.mark.asyncio
    async def test_match_job_vector_score_populated(
        self,
        mock_embedding: MockEmbeddingProvider,
        profile: CandidateProfile,
        jobs: list[JobDescription],
    ) -> None:
        matcher = create_matcher(mock_embedding, vector_threshold=0.0)
        result = await match_job(matcher, profile, jobs[0])
        assert result.vector_score is not None
        assert 0.0 <= result.vector_score <= 100.0


# ---------------------------------------------------------------------------
# match_batch — batch function integration tests
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestMatchBatchIntegration:
    @pytest.mark.asyncio
    async def test_match_batch_returns_batch_result(
        self,
        mock_embedding: MockEmbeddingProvider,
        profile: CandidateProfile,
        jobs: list[JobDescription],
    ) -> None:
        matcher = create_matcher(mock_embedding, vector_threshold=0.0)
        result = await match_batch(matcher, profile, jobs)
        assert isinstance(result, BatchMatchResult)
        assert result.jobs_evaluated == 3
        assert len(result.results) <= 3

    @pytest.mark.asyncio
    async def test_match_batch_with_llm_scoring(
        self,
        mock_embedding: MockEmbeddingProvider,
        mock_llm: MockLLMProvider,
        profile: CandidateProfile,
        jobs: list[JobDescription],
    ) -> None:
        """Batch with LLM scoring: tracks total tokens and LLM-scored count."""
        matcher = create_matcher(
            mock_embedding,
            scoring_provider=mock_llm,
            vector_threshold=0.0,
        )
        result = await match_batch(matcher, profile, jobs)
        assert result.llm_scored_count == 3
        assert result.total_tokens == 3 * 700

    @pytest.mark.asyncio
    async def test_match_batch_threshold_skips_low_scoring_jobs(
        self,
        mock_embedding: MockEmbeddingProvider,
        mock_llm: MockLLMProvider,
        profile: CandidateProfile,
        jobs: list[JobDescription],
    ) -> None:
        """High threshold should skip jobs and avoid LLM calls for those below."""
        matcher = create_matcher(
            mock_embedding,
            scoring_provider=mock_llm,
            vector_threshold=0.99,
        )
        result = await match_batch(matcher, profile, jobs)
        assert result.jobs_skipped >= 0
        assert result.jobs_evaluated == 3
        assert mock_llm.call_count == result.llm_scored_count

    @pytest.mark.asyncio
    async def test_match_batch_results_sorted_by_score(
        self,
        mock_embedding: MockEmbeddingProvider,
        profile: CandidateProfile,
        jobs: list[JobDescription],
    ) -> None:
        matcher = create_matcher(mock_embedding, vector_threshold=0.0)
        result = await match_batch(matcher, profile, jobs)
        scores = [r.score for r in result.results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_match_batch_empty_jobs_list(
        self,
        mock_embedding: MockEmbeddingProvider,
        profile: CandidateProfile,
    ) -> None:
        matcher = create_matcher(mock_embedding, vector_threshold=0.0)
        result = await match_batch(matcher, profile, [])
        assert result.jobs_evaluated == 0
        assert result.results == []

    @pytest.mark.asyncio
    async def test_match_batch_token_tracking(
        self,
        mock_embedding: MockEmbeddingProvider,
        mock_llm: MockLLMProvider,
        profile: CandidateProfile,
        jobs: list[JobDescription],
    ) -> None:
        """Token usage should accumulate across all LLM-scored jobs."""
        matcher = create_matcher(
            mock_embedding,
            scoring_provider=mock_llm,
            vector_threshold=0.0,
        )
        result = await match_batch(matcher, profile, jobs)
        assert result.total_tokens > 0
        individual_tokens = sum(r.tokens_used for r in result.results if r.llm_scored)
        assert result.total_tokens == individual_tokens


# ---------------------------------------------------------------------------
# Full pipeline integration test
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestFullPipelineIntegration:
    @pytest.mark.asyncio
    async def test_end_to_end_with_mock_providers(
        self,
        profile: CandidateProfile,
        jobs: list[JobDescription],
    ) -> None:
        """Full integration: create_matcher → match_job → match_batch with mocks."""
        embed = MockEmbeddingProvider(dimension=8)
        llm = MockLLMProvider()

        matcher = create_matcher(
            embed,
            scoring_provider=llm,
            vector_threshold=0.0,
        )

        single = await match_job(matcher, profile, jobs[0])
        assert isinstance(single, MatchResult)
        assert single.llm_scored is True
        assert single.score == 82.0
        assert single.tokens_used == 700
        assert single.job_company == "Acme Corp"

        batch = await match_batch(matcher, profile, jobs)
        assert isinstance(batch, BatchMatchResult)
        assert batch.jobs_evaluated == 3
        assert batch.llm_scored_count == 3
        assert batch.total_tokens == 3 * 700
        for r in batch.results:
            assert r.llm_scored is True
            assert r.strengths == ["Python expertise", "System design"]
            assert r.gaps == ["No leadership experience"]

    @pytest.mark.asyncio
    async def test_vector_only_pipeline(
        self,
        profile: CandidateProfile,
        jobs: list[JobDescription],
    ) -> None:
        """Pipeline without LLM: only vector scoring."""
        embed = MockEmbeddingProvider(dimension=8)
        matcher = create_matcher(embed, vector_threshold=0.0)

        single = await match_job(matcher, profile, jobs[0])
        assert single.llm_scored is False
        assert single.vector_score is not None
        assert single.tokens_used == 0

        batch = await match_batch(matcher, profile, jobs)
        assert batch.llm_scored_count == 0
        assert batch.total_tokens == 0

    @pytest.mark.asyncio
    async def test_configurable_thresholds_affect_behavior(
        self,
        profile: CandidateProfile,
        jobs: list[JobDescription],
    ) -> None:
        """Verify that vector_threshold controls which jobs get LLM scoring."""
        embed = MockEmbeddingProvider(dimension=8)
        llm_low = MockLLMProvider()
        llm_high = MockLLMProvider()

        matcher_low = create_matcher(
            embed, scoring_provider=llm_low, vector_threshold=0.0
        )
        matcher_high = create_matcher(
            embed, scoring_provider=llm_high, vector_threshold=0.99
        )

        batch_low = await match_batch(matcher_low, profile, jobs)
        batch_high = await match_batch(matcher_high, profile, jobs)

        assert batch_low.llm_scored_count >= batch_high.llm_scored_count
        assert llm_low.call_count >= llm_high.call_count


# ---------------------------------------------------------------------------
# Exports verification
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestPublicExports:
    def test_all_functions_exported_from_init(self) -> None:
        """Documented public API must be importable from strata_match."""
        import strata_match

        assert hasattr(strata_match, "create_matcher")
        assert hasattr(strata_match, "match_job")
        assert hasattr(strata_match, "match_batch")
        assert hasattr(strata_match, "Matcher")
        assert hasattr(strata_match, "MatchResult")
        assert hasattr(strata_match, "BatchMatchResult")
        assert hasattr(strata_match, "CandidateProfile")
        assert hasattr(strata_match, "JobDescription")
        assert hasattr(strata_match, "ConfidenceTier")
        assert not hasattr(strata_match, "LLMProvider")
        assert not hasattr(strata_match, "EmbeddingProvider")

    def test_create_matcher_in_all(self) -> None:
        import strata_match

        assert "create_matcher" in strata_match.__all__
        assert "match_job" in strata_match.__all__
        assert "match_batch" in strata_match.__all__
        assert strata_match.__all__ == sorted(strata_match.__all__)

    def test_provider_factories_not_in_all(self) -> None:
        """Low-level providers stay in submodules, not the package root."""
        import strata_match

        assert not hasattr(strata_match, "create_llm_provider")
        assert "create_llm_provider" not in strata_match.__all__
        assert "create_embedding_provider" not in strata_match.__all__

    def test_help_shows_module_summary(self) -> None:
        """Package docstring should describe the public API (pydoc render)."""
        import pydoc

        import strata_match

        text = pydoc.plain(pydoc.render_doc(strata_match))
        assert "create_matcher" in text
        assert "match_job" in text
        assert "CandidateProfile" in text
        assert "Advanced integrations" in text or "public API" in text
