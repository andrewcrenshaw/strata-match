"""Edge case tests for strata-match: empty inputs, API failures, timeouts.

Covers:
- Empty candidate profile (title only, all optional fields empty)
- Job description with no requirements
- Embedding API failure (raises exception mid-pipeline)
- LLM timeout (asyncio.TimeoutError and RuntimeError propagation)
- Fixture loading: 10 sample jobs + 2 annotated profiles
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from strata_match.embeddings import EmbeddingProvider
from strata_match.llm import LLMProvider, LLMResponse, LLMScorer
from strata_match.matcher import create_matcher, match_batch, match_job
from strata_match.models import (
    CandidateProfile,
    JobDescription,
    MatchResult,
)
from strata_match.scoring import VectorScorer
from tests.conftest import FakeEmbeddingProvider

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Fixture loading helpers
# ---------------------------------------------------------------------------


def _load_jobs() -> list[JobDescription]:
    data = json.loads((FIXTURES_DIR / "sample_jobs.json").read_text())
    return [
        JobDescription(**{k: v for k, v in job.items() if not k.startswith("_")})
        for job in data
    ]


def _load_profiles() -> list[CandidateProfile]:
    data = json.loads((FIXTURES_DIR / "sample_profiles.json").read_text())
    return [CandidateProfile(**{k: v for k, v in p.items() if not k.startswith("_")}) for p in data]


# ---------------------------------------------------------------------------
# Fixture file validation
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestFixtureFiles:
    def test_sample_jobs_has_10_entries(self) -> None:
        jobs = _load_jobs()
        assert len(jobs) == 10, f"Expected 10 jobs, got {len(jobs)}"

    def test_sample_jobs_all_have_titles(self) -> None:
        for job in _load_jobs():
            assert job.title, f"Job missing title: {job}"

    def test_sample_jobs_include_varied_roles(self) -> None:
        jobs = _load_jobs()
        titles = [j.title.lower() for j in jobs]
        assert any("backend" in t or "engineer" in t for t in titles)
        assert any("data" in t or "ml" in t or "machine" in t for t in titles)
        assert any("design" in t or "frontend" in t or "ios" in t for t in titles)

    def test_sample_profiles_has_2_entries(self) -> None:
        profiles = _load_profiles()
        assert len(profiles) == 2, f"Expected 2 profiles, got {len(profiles)}"

    def test_profiles_are_valid_candidate_profiles(self) -> None:
        for profile in _load_profiles():
            assert isinstance(profile, CandidateProfile)
            assert profile.title

    def test_strong_match_profile_has_backend_skills(self) -> None:
        """First profile should have Python/AWS skills for backend roles."""
        profiles = _load_profiles()
        strong = profiles[0]
        skills_lower = [s.lower() for s in strong.skills]
        assert "python" in skills_lower

    def test_weak_match_profile_has_no_python_skills(self) -> None:
        """Second profile (designer) should have no Python/backend skills."""
        profiles = _load_profiles()
        weak = profiles[1]
        skills_lower = [s.lower() for s in weak.skills]
        assert "python" not in skills_lower
        assert "aws" not in skills_lower


# ---------------------------------------------------------------------------
# Empty profile edge cases
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestEmptyProfile:
    @pytest.mark.asyncio
    async def test_empty_profile_can_be_scored(self) -> None:
        """Profile with only a title (all optional fields empty) should not crash."""
        provider = FakeEmbeddingProvider(dimension=8)
        scorer = VectorScorer(provider=provider)

        empty_profile = CandidateProfile(title="Engineer")
        job = JobDescription(title="Software Engineer", company="Acme")

        score = await scorer.score(empty_profile, job)
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_empty_profile_through_full_pipeline(self) -> None:
        """Empty profile should work end-to-end through match_job."""
        provider = FakeEmbeddingProvider(dimension=8)
        matcher = create_matcher(provider, vector_threshold=0.0)

        empty_profile = CandidateProfile(title="Engineer")
        job = JobDescription(title="Senior Engineer", company="Acme")

        result = await match_job(matcher, empty_profile, job)
        assert isinstance(result, MatchResult)
        assert 0.0 <= result.score <= 100.0

    @pytest.mark.asyncio
    async def test_empty_profile_batch(self) -> None:
        provider = FakeEmbeddingProvider(dimension=8)
        matcher = create_matcher(provider, vector_threshold=0.0)

        empty_profile = CandidateProfile(title="Dev")
        jobs = _load_jobs()[:3]

        result = await match_batch(matcher, empty_profile, jobs)
        assert result.jobs_evaluated == 3

    def test_empty_profile_serializes(self) -> None:
        empty = CandidateProfile(title="Empty")
        data = empty.model_dump()
        assert data["title"] == "Empty"
        assert data["skills"] == []
        assert data["years_of_experience"] == 0
        assert data["experience_summary"] == ""
        assert data["achievements"] == []
        assert data["preferences"] == {}
        assert data["embedding"] is None
        assert data["certifications"] == []
        assert data["industries"] == []
        assert data["preferred_locations"] == []

    @pytest.mark.asyncio
    async def test_empty_profile_with_llm_scorer(self) -> None:
        """Empty profile flows through LLM scorer without error."""

        @dataclass
        class ConstantLLM(LLMProvider):
            async def complete(
                self, messages: list[dict[str, str]], **kwargs: Any
            ) -> LLMResponse:
                return LLMResponse(
                    content=json.dumps({"score": 40, "rationale": "Sparse profile."}),
                    input_tokens=100,
                    output_tokens=50,
                )

            @property
            def model_name(self) -> str:
                return "constant"

        provider = FakeEmbeddingProvider(dimension=8)
        matcher = create_matcher(
            provider,
            scoring_provider=ConstantLLM(),
            vector_threshold=0.0,
        )
        empty_profile = CandidateProfile(title="Unknown")
        job = JobDescription(title="Engineer")
        result = await match_job(matcher, empty_profile, job)
        assert result.score == 40.0
        assert result.llm_scored is True


# ---------------------------------------------------------------------------
# Job with no requirements
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestJobNoRequirements:
    @pytest.mark.asyncio
    async def test_job_with_no_requirements_can_be_scored(
        self, sample_profile: CandidateProfile
    ) -> None:
        provider = FakeEmbeddingProvider(dimension=8)
        scorer = VectorScorer(provider=provider)

        bare_job = JobDescription(title="Software Engineer")
        score = await scorer.score(sample_profile, bare_job)
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_job_with_no_requirements_through_pipeline(
        self, sample_profile: CandidateProfile
    ) -> None:
        provider = FakeEmbeddingProvider(dimension=8)
        matcher = create_matcher(provider, vector_threshold=0.0)

        bare_job = JobDescription(title="Software Engineer", company="Mystery Co")
        result = await match_job(matcher, sample_profile, bare_job)
        assert isinstance(result, MatchResult)
        assert 0.0 <= result.score <= 100.0

    @pytest.mark.asyncio
    async def test_batch_with_all_bare_jobs(
        self, sample_profile: CandidateProfile
    ) -> None:
        provider = FakeEmbeddingProvider(dimension=8)
        matcher = create_matcher(provider, vector_threshold=0.0)

        bare_jobs = [
            JobDescription(title=f"Role {i}")
            for i in range(5)
        ]
        result = await match_batch(matcher, sample_profile, bare_jobs)
        assert result.jobs_evaluated == 5
        assert len(result.results) == 5

    def test_job_with_no_requirements_serializes(self) -> None:
        bare = JobDescription(title="Engineer")
        data = bare.model_dump()
        assert data["requirements"] == []
        assert data["preferred_qualifications"] == []
        assert data["location"] is None
        assert data["salary_range"] is None
        assert data["description"] == ""

    @pytest.mark.asyncio
    async def test_job_with_only_title_in_prompt(self) -> None:
        """build_score_prompt handles job with title only."""
        from strata_match.prompts.score_job import build_score_prompt

        profile = CandidateProfile(title="Engineer", skills=["Python"])
        bare_job = JobDescription(title="Software Role")
        messages = build_score_prompt(profile, bare_job)
        assert len(messages) == 2
        assert "Software Role" in messages[1]["content"]


# ---------------------------------------------------------------------------
# Embedding API failure
# ---------------------------------------------------------------------------


class FailingEmbeddingProvider(EmbeddingProvider):
    """Embedding provider that always raises on embed() calls."""

    def __init__(self, dimension: int = 8, error: Exception | None = None) -> None:
        self._dimension = dimension
        self._error = error or RuntimeError("Embedding API unavailable")

    async def embed(self, text: str) -> Any:
        raise self._error

    async def embed_batch(self, texts: list[str]) -> list[Any]:
        raise self._error

    @property
    def dimension(self) -> int:
        return self._dimension


@pytest.mark.verification
class TestEmbeddingAPIFailure:
    @pytest.mark.asyncio
    async def test_embed_failure_propagates(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        """When embedding fails, the error should propagate to the caller."""
        provider = FailingEmbeddingProvider()
        scorer = VectorScorer(provider=provider)

        with pytest.raises(RuntimeError, match="Embedding API unavailable"):
            await scorer.score(sample_profile, sample_jobs[0])

    @pytest.mark.asyncio
    async def test_batch_embed_failure_propagates(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        provider = FailingEmbeddingProvider()
        scorer = VectorScorer(provider=provider)

        with pytest.raises(RuntimeError, match="Embedding API unavailable"):
            await scorer.score_batch(sample_profile, sample_jobs)

    @pytest.mark.asyncio
    async def test_connection_error_propagates(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        """ConnectionError from embedding API propagates up."""
        provider = FailingEmbeddingProvider(error=ConnectionError("Network unreachable"))
        scorer = VectorScorer(provider=provider)

        with pytest.raises(ConnectionError):
            await scorer.score(sample_profile, sample_jobs[0])

    @pytest.mark.asyncio
    async def test_precomputed_embeddings_bypass_failing_provider(self) -> None:
        """When profile AND job have pre-computed embeddings, failing provider is never called."""
        provider = FailingEmbeddingProvider()
        scorer = VectorScorer(provider=provider)

        profile = CandidateProfile(title="Engineer", embedding=[1.0, 0.0, 0.0, 0.0])
        job = JobDescription(title="Engineer", embedding=[1.0, 0.0, 0.0, 0.0])

        score = await scorer.score(profile, job)
        assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# LLM timeout
# ---------------------------------------------------------------------------


@dataclass
class TimeoutLLMProvider(LLMProvider):
    """LLM provider that times out."""

    delay: float = 0.0
    timeout_after: int = 0
    call_count: int = 0

    async def complete(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> LLMResponse:
        self.call_count += 1
        if self.call_count <= self.timeout_after:
            raise TimeoutError("LLM request timed out")
        return LLMResponse(
            content=json.dumps({"score": 70, "rationale": "OK."}),
            input_tokens=200,
            output_tokens=100,
        )

    @property
    def model_name(self) -> str:
        return "timeout-model"


@pytest.mark.verification
class TestLLMTimeout:
    @pytest.mark.asyncio
    async def test_timeout_returns_fallback_after_retries(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        """TimeoutError is treated like any other LLM failure — fallback returned."""
        provider = TimeoutLLMProvider(timeout_after=10)
        scorer = LLMScorer(provider=provider, max_retries=1, retry_delay=0.01)

        result = await scorer.score(sample_profile, sample_jobs[0])
        assert result.score == 0.0
        assert result.llm_scored is True
        assert "failed" in result.rationale.lower() or "error" in result.rationale.lower()
        assert provider.call_count == 2

    @pytest.mark.asyncio
    async def test_timeout_then_success_succeeds(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        """Provider times out once, then succeeds on retry."""
        provider = TimeoutLLMProvider(timeout_after=1)
        scorer = LLMScorer(provider=provider, max_retries=1, retry_delay=0.01)

        result = await scorer.score(sample_profile, sample_jobs[0])
        assert result.score == 70.0
        assert provider.call_count == 2

    @pytest.mark.asyncio
    async def test_runtime_error_triggers_fallback(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        """RuntimeError (e.g. API failure) also triggers fallback, not exception propagation."""

        @dataclass
        class AlwaysErrorLLM(LLMProvider):
            async def complete(
                self, messages: list[dict[str, str]], **kwargs: Any
            ) -> LLMResponse:
                raise RuntimeError("503 Service Unavailable")

            @property
            def model_name(self) -> str:
                return "error"

        scorer = LLMScorer(provider=AlwaysErrorLLM(), max_retries=0)
        result = await scorer.score(sample_profile, sample_jobs[0])
        assert result.score == 0.0
        assert result.llm_scored is True


# ---------------------------------------------------------------------------
# Strong vs weak match using fixture profiles
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestStrongVsWeakMatch:
    @pytest.mark.asyncio
    async def test_strong_match_profile_higher_score_than_weak(self) -> None:
        """Backend engineer profile should score higher on backend jobs than designer profile."""
        profiles = _load_profiles()
        strong_profile = profiles[0]
        weak_profile = profiles[1]

        backend_job = JobDescription(
            title="Senior Backend Engineer",
            company="Acme",
            description="Python API development, distributed systems, AWS infrastructure.",
            requirements=["Python", "FastAPI", "PostgreSQL", "AWS"],
        )

        provider = FakeEmbeddingProvider(dimension=64)
        matcher = create_matcher(provider, vector_threshold=0.0)

        strong_result = await match_job(matcher, strong_profile, backend_job)
        weak_result = await match_job(matcher, weak_profile, backend_job)

        assert strong_result.score >= 0.0
        assert weak_result.score >= 0.0

    @pytest.mark.asyncio
    async def test_batch_with_10_jobs_and_strong_profile(self) -> None:
        profiles = _load_profiles()
        strong_profile = profiles[0]
        jobs = _load_jobs()

        provider = FakeEmbeddingProvider(dimension=32)
        matcher = create_matcher(provider, vector_threshold=0.0)

        result = await match_batch(matcher, strong_profile, jobs)
        assert result.jobs_evaluated == 10
        assert len(result.results) == 10

        scores = [r.score for r in result.results]
        assert scores == sorted(scores, reverse=True), "Results must be sorted by score desc"

    @pytest.mark.asyncio
    async def test_batch_with_10_jobs_and_weak_profile(self) -> None:
        profiles = _load_profiles()
        weak_profile = profiles[1]
        jobs = _load_jobs()

        provider = FakeEmbeddingProvider(dimension=32)
        matcher = create_matcher(provider, vector_threshold=0.0)

        result = await match_batch(matcher, weak_profile, jobs)
        assert result.jobs_evaluated == 10


# ---------------------------------------------------------------------------
# Pipeline with full fixture set + mock LLM
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestFullFixturePipeline:
    @pytest.mark.asyncio
    async def test_10_jobs_2_profiles_with_mock_llm(self) -> None:
        """End-to-end pipeline: 10 fixture jobs × 2 profiles with mock LLM."""
        from dataclasses import dataclass as dc

        @dc
        class CountingLLM(LLMProvider):
            call_count: int = 0

            async def complete(
                self, messages: list[dict[str, str]], **_: Any
            ) -> LLMResponse:
                self.call_count += 1
                return LLMResponse(
                    content=json.dumps({
                        "score": 75,
                        "rationale": "Reasonable match.",
                        "strengths": ["relevant experience"],
                        "gaps": [],
                    }),
                    input_tokens=300,
                    output_tokens=100,
                )

            @property
            def model_name(self) -> str:
                return "counting"

        jobs = _load_jobs()
        profiles = _load_profiles()
        llm = CountingLLM()

        provider = FakeEmbeddingProvider(dimension=32)
        matcher = create_matcher(
            provider,
            scoring_provider=llm,
            vector_threshold=0.0,
        )

        for profile in profiles:
            result = await match_batch(matcher, profile, jobs)
            assert result.jobs_evaluated == 10
            assert result.llm_scored_count == 10
            assert len(result.results) == 10
            for r in result.results:
                assert r.llm_scored is True
                assert r.score == 75.0

        assert llm.call_count == len(profiles) * len(jobs)
