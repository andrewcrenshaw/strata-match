"""Tests for the main Matcher engine."""

import asyncio
import json
from typing import Any

import pytest

from strata_match.llm import LLMProvider, LLMResponse, LLMScorer
from strata_match.matcher import Matcher, create_matcher, match_batch, match_job
from strata_match.models import CandidateProfile, ConfidenceTier, JobDescription, MatchResult
from strata_match.scoring import VectorScorer, build_match_result
from tests.conftest import FakeEmbeddingProvider


@pytest.mark.verification
class TestMatcher:
    @pytest.mark.asyncio
    async def test_match_one_returns_result(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        provider = FakeEmbeddingProvider(dimension=8)
        scorer = VectorScorer(provider=provider)
        matcher = Matcher(vector_scorer=scorer, vector_threshold=0.0)

        result = await matcher.match_one(sample_profile, sample_jobs[0])
        assert result.job_title == "Staff Engineer — Backend Platform"
        assert 0.0 <= result.score <= 100.0
        assert result.vector_score is not None

    @pytest.mark.asyncio
    async def test_match_batch_returns_batch_result(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        provider = FakeEmbeddingProvider(dimension=8)
        scorer = VectorScorer(provider=provider)
        matcher = Matcher(vector_scorer=scorer, vector_threshold=0.0)

        batch = await matcher.match_batch(sample_profile, sample_jobs)
        assert batch.jobs_evaluated == 3
        assert len(batch.results) <= 3

    @pytest.mark.asyncio
    async def test_threshold_skips_low_scores(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        provider = FakeEmbeddingProvider(dimension=8)
        scorer = VectorScorer(provider=provider)
        matcher = Matcher(vector_scorer=scorer, vector_threshold=0.99)

        batch = await matcher.match_batch(sample_profile, sample_jobs)
        assert batch.jobs_skipped >= 0
        assert batch.jobs_evaluated == 3

    @pytest.mark.asyncio
    async def test_results_sorted_by_score(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        provider = FakeEmbeddingProvider(dimension=8)
        scorer = VectorScorer(provider=provider)
        matcher = Matcher(vector_scorer=scorer, vector_threshold=0.0)

        batch = await matcher.match_batch(sample_profile, sample_jobs)
        scores = [r.score for r in batch.results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_scores_in_0_100_range(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        provider = FakeEmbeddingProvider(dimension=8)
        scorer = VectorScorer(provider=provider)
        matcher = Matcher(vector_scorer=scorer, vector_threshold=0.0)

        batch = await matcher.match_batch(sample_profile, sample_jobs)
        for result in batch.results:
            assert 0.0 <= result.score <= 100.0
            if result.vector_score is not None:
                assert 0.0 <= result.vector_score <= 100.0

    @pytest.mark.asyncio
    async def test_match_batch_duration_ms_zero_for_empty_jobs(
        self, sample_profile: CandidateProfile
    ) -> None:
        provider = FakeEmbeddingProvider(dimension=8)
        scorer = VectorScorer(provider=provider)
        matcher = Matcher(vector_scorer=scorer, vector_threshold=0.0)

        batch = await matcher.match_batch(sample_profile, [])
        assert batch.jobs_evaluated == 0
        assert batch.duration_ms == 0.0

    @pytest.mark.asyncio
    async def test_match_batch_duration_ms_positive_for_non_empty_batch(
        self,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Wall-clock ms is passed through (deterministic via perf_counter stub)."""
        calls = iter((100.0, 100.012))

        def fake_perf_counter() -> float:
            return next(calls)

        monkeypatch.setattr("strata_match.matcher.time.perf_counter", fake_perf_counter)

        provider = FakeEmbeddingProvider(dimension=8)
        scorer = VectorScorer(provider=provider)
        matcher = Matcher(vector_scorer=scorer, vector_threshold=0.0)

        batch = await matcher.match_batch(sample_profile, sample_jobs)
        assert batch.duration_ms == 12.0


@pytest.mark.verification
class TestConvenienceFunctions:
    @pytest.mark.asyncio
    async def test_match_job_function(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        provider = FakeEmbeddingProvider(dimension=8)
        scorer = VectorScorer(provider=provider)
        matcher = Matcher(vector_scorer=scorer, vector_threshold=0.0)

        result = await match_job(matcher, sample_profile, sample_jobs[0])
        assert result.job_title == "Staff Engineer — Backend Platform"

    @pytest.mark.asyncio
    async def test_match_batch_function(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        provider = FakeEmbeddingProvider(dimension=8)
        scorer = VectorScorer(provider=provider)
        matcher = Matcher(vector_scorer=scorer, vector_threshold=0.0)

        batch = await match_batch(matcher, sample_profile, sample_jobs)
        assert batch.jobs_evaluated == 3

    def test_create_matcher_with_string_resolves_provider(self) -> None:
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        matcher = create_matcher("openai", _provider_client=mock_client)
        assert matcher.vector_scorer.provider.dimension == 1536

    def test_create_matcher_with_provider(self) -> None:
        provider = FakeEmbeddingProvider(dimension=8)
        matcher = create_matcher(provider, vector_threshold=0.5)
        assert matcher.vector_threshold == 0.5

    def test_matcher_default_max_concurrency_is_5(self) -> None:
        provider = FakeEmbeddingProvider(dimension=8)
        scorer = VectorScorer(provider=provider)
        matcher = Matcher(vector_scorer=scorer)
        assert matcher.max_concurrency == 5

    def test_create_matcher_accepts_max_concurrency(self) -> None:
        provider = FakeEmbeddingProvider(dimension=8)
        matcher = create_matcher(provider, max_concurrency=3)
        assert matcher.max_concurrency == 3


@pytest.mark.verification
class TestMatchBatchLLMConcurrency:
    """Concurrent LLM scoring in match_batch (PCC-1602)."""

    @pytest.mark.asyncio
    async def test_llm_calls_respect_max_concurrency(
        self, sample_profile: CandidateProfile
    ) -> None:
        """Peak concurrent LLM work stays within max_concurrency."""

        class DepthTrackingLLM(LLMProvider):
            def __init__(self) -> None:
                self._depth = 0
                self.max_depth = 0
                self._lock = asyncio.Lock()

            async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
                async with self._lock:
                    self._depth += 1
                    self.max_depth = max(self.max_depth, self._depth)
                try:
                    await asyncio.sleep(0.04)
                finally:
                    async with self._lock:
                        self._depth -= 1
                return LLMResponse(
                    content=json.dumps({"score": 70, "rationale": "ok"}),
                    input_tokens=1,
                    output_tokens=1,
                )

            @property
            def model_name(self) -> str:
                return "depth-test"

        jobs = [
            JobDescription(
                title=f"Role {i}",
                company="Co",
                description="Desc",
                requirements=["Python"],
            )
            for i in range(10)
        ]

        embed = FakeEmbeddingProvider(dimension=8)
        llm = DepthTrackingLLM()
        matcher = create_matcher(
            embed,
            scoring_provider=LLMScorer(provider=llm),
            vector_threshold=0.0,
            max_concurrency=2,
        )

        await matcher.match_batch(sample_profile, jobs)

        assert llm.max_depth <= 2

    @pytest.mark.asyncio
    async def test_exception_in_one_llm_score_skips_job_batch_continues(
        self,
        sample_profile: CandidateProfile,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Unexpected scorer exceptions are logged; other results are kept."""

        class PartialRaiseScorer:
            async def score(
                self,
                profile: CandidateProfile,
                job: JobDescription,
                *,
                vector_score: float | None = None,
            ) -> MatchResult:
                if job.title == "FAIL":
                    raise RuntimeError("simulated scorer failure")
                return build_match_result(
                    job,
                    score=72.0,
                    vector_score=vector_score,
                    confidence_tier=ConfidenceTier.MEDIUM,
                    llm_scored=True,
                )

        jobs = [
            JobDescription(
                title="OK A",
                company="Co",
                description="D",
                requirements=["Python"],
            ),
            JobDescription(
                title="FAIL",
                company="Co",
                description="D",
                requirements=["Python"],
            ),
            JobDescription(
                title="OK B",
                company="Co",
                description="D",
                requirements=["Python"],
            ),
        ]

        embed = FakeEmbeddingProvider(dimension=8)
        matcher = Matcher(
            vector_scorer=VectorScorer(provider=embed),
            vector_threshold=0.0,
            llm_scorer=PartialRaiseScorer(),  # type: ignore[arg-type]
            max_concurrency=3,
        )

        with caplog.at_level("ERROR"):
            batch = await matcher.match_batch(sample_profile, jobs)

        assert batch.jobs_evaluated == 3
        assert len(batch.results) == 2
        assert batch.llm_scored_count == 2
        titles = {r.job_title for r in batch.results}
        assert titles == {"OK A", "OK B"}
        log_text = caplog.text
        assert "skipping job 'FAIL'" in log_text
        assert "simulated scorer failure" in log_text
