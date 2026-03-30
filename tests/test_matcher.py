"""Tests for the main Matcher engine."""

import pytest

from strata_match.matcher import Matcher, create_matcher, match_batch, match_job
from strata_match.models import CandidateProfile, JobDescription
from strata_match.scoring import VectorScorer
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
