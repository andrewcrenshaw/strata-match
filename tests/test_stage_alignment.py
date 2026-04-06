"""Regression tests for PCC-1632 — stage alignment and batch failure semantics.

AC1: Stage-1 embedding text includes all profile/job fit fields present in Stage-2 prompts.
AC2: Batch matching preserves candidate rows when LLM scoring errors occur (fallback, not drop).
AC3: BatchMatchResult separates threshold skips from LLM fallback/error outcomes.
AC4: _profile_to_text / _job_to_text are semantically aligned with prompt formatters.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from strata_match.llm import LLMProvider, LLMResponse, LLMScorer
from strata_match.matcher import Matcher
from strata_match.models import (
    BatchMatchResult,
    CandidateProfile,
    JobDescription,
)
from strata_match.scoring import VectorScorer, _job_to_text, _profile_to_text
from tests.conftest import FakeEmbeddingProvider

# ---------------------------------------------------------------------------
# AC1 / AC4 — vector serialisation field alignment
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestProfileTextAlignment:
    """_profile_to_text must include all fields that _format_profile includes."""

    def test_education_included_in_profile_text(self) -> None:
        profile = CandidateProfile(
            title="Engineer",
            education=["BS Computer Science", "MBA"],
        )
        text = _profile_to_text(profile)
        assert "BS Computer Science" in text
        assert "MBA" in text

    def test_certifications_included_in_profile_text(self) -> None:
        profile = CandidateProfile(
            title="Engineer",
            certifications=["AWS Certified", "CKA"],
        )
        text = _profile_to_text(profile)
        assert "AWS Certified" in text
        assert "CKA" in text

    def test_achievements_included_in_profile_text(self) -> None:
        profile = CandidateProfile(
            title="Engineer",
            achievements=["Led team of 20", "Reduced latency 40%"],
        )
        text = _profile_to_text(profile)
        assert "Led team of 20" in text
        assert "Reduced latency 40%" in text

    def test_preferred_locations_included_in_profile_text(self) -> None:
        profile = CandidateProfile(
            title="Engineer",
            preferred_locations=["Remote", "New York"],
        )
        text = _profile_to_text(profile)
        assert "Remote" in text
        assert "New York" in text

    def test_preferences_included_in_profile_text(self) -> None:
        profile = CandidateProfile(
            title="Engineer",
            preferences={"remote": "yes", "equity": "required"},
        )
        text = _profile_to_text(profile)
        assert "remote" in text
        assert "equity" in text

    def test_empty_optional_fields_omitted(self) -> None:
        """Empty lists/dicts must not produce blank lines or labels."""
        profile = CandidateProfile(title="Engineer")
        text = _profile_to_text(profile)
        assert "Education:" not in text
        assert "Certifications:" not in text
        assert "Achievements:" not in text
        assert "Preferred Locations:" not in text
        assert "Preferences:" not in text


@pytest.mark.verification
class TestJobTextAlignment:
    """_job_to_text must include all fields that _format_job includes."""

    def test_preferred_qualifications_in_job_text(self) -> None:
        job = JobDescription(
            title="Staff Engineer",
            preferred_qualifications=["PhD", "TensorFlow"],
        )
        text = _job_to_text(job)
        assert "PhD" in text
        assert "TensorFlow" in text

    def test_location_in_job_text(self) -> None:
        job = JobDescription(
            title="Staff Engineer",
            location="San Francisco, CA",
        )
        text = _job_to_text(job)
        assert "San Francisco" in text

    def test_salary_range_in_job_text(self) -> None:
        job = JobDescription(
            title="Staff Engineer",
            salary_range="$180k-$220k",
        )
        text = _job_to_text(job)
        assert "$180k" in text

    def test_employment_type_in_job_text(self) -> None:
        job = JobDescription(
            title="Staff Engineer",
            employment_type="Full-time",
        )
        text = _job_to_text(job)
        assert "Full-time" in text

    def test_empty_optional_job_fields_omitted(self) -> None:
        job = JobDescription(title="Engineer")
        text = _job_to_text(job)
        assert "Preferred:" not in text
        assert "Location:" not in text
        assert "Salary:" not in text
        assert "Employment Type:" not in text


# ---------------------------------------------------------------------------
# AC2 — batch cardinality stability (no silent drops on LLM error)
# ---------------------------------------------------------------------------


class _AlwaysFailLLMProvider(LLMProvider):
    """Provider that always raises on complete()."""

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        raise RuntimeError("simulated LLM error")

    @property
    def model_name(self) -> str:
        return "always-fail"


class _AlwaysOKLLMProvider(LLMProvider):
    """Provider that always succeeds."""

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        return LLMResponse(
            content=json.dumps({"score": 75, "rationale": "Good fit"}),
            input_tokens=10,
            output_tokens=20,
        )

    @property
    def model_name(self) -> str:
        return "always-ok"


@pytest.mark.verification
class TestBatchCardinalityStability:
    """match_batch must preserve a row for every job that passes the vector threshold,
    even when LLM scoring raises an exception (AC2)."""

    @pytest.mark.asyncio
    async def test_llm_error_produces_fallback_row_not_drop(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        embed = FakeEmbeddingProvider(dimension=8)
        llm_scorer = LLMScorer(provider=_AlwaysFailLLMProvider(), max_retries=0)
        matcher = Matcher(
            vector_scorer=VectorScorer(provider=embed),
            vector_threshold=0.0,
            llm_scorer=llm_scorer,
            max_concurrency=5,
        )

        batch = await matcher.match_batch(sample_profile, sample_jobs)

        # All jobs passed threshold → all must appear in results
        assert len(batch.results) == len(sample_jobs), (
            f"Expected {len(sample_jobs)} rows, got {len(batch.results)}; "
            "LLM errors should produce fallback rows, not drops."
        )

    @pytest.mark.asyncio
    async def test_llm_error_row_has_llm_error_set(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        embed = FakeEmbeddingProvider(dimension=8)
        llm_scorer = LLMScorer(provider=_AlwaysFailLLMProvider(), max_retries=0)
        matcher = Matcher(
            vector_scorer=VectorScorer(provider=embed),
            vector_threshold=0.0,
            llm_scorer=llm_scorer,
        )

        batch = await matcher.match_batch(sample_profile, sample_jobs)

        for result in batch.results:
            assert result.llm_error is not None, (
                f"Fallback row for {result.job_title!r} must have llm_error set"
            )
            assert result.llm_scored is False

    @pytest.mark.asyncio
    async def test_partial_fail_preserves_all_rows(
        self, sample_profile: CandidateProfile
    ) -> None:
        """Mix of failing and succeeding LLM calls → all rows preserved."""

        class _PartialFailProvider(LLMProvider):
            async def complete(
                self, messages: list[dict[str, str]], **kwargs: Any
            ) -> LLMResponse:
                # Fail for the first call (detected via title in messages text)
                msg_text = str(messages)
                if "Role 0" in msg_text:
                    raise RuntimeError("partial failure")
                return LLMResponse(
                    content=json.dumps({"score": 80, "rationale": "ok"}),
                    input_tokens=5,
                    output_tokens=10,
                )

            @property
            def model_name(self) -> str:
                return "partial-fail"

        jobs = [
            JobDescription(
                title=f"Role {i}", company="Co", description="D", requirements=["Python"]
            )
            for i in range(3)
        ]

        embed = FakeEmbeddingProvider(dimension=8)
        llm_scorer = LLMScorer(provider=_PartialFailProvider(), max_retries=0)
        matcher = Matcher(
            vector_scorer=VectorScorer(provider=embed),
            vector_threshold=0.0,
            llm_scorer=llm_scorer,
        )

        batch = await matcher.match_batch(sample_profile, jobs)
        assert len(batch.results) == 3, "All rows must be present regardless of per-job LLM failure"


# ---------------------------------------------------------------------------
# AC3 — explicit counter separation
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestBatchCounterSeparation:
    """BatchMatchResult must expose llm_fallback_count separate from jobs_skipped."""

    def test_batch_match_result_has_llm_fallback_count_field(self) -> None:
        result = BatchMatchResult(
            results=[],
            jobs_evaluated=5,
            jobs_skipped=2,
            total_tokens=0,
            duration_ms=0.0,
            llm_scored_count=2,
            llm_fallback_count=1,
        )
        assert result.llm_fallback_count == 1

    def test_llm_fallback_count_defaults_to_zero(self) -> None:
        result = BatchMatchResult()
        assert result.llm_fallback_count == 0

    @pytest.mark.asyncio
    async def test_match_batch_counts_llm_fallbacks_separately(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        embed = FakeEmbeddingProvider(dimension=8)
        llm_scorer = LLMScorer(provider=_AlwaysFailLLMProvider(), max_retries=0)
        matcher = Matcher(
            vector_scorer=VectorScorer(provider=embed),
            vector_threshold=0.0,
            llm_scorer=llm_scorer,
        )

        batch = await matcher.match_batch(sample_profile, sample_jobs)

        # jobs_skipped = threshold skips (0 with threshold=0.0)
        assert batch.jobs_skipped == 0
        # llm_fallback_count = rows where LLM failed → should equal number of jobs
        assert batch.llm_fallback_count == len(sample_jobs)
        # llm_scored_count = rows where LLM succeeded → 0
        assert batch.llm_scored_count == 0

    @pytest.mark.asyncio
    async def test_match_batch_threshold_skips_not_counted_as_fallbacks(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        embed = FakeEmbeddingProvider(dimension=8)
        llm_scorer = LLMScorer(provider=_AlwaysOKLLMProvider(), max_retries=0)
        matcher = Matcher(
            vector_scorer=VectorScorer(provider=embed),
            vector_threshold=0.99,  # High threshold → all skipped
            llm_scorer=llm_scorer,
        )

        batch = await matcher.match_batch(sample_profile, sample_jobs)

        assert batch.jobs_skipped == len(sample_jobs)
        assert batch.llm_fallback_count == 0

    @pytest.mark.asyncio
    async def test_match_batch_mixed_success_and_fallback(
        self, sample_profile: CandidateProfile
    ) -> None:
        """When some LLM calls succeed and others fail, counters are correct."""

        class _MixedProvider(LLMProvider):
            def __init__(self) -> None:
                self._call_count = 0

            async def complete(
                self, messages: list[dict[str, str]], **kwargs: Any
            ) -> LLMResponse:
                self._call_count += 1
                if self._call_count % 2 == 0:
                    raise RuntimeError("even calls fail")
                return LLMResponse(
                    content=json.dumps({"score": 70, "rationale": "ok"}),
                    input_tokens=5,
                    output_tokens=10,
                )

            @property
            def model_name(self) -> str:
                return "mixed"

        jobs = [
            JobDescription(title=f"Job {i}", company="Co", description="D", requirements=["Python"])
            for i in range(4)
        ]

        embed = FakeEmbeddingProvider(dimension=8)
        llm_scorer = LLMScorer(provider=_MixedProvider(), max_retries=0)
        matcher = Matcher(
            vector_scorer=VectorScorer(provider=embed),
            vector_threshold=0.0,
            llm_scorer=llm_scorer,
        )

        batch = await matcher.match_batch(sample_profile, jobs)

        assert len(batch.results) == 4  # All rows preserved
        assert batch.llm_scored_count + batch.llm_fallback_count == 4
        assert batch.jobs_skipped == 0
