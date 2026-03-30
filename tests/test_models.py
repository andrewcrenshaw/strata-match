"""Tests for strata_match data models."""

import pytest

from strata_match.models import (
    BatchMatchResult,
    CandidateProfile,
    ConfidenceTier,
    JobDescription,
    MatchResult,
)


@pytest.mark.verification
class TestCandidateProfile:
    def test_minimal_profile(self) -> None:
        profile = CandidateProfile(title="Engineer")
        assert profile.title == "Engineer"
        assert profile.skills == []
        assert profile.experience_years == 0
        assert profile.summary == ""

    def test_full_profile(self) -> None:
        profile = CandidateProfile(
            title="Senior Engineer",
            skills=["Python", "FastAPI"],
            experience_years=8,
            summary="Backend specialist.",
            education=["BS CS"],
            certifications=["AWS SAA"],
            industries=["SaaS"],
            preferred_locations=["Remote"],
        )
        assert len(profile.skills) == 2
        assert profile.experience_years == 8
        assert profile.preferred_locations == ["Remote"]


@pytest.mark.verification
class TestJobDescription:
    def test_minimal_job(self) -> None:
        job = JobDescription(title="Engineer")
        assert job.title == "Engineer"
        assert job.company == ""
        assert job.requirements == []
        assert job.location is None

    def test_full_job(self) -> None:
        job = JobDescription(
            title="Staff Engineer",
            company="Acme Corp",
            description="Lead platform team.",
            requirements=["Python", "System Design"],
            preferred_qualifications=["PhD"],
            location="Remote",
            salary_range="$180k-$220k",
            employment_type="Full-time",
            external_id="acme-123",
        )
        assert job.company == "Acme Corp"
        assert len(job.requirements) == 2
        assert job.external_id == "acme-123"


@pytest.mark.verification
class TestMatchResult:
    def test_minimal_result(self) -> None:
        result = MatchResult(job_title="Engineer", score=0.5)
        assert result.score == 0.5
        assert result.confidence_tier == ConfidenceTier.LOW
        assert result.is_strong_match is False

    def test_strong_match(self) -> None:
        result = MatchResult(
            job_title="Engineer",
            score=0.85,
            confidence_tier=ConfidenceTier.HIGH,
        )
        assert result.is_strong_match is True

    def test_high_score_low_confidence_not_strong(self) -> None:
        result = MatchResult(
            job_title="Engineer",
            score=0.9,
            confidence_tier=ConfidenceTier.MEDIUM,
        )
        assert result.is_strong_match is False

    def test_score_bounds(self) -> None:
        result = MatchResult(job_title="Engineer", score=0.0)
        assert result.score == 0.0

        result = MatchResult(job_title="Engineer", score=1.0)
        assert result.score == 1.0

    def test_score_out_of_bounds_raises(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MatchResult(job_title="Engineer", score=1.5)

        with pytest.raises(ValidationError):
            MatchResult(job_title="Engineer", score=-0.1)


@pytest.mark.verification
class TestBatchMatchResult:
    def test_empty_batch(self) -> None:
        batch = BatchMatchResult()
        assert batch.results == []
        assert batch.strong_matches == []
        assert batch.total_jobs == 0

    def test_batch_with_mixed_results(self) -> None:
        results = [
            MatchResult(
                job_title="A", score=0.9, confidence_tier=ConfidenceTier.HIGH
            ),
            MatchResult(
                job_title="B", score=0.6, confidence_tier=ConfidenceTier.MEDIUM
            ),
            MatchResult(
                job_title="C", score=0.3, confidence_tier=ConfidenceTier.LOW
            ),
        ]
        batch = BatchMatchResult(
            results=results,
            total_jobs=5,
            skipped_below_threshold=2,
            llm_scored_count=1,
        )
        assert len(batch.strong_matches) == 1
        assert batch.strong_matches[0].job_title == "A"
        assert batch.total_jobs == 5
        assert batch.skipped_below_threshold == 2
