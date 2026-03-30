"""Tests for strata_match data models."""

import pytest
from pydantic import ValidationError

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
        assert profile.years_of_experience == 0
        assert profile.experience_summary == ""
        assert profile.achievements == []
        assert profile.preferences == {}
        assert profile.embedding is None

    def test_full_profile(self) -> None:
        profile = CandidateProfile(
            title="Senior Engineer",
            skills=["Python", "FastAPI"],
            years_of_experience=8,
            experience_summary="Backend specialist with 8 years in distributed systems.",
            education=["BS CS"],
            achievements=["Led platform migration", "Reduced latency 40%"],
            preferences={"remote": True, "min_salary": 180000},
            embedding=[0.1, 0.2, 0.3],
            certifications=["AWS SAA"],
            industries=["SaaS"],
            preferred_locations=["Remote"],
        )
        assert len(profile.skills) == 2
        assert profile.years_of_experience == 8
        assert profile.experience_summary.startswith("Backend specialist")
        assert len(profile.achievements) == 2
        assert profile.preferences["remote"] is True
        assert profile.embedding == [0.1, 0.2, 0.3]
        assert profile.preferred_locations == ["Remote"]

    def test_years_of_experience_rejects_negative(self) -> None:
        with pytest.raises(ValidationError):
            CandidateProfile(title="Engineer", years_of_experience=-1)

    def test_embedding_optional(self) -> None:
        with_embedding = CandidateProfile(
            title="Engineer", embedding=[0.5, -0.3, 0.1]
        )
        assert with_embedding.embedding == [0.5, -0.3, 0.1]

        without_embedding = CandidateProfile(title="Engineer")
        assert without_embedding.embedding is None

    def test_preferences_accepts_nested_dict(self) -> None:
        profile = CandidateProfile(
            title="Engineer",
            preferences={
                "remote": True,
                "salary_range": {"min": 150000, "max": 200000},
                "industries": ["SaaS", "FinTech"],
            },
        )
        assert profile.preferences["salary_range"]["min"] == 150000

    def test_serialization_roundtrip(self) -> None:
        profile = CandidateProfile(
            title="Staff Engineer",
            skills=["Python", "Go"],
            years_of_experience=10,
            experience_summary="Platform lead.",
            education=["MS CS"],
            achievements=["Built CI/CD pipeline"],
            preferences={"remote": True},
            embedding=[0.1, 0.2],
            certifications=["CKA"],
            industries=["Cloud"],
            preferred_locations=["Remote"],
        )
        data = profile.model_dump()
        restored = CandidateProfile.model_validate(data)
        assert restored == profile

    def test_json_serialization_roundtrip(self) -> None:
        profile = CandidateProfile(
            title="Engineer",
            skills=["Rust"],
            embedding=[1.0, 2.0, 3.0],
            preferences={"level": "senior"},
        )
        json_str = profile.model_dump_json()
        restored = CandidateProfile.model_validate_json(json_str)
        assert restored == profile


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
        result = MatchResult(job_title="Engineer", score=50.0)
        assert result.score == 50.0
        assert result.confidence_tier == ConfidenceTier.LOW
        assert result.is_strong_match is False
        assert result.salary_match is None
        assert result.culture_signals == []
        assert result.tokens_used == 0

    def test_strong_match(self) -> None:
        result = MatchResult(
            job_title="Engineer",
            score=85.0,
            confidence_tier=ConfidenceTier.HIGH,
        )
        assert result.is_strong_match is True

    def test_high_score_low_confidence_not_strong(self) -> None:
        result = MatchResult(
            job_title="Engineer",
            score=90.0,
            confidence_tier=ConfidenceTier.MEDIUM,
        )
        assert result.is_strong_match is False

    def test_score_at_threshold_boundary(self) -> None:
        at_threshold = MatchResult(
            job_title="Engineer",
            score=70.0,
            confidence_tier=ConfidenceTier.HIGH,
        )
        assert at_threshold.is_strong_match is True

        below_threshold = MatchResult(
            job_title="Engineer",
            score=69.9,
            confidence_tier=ConfidenceTier.HIGH,
        )
        assert below_threshold.is_strong_match is False

    def test_score_bounds(self) -> None:
        result = MatchResult(job_title="Engineer", score=0.0)
        assert result.score == 0.0

        result = MatchResult(job_title="Engineer", score=100.0)
        assert result.score == 100.0

    def test_score_out_of_bounds_raises(self) -> None:
        with pytest.raises(ValidationError):
            MatchResult(job_title="Engineer", score=100.1)

        with pytest.raises(ValidationError):
            MatchResult(job_title="Engineer", score=-0.1)

    def test_vector_score_bounds(self) -> None:
        result = MatchResult(job_title="Engineer", score=50.0, vector_score=0.0)
        assert result.vector_score == 0.0

        result = MatchResult(job_title="Engineer", score=50.0, vector_score=100.0)
        assert result.vector_score == 100.0

    def test_vector_score_out_of_bounds_raises(self) -> None:
        with pytest.raises(ValidationError):
            MatchResult(job_title="Engineer", score=50.0, vector_score=100.1)

        with pytest.raises(ValidationError):
            MatchResult(job_title="Engineer", score=50.0, vector_score=-0.1)

    def test_salary_match_field(self) -> None:
        with_salary = MatchResult(job_title="Engineer", score=75.0, salary_match=True)
        assert with_salary.salary_match is True

        no_data = MatchResult(job_title="Engineer", score=75.0)
        assert no_data.salary_match is None

        mismatch = MatchResult(job_title="Engineer", score=75.0, salary_match=False)
        assert mismatch.salary_match is False

    def test_culture_signals_field(self) -> None:
        result = MatchResult(
            job_title="Engineer",
            score=80.0,
            culture_signals=["remote-friendly", "async-first", "engineering-led"],
        )
        assert len(result.culture_signals) == 3
        assert "remote-friendly" in result.culture_signals

    def test_tokens_used_field(self) -> None:
        result = MatchResult(job_title="Engineer", score=80.0, tokens_used=1500)
        assert result.tokens_used == 1500

    def test_tokens_used_rejects_negative(self) -> None:
        with pytest.raises(ValidationError):
            MatchResult(job_title="Engineer", score=50.0, tokens_used=-1)

    def test_full_result_with_all_fields(self) -> None:
        result = MatchResult(
            job_title="Staff Engineer",
            job_company="Acme Corp",
            score=88.5,
            vector_score=72.3,
            confidence_tier=ConfidenceTier.HIGH,
            rationale="Strong backend experience with matching tech stack.",
            strengths=["Python expertise", "System design", "AWS experience"],
            gaps=["No leadership experience listed"],
            salary_match=True,
            culture_signals=["remote-ok", "startup-pace"],
            llm_scored=True,
            tokens_used=2450,
        )
        assert result.is_strong_match is True
        assert result.llm_scored is True
        assert result.tokens_used == 2450
        assert result.salary_match is True

    def test_serialization_roundtrip(self) -> None:
        result = MatchResult(
            job_title="Engineer",
            job_company="Co",
            score=75.0,
            vector_score=68.0,
            confidence_tier=ConfidenceTier.MEDIUM,
            rationale="Decent fit.",
            strengths=["Python"],
            gaps=["Go"],
            salary_match=True,
            culture_signals=["remote"],
            llm_scored=True,
            tokens_used=500,
        )
        data = result.model_dump()
        restored = MatchResult.model_validate(data)
        assert restored.score == result.score
        assert restored.salary_match == result.salary_match
        assert restored.tokens_used == result.tokens_used
        assert restored.culture_signals == result.culture_signals

    def test_json_serialization_roundtrip(self) -> None:
        result = MatchResult(
            job_title="Engineer",
            score=82.0,
            salary_match=False,
            culture_signals=["async"],
            tokens_used=300,
        )
        json_str = result.model_dump_json()
        restored = MatchResult.model_validate_json(json_str)
        assert restored == result


@pytest.mark.verification
class TestBatchMatchResult:
    def test_empty_batch(self) -> None:
        batch = BatchMatchResult()
        assert batch.results == []
        assert batch.strong_matches == []
        assert batch.jobs_evaluated == 0
        assert batch.jobs_skipped == 0
        assert batch.total_tokens == 0
        assert batch.duration_ms == 0.0

    def test_batch_with_mixed_results(self) -> None:
        results = [
            MatchResult(
                job_title="A", score=90.0, confidence_tier=ConfidenceTier.HIGH
            ),
            MatchResult(
                job_title="B", score=60.0, confidence_tier=ConfidenceTier.MEDIUM
            ),
            MatchResult(
                job_title="C", score=30.0, confidence_tier=ConfidenceTier.LOW
            ),
        ]
        batch = BatchMatchResult(
            results=results,
            jobs_evaluated=5,
            jobs_skipped=2,
            total_tokens=3500,
            duration_ms=1250.5,
            llm_scored_count=1,
        )
        assert len(batch.strong_matches) == 1
        assert batch.strong_matches[0].job_title == "A"
        assert batch.jobs_evaluated == 5
        assert batch.jobs_skipped == 2
        assert batch.total_tokens == 3500
        assert batch.duration_ms == 1250.5

    def test_total_tokens_rejects_negative(self) -> None:
        with pytest.raises(ValidationError):
            BatchMatchResult(total_tokens=-1)

    def test_duration_ms_rejects_negative(self) -> None:
        with pytest.raises(ValidationError):
            BatchMatchResult(duration_ms=-1.0)

    def test_serialization_roundtrip(self) -> None:
        batch = BatchMatchResult(
            results=[
                MatchResult(job_title="A", score=85.0, tokens_used=1000),
                MatchResult(job_title="B", score=45.0, tokens_used=800),
            ],
            jobs_evaluated=10,
            jobs_skipped=8,
            total_tokens=1800,
            duration_ms=500.0,
            llm_scored_count=2,
        )
        data = batch.model_dump()
        restored = BatchMatchResult.model_validate(data)
        assert len(restored.results) == 2
        assert restored.jobs_evaluated == batch.jobs_evaluated
        assert restored.total_tokens == batch.total_tokens
        assert restored.duration_ms == batch.duration_ms

    def test_json_serialization_roundtrip(self) -> None:
        batch = BatchMatchResult(
            results=[MatchResult(job_title="X", score=77.0)],
            jobs_evaluated=5,
            jobs_skipped=4,
            total_tokens=200,
            duration_ms=100.0,
        )
        json_str = batch.model_dump_json()
        restored = BatchMatchResult.model_validate_json(json_str)
        assert restored == batch
