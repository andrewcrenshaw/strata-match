"""Tests for LLM prompt templates."""

import pytest

from strata_match.models import (
    CandidateProfile,
    ConfidenceTier,
    JobDescription,
    MatchResult,
)
from strata_match.prompts.rationale import PROMPT_VERSION as RATIONALE_VERSION
from strata_match.prompts.rationale import build_rationale_prompt
from strata_match.prompts.score_job import PROMPT_VERSION as SCORE_VERSION
from strata_match.prompts.score_job import build_score_prompt, build_score_prompt_parts


@pytest.mark.verification
class TestScoreJobPrompt:
    def test_builds_message_list(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        messages = build_score_prompt(sample_profile, sample_jobs[0])
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_system_prompt_contains_scoring_guidance(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        messages = build_score_prompt(sample_profile, sample_jobs[0])
        system = messages[0]["content"]
        assert "score" in system.lower()
        assert "strengths" in system.lower()
        assert "gaps" in system.lower()

    def test_user_message_contains_profile_and_job(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        messages = build_score_prompt(sample_profile, sample_jobs[0])
        user_msg = messages[1]["content"]
        assert "Senior Software Engineer" in user_msg
        assert "Staff Engineer" in user_msg
        assert "Python" in user_msg

    def test_minimal_profile(self) -> None:
        profile = CandidateProfile(title="Engineer")
        job = JobDescription(title="Role")
        messages = build_score_prompt(profile, job)
        assert len(messages) == 2
        assert "Engineer" in messages[1]["content"]

    def test_profile_appears_before_job_in_user_message(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        """Profile (static) must precede job (dynamic) for caching to work."""
        messages = build_score_prompt(sample_profile, sample_jobs[0])
        user_msg = messages[1]["content"]
        profile_pos = user_msg.index("Senior Software Engineer")
        job_pos = user_msg.index("Staff Engineer")
        assert profile_pos < job_pos, "Profile block must appear before job block"


@pytest.mark.verification
class TestScorePromptParts:
    """Verify the static/dynamic split used for Anthropic prompt caching."""

    def test_returns_two_strings(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        result = build_score_prompt_parts(sample_profile, sample_jobs[0])
        assert isinstance(result, tuple)
        assert len(result) == 2
        static_prefix, dynamic_suffix = result
        assert isinstance(static_prefix, str)
        assert isinstance(dynamic_suffix, str)

    def test_static_prefix_contains_profile(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        static_prefix, _ = build_score_prompt_parts(sample_profile, sample_jobs[0])
        assert "Senior Software Engineer" in static_prefix
        assert "Python" in static_prefix

    def test_dynamic_suffix_contains_job(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        _, dynamic_suffix = build_score_prompt_parts(sample_profile, sample_jobs[0])
        assert "Staff Engineer" in dynamic_suffix
        assert "Acme Corp" in dynamic_suffix

    def test_static_prefix_excludes_job(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        static_prefix, _ = build_score_prompt_parts(sample_profile, sample_jobs[0])
        assert "Acme Corp" not in static_prefix

    def test_dynamic_suffix_excludes_profile_summary(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        _, dynamic_suffix = build_score_prompt_parts(sample_profile, sample_jobs[0])
        assert "Full-stack engineer" not in dynamic_suffix

    def test_static_prefix_is_identical_across_jobs(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        """Same profile → same static prefix regardless of which job is evaluated."""
        prefix_a, _ = build_score_prompt_parts(sample_profile, sample_jobs[0])
        prefix_b, _ = build_score_prompt_parts(sample_profile, sample_jobs[1])
        assert prefix_a == prefix_b

    def test_dynamic_suffix_differs_across_jobs(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        _, suffix_a = build_score_prompt_parts(sample_profile, sample_jobs[0])
        _, suffix_b = build_score_prompt_parts(sample_profile, sample_jobs[1])
        assert suffix_a != suffix_b

    def test_parts_consistent_with_full_prompt(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        """Content from parts must all appear in the full build_score_prompt user message."""
        static_prefix, dynamic_suffix = build_score_prompt_parts(
            sample_profile, sample_jobs[0]
        )
        messages = build_score_prompt(sample_profile, sample_jobs[0])
        user_msg = messages[1]["content"]
        assert static_prefix in user_msg
        assert dynamic_suffix in user_msg

    def test_minimal_profile_and_job(self) -> None:
        profile = CandidateProfile(title="Dev")
        job = JobDescription(title="Role")
        static_prefix, dynamic_suffix = build_score_prompt_parts(profile, job)
        assert "Dev" in static_prefix
        assert "Role" in dynamic_suffix

    def test_static_prefix_includes_certifications_when_present(self) -> None:
        """Certifications should appear in the static prefix (prompt line 103)."""
        profile = CandidateProfile(
            title="Engineer",
            certifications=["AWS SAA", "CKA"],
        )
        job = JobDescription(title="Role")
        static_prefix, _ = build_score_prompt_parts(profile, job)
        assert "AWS SAA" in static_prefix
        assert "CKA" in static_prefix
        assert "Certifications" in static_prefix

    def test_dynamic_suffix_includes_preferred_qualifications_when_present(self) -> None:
        """Preferred qualifications should appear in the dynamic suffix (prompt line 118)."""
        profile = CandidateProfile(title="Engineer")
        job = JobDescription(
            title="Staff Engineer",
            company="Acme",
            preferred_qualifications=["PhD", "10+ years experience"],
        )
        _, dynamic_suffix = build_score_prompt_parts(profile, job)
        assert "PhD" in dynamic_suffix
        assert "10+ years experience" in dynamic_suffix
        assert "Preferred" in dynamic_suffix


@pytest.mark.verification
class TestPromptVersioning:
    def test_score_prompt_version_is_non_empty_string(self) -> None:
        assert isinstance(SCORE_VERSION, str)
        assert len(SCORE_VERSION) > 0

    def test_rationale_prompt_version_is_non_empty_string(self) -> None:
        assert isinstance(RATIONALE_VERSION, str)
        assert len(RATIONALE_VERSION) > 0

    def test_match_result_stores_prompt_version(self) -> None:
        result = MatchResult(
            job_title="Engineer",
            score=80.0,
            prompt_version=SCORE_VERSION,
        )
        assert result.prompt_version == SCORE_VERSION

    def test_match_result_prompt_version_defaults_to_none(self) -> None:
        result = MatchResult(job_title="Engineer", score=50.0)
        assert result.prompt_version is None

    def test_match_result_prompt_version_is_serialised(self) -> None:
        result = MatchResult(
            job_title="Engineer",
            score=80.0,
            prompt_version="v1",
        )
        data = result.model_dump()
        assert data["prompt_version"] == "v1"


@pytest.mark.verification
class TestRationalePrompt:
    def test_builds_message_list(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        initial = MatchResult(
            job_title="Staff Engineer",
            score=75.0,
            confidence_tier=ConfidenceTier.MEDIUM,
            strengths=["Python"],
            gaps=["Leadership"],
        )
        messages = build_rationale_prompt(
            sample_profile, sample_jobs[0], initial
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"

    def test_includes_initial_assessment(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        initial = MatchResult(
            job_title="Staff Engineer",
            score=75.0,
            confidence_tier=ConfidenceTier.MEDIUM,
            strengths=["Python"],
            gaps=["Leadership"],
        )
        messages = build_rationale_prompt(
            sample_profile, sample_jobs[0], initial
        )
        user_msg = messages[1]["content"]
        assert "75.00" in user_msg
        assert "medium" in user_msg
        assert "Python" in user_msg
        assert "Leadership" in user_msg

    def test_system_prompt_mentions_strengths_and_gaps(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        initial = MatchResult(job_title="Staff Engineer", score=60.0)
        messages = build_rationale_prompt(sample_profile, sample_jobs[0], initial)
        system = messages[0]["content"]
        assert "strengths" in system.lower() or "selling" in system.lower()
        assert "risk" in system.lower() or "gaps" in system.lower()
