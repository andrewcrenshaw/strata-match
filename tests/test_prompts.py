"""Tests for LLM prompt templates."""

import pytest

from strata_match.models import (
    CandidateProfile,
    ConfidenceTier,
    JobDescription,
    MatchResult,
)
from strata_match.prompts.rationale import build_rationale_prompt
from strata_match.prompts.score_job import build_score_prompt


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


@pytest.mark.verification
class TestRationalePrompt:
    def test_builds_message_list(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        initial = MatchResult(
            job_title="Staff Engineer",
            score=0.75,
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
            score=0.75,
            confidence_tier=ConfidenceTier.MEDIUM,
            strengths=["Python"],
            gaps=["Leadership"],
        )
        messages = build_rationale_prompt(
            sample_profile, sample_jobs[0], initial
        )
        user_msg = messages[1]["content"]
        assert "0.75" in user_msg
        assert "medium" in user_msg
        assert "Python" in user_msg
        assert "Leadership" in user_msg
