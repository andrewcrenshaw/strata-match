"""Prompt template for generating detailed match rationale.

Used when a deeper explanation is needed beyond the initial score_job output —
e.g., for candidate-facing reports or application cover letter context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strata_match.models import CandidateProfile, JobDescription, MatchResult

SYSTEM_PROMPT = """\
You are writing a detailed match rationale for a job seeker. Given a candidate profile,
job description, and initial match assessment, provide an actionable analysis.

Return a JSON object with:
- detailed_rationale: 2-3 paragraph analysis of the match quality
- key_selling_points: list of the candidate's strongest qualifications for this role
- preparation_tips: list of specific actions the candidate should take before applying
- risk_factors: list of potential concerns a hiring manager might have
"""


def build_rationale_prompt(
    profile: CandidateProfile,
    job: JobDescription,
    initial_result: MatchResult,
) -> list[dict[str, str]]:
    """Build the LLM message sequence for detailed rationale generation.

    Returns a list of message dicts compatible with OpenAI/LiteLLM chat format.
    """
    context = (
        f"## Initial Assessment\n\n"
        f"**Score:** {initial_result.score:.2f}\n"
        f"**Confidence:** {initial_result.confidence_tier.value}\n"
        f"**Strengths:** {', '.join(initial_result.strengths) or 'None identified'}\n"
        f"**Gaps:** {', '.join(initial_result.gaps) or 'None identified'}\n"
    )

    from strata_match.prompts.score_job import _format_job, _format_profile

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{_format_profile(profile)}\n\n---\n\n"
                f"{_format_job(job)}\n\n---\n\n"
                f"{context}"
            ),
        },
    ]
