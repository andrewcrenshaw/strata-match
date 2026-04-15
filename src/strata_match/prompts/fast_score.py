"""Simplified prompt for fast-pass LLM triage scoring (Stage 2A, PCC-1891).

Designed to produce ~30-token output (~1-2s/call) for quick triage of all
above-threshold listings.  Only a numeric score is returned; listings that
pass the threshold go on to full deep scoring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strata_match.models import CandidateProfile, JobDescription

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """\
You are a job-match triage engine. Given a candidate profile and a job description,
output ONLY a JSON object with one field:

{"score": N}

where N is an integer 0-100 representing match quality. No explanation, no other
fields — just the score.
"""


def build_fast_score_prompt(
    profile: CandidateProfile,
    job: JobDescription,
) -> list[dict[str, str]]:
    """Build the triage scoring message sequence (fast pass, ~30 token output).

    Reuses the same profile/job formatting as the full scorer so the candidate
    data is comparable.  Only the system prompt differs — instructing the model
    to return a bare score JSON.
    """
    from strata_match.prompts.score_job import _format_job, _format_profile

    profile_block = _format_profile(profile)
    job_block = _format_job(job)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{profile_block}\n\n---\n\n{job_block}"},
    ]
