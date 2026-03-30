"""Prompt template for LLM-based job match scoring.

Design: static prefix (CIP) + dynamic suffix (job) to maximize prompt caching.
The candidate profile is placed first so it can be cached across multiple
job evaluations within a single matching session.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strata_match.models import CandidateProfile, JobDescription

SYSTEM_PROMPT = """\
You are a job-match scoring engine. Given a candidate profile and a job description,
evaluate the match quality on a scale of 0.0 to 1.0.

Return a JSON object with these fields:
- score: float between 0.0 and 1.0
- strengths: list of specific strengths the candidate brings to this role
- gaps: list of specific gaps or missing qualifications
- rationale: one-paragraph explanation of the score

Be precise. A 0.8+ score means the candidate is a strong fit with minimal gaps.
A 0.5-0.8 score means a reasonable fit with some gaps. Below 0.5 means weak alignment.
"""


def build_score_prompt(
    profile: CandidateProfile,
    job: JobDescription,
) -> list[dict[str, str]]:
    """Build the LLM message sequence for job match scoring.

    Returns a list of message dicts compatible with OpenAI/LiteLLM chat format.
    """
    profile_block = _format_profile(profile)
    job_block = _format_job(job)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{profile_block}\n\n---\n\n{job_block}"},
    ]


def _format_profile(profile: CandidateProfile) -> str:
    lines = [f"## Candidate Profile\n\n**Title:** {profile.title}"]
    if profile.summary:
        lines.append(f"**Summary:** {profile.summary}")
    if profile.skills:
        lines.append(f"**Skills:** {', '.join(profile.skills)}")
    if profile.experience_years:
        lines.append(f"**Experience:** {profile.experience_years} years")
    if profile.education:
        lines.append(f"**Education:** {', '.join(profile.education)}")
    if profile.certifications:
        lines.append(f"**Certifications:** {', '.join(profile.certifications)}")
    if profile.industries:
        lines.append(f"**Industries:** {', '.join(profile.industries)}")
    return "\n".join(lines)


def _format_job(job: JobDescription) -> str:
    lines = [f"## Job Description\n\n**Title:** {job.title}"]
    if job.company:
        lines.append(f"**Company:** {job.company}")
    if job.description:
        lines.append(f"**Description:** {job.description}")
    if job.requirements:
        lines.append(f"**Requirements:** {', '.join(job.requirements)}")
    if job.preferred_qualifications:
        lines.append(f"**Preferred:** {', '.join(job.preferred_qualifications)}")
    if job.location:
        lines.append(f"**Location:** {job.location}")
    if job.salary_range:
        lines.append(f"**Salary:** {job.salary_range}")
    return "\n".join(lines)
