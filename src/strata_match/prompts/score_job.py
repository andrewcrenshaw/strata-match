"""Prompt template for LLM-based job match scoring.

Design: static prefix (CIP) + dynamic suffix (job) to maximize prompt caching.
The candidate profile is placed first so it can be cached across multiple
job evaluations within a single matching session.

For Anthropic prompt caching, use ``build_score_prompt_parts`` to obtain the
static and dynamic portions separately. Mark the static prefix with
``cache_control: {"type": "ephemeral"}`` — this achieves ~90% cache hit rate
when scoring many jobs against the same profile.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strata_match.models import CandidateProfile, JobDescription

PROMPT_VERSION = "v4"

SYSTEM_PROMPT = """\
You are a job-match scoring engine. Given a candidate profile and a job description,
evaluate the match quality on a scale of 0 to 100.

Return a JSON object with these fields:
- score: integer between 0 and 100
- strengths: list of specific strengths the candidate brings to this role
- gaps: list of actionable gaps. Each gap must name exactly ONE learnable skill, tool, or \
credential the candidate is missing — one skill per gap. Cite the specific profile attribute \
(skill, title, or experience entry) that makes this gap concrete. Do not infer personality \
traits, leadership style, cultural fit, or working-style mismatches; route seniority \
expectations, scope preferences, title concerns, compensation mismatches, and culture \
observations to job_fit_signals instead.
- rationale: one-paragraph explanation of the score
- salary_match: boolean indicating whether salary expectations align (true/false), \
or null if salary info is unavailable
- culture_signals: list of cultural fit indicators observed in the job description \
(e.g. "remote-friendly", "startup-pace", "engineering-led")

When culture_signals are present in the JD, note alignment or tension with the \
candidate's values and working style in the rationale. Do not penalize mismatches \
unless they are fundamental (e.g., candidate is async-first but JD explicitly \
requires 5-day onsite).
- what_they_want: a structured string that reads between the lines of the job description \
to surface what the hiring team *actually* needs. Follow this exact format:

  This is a **[Role Archetype]** role. You will need to:
  1. **[Key Need 1]:** [Why this matters / what it signals about the team]
  2. **[Key Need 2]:** [Why this matters / what it signals about the team]
  3. **[Key Need 3]:** [Why this matters / what it signals about the team]
  Your **"[candidate proof point from profile]"** and \
**"[second candidate proof point]"** story is the ultimate proof that you can \
[deliver what they need].

  Rules for what_they_want:
  * Role Archetype: one compact phrase (e.g. "senior IC infrastructure owner", \
"player-coach engineering manager", "growth-focused product generalist").
  * Key Needs: infer the real priorities by reading between the JD lines — \
not just the bullet points. Look for signals in language, team stage, and company context.
  * Proof points: cite specific, concrete items from the candidate profile \
(achievements, projects, skills, titles) that map directly to each need.
  * If you cannot infer sufficient context, set what_they_want to an empty string.

Be precise. A 80+ score means the candidate is a strong fit with minimal gaps.
A 50-80 score means a reasonable fit with some gaps. Below 50 means weak alignment.
"""


def build_score_prompt(
    profile: CandidateProfile,
    job: JobDescription,
) -> list[dict[str, str]]:
    """Build the LLM message sequence for job match scoring.

    Returns a list of message dicts compatible with OpenAI/LiteLLM chat format.
    Profile content precedes the job description so the static prefix can be
    cached by prompt-caching-aware callers (see ``build_score_prompt_parts``).
    """
    static_prefix, dynamic_suffix = build_score_prompt_parts(profile, job)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{static_prefix}\n\n---\n\n{dynamic_suffix}"},
    ]


def build_score_prompt_parts(
    profile: CandidateProfile,
    job: JobDescription,
) -> tuple[str, str]:
    """Return ``(static_prefix, dynamic_suffix)`` for the score prompt.

    The split enables Anthropic-style prompt caching:

    - **static_prefix** — candidate profile block (identical across all job
      evaluations for the same profile; mark with ``cache_control`` for caching).
    - **dynamic_suffix** — job description block (varies per job; always sent
      fresh, never cached).

    Example — Anthropic cache-aware usage::

        static, dynamic = build_score_prompt_parts(profile, job)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": static,
                     "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": dynamic},
                ],
            },
        ]

    When scoring *N* jobs for a single profile the cache hit rate approaches
    ``(N-1)/N``  — ~90 % at N=10.
    """
    return _format_profile(profile), _format_job(job)


def _format_profile(profile: CandidateProfile) -> str:
    lines = [f"## Candidate Profile\n\n**Title:** {profile.title}"]
    if profile.experience_summary:
        lines.append(f"**Summary:** {profile.experience_summary}")
    if profile.skills:
        lines.append(f"**Skills:** {', '.join(profile.skills)}")
    if profile.years_of_experience:
        lines.append(f"**Experience:** {profile.years_of_experience} years")
    if profile.education:
        lines.append(f"**Education:** {', '.join(profile.education)}")
    if profile.certifications:
        lines.append(f"**Certifications:** {', '.join(profile.certifications)}")
    if profile.industries:
        lines.append(f"**Industries:** {', '.join(profile.industries)}")
    if profile.achievements:
        lines.append(f"**Achievements:** {', '.join(profile.achievements)}")
    if profile.preferred_locations:
        lines.append(f"**Preferred Locations:** {', '.join(profile.preferred_locations)}")
    if profile.values_and_culture:
        lines.append(f"**Candidate Values:** {', '.join(profile.values_and_culture)}")
    if profile.working_style:
        lines.append(f"**Working Style:** {', '.join(profile.working_style)}")
    if profile.preferences:
        formatted = ", ".join(f"{k}: {v}" for k, v in profile.preferences.items())
        lines.append(f"**Preferences:** {formatted}")
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
    if job.employment_type:
        lines.append(f"**Employment Type:** {job.employment_type}")
    return "\n".join(lines)
