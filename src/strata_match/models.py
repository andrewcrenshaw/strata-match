"""Public data models for strata-match."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ConfidenceTier(StrEnum):
    """Match confidence classification — four tiers from strongest to weakest.

    VERY_HIGH: Exceptional semantic AND LLM-confirmed fit.  Display as "Very High".
    HIGH:      Strong fit confirmed by both vector and LLM.  Display as "High".
    MEDIUM:    Meaningful overlap; worth exploring.            Display as "Medium".
    LOW:       Weak fit — above the vector floor but not actionable. Display as "Low".
    """

    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CandidateProfile(BaseModel):
    """Public candidate profile for matching (no PII fields).

    Attributes map to embedding text and optional pre-computed ``embedding``
    vectors. Use :class:`JobDescription` for the job side.

    Example::

        CandidateProfile(
            title="Senior Engineer",
            skills=["Python", "PostgreSQL"],
            years_of_experience=8,
            experience_summary="Backend and distributed systems.",
        )
    """

    title: str
    skills: list[str] = Field(default_factory=list)
    experience_summary: str = ""
    years_of_experience: int = Field(default=0, ge=0)
    education: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    preferences: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = Field(
        default=None, description="Optional pre-computed embedding vector"
    )
    certifications: list[str] = Field(default_factory=list)
    industries: list[str] = Field(default_factory=list)
    preferred_locations: list[str] = Field(default_factory=list)
    values_and_culture: list[str] = Field(default_factory=list)
    working_style: list[str] = Field(default_factory=list)


class JobDescription(BaseModel):
    """Structured job description for matching.

    Together with :class:`CandidateProfile`, this is the input to
    :func:`~strata_match.matcher.match_job` / :func:`~strata_match.matcher.match_batch`.

    Example::

        JobDescription(
            title="Staff Engineer",
            company="Acme",
            requirements=["Python", "Leadership"],
            description="Lead the platform team.",
        )
    """

    title: str
    company: str = ""
    description: str = ""
    requirements: list[str] = Field(default_factory=list)
    preferred_qualifications: list[str] = Field(default_factory=list)
    location: str | None = None
    salary_range: str | None = None
    employment_type: str | None = None
    external_id: str | None = None
    embedding: list[float] | None = Field(
        default=None, description="Optional pre-computed embedding vector"
    )
    ann_score: float | None = Field(
        default=None,
        description=(
            "Pre-computed ANN cosine similarity in [0, 1] from SQL ANN retrieval (PCC-1895). "
            "When set, VectorScorer returns this directly, skipping embed + cosine."
        ),
    )


class MatchResult(BaseModel):
    """Result of matching a single job against a candidate profile.

    ``score`` is the primary signal (0–100). When LLM scoring **succeeded**,
    ``llm_scored`` is ``True``, ``llm_error`` is ``None``, and ``rationale`` /
    ``strengths`` / ``gaps`` are populated. When LLM scoring was **attempted**
    but failed, ``llm_scored`` is ``False`` and ``llm_error`` explains why.
    When no LLM stage ran, ``llm_scored`` is ``False`` and ``llm_error`` is
    ``None``.

    Example::

        assert 0 <= result.score <= 100
        if result.confidence_tier == ConfidenceTier.HIGH:
            ...
    """

    job_title: str
    job_company: str = ""
    score: float = Field(ge=0.0, le=100.0, description="Overall match score 0-100")
    vector_score: float | None = Field(
        default=None, ge=0.0, le=100.0, description="Vector similarity score 0-100"
    )
    confidence_tier: ConfidenceTier = ConfidenceTier.LOW
    rationale: str = ""
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    salary_match: bool | None = Field(default=None, description="Whether salary expectations align")
    culture_signals: list[str] = Field(default_factory=list)
    what_they_want: str = Field(
        default="",
        description=(
            "Structured WTAW analysis: role archetype + 3 key needs read between JD lines "
            "+ candidate proof points mapped to each need. Empty string when LLM omits it."
        ),
    )
    llm_scored: bool = False
    llm_error: str | None = Field(
        default=None,
        description=(
            "Set when LLM scoring was attempted but failed; None if not attempted or succeeded"
        ),
    )
    tokens_used: int = Field(default=0, ge=0)
    prompt_version: str | None = Field(
        default=None, description="Version of the scoring prompt template used"
    )

    @property
    def is_strong_match(self) -> bool:
        """True for HIGH and VERY_HIGH tiers with a score >= 70."""
        return (
            self.confidence_tier
            in (
                ConfidenceTier.VERY_HIGH,
                ConfidenceTier.HIGH,
            )
            and self.score >= 70.0
        )


class BatchMatchResult(BaseModel):
    """Result of matching multiple jobs against a candidate profile.

    ``results`` is sorted by ``score`` descending. ``jobs_skipped`` counts jobs
    that fell below the matcher's vector threshold.

    Example::

        assert batch.jobs_evaluated == len(jobs)
        for r in batch.results:
            print(r.job_title, r.score)
    """

    results: list[MatchResult] = Field(default_factory=list)
    jobs_evaluated: int = 0
    jobs_skipped: int = 0
    total_tokens: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0.0, ge=0.0)
    llm_scored_count: int = 0
    llm_fallback_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Rows where LLM scoring was attempted but failed (fallback returned). "
            "Distinct from jobs_skipped (below vector threshold) and "
            "llm_scored_count (successful LLM calls)."
        ),
    )

    @property
    def strong_matches(self) -> list[MatchResult]:
        return [r for r in self.results if r.is_strong_match]
