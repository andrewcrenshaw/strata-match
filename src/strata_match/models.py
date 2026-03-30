"""Public data models for strata-match."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ConfidenceTier(StrEnum):
    """Match confidence classification."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CandidateProfile(BaseModel):
    """Public candidate profile for matching — no PII fields."""

    title: str
    skills: list[str] = Field(default_factory=list)
    experience_years: int = 0
    summary: str = ""
    education: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    industries: list[str] = Field(default_factory=list)
    preferred_locations: list[str] = Field(default_factory=list)


class JobDescription(BaseModel):
    """Structured job description for matching."""

    title: str
    company: str = ""
    description: str = ""
    requirements: list[str] = Field(default_factory=list)
    preferred_qualifications: list[str] = Field(default_factory=list)
    location: str | None = None
    salary_range: str | None = None
    employment_type: str | None = None
    external_id: str | None = None


class MatchResult(BaseModel):
    """Result of matching a single job against a candidate profile."""

    job_title: str
    job_company: str = ""
    score: float = Field(ge=0.0, le=1.0)
    vector_score: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_tier: ConfidenceTier = ConfidenceTier.LOW
    rationale: str = ""
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    llm_scored: bool = False

    @property
    def is_strong_match(self) -> bool:
        return self.confidence_tier == ConfidenceTier.HIGH and self.score >= 0.7


class BatchMatchResult(BaseModel):
    """Result of matching multiple jobs against a candidate profile."""

    results: list[MatchResult] = Field(default_factory=list)
    total_jobs: int = 0
    skipped_below_threshold: int = 0
    llm_scored_count: int = 0

    @property
    def strong_matches(self) -> list[MatchResult]:
        return [r for r in self.results if r.is_strong_match]
