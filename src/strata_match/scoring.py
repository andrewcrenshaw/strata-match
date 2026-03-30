"""Scoring logic — vector similarity (Stage 1) and LLM nuance scoring (Stage 2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from strata_match.embeddings import EmbeddingProvider, cosine_similarity
from strata_match.models import (
    CandidateProfile,
    ConfidenceTier,
    JobDescription,
    MatchResult,
)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


def _profile_to_text(profile: CandidateProfile) -> str:
    """Serialize a candidate profile into a single text block for embedding."""
    parts = [profile.title]
    if profile.summary:
        parts.append(profile.summary)
    if profile.skills:
        parts.append("Skills: " + ", ".join(profile.skills))
    if profile.experience_years:
        parts.append(f"Experience: {profile.experience_years} years")
    if profile.industries:
        parts.append("Industries: " + ", ".join(profile.industries))
    return "\n".join(parts)


def _job_to_text(job: JobDescription) -> str:
    """Serialize a job description into a single text block for embedding."""
    parts = [job.title]
    if job.company:
        parts.append(f"Company: {job.company}")
    if job.description:
        parts.append(job.description)
    if job.requirements:
        parts.append("Requirements: " + ", ".join(job.requirements))
    return "\n".join(parts)


@dataclass
class VectorScorer:
    """Stage 1: fast vector similarity scoring via embeddings."""

    provider: EmbeddingProvider
    _profile_cache: dict[str, NDArray[np.float32]] = field(default_factory=dict, repr=False)

    async def score(self, profile: CandidateProfile, job: JobDescription) -> float:
        """Return cosine similarity between profile and job embeddings."""
        profile_text = _profile_to_text(profile)
        cache_key = profile_text

        if cache_key not in self._profile_cache:
            self._profile_cache[cache_key] = await self.provider.embed(profile_text)

        profile_vec = self._profile_cache[cache_key]
        job_vec = await self.provider.embed(_job_to_text(job))
        return cosine_similarity(profile_vec, job_vec)

    async def score_batch(
        self, profile: CandidateProfile, jobs: list[JobDescription]
    ) -> list[float]:
        """Score multiple jobs against a single profile."""
        profile_text = _profile_to_text(profile)
        cache_key = profile_text

        if cache_key not in self._profile_cache:
            self._profile_cache[cache_key] = await self.provider.embed(profile_text)

        job_texts = [_job_to_text(j) for j in jobs]
        job_vecs = await self.provider.embed_batch(job_texts)
        profile_vec = self._profile_cache[cache_key]
        return [cosine_similarity(profile_vec, jv) for jv in job_vecs]


def classify_confidence(
    vector_score: float,
    llm_confirmed: bool,
    *,
    high_threshold: float = 0.7,
    medium_threshold: float = 0.5,
) -> ConfidenceTier:
    """Classify match confidence based on vector score and LLM confirmation.

    HIGH:   vector >= high_threshold AND LLM confirms
    MEDIUM: vector >= medium_threshold OR LLM-only
    LOW:    everything else
    """
    if vector_score >= high_threshold and llm_confirmed:
        return ConfidenceTier.HIGH
    if vector_score >= medium_threshold or llm_confirmed:
        return ConfidenceTier.MEDIUM
    return ConfidenceTier.LOW


def build_match_result(
    job: JobDescription,
    *,
    score: float,
    vector_score: float | None = None,
    rationale: str = "",
    strengths: list[str] | None = None,
    gaps: list[str] | None = None,
    confidence_tier: ConfidenceTier = ConfidenceTier.LOW,
    llm_scored: bool = False,
) -> MatchResult:
    """Construct a MatchResult from scoring outputs."""
    return MatchResult(
        job_title=job.title,
        job_company=job.company,
        score=score,
        vector_score=vector_score,
        confidence_tier=confidence_tier,
        rationale=rationale,
        strengths=strengths or [],
        gaps=gaps or [],
        llm_scored=llm_scored,
    )
