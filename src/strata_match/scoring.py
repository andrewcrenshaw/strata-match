"""Scoring logic — vector similarity (Stage 1) and LLM nuance scoring (Stage 2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from strata_match.embeddings import EmbeddingProvider, cosine_similarity
from strata_match.exceptions import EmbeddingError
from strata_match.models import (
    CandidateProfile,
    ConfidenceTier,
    JobDescription,
    MatchResult,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _profile_to_text(profile: CandidateProfile) -> str:
    """Serialize a candidate profile into a single text block for embedding."""
    parts = [profile.title]
    if profile.experience_summary:
        parts.append(profile.experience_summary)
    if profile.skills:
        parts.append("Skills: " + ", ".join(profile.skills))
    if profile.years_of_experience:
        parts.append(f"Experience: {profile.years_of_experience} years")
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
    """Stage 1: fast vector similarity scoring via embeddings.

    Scores are clamped to [0, 1].  When pre-computed embeddings are present
    on the profile or job, embedding generation is skipped for that side.
    """

    provider: EmbeddingProvider
    vector_threshold: float = 0.6
    _profile_cache: dict[str, NDArray[np.float32]] = field(default_factory=dict, repr=False)

    # -- helpers ---------------------------------------------------------

    async def _get_profile_vec(self, profile: CandidateProfile) -> NDArray[np.float32]:
        if profile.embedding is not None:
            return np.asarray(profile.embedding, dtype=np.float32)

        text = _profile_to_text(profile)
        if text not in self._profile_cache:
            self._profile_cache[text] = await self.provider.embed(text)
        return self._profile_cache[text]

    async def _get_job_vec(self, job: JobDescription) -> NDArray[np.float32]:
        if job.embedding is not None:
            return np.asarray(job.embedding, dtype=np.float32)
        return await self.provider.embed(_job_to_text(job))

    @staticmethod
    def _clamp(raw: float) -> float:
        """Clamp cosine similarity from [-1, 1] to [0, 1]."""
        return max(0.0, min(raw, 1.0))

    # -- public API ------------------------------------------------------

    async def score(self, profile: CandidateProfile, job: JobDescription) -> float:
        """Return similarity score in [0, 1] between profile and job."""
        profile_vec = await self._get_profile_vec(profile)
        job_vec = await self._get_job_vec(job)
        return self._clamp(cosine_similarity(profile_vec, job_vec))

    async def score_with_threshold(
        self, profile: CandidateProfile, job: JobDescription
    ) -> tuple[float, bool]:
        """Score and report whether the result was filtered by threshold.

        Returns:
            ``(score, filtered)`` where *filtered* is ``True`` when the score
            falls below ``self.vector_threshold``.
        """
        s = await self.score(profile, job)
        return s, s < self.vector_threshold

    async def score_batch(
        self, profile: CandidateProfile, jobs: list[JobDescription]
    ) -> list[float]:
        """Score multiple jobs against a single profile. Returns [0, 1] scores."""
        if not jobs:
            return []

        profile_vec = await self._get_profile_vec(profile)

        texts_to_embed: list[str] = []
        text_indices: list[int] = []
        job_vecs: list[NDArray[np.float32] | None] = [None] * len(jobs)

        for i, job in enumerate(jobs):
            if job.embedding is not None:
                job_vecs[i] = np.asarray(job.embedding, dtype=np.float32)
            else:
                texts_to_embed.append(_job_to_text(job))
                text_indices.append(i)

        if texts_to_embed:
            embedded = await self.provider.embed_batch(texts_to_embed)
            for idx, vec in zip(text_indices, embedded, strict=True):
                job_vecs[idx] = vec

        resolved: list[NDArray[np.float32]] = []
        for i, jv in enumerate(job_vecs):
            if jv is None:
                raise EmbeddingError(f"Embedding for job index {i} was not resolved")
            resolved.append(jv)

        return [self._clamp(cosine_similarity(profile_vec, jv)) for jv in resolved]

    async def score_batch_filtered(
        self,
        profile: CandidateProfile,
        jobs: list[JobDescription],
    ) -> tuple[list[tuple[JobDescription, float]], list[tuple[JobDescription, float]]]:
        """Score a batch and split into above/below threshold.

        Returns:
            ``(above, below)`` — each a list of ``(job, score)`` pairs.
        """
        scores = await self.score_batch(profile, jobs)
        above: list[tuple[JobDescription, float]] = []
        below: list[tuple[JobDescription, float]] = []
        for job, s in zip(jobs, scores, strict=True):
            (above if s >= self.vector_threshold else below).append((job, s))
        return above, below


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
    salary_match: bool | None = None,
    culture_signals: list[str] | None = None,
    confidence_tier: ConfidenceTier = ConfidenceTier.LOW,
    llm_scored: bool = False,
    llm_error: str | None = None,
    tokens_used: int = 0,
    prompt_version: str | None = None,
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
        salary_match=salary_match,
        culture_signals=culture_signals or [],
        llm_scored=llm_scored,
        llm_error=llm_error,
        tokens_used=tokens_used,
        prompt_version=prompt_version,
    )
