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


# ---------------------------------------------------------------------------
# Configurable classification thresholds
# ---------------------------------------------------------------------------
# These are module-level defaults that callers can override via keyword args
# in classify_confidence().
#
# Calibration notes (Gemini text-embedding-004, cosine similarity):
#   0.85+ = very strong semantic overlap (title match + most skills present)
#   0.70+ = good fit (solid role match, minor gaps expected)
#   0.50+ = meaningful overlap (transferable skills, adjacent domain)
#   0.30  = pipeline vector floor — below this, no MatchResult is created
#
# LLM score thresholds map the 0–100 LLM output to tier gates:
#   85+ = LLM sees excellent candidate-role alignment
#   70+ = LLM confirms a strong match (llm_confirm_threshold default)
# ---------------------------------------------------------------------------

#: Minimum vector score to qualify for VERY_HIGH (raw cosine 0–1).
# Calibrated for ollama/nomic-embed-text which scores ~0.10 lower than
# Gemini text-embedding-004 for equivalent semantic matches.
# Observed range on real data: 0.62–0.70 for strong matches (LLM score 85–92).
DEFAULT_VERY_HIGH_VECTOR: float = 0.85
#: Minimum LLM score (0–100) to qualify for VERY_HIGH.
DEFAULT_VERY_HIGH_LLM: float = 85.0

#: Minimum vector score to qualify for HIGH (requires llm_confirmed).
DEFAULT_HIGH_VECTOR: float = 0.70
#: Minimum LLM score (0–100) to qualify for HIGH.
DEFAULT_HIGH_LLM: float = 70.0

#: Minimum vector score to qualify for MEDIUM when LLM-confirmed.
DEFAULT_MEDIUM_VECTOR: float = 0.50
#: Minimum vector score for MEDIUM when LLM did NOT confirm (vector-only path).
DEFAULT_MEDIUM_VECTOR_NO_LLM: float = 0.65


def _profile_to_text(profile: CandidateProfile) -> str:
    """Serialize a candidate profile into a single text block for embedding.

    All fields here are kept in sync with ``_format_profile`` in
    ``prompts/score_job.py`` so Stage-1 vector similarity is computed over
    the same semantic content as Stage-2 LLM prompts (AC1/AC4).
    """
    parts = [profile.title]
    if profile.experience_summary:
        parts.append(profile.experience_summary)
    if profile.skills:
        parts.append("Skills: " + ", ".join(profile.skills))
    if profile.years_of_experience:
        parts.append(f"Experience: {profile.years_of_experience} years")
    if profile.education:
        parts.append("Education: " + ", ".join(profile.education))
    if profile.certifications:
        parts.append("Certifications: " + ", ".join(profile.certifications))
    if profile.industries:
        parts.append("Industries: " + ", ".join(profile.industries))
    if profile.achievements:
        parts.append("Achievements: " + ", ".join(profile.achievements))
    if profile.preferred_locations:
        parts.append("Preferred Locations: " + ", ".join(profile.preferred_locations))
    if profile.preferences:
        formatted = ", ".join(f"{k}: {v}" for k, v in profile.preferences.items())
        parts.append("Preferences: " + formatted)
    return "\n".join(parts)


def _job_to_text(job: JobDescription) -> str:
    """Serialize a job description into a single text block for embedding.

    All fields here are kept in sync with ``_format_job`` in
    ``prompts/score_job.py`` so Stage-1 vector similarity is computed over
    the same semantic content as Stage-2 LLM prompts (AC1/AC4).
    """
    parts = [job.title]
    if job.company:
        parts.append(f"Company: {job.company}")
    if job.description:
        parts.append(job.description)
    if job.requirements:
        parts.append("Requirements: " + ", ".join(job.requirements))
    if job.preferred_qualifications:
        parts.append("Preferred: " + ", ".join(job.preferred_qualifications))
    if job.location:
        parts.append(f"Location: {job.location}")
    if job.salary_range:
        parts.append(f"Salary: {job.salary_range}")
    if job.employment_type:
        parts.append(f"Employment Type: {job.employment_type}")
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
    llm_score: float | None = None,
    very_high_vector: float = DEFAULT_VERY_HIGH_VECTOR,
    very_high_llm: float = DEFAULT_VERY_HIGH_LLM,
    high_vector: float = DEFAULT_HIGH_VECTOR,
    high_llm: float = DEFAULT_HIGH_LLM,
    medium_vector: float = DEFAULT_MEDIUM_VECTOR,
    medium_vector_no_llm: float = DEFAULT_MEDIUM_VECTOR_NO_LLM,
    # Legacy aliases kept for backward compatibility:
    high_threshold: float | None = None,
    medium_threshold: float | None = None,
) -> ConfidenceTier:
    """Classify match confidence into one of four tiers.

    Uses a two-signal gate: vector cosine similarity (Stage 1, fast) and
    the LLM's numeric score (Stage 2, deep).  Both signals must agree for
    the top two tiers, preventing false positives from either channel alone.

    Tier rules (evaluated top-to-bottom, first match wins):

    VERY_HIGH
        ``vector_score >= very_high_vector``
        AND ``llm_confirmed is True``
        AND ``llm_score >= very_high_llm``.
        Both channels show exceptional alignment.

    HIGH
        ``vector_score >= high_vector``
        AND ``llm_confirmed is True``
        AND ``llm_score >= high_llm``.
        Strong semantic fit confirmed by the LLM.

    MEDIUM
        ``(vector_score >= medium_vector AND llm_confirmed)``
        OR ``vector_score >= medium_vector_no_llm``.
        Meaningful overlap; LLM either confirms or vector is strong alone.

    LOW
        Anything above the pipeline vector floor (default 0.30) that does
        not meet MEDIUM criteria.  Filtered out of the Jobs page by default.

    Args:
        vector_score: Raw cosine similarity in [0, 1] from Stage 1.
        llm_confirmed: True when the LLM score meets the confirmation
            threshold (typically ``llm_score >= 70``).
        llm_score: The raw LLM score in [0, 100].  Required for VERY_HIGH
            and HIGH gates; treated as 0 when not provided.
        very_high_vector: Vector floor for VERY_HIGH (default 0.85).
        very_high_llm: LLM score floor for VERY_HIGH (default 85).
        high_vector: Vector floor for HIGH (default 0.70).
        high_llm: LLM score floor for HIGH (default 70).
        medium_vector: Vector floor for MEDIUM when LLM-confirmed (default 0.50).
        medium_vector_no_llm: Vector floor for MEDIUM without LLM (default 0.65).
        high_threshold: Deprecated alias for ``high_vector``.
        medium_threshold: Deprecated alias for ``medium_vector``.
    """
    # Backward-compat: legacy callers pass high_threshold/medium_threshold.
    if high_threshold is not None:
        high_vector = high_threshold
    if medium_threshold is not None:
        medium_vector = medium_threshold

    score = llm_score if llm_score is not None else 0.0

    # VERY_HIGH: vector AND LLM both excellent.
    if vector_score >= very_high_vector and llm_confirmed and score >= very_high_llm:
        return ConfidenceTier.VERY_HIGH

    # HIGH: vector AND LLM both strong.
    if vector_score >= high_vector and llm_confirmed and score >= high_llm:
        return ConfidenceTier.HIGH

    # MEDIUM: LLM-confirmed OR strong vector alone.
    if (vector_score >= medium_vector and llm_confirmed) or vector_score >= medium_vector_no_llm:
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
    what_they_want: str = "",
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
        what_they_want=what_they_want,
        llm_scored=llm_scored,
        llm_error=llm_error,
        tokens_used=tokens_used,
        prompt_version=prompt_version,
    )
