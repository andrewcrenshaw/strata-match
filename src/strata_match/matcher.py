"""Main matching engine — two-stage vector + LLM job-to-profile matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from strata_match.models import (
    BatchMatchResult,
    ConfidenceTier,
    MatchResult,
)
from strata_match.scoring import (
    VectorScorer,
    build_match_result,
    classify_confidence,
)

if TYPE_CHECKING:
    from strata_match.embeddings import EmbeddingProvider
    from strata_match.models import CandidateProfile, JobDescription


@dataclass
class Matcher:
    """Two-stage matching engine.

    Stage 1 (vector): Fast cosine similarity via embeddings.
    Stage 2 (LLM):    Rich nuance scoring with rationale — only for candidates
                       that pass the vector threshold.
    """

    vector_scorer: VectorScorer
    vector_threshold: float = 0.3
    llm_provider: object | None = None

    async def match_one(self, profile: CandidateProfile, job: JobDescription) -> MatchResult:
        """Score a single job against the candidate profile."""
        vector_score = max(0.0, await self.vector_scorer.score(profile, job))

        if vector_score < self.vector_threshold:
            return build_match_result(
                job,
                score=vector_score,
                vector_score=vector_score,
                confidence_tier=ConfidenceTier.LOW,
            )

        # Stage 2 placeholder — LLM scoring will be implemented in Phase 2B
        confidence = classify_confidence(vector_score, llm_confirmed=False)
        return build_match_result(
            job,
            score=vector_score,
            vector_score=vector_score,
            confidence_tier=confidence,
            llm_scored=False,
        )

    async def match_batch(
        self, profile: CandidateProfile, jobs: list[JobDescription]
    ) -> BatchMatchResult:
        """Score multiple jobs against the candidate profile."""
        raw_scores = await self.vector_scorer.score_batch(profile, jobs)
        vector_scores = [max(0.0, s) for s in raw_scores]

        results: list[MatchResult] = []
        skipped = 0
        llm_count = 0

        for job, vs in zip(jobs, vector_scores, strict=True):
            if vs < self.vector_threshold:
                skipped += 1
                continue

            confidence = classify_confidence(vs, llm_confirmed=False)
            results.append(
                build_match_result(
                    job,
                    score=vs,
                    vector_score=vs,
                    confidence_tier=confidence,
                    llm_scored=False,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)

        return BatchMatchResult(
            results=results,
            total_jobs=len(jobs),
            skipped_below_threshold=skipped,
            llm_scored_count=llm_count,
        )


def create_matcher(
    embedding_provider: EmbeddingProvider | str = "openai",
    *,
    vector_threshold: float = 0.3,
) -> Matcher:
    """Factory: create a configured Matcher instance.

    Args:
        embedding_provider: An EmbeddingProvider instance, or a string key
            ("openai", "gemini", "ollama") to auto-resolve. Provider resolution
            is deferred to Phase 2B implementation.
        vector_threshold: Minimum vector similarity to proceed to LLM scoring.

    Returns:
        A configured Matcher ready for use.
    """
    if isinstance(embedding_provider, str):
        raise NotImplementedError(
            f"Auto-resolving provider '{embedding_provider}' is not yet implemented. "
            "Pass an EmbeddingProvider instance directly. "
            "Provider auto-resolution is planned for Phase 2B."
        )

    vector_scorer = VectorScorer(provider=embedding_provider)
    return Matcher(
        vector_scorer=vector_scorer,
        vector_threshold=vector_threshold,
    )


async def match_job(
    matcher: Matcher,
    profile: CandidateProfile,
    job: JobDescription,
) -> MatchResult:
    """Convenience: match a single job against a profile."""
    return await matcher.match_one(profile, job)


async def match_batch(
    matcher: Matcher,
    profile: CandidateProfile,
    jobs: list[JobDescription],
) -> BatchMatchResult:
    """Convenience: match multiple jobs against a profile."""
    return await matcher.match_batch(profile, jobs)
