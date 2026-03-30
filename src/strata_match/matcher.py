"""Main matching engine — two-stage vector + LLM job-to-profile matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
    from strata_match.llm import LLMScorer
    from strata_match.models import CandidateProfile, JobDescription


def _to_score_100(raw: float) -> float:
    """Convert a raw cosine similarity (0.0-1.0) to the 0-100 score range."""
    return round(max(0.0, min(raw, 1.0)) * 100.0, 2)


@dataclass
class Matcher:
    """Two-stage matching engine.

    Stage 1 (vector): Fast cosine similarity via embeddings.
    Stage 2 (LLM):    Rich nuance scoring with rationale — only for candidates
                       that pass the vector threshold.
    """

    vector_scorer: VectorScorer
    vector_threshold: float = 0.3
    llm_scorer: LLMScorer | None = None

    async def match_one(
        self, profile: CandidateProfile, job: JobDescription
    ) -> MatchResult:
        """Score a single job against the candidate profile."""
        raw_score = await self.vector_scorer.score(profile, job)
        score_100 = _to_score_100(raw_score)

        if raw_score < self.vector_threshold:
            return build_match_result(
                job,
                score=score_100,
                vector_score=score_100,
                confidence_tier=ConfidenceTier.LOW,
            )

        if self.llm_scorer is not None:
            llm_result = await self.llm_scorer.score(
                profile, job, vector_score=score_100
            )
            confidence = classify_confidence(
                raw_score, llm_confirmed=llm_result.score >= 50.0
            )
            return MatchResult(
                job_title=llm_result.job_title,
                job_company=llm_result.job_company,
                score=llm_result.score,
                vector_score=score_100,
                confidence_tier=confidence,
                rationale=llm_result.rationale,
                strengths=llm_result.strengths,
                gaps=llm_result.gaps,
                salary_match=llm_result.salary_match,
                culture_signals=llm_result.culture_signals,
                llm_scored=True,
                tokens_used=llm_result.tokens_used,
            )

        confidence = classify_confidence(raw_score, llm_confirmed=False)
        return build_match_result(
            job,
            score=score_100,
            vector_score=score_100,
            confidence_tier=confidence,
            llm_scored=False,
        )

    async def match_batch(
        self, profile: CandidateProfile, jobs: list[JobDescription]
    ) -> BatchMatchResult:
        """Score multiple jobs against the candidate profile."""
        scores = await self.vector_scorer.score_batch(profile, jobs)

        results: list[MatchResult] = []
        skipped = 0
        llm_count = 0
        total_tokens = 0

        for job, raw in zip(jobs, scores, strict=True):
            if raw < self.vector_threshold:
                skipped += 1
                continue

            score_100 = _to_score_100(raw)

            if self.llm_scorer is not None:
                llm_result = await self.llm_scorer.score(
                    profile, job, vector_score=score_100
                )
                confidence = classify_confidence(
                    raw, llm_confirmed=llm_result.score >= 50.0
                )
                results.append(
                    MatchResult(
                        job_title=llm_result.job_title,
                        job_company=llm_result.job_company,
                        score=llm_result.score,
                        vector_score=score_100,
                        confidence_tier=confidence,
                        rationale=llm_result.rationale,
                        strengths=llm_result.strengths,
                        gaps=llm_result.gaps,
                        salary_match=llm_result.salary_match,
                        culture_signals=llm_result.culture_signals,
                        llm_scored=True,
                        tokens_used=llm_result.tokens_used,
                    )
                )
                llm_count += 1
                total_tokens += llm_result.tokens_used
            else:
                confidence = classify_confidence(raw, llm_confirmed=False)
                results.append(
                    build_match_result(
                        job,
                        score=score_100,
                        vector_score=score_100,
                        confidence_tier=confidence,
                        llm_scored=False,
                    )
                )

        results.sort(key=lambda r: r.score, reverse=True)

        return BatchMatchResult(
            results=results,
            jobs_evaluated=len(jobs),
            jobs_skipped=skipped,
            total_tokens=total_tokens,
            llm_scored_count=llm_count,
        )


def create_matcher(
    embedding_provider: EmbeddingProvider | str = "openai",
    *,
    model: str | None = None,
    vector_threshold: float = 0.3,
    llm_scorer: LLMScorer | None = None,
    _provider_client: object | None = None,
    **provider_config: Any,
) -> Matcher:
    """Factory: create a configured Matcher instance.

    Args:
        embedding_provider: An EmbeddingProvider instance, or a string key
            ("openai", "gemini", "ollama") to auto-resolve via the provider
            factory.
        model: Model identifier forwarded to the provider factory when
            *embedding_provider* is a string.
        vector_threshold: Minimum vector similarity to proceed to LLM scoring.
        llm_scorer: Optional LLM scorer for Stage 2 nuance scoring.
        _provider_client: Pre-built API client forwarded to the provider
            factory (for testing).
        **provider_config: Extra keyword arguments forwarded to the provider
            factory (e.g. ``api_key``, ``base_url``).

    Returns:
        A configured Matcher ready for use.
    """
    if isinstance(embedding_provider, str):
        from strata_match.providers import create_embedding_provider

        embedding_provider = create_embedding_provider(
            embedding_provider,
            model=model,
            _client=_provider_client,
            **provider_config,
        )

    vector_scorer = VectorScorer(provider=embedding_provider)
    return Matcher(
        vector_scorer=vector_scorer,
        vector_threshold=vector_threshold,
        llm_scorer=llm_scorer,
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
