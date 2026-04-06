"""Main matching engine — two-stage vector + LLM job-to-profile matching."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from strata_match.llm import LLMProvider, LLMScorer
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

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from strata_match.embeddings import EmbeddingProvider
    from strata_match.models import CandidateProfile, JobDescription


def _to_score_100(raw: float) -> float:
    """Convert a raw cosine similarity (0.0-1.0) to the 0-100 score range."""
    return round(max(0.0, min(raw, 1.0)) * 100.0, 2)


@dataclass
class Matcher:
    """Two-stage matching engine.

    **Stage 1 (vector):** Cosine similarity between profile and job embeddings
    (or pre-computed vectors on the models). Produces a score in ``[0, 100]``.

    **Stage 2 (LLM):** When ``llm_scorer`` is set and the Stage 1 raw score is
    at or above ``vector_threshold``, the job is sent to the LLM for nuanced
    scoring (rationale, strengths, gaps). Below the threshold, Stage 2 is
    skipped and confidence is typically LOW.

    Args:
        vector_scorer: Stage-1 scorer wrapping an embedding provider.
        vector_threshold: Minimum raw cosine similarity in ``[0, 1]`` before
            Stage 2 runs (when an LLM scorer is configured).
        llm_scorer: Optional Stage-2 scorer; ``None`` means vector-only matching.
        max_concurrency: Maximum concurrent LLM scoring calls in :meth:`match_batch`
            (when ``llm_scorer`` is set). Values below ``1`` are treated as ``1``.

    Example::

        matcher = create_matcher("openai", vector_threshold=0.35)
        result = await matcher.match_one(profile, job)
    """

    vector_scorer: VectorScorer
    vector_threshold: float = 0.3
    llm_scorer: LLMScorer | None = None
    llm_confirm_threshold: float = 70.0
    max_concurrency: int = 5

    async def match_one(self, profile: CandidateProfile, job: JobDescription) -> MatchResult:
        """Score a single job against the candidate profile.

        Args:
            profile: Candidate data (skills, summary, optional embedding).
            job: Job posting data (title, requirements, optional embedding).

        Returns:
            A :class:`~strata_match.models.MatchResult` with score, tier, and
            optional LLM fields.

        Raises:
            StrataMatchError: Propagates embedding or LLM provider errors (network,
                authentication, rate limits) from the configured backends.

        Example::

            r = await matcher.match_one(profile, job)
            print(r.score, r.confidence_tier, r.rationale)
        """
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
            llm_result = await self.llm_scorer.score(profile, job, vector_score=score_100)
            confidence = classify_confidence(
                raw_score,
                llm_confirmed=llm_result.score >= self.llm_confirm_threshold,
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
                llm_scored=llm_result.llm_scored,
                llm_error=llm_result.llm_error,
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
        """Score multiple jobs against the candidate profile.

        Jobs below ``vector_threshold`` are counted in ``jobs_skipped`` and do
        not receive LLM scoring. Remaining results are sorted by ``score``
        descending.

        Args:
            profile: Candidate data shared across all jobs.
            jobs: List of job postings to evaluate (may be empty).

        Returns:
            :class:`~strata_match.models.BatchMatchResult` with per-job
            results, counts, and token totals.

        Raises:
            StrataMatchError: Propagates embedding or LLM provider errors from the
                configured backends.

        Example::

            batch = await matcher.match_batch(profile, open_jobs)
            for r in batch.results:
                print(r.job_title, r.score)
        """
        if not jobs:
            return BatchMatchResult(
                results=[],
                jobs_evaluated=0,
                jobs_skipped=0,
                total_tokens=0,
                duration_ms=0.0,
                llm_scored_count=0,
            )

        start = time.perf_counter()
        scores = await self.vector_scorer.score_batch(profile, jobs)

        results: list[MatchResult] = []
        skipped = 0
        llm_count = 0
        llm_fallback_count = 0
        total_tokens = 0

        llm_pairs: list[tuple[JobDescription, float]] = []

        for job, raw in zip(jobs, scores, strict=True):
            if raw < self.vector_threshold:
                skipped += 1
                continue

            score_100 = _to_score_100(raw)

            if self.llm_scorer is not None:
                llm_pairs.append((job, raw))
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

        if self.llm_scorer is not None and llm_pairs:
            cap = max(1, self.max_concurrency)
            semaphore = asyncio.Semaphore(cap)
            llm = self.llm_scorer

            async def _score_one_llm(job: JobDescription, raw: float) -> MatchResult:
                score_100 = _to_score_100(raw)
                async with semaphore:
                    return await llm.score(profile, job, vector_score=score_100)

            gathered = await asyncio.gather(
                *(_score_one_llm(j, r) for j, r in llm_pairs),
                return_exceptions=True,
            )

            for (job, raw), item in zip(llm_pairs, gathered, strict=True):
                score_100 = _to_score_100(raw)
                if isinstance(item, BaseException):
                    # AC2: preserve the row as a fallback instead of dropping
                    logger.error(
                        "match_batch: LLM scoring error for job %r; returning fallback row",
                        job.title,
                        exc_info=(type(item), item, item.__traceback__),
                    )
                    results.append(
                        build_match_result(
                            job,
                            score=score_100,
                            vector_score=score_100,
                            llm_scored=False,
                            llm_error=f"LLM scoring error: {item}",
                        )
                    )
                    llm_fallback_count += 1  # AC3: distinct counter
                    continue

                llm_result = item
                score_100 = _to_score_100(raw)
                confidence = classify_confidence(
                    raw,
                    llm_confirmed=llm_result.score >= self.llm_confirm_threshold,
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
                        llm_scored=llm_result.llm_scored,
                        llm_error=llm_result.llm_error,
                        tokens_used=llm_result.tokens_used,
                    )
                )
                if llm_result.llm_scored:
                    llm_count += 1
                elif llm_result.llm_error is not None:
                    llm_fallback_count += 1  # AC3: LLMScorer returned a fallback
                total_tokens += llm_result.tokens_used

        results.sort(key=lambda r: r.score, reverse=True)

        elapsed_ms = (time.perf_counter() - start) * 1000
        return BatchMatchResult(
            results=results,
            jobs_evaluated=len(jobs),
            jobs_skipped=skipped,
            total_tokens=total_tokens,
            duration_ms=round(elapsed_ms, 2),
            llm_scored_count=llm_count,
            llm_fallback_count=llm_fallback_count,
        )


def create_matcher(
    embedding_provider: EmbeddingProvider | str = "openai",
    *,
    embedding_model: str | None = None,
    model: str | None = None,
    scoring_provider: LLMProvider | LLMScorer | str | None = None,
    scoring_model: str | None = None,
    vector_threshold: float = 0.3,
    llm_confirm_threshold: float = 70.0,
    max_concurrency: int = 5,
    llm_scorer: LLMScorer | None = None,
    _provider_client: object | None = None,
    _scoring_client: object | None = None,
    **provider_config: Any,
) -> Matcher:
    """Factory: create a configured Matcher instance.

    Args:
        embedding_provider: An EmbeddingProvider instance, or a string key
            ("openai", "gemini", "ollama") to auto-resolve via the provider
            factory.
        embedding_model: Model identifier for the embedding provider (forwarded
            when *embedding_provider* is a string).  Alias for *model*.
        model: Deprecated alias for *embedding_model*.  If both are supplied,
            *embedding_model* wins.
        scoring_provider: An LLMProvider, LLMScorer, or string key ("openai",
            "litellm") for Stage 2 nuance scoring.  When a string is given the
            provider is auto-resolved via the LLM provider factory.  ``None``
            disables LLM scoring.
        scoring_model: Model identifier forwarded to the LLM provider factory
            when *scoring_provider* is a string.
        vector_threshold: Minimum vector similarity to proceed to LLM scoring.
        llm_confirm_threshold: Minimum LLM score (0–100) to consider the LLM
            as "confirming" the match for tier classification.  Defaults to 70.
        max_concurrency: Maximum concurrent LLM calls in :meth:`Matcher.match_batch`.
            Defaults to 5. Values below ``1`` are treated as ``1``.
        llm_scorer: Pre-built LLM scorer (backward-compatible).  Ignored when
            *scoring_provider* is supplied.
        _provider_client: Pre-built API client forwarded to the embedding
            provider factory (for testing).
        _scoring_client: Pre-built API client forwarded to the LLM provider
            factory (for testing).
        **provider_config: Extra keyword arguments forwarded to the embedding
            provider factory (e.g. ``api_key``, ``base_url``).

    Returns:
        A configured Matcher ready for use.

    Raises:
        ProviderError: If a string provider name is used but the optional
            dependency for that backend is not installed (see extras in
            ``pyproject.toml``).
        ConfigurationError: If an unknown embedding or LLM provider name is given.

    Example::

        # Vector-only (no LLM)
        m = create_matcher("openai", api_key="...", vector_threshold=0.4)

        # Two-stage with LLM via provider key
        m = create_matcher(
            "openai",
            scoring_provider="openai",
            scoring_model="gpt-4o-mini",
            api_key="...",
        )
    """
    resolved_emb_model = embedding_model or model

    if isinstance(embedding_provider, str):
        from strata_match.providers import create_embedding_provider

        embedding_provider = create_embedding_provider(
            embedding_provider,
            model=resolved_emb_model,
            _client=_provider_client,
            **provider_config,
        )

    resolved_scorer = _resolve_llm_scorer(
        scoring_provider, scoring_model, _scoring_client, llm_scorer
    )

    vector_scorer = VectorScorer(provider=embedding_provider)
    return Matcher(
        vector_scorer=vector_scorer,
        vector_threshold=vector_threshold,
        llm_scorer=resolved_scorer,
        llm_confirm_threshold=llm_confirm_threshold,
        max_concurrency=max_concurrency,
    )


def _resolve_llm_scorer(
    scoring_provider: LLMProvider | LLMScorer | str | None,
    scoring_model: str | None,
    _scoring_client: object | None,
    llm_scorer: LLMScorer | None,
) -> LLMScorer | None:
    """Resolve a scoring_provider arg into an LLMScorer (or None)."""
    if scoring_provider is None:
        return llm_scorer

    if isinstance(scoring_provider, LLMScorer):
        return scoring_provider

    if isinstance(scoring_provider, LLMProvider):
        return LLMScorer(provider=scoring_provider)

    from strata_match.llm_providers import create_llm_provider

    provider = create_llm_provider(
        scoring_provider,
        model=scoring_model,
        _client=_scoring_client,
    )
    return LLMScorer(provider=provider)


async def match_job(
    matcher: Matcher,
    profile: CandidateProfile,
    job: JobDescription,
) -> MatchResult:
    """Match one job against a profile (delegates to :meth:`Matcher.match_one`).

    Args:
        matcher: Engine from :func:`create_matcher`.
        profile: Candidate profile.
        job: Job description.

    Returns:
        Match outcome for the pair.

    Raises:
        StrataMatchError: Same as :meth:`Matcher.match_one`.

    Example::

        result = await match_job(matcher, profile, job)
    """
    return await matcher.match_one(profile, job)


async def match_batch(
    matcher: Matcher,
    profile: CandidateProfile,
    jobs: list[JobDescription],
) -> BatchMatchResult:
    """Match many jobs against a profile (delegates to :meth:`Matcher.match_batch`).

    Args:
        matcher: Engine from :func:`create_matcher`.
        profile: Candidate profile.
        jobs: Jobs to score.

    Returns:
        Batch outcome with sorted ``results``.

    Raises:
        StrataMatchError: Same as :meth:`Matcher.match_batch`.

    Example::

        batch = await match_batch(matcher, profile, jobs)
        print(batch.strong_matches)
    """
    return await matcher.match_batch(profile, jobs)
