"""LLM provider abstraction and nuance scorer (Stage 2).

Stage 2 sends jobs that pass the vector threshold to a configurable LLM
for rich scoring with rationale, strengths, gaps, salary match, and culture signals.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from strata_match.scoring import build_match_result

if TYPE_CHECKING:
    from strata_match.models import CandidateProfile, JobDescription, MatchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM response container
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """Container for an LLM chat-completion response."""

    content: str
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ---------------------------------------------------------------------------
# Abstract LLM provider
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    """Abstract base for LLM chat-completion providers."""

    @abstractmethod
    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        """Send a chat completion request and return the response."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier for the underlying model."""


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a JSON object from LLM output.

    Handles:
    - Raw JSON
    - JSON inside markdown code blocks (```json ... ```)
    - JSON with surrounding prose
    """
    text = text.strip()

    # Try direct parse first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting from markdown code block
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1).strip())
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    # Try finding a JSON object in the text
    m = _JSON_OBJECT_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    return None


# ---------------------------------------------------------------------------
# Fast Triage Scorer (Stage 2A — PCC-1891)
# ---------------------------------------------------------------------------


@dataclass
class FastScorer:
    """Stage 2A: lightweight triage scorer returning only a numeric score.

    Uses a simplified ~30-token output prompt to quickly triage all
    above-threshold listings.  Listings with fast_score >= the configured
    threshold proceed to full deep scoring (Stage 2B).

    Returns -1.0 on LLM failure or unparseable output so callers can
    distinguish a genuine low score from an error.
    """

    provider: LLMProvider
    max_retries: int = 1
    retry_delay: float = 1.0

    async def score(
        self,
        profile: CandidateProfile,
        job: JobDescription,
    ) -> float:
        """Return a triage score 0-100, or -1.0 on failure.

        A return value of -1.0 signals that the LLM call failed or produced
        unparseable output.  Callers should fall back to vector score only in
        that case rather than routing the job to deep scoring.
        """
        from strata_match.prompts.fast_score import build_fast_score_prompt

        messages = build_fast_score_prompt(profile, job)
        response, _ = await self._call_with_retry(messages)

        if response is None:
            return -1.0

        parsed = _extract_json(response.content)
        if parsed is None:
            logger.warning("FastScorer: unparseable response for job=%s", job.title)
            return -1.0

        try:
            score = float(parsed.get("score", -1))
        except (TypeError, ValueError):
            logger.warning("FastScorer: non-numeric score for job=%s", job.title)
            return -1.0

        if score < 0:
            return -1.0
        return max(0.0, min(score, 100.0))

    async def _call_with_retry(
        self, messages: list[dict[str, str]]
    ) -> tuple[LLMResponse | None, Exception | None]:
        """Call the LLM provider with retry logic (mirrors LLMScorer)."""
        attempts = 1 + self.max_retries
        last_error: Exception | None = None

        for attempt in range(attempts):
            try:
                return await self.provider.complete(messages), None
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "FastScorer attempt %d/%d failed: %s",
                    attempt + 1,
                    attempts,
                    exc,
                )
                if attempt < attempts - 1:
                    await asyncio.sleep(self.retry_delay)

        logger.error("FastScorer failed after %d attempts: %s", attempts, last_error)
        return None, last_error


# ---------------------------------------------------------------------------
# LLM Nuance Scorer (Stage 2)
# ---------------------------------------------------------------------------


@dataclass
class LLMScorer:
    """Stage 2: rich LLM-based nuance scoring for jobs above the vector threshold.

    Sends structured profile + job to a configurable LLM provider and parses
    the response into a MatchResult with score, rationale, strengths, gaps,
    salary_match, and culture_signals.
    """

    provider: LLMProvider
    max_retries: int = 1
    retry_delay: float = 1.0

    async def score(
        self,
        profile: CandidateProfile,
        job: JobDescription,
        *,
        vector_score: float | None = None,
    ) -> MatchResult:
        """Score a single job against a candidate profile using the LLM.

        On LLM failure (after retries) or unparseable output, returns a
        fallback MatchResult with score=0, ``llm_scored=False``, ``llm_error``
        set, and an error ``rationale``.
        """
        from strata_match.prompts.score_job import PROMPT_VERSION, build_score_prompt

        messages = build_score_prompt(profile, job)
        response, last_error = await self._call_with_retry(messages)

        if response is None:
            err_detail = f"{last_error!r}" if last_error is not None else "unknown error"
            llm_err = f"LLM scoring failed after retries: {err_detail}"
            return self._fallback_result(
                job,
                vector_score=vector_score,
                rationale="LLM scoring failed after retries.",
                llm_error=llm_err,
                prompt_version=PROMPT_VERSION,
            )

        parsed = _extract_json(response.content)
        if parsed is None:
            return self._fallback_result(
                job,
                vector_score=vector_score,
                rationale="Failed to parse LLM response as JSON.",
                llm_error="Failed to parse LLM response as JSON.",
                tokens_used=response.total_tokens,
                prompt_version=PROMPT_VERSION,
            )

        return self._build_from_parsed(
            job,
            parsed,
            vector_score=vector_score,
            tokens_used=response.total_tokens,
            prompt_version=PROMPT_VERSION,
        )

    async def _call_with_retry(
        self, messages: list[dict[str, str]]
    ) -> tuple[LLMResponse | None, Exception | None]:
        """Call the LLM provider with retry logic."""
        attempts = 1 + self.max_retries
        last_error: Exception | None = None

        for attempt in range(attempts):
            try:
                return await self.provider.complete(messages), None
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "LLM call attempt %d/%d failed: %s",
                    attempt + 1,
                    attempts,
                    exc,
                )
                if attempt < attempts - 1:
                    await asyncio.sleep(self.retry_delay)

        logger.error("LLM scoring failed after %d attempts: %s", attempts, last_error)
        return None, last_error

    @staticmethod
    def _build_from_parsed(
        job: JobDescription,
        data: dict[str, Any],
        *,
        vector_score: float | None = None,
        tokens_used: int = 0,
        prompt_version: str | None = None,
    ) -> MatchResult:
        """Build a MatchResult from parsed LLM JSON output."""
        raw_score = data.get("score", 0)
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(score, 100.0))

        rationale = str(data.get("rationale", ""))

        def _coerce_str_list(raw: object) -> list[str]:
            items = raw if isinstance(raw, list) else []
            return [
                " — ".join(str(v) for v in item.values()) if isinstance(item, dict) else str(item)
                for item in items
            ]

        strengths = _coerce_str_list(data.get("strengths"))
        gaps = _coerce_str_list(data.get("gaps"))

        salary_raw = data.get("salary_match")
        salary_match: bool | None = None
        if isinstance(salary_raw, bool):
            salary_match = salary_raw

        culture_signals = _coerce_str_list(data.get("culture_signals"))
        what_they_want = str(data.get("what_they_want", ""))

        return build_match_result(
            job,
            score=score,
            vector_score=vector_score,
            rationale=rationale,
            strengths=strengths,
            gaps=gaps,
            salary_match=salary_match,
            culture_signals=culture_signals,
            what_they_want=what_they_want,
            llm_scored=True,
            llm_error=None,
            tokens_used=tokens_used,
            prompt_version=prompt_version,
        )

    @staticmethod
    def _fallback_result(
        job: JobDescription,
        *,
        vector_score: float | None = None,
        rationale: str = "",
        llm_error: str,
        tokens_used: int = 0,
        prompt_version: str | None = None,
    ) -> MatchResult:
        """Return a safe fallback MatchResult when LLM scoring fails."""
        return build_match_result(
            job,
            score=0.0,
            vector_score=vector_score,
            rationale=rationale,
            llm_scored=False,
            llm_error=llm_error,
            tokens_used=tokens_used,
            prompt_version=prompt_version,
        )
