"""Tests for confidence tier classification boundary conditions (PCC-1459).

Covers:
- HIGH: vector_score >= 0.7 AND LLM score >= 70
- MEDIUM: vector_score >= 0.5 OR LLM-only score >= 70
- LOW: everything else above floor
- Configurable floor: vector_score < 0.3 → skipped (no result)
- llm_confirm_threshold is configurable on Matcher
- Exact boundary values for each tier transition

Test strategy:
- classify_confidence: pure function, direct boundary tests
- Matcher: mock LLM scorer to control scores precisely, verify end-to-end tier assignment
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from strata_match.matcher import Matcher, create_matcher
from strata_match.models import (
    CandidateProfile,
    ConfidenceTier,
    JobDescription,
    MatchResult,
)
from strata_match.scoring import VectorScorer, classify_confidence
from tests.conftest import FakeEmbeddingProvider

pytestmark = pytest.mark.verification


# ---------------------------------------------------------------------------
# classify_confidence boundary conditions
# ---------------------------------------------------------------------------


class TestClassifyConfidenceBoundaries:
    """Boundary condition tests for the classify_confidence function."""

    def test_high_exact_boundary(self) -> None:
        """vector=0.7, llm_confirmed=True → HIGH."""
        assert classify_confidence(0.7, llm_confirmed=True) == ConfidenceTier.HIGH

    def test_high_just_below_vector(self) -> None:
        """vector=0.699, llm_confirmed=True → MEDIUM (vector too low for HIGH)."""
        assert classify_confidence(0.699, llm_confirmed=True) == ConfidenceTier.MEDIUM

    def test_high_no_llm(self) -> None:
        """vector=0.8, llm_confirmed=False → MEDIUM (LLM didn't confirm)."""
        assert classify_confidence(0.8, llm_confirmed=False) == ConfidenceTier.MEDIUM

    def test_medium_exact_boundary(self) -> None:
        """vector=0.5, llm_confirmed=False → MEDIUM."""
        assert classify_confidence(0.5, llm_confirmed=False) == ConfidenceTier.MEDIUM

    def test_medium_just_below_vector(self) -> None:
        """vector=0.499, llm_confirmed=False → LOW."""
        assert classify_confidence(0.499, llm_confirmed=False) == ConfidenceTier.LOW

    def test_medium_via_llm_only(self) -> None:
        """vector=0.4, llm_confirmed=True → MEDIUM (LLM override into medium)."""
        assert classify_confidence(0.4, llm_confirmed=True) == ConfidenceTier.MEDIUM

    def test_low_no_signals(self) -> None:
        """vector=0.3, llm_confirmed=False → LOW."""
        assert classify_confidence(0.3, llm_confirmed=False) == ConfidenceTier.LOW

    def test_low_zero_vector(self) -> None:
        """vector=0.0, llm_confirmed=False → LOW."""
        assert classify_confidence(0.0, llm_confirmed=False) == ConfidenceTier.LOW


# ---------------------------------------------------------------------------
# Matcher-level tier assignment with LLM confirm threshold
# ---------------------------------------------------------------------------


def _make_mock_llm_scorer(llm_score: float) -> AsyncMock:
    """Create a mock LLM scorer that returns a MatchResult with the given score."""
    scorer = AsyncMock()
    scorer.score = AsyncMock(
        return_value=MatchResult(
            job_title="Test Job",
            job_company="TestCo",
            score=llm_score,
            rationale="Test rationale",
            strengths=["skill"],
            gaps=[],
            llm_scored=True,
            tokens_used=100,
        )
    )
    return scorer


def _make_fixed_scorer(raw_cosine: float) -> VectorScorer:
    """Create a VectorScorer that always returns a fixed cosine similarity."""
    scorer = AsyncMock(spec=VectorScorer)
    scorer.score = AsyncMock(return_value=raw_cosine)
    scorer.score_batch = AsyncMock(return_value=[raw_cosine])
    return scorer


class TestMatcherLLMConfirmThreshold:
    """Verify Matcher uses llm_confirm_threshold=70 (not 50) for tier classification."""

    @pytest.mark.asyncio
    async def test_default_llm_confirm_threshold_is_70(self) -> None:
        """Matcher.llm_confirm_threshold defaults to 70.0."""
        provider = FakeEmbeddingProvider(dimension=8)
        scorer = VectorScorer(provider=provider)
        matcher = Matcher(vector_scorer=scorer)
        assert matcher.llm_confirm_threshold == 70.0

    @pytest.mark.asyncio
    async def test_high_tier_vector_07_llm_70(self) -> None:
        """vector=0.7, LLM score=70 → HIGH tier."""
        vector_scorer = _make_fixed_scorer(0.7)
        llm_scorer = _make_mock_llm_scorer(70.0)
        matcher = Matcher(
            vector_scorer=vector_scorer,
            vector_threshold=0.3,
            llm_scorer=llm_scorer,
        )

        profile = CandidateProfile(title="Engineer")
        job = JobDescription(title="Test Job", company="TestCo")
        result = await matcher.match_one(profile, job)

        assert result.confidence_tier == ConfidenceTier.HIGH

    @pytest.mark.asyncio
    async def test_not_high_when_llm_69(self) -> None:
        """vector=0.8, LLM score=69 → NOT HIGH (LLM below 70 threshold)."""
        vector_scorer = _make_fixed_scorer(0.8)
        llm_scorer = _make_mock_llm_scorer(69.0)
        matcher = Matcher(
            vector_scorer=vector_scorer,
            vector_threshold=0.3,
            llm_scorer=llm_scorer,
        )

        profile = CandidateProfile(title="Engineer")
        job = JobDescription(title="Test Job", company="TestCo")
        result = await matcher.match_one(profile, job)

        assert result.confidence_tier == ConfidenceTier.MEDIUM

    @pytest.mark.asyncio
    async def test_medium_via_llm_override_low_vector(self) -> None:
        """vector=0.4, LLM score=75 → MEDIUM (LLM override with weak vector)."""
        vector_scorer = _make_fixed_scorer(0.4)
        llm_scorer = _make_mock_llm_scorer(75.0)
        matcher = Matcher(
            vector_scorer=vector_scorer,
            vector_threshold=0.3,
            llm_scorer=llm_scorer,
        )

        profile = CandidateProfile(title="Engineer")
        job = JobDescription(title="Test Job", company="TestCo")
        result = await matcher.match_one(profile, job)

        assert result.confidence_tier == ConfidenceTier.MEDIUM

    @pytest.mark.asyncio
    async def test_low_when_both_weak(self) -> None:
        """vector=0.35, LLM score=40 → LOW (both signals weak)."""
        vector_scorer = _make_fixed_scorer(0.35)
        llm_scorer = _make_mock_llm_scorer(40.0)
        matcher = Matcher(
            vector_scorer=vector_scorer,
            vector_threshold=0.3,
            llm_scorer=llm_scorer,
        )

        profile = CandidateProfile(title="Engineer")
        job = JobDescription(title="Test Job", company="TestCo")
        result = await matcher.match_one(profile, job)

        assert result.confidence_tier == ConfidenceTier.LOW

    @pytest.mark.asyncio
    async def test_medium_vector_only_no_llm(self) -> None:
        """vector=0.6, no LLM scorer → MEDIUM (vector alone sufficient)."""
        vector_scorer = _make_fixed_scorer(0.6)
        matcher = Matcher(
            vector_scorer=vector_scorer,
            vector_threshold=0.3,
            llm_scorer=None,
        )

        profile = CandidateProfile(title="Engineer")
        job = JobDescription(title="Test Job", company="TestCo")
        result = await matcher.match_one(profile, job)

        assert result.confidence_tier == ConfidenceTier.MEDIUM

    @pytest.mark.asyncio
    async def test_floor_skips_below_threshold(self) -> None:
        """vector=0.2 (below floor 0.3) → skipped in batch, no result."""
        vector_scorer = _make_fixed_scorer(0.2)
        matcher = Matcher(
            vector_scorer=vector_scorer,
            vector_threshold=0.3,
            llm_scorer=None,
        )

        profile = CandidateProfile(title="Engineer")
        jobs = [JobDescription(title="Test Job", company="TestCo")]
        batch = await matcher.match_batch(profile, jobs)

        assert batch.jobs_skipped == 1
        assert len(batch.results) == 0

    @pytest.mark.asyncio
    async def test_custom_llm_confirm_threshold(self) -> None:
        """Custom llm_confirm_threshold=80: LLM score=75 → not confirmed."""
        vector_scorer = _make_fixed_scorer(0.8)
        llm_scorer = _make_mock_llm_scorer(75.0)
        matcher = Matcher(
            vector_scorer=vector_scorer,
            vector_threshold=0.3,
            llm_scorer=llm_scorer,
            llm_confirm_threshold=80.0,
        )

        profile = CandidateProfile(title="Engineer")
        job = JobDescription(title="Test Job", company="TestCo")
        result = await matcher.match_one(profile, job)

        assert result.confidence_tier == ConfidenceTier.MEDIUM

    @pytest.mark.asyncio
    async def test_batch_tier_assignment(self) -> None:
        """Batch match_batch applies tier classification consistently."""
        vector_scorer = _make_fixed_scorer(0.75)
        llm_scorer = _make_mock_llm_scorer(80.0)
        matcher = Matcher(
            vector_scorer=vector_scorer,
            vector_threshold=0.3,
            llm_scorer=llm_scorer,
        )

        profile = CandidateProfile(title="Engineer")
        jobs = [JobDescription(title="Test Job", company="TestCo")]
        batch = await matcher.match_batch(profile, jobs)

        assert len(batch.results) == 1
        assert batch.results[0].confidence_tier == ConfidenceTier.HIGH


class TestCreateMatcherLLMThreshold:
    """Verify create_matcher passes llm_confirm_threshold through."""

    def test_create_matcher_accepts_llm_confirm_threshold(self) -> None:
        provider = FakeEmbeddingProvider(dimension=8)
        matcher = create_matcher(provider, llm_confirm_threshold=80.0)
        assert matcher.llm_confirm_threshold == 80.0

    def test_create_matcher_default_llm_confirm_threshold(self) -> None:
        provider = FakeEmbeddingProvider(dimension=8)
        matcher = create_matcher(provider)
        assert matcher.llm_confirm_threshold == 70.0
