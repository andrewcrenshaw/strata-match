"""Tests for FastScorer — Stage 2A triage scoring (PCC-1891)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

from strata_match.llm import FastScorer, LLMProvider, LLMResponse
from strata_match.prompts.fast_score import PROMPT_VERSION, build_fast_score_prompt

if TYPE_CHECKING:
    from strata_match.models import CandidateProfile, JobDescription


# ---------------------------------------------------------------------------
# Fake provider
# ---------------------------------------------------------------------------


@dataclass
class FakeFastProvider(LLMProvider):
    """Deterministic LLM provider for FastScorer tests."""

    response_content: str = json.dumps({"score": 75})
    input_tokens: int = 200
    output_tokens: int = 10
    call_count: int = 0
    should_fail_times: int = 0

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        self.call_count += 1
        if self.call_count <= self.should_fail_times:
            raise RuntimeError("provider unavailable")
        return LLMResponse(
            content=self.response_content,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )

    @property
    def model_name(self) -> str:
        return "fake-fast-model"


@pytest.fixture
def fake_provider() -> FakeFastProvider:
    return FakeFastProvider()


@pytest.fixture
def fast_scorer(fake_provider: FakeFastProvider) -> FastScorer:
    return FastScorer(provider=fake_provider)


# ---------------------------------------------------------------------------
# build_fast_score_prompt
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestBuildFastScorePrompt:
    def test_returns_two_messages(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        msgs = build_fast_score_prompt(sample_profile, sample_jobs[0])
        assert len(msgs) == 2

    def test_system_message_instructs_score_only(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        msgs = build_fast_score_prompt(sample_profile, sample_jobs[0])
        assert msgs[0]["role"] == "system"
        assert "score" in msgs[0]["content"].lower()
        # No "strengths" or "gaps" instructions — this is the triage prompt
        assert "strengths" not in msgs[0]["content"].lower()
        assert "gaps" not in msgs[0]["content"].lower()

    def test_user_message_contains_profile_and_job(
        self, sample_profile: CandidateProfile, sample_jobs: list[JobDescription]
    ) -> None:
        msgs = build_fast_score_prompt(sample_profile, sample_jobs[0])
        assert msgs[1]["role"] == "user"
        assert "Senior Software Engineer" in msgs[1]["content"]
        assert "Staff Engineer" in msgs[1]["content"]

    def test_prompt_version_is_set(self) -> None:
        assert PROMPT_VERSION == "v1"


# ---------------------------------------------------------------------------
# FastScorer — happy path
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestFastScorerHappyPath:
    @pytest.mark.asyncio
    async def test_returns_float_score(
        self,
        fast_scorer: FastScorer,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
    ) -> None:
        result = await fast_scorer.score(sample_profile, sample_jobs[0])
        assert isinstance(result, float)
        assert result == 75.0

    @pytest.mark.asyncio
    async def test_clamps_score_above_100(
        self,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
    ) -> None:
        provider = FakeFastProvider(response_content=json.dumps({"score": 150}))
        scorer = FastScorer(provider=provider)
        result = await scorer.score(sample_profile, sample_jobs[0])
        assert result == 100.0

    @pytest.mark.asyncio
    async def test_clamps_score_below_0(
        self,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
    ) -> None:
        provider = FakeFastProvider(response_content=json.dumps({"score": -10}))
        scorer = FastScorer(provider=provider)
        result = await scorer.score(sample_profile, sample_jobs[0])
        # -10 → negative → returns -1.0 (failure sentinel) since score < 0
        assert result == -1.0

    @pytest.mark.asyncio
    async def test_handles_float_score(
        self,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
    ) -> None:
        provider = FakeFastProvider(response_content=json.dumps({"score": 62.5}))
        scorer = FastScorer(provider=provider)
        result = await scorer.score(sample_profile, sample_jobs[0])
        assert result == 62.5

    @pytest.mark.asyncio
    async def test_score_zero_returns_zero(
        self,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
    ) -> None:
        provider = FakeFastProvider(response_content=json.dumps({"score": 0}))
        scorer = FastScorer(provider=provider)
        result = await scorer.score(sample_profile, sample_jobs[0])
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_handles_json_in_markdown_block(
        self,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
    ) -> None:
        wrapped = '```json\n{"score": 88}\n```'
        provider = FakeFastProvider(response_content=wrapped)
        scorer = FastScorer(provider=provider)
        result = await scorer.score(sample_profile, sample_jobs[0])
        assert result == 88.0


# ---------------------------------------------------------------------------
# FastScorer — failure modes
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestFastScorerFailures:
    @pytest.mark.asyncio
    async def test_llm_failure_returns_sentinel(
        self,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
    ) -> None:
        provider = FakeFastProvider(should_fail_times=10)
        scorer = FastScorer(provider=provider, max_retries=0, retry_delay=0.0)
        result = await scorer.score(sample_profile, sample_jobs[0])
        assert result == -1.0

    @pytest.mark.asyncio
    async def test_unparseable_json_returns_sentinel(
        self,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
    ) -> None:
        provider = FakeFastProvider(response_content="Just a number: 72")
        scorer = FastScorer(provider=provider)
        result = await scorer.score(sample_profile, sample_jobs[0])
        assert result == -1.0

    @pytest.mark.asyncio
    async def test_non_numeric_score_returns_sentinel(
        self,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
    ) -> None:
        provider = FakeFastProvider(response_content=json.dumps({"score": "high"}))
        scorer = FastScorer(provider=provider)
        result = await scorer.score(sample_profile, sample_jobs[0])
        assert result == -1.0

    @pytest.mark.asyncio
    async def test_missing_score_field_returns_sentinel(
        self,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
    ) -> None:
        provider = FakeFastProvider(response_content=json.dumps({"result": 80}))
        scorer = FastScorer(provider=provider)
        result = await scorer.score(sample_profile, sample_jobs[0])
        # score field missing → defaults to -1 → returns -1.0 sentinel
        assert result == -1.0


# ---------------------------------------------------------------------------
# FastScorer — retry logic
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestFastScorerRetry:
    @pytest.mark.asyncio
    async def test_retries_on_failure(
        self,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
    ) -> None:
        provider = FakeFastProvider(should_fail_times=1)
        scorer = FastScorer(provider=provider, max_retries=1, retry_delay=0.01)
        result = await scorer.score(sample_profile, sample_jobs[0])
        assert result == 75.0
        assert provider.call_count == 2

    @pytest.mark.asyncio
    async def test_exhausts_retries_returns_sentinel(
        self,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
    ) -> None:
        provider = FakeFastProvider(should_fail_times=10)
        scorer = FastScorer(provider=provider, max_retries=1, retry_delay=0.01)
        result = await scorer.score(sample_profile, sample_jobs[0])
        assert result == -1.0
        assert provider.call_count == 2  # initial + 1 retry

    @pytest.mark.asyncio
    async def test_no_retry_when_max_retries_zero(
        self,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
    ) -> None:
        provider = FakeFastProvider(should_fail_times=10)
        scorer = FastScorer(provider=provider, max_retries=0, retry_delay=0.01)
        result = await scorer.score(sample_profile, sample_jobs[0])
        assert result == -1.0
        assert provider.call_count == 1
