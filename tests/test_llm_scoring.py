"""Tests for LLM nuance scoring (Stage 2)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

from strata_match.llm import LLMProvider, LLMResponse, LLMScorer

if TYPE_CHECKING:
    from strata_match.models import CandidateProfile, JobDescription


# ---------------------------------------------------------------------------
# Fake / helper fixtures
# ---------------------------------------------------------------------------

GOOD_LLM_RESPONSE = json.dumps(
    {
        "score": 82,
        "strengths": ["Strong Python background", "Distributed systems experience"],
        "gaps": ["No explicit leadership experience"],
        "rationale": "Candidate is a solid fit for backend platform work.",
        "salary_match": True,
        "culture_signals": ["remote-friendly", "engineering-led"],
    }
)


@dataclass
class FakeLLMProvider(LLMProvider):
    """Deterministic LLM provider for tests."""

    response_content: str = GOOD_LLM_RESPONSE
    input_tokens: int = 500
    output_tokens: int = 200
    model: str = "fake-model"
    call_count: int = 0
    should_fail_times: int = 0

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        self.call_count += 1
        if self.call_count <= self.should_fail_times:
            raise RuntimeError("LLM provider temporarily unavailable")
        return LLMResponse(
            content=self.response_content,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )

    @property
    def model_name(self) -> str:
        return self.model


@pytest.fixture
def fake_llm() -> FakeLLMProvider:
    return FakeLLMProvider()


@pytest.fixture
def llm_scorer(fake_llm: FakeLLMProvider) -> LLMScorer:
    return LLMScorer(provider=fake_llm)


# ---------------------------------------------------------------------------
# LLMResponse
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestLLMResponse:
    def test_total_tokens(self) -> None:
        resp = LLMResponse(content="hello", input_tokens=100, output_tokens=50)
        assert resp.total_tokens == 150

    def test_defaults(self) -> None:
        resp = LLMResponse(content="hi")
        assert resp.input_tokens == 0
        assert resp.output_tokens == 0
        assert resp.total_tokens == 0


# ---------------------------------------------------------------------------
# LLMProvider abstract interface
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestLLMProvider:
    def test_fake_provider_implements_interface(self, fake_llm: FakeLLMProvider) -> None:
        assert isinstance(fake_llm, LLMProvider)
        assert fake_llm.model_name == "fake-model"

    @pytest.mark.asyncio
    async def test_fake_provider_returns_response(self, fake_llm: FakeLLMProvider) -> None:
        resp = await fake_llm.complete([{"role": "user", "content": "hi"}])
        assert isinstance(resp, LLMResponse)
        assert resp.input_tokens == 500


# ---------------------------------------------------------------------------
# LLMScorer — happy path
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestLLMScorerHappyPath:
    @pytest.mark.asyncio
    async def test_returns_match_result(
        self,
        llm_scorer: LLMScorer,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
    ) -> None:
        result = await llm_scorer.score(sample_profile, sample_jobs[0])
        assert result.job_title == "Staff Engineer — Backend Platform"
        assert result.job_company == "Acme Corp"
        assert result.score == 82.0
        assert result.llm_scored is True
        assert result.llm_error is None

    @pytest.mark.asyncio
    async def test_populates_strengths_and_gaps(
        self,
        llm_scorer: LLMScorer,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
    ) -> None:
        result = await llm_scorer.score(sample_profile, sample_jobs[0])
        assert "Strong Python background" in result.strengths
        assert "No explicit leadership experience" in result.gaps

    @pytest.mark.asyncio
    async def test_populates_rationale(
        self,
        llm_scorer: LLMScorer,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
    ) -> None:
        result = await llm_scorer.score(sample_profile, sample_jobs[0])
        assert "solid fit" in result.rationale

    @pytest.mark.asyncio
    async def test_populates_salary_match(
        self,
        llm_scorer: LLMScorer,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
    ) -> None:
        result = await llm_scorer.score(sample_profile, sample_jobs[0])
        assert result.salary_match is True

    @pytest.mark.asyncio
    async def test_populates_culture_signals(
        self,
        llm_scorer: LLMScorer,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
    ) -> None:
        result = await llm_scorer.score(sample_profile, sample_jobs[0])
        assert result.culture_signals == ["remote-friendly", "engineering-led"]

    @pytest.mark.asyncio
    async def test_tracks_token_usage(
        self,
        llm_scorer: LLMScorer,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
    ) -> None:
        result = await llm_scorer.score(sample_profile, sample_jobs[0])
        assert result.tokens_used == 700  # 500 input + 200 output


# ---------------------------------------------------------------------------
# LLMScorer — structured output parsing (response variations)
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestLLMScorerResponseParsing:
    @pytest.mark.asyncio
    async def test_handles_json_in_markdown_code_block(
        self,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
    ) -> None:
        wrapped = f"```json\n{GOOD_LLM_RESPONSE}\n```"
        provider = FakeLLMProvider(response_content=wrapped)
        scorer = LLMScorer(provider=provider)
        result = await scorer.score(sample_profile, sample_jobs[0])
        assert result.score == 82.0

    @pytest.mark.asyncio
    async def test_handles_json_with_surrounding_text(
        self,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
    ) -> None:
        wrapped = f"Here is the assessment:\n{GOOD_LLM_RESPONSE}\nThat's my evaluation."
        provider = FakeLLMProvider(response_content=wrapped)
        scorer = LLMScorer(provider=provider)
        result = await scorer.score(sample_profile, sample_jobs[0])
        assert result.score == 82.0

    @pytest.mark.asyncio
    async def test_handles_missing_optional_fields(
        self,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
    ) -> None:
        minimal = json.dumps({"score": 60, "rationale": "Okay fit."})
        provider = FakeLLMProvider(response_content=minimal)
        scorer = LLMScorer(provider=provider)
        result = await scorer.score(sample_profile, sample_jobs[0])
        assert result.score == 60.0
        assert result.rationale == "Okay fit."
        assert result.strengths == []
        assert result.gaps == []
        assert result.salary_match is None
        assert result.culture_signals == []

    @pytest.mark.asyncio
    async def test_clamps_score_above_100(
        self,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
    ) -> None:
        bad = json.dumps({"score": 150, "rationale": "Perfect."})
        provider = FakeLLMProvider(response_content=bad)
        scorer = LLMScorer(provider=provider)
        result = await scorer.score(sample_profile, sample_jobs[0])
        assert result.score == 100.0

    @pytest.mark.asyncio
    async def test_clamps_score_below_0(
        self,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
    ) -> None:
        bad = json.dumps({"score": -10, "rationale": "No fit."})
        provider = FakeLLMProvider(response_content=bad)
        scorer = LLMScorer(provider=provider)
        result = await scorer.score(sample_profile, sample_jobs[0])
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_handles_score_as_float(
        self,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
    ) -> None:
        resp = json.dumps({"score": 75.5, "rationale": "Decent."})
        provider = FakeLLMProvider(response_content=resp)
        scorer = LLMScorer(provider=provider)
        result = await scorer.score(sample_profile, sample_jobs[0])
        assert result.score == 75.5

    @pytest.mark.asyncio
    async def test_unparseable_json_returns_fallback(
        self,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
    ) -> None:
        """When the LLM returns garbage, score falls back to 0 with error rationale."""
        provider = FakeLLMProvider(response_content="I cannot evaluate this.")
        scorer = LLMScorer(provider=provider)
        result = await scorer.score(sample_profile, sample_jobs[0])
        assert result.score == 0.0
        assert result.llm_scored is False
        assert result.llm_error == "Failed to parse LLM response as JSON."
        assert "parse" in result.rationale.lower() or "failed" in result.rationale.lower()

    @pytest.mark.asyncio
    async def test_non_numeric_score_field_defaults_to_zero(
        self,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
    ) -> None:
        """When score field is a non-numeric string, it defaults to 0.0 (not a crash)."""
        bad = json.dumps({"score": "high", "rationale": "Looks good."})
        provider = FakeLLMProvider(response_content=bad)
        scorer = LLMScorer(provider=provider)
        result = await scorer.score(sample_profile, sample_jobs[0])
        assert result.score == 0.0
        assert result.llm_scored is True
        assert result.llm_error is None

    @pytest.mark.asyncio
    async def test_null_score_field_defaults_to_zero(
        self,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
    ) -> None:
        """When score is null/None in JSON, it should default to 0.0."""
        bad = json.dumps({"score": None, "rationale": "No score provided."})
        provider = FakeLLMProvider(response_content=bad)
        scorer = LLMScorer(provider=provider)
        result = await scorer.score(sample_profile, sample_jobs[0])
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_json_code_block_with_invalid_inner_json_falls_back(
        self,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
    ) -> None:
        """Code block wrapper with broken inner JSON → fallback."""
        bad = "```json\n{not valid json at all\n```"
        provider = FakeLLMProvider(response_content=bad)
        scorer = LLMScorer(provider=provider)
        result = await scorer.score(sample_profile, sample_jobs[0])
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_vector_score_passed_through_on_success(
        self,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
    ) -> None:
        """vector_score kwarg is propagated to the MatchResult on LLM success."""
        scorer = LLMScorer(provider=FakeLLMProvider())
        result = await scorer.score(sample_profile, sample_jobs[0], vector_score=72.5)
        assert result.vector_score == 72.5

    @pytest.mark.asyncio
    async def test_vector_score_passed_through_on_fallback(
        self,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
    ) -> None:
        """vector_score is preserved even when LLM call fails."""
        provider = FakeLLMProvider(should_fail_times=10)
        scorer = LLMScorer(provider=provider, max_retries=0, retry_delay=0.0)
        result = await scorer.score(sample_profile, sample_jobs[0], vector_score=55.0)
        assert result.vector_score == 55.0
        assert result.score == 0.0
        assert result.llm_scored is False
        assert result.llm_error is not None


# ---------------------------------------------------------------------------
# LLMScorer — retry logic
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestLLMScorerRetry:
    @pytest.mark.asyncio
    async def test_retries_on_failure(
        self,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
    ) -> None:
        provider = FakeLLMProvider(should_fail_times=1)
        scorer = LLMScorer(provider=provider, max_retries=1, retry_delay=0.01)
        result = await scorer.score(sample_profile, sample_jobs[0])
        assert result.score == 82.0
        assert result.llm_error is None
        assert provider.call_count == 2

    @pytest.mark.asyncio
    async def test_exhausts_retries_returns_fallback(
        self,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
    ) -> None:
        provider = FakeLLMProvider(should_fail_times=10)
        scorer = LLMScorer(provider=provider, max_retries=1, retry_delay=0.01)
        result = await scorer.score(sample_profile, sample_jobs[0])
        assert result.score == 0.0
        assert result.llm_scored is False
        assert result.llm_error is not None
        assert "LLM scoring failed after retries" in result.llm_error
        assert "failed" in result.rationale.lower() or "error" in result.rationale.lower()
        assert provider.call_count == 2  # initial + 1 retry

    @pytest.mark.asyncio
    async def test_no_retry_when_max_retries_zero(
        self,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
    ) -> None:
        provider = FakeLLMProvider(should_fail_times=10)
        scorer = LLMScorer(provider=provider, max_retries=0, retry_delay=0.01)
        result = await scorer.score(sample_profile, sample_jobs[0])
        assert result.score == 0.0
        assert result.llm_scored is False
        assert result.llm_error is not None
        assert provider.call_count == 1


# ---------------------------------------------------------------------------
# LLMScorer — prompt construction
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestLLMScorerPromptConstruction:
    @pytest.mark.asyncio
    async def test_uses_score_prompt_builder(
        self,
        sample_profile: CandidateProfile,
        sample_jobs: list[JobDescription],
    ) -> None:
        """Verify the scorer passes profile/job through the prompt builder."""
        captured_messages: list[list[dict[str, str]]] = []

        class SpyProvider(LLMProvider):
            async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
                captured_messages.append(messages)
                return LLMResponse(
                    content=GOOD_LLM_RESPONSE,
                    input_tokens=100,
                    output_tokens=50,
                )

            @property
            def model_name(self) -> str:
                return "spy"

        scorer = LLMScorer(provider=SpyProvider())
        await scorer.score(sample_profile, sample_jobs[0])

        assert len(captured_messages) == 1
        msgs = captured_messages[0]
        assert msgs[0]["role"] == "system"
        assert "score" in msgs[0]["content"].lower()
        assert "Senior Software Engineer" in msgs[1]["content"]
        assert "Staff Engineer" in msgs[1]["content"]
