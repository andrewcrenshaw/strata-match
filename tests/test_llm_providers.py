"""Tests for concrete LLM providers and the LLM provider factory."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from strata_match.llm import LLMProvider, LLMResponse, LLMScorer
from strata_match.llm_providers import (
    LiteLLMProvider,
    OpenAILLMProvider,
    create_llm_provider,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_openai_llm_client(
    content: str = '{"score": 75, "rationale": "Good fit."}',
    input_tokens: int = 300,
    output_tokens: int = 150,
) -> MagicMock:
    """Build a mock AsyncOpenAI-style chat client."""
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    usage = MagicMock()
    usage.prompt_tokens = input_tokens
    usage.completion_tokens = output_tokens
    resp.usage = usage
    client.chat.completions.create = AsyncMock(return_value=resp)
    return client


def _mock_litellm(
    content: str = '{"score": 80, "rationale": "Solid."}',
    input_tokens: int = 200,
    output_tokens: int = 100,
) -> MagicMock:
    """Build a mock litellm module with acompletion."""
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    usage = MagicMock()
    usage.prompt_tokens = input_tokens
    usage.completion_tokens = output_tokens
    resp.usage = usage

    litellm_mod = MagicMock()
    litellm_mod.acompletion = AsyncMock(return_value=resp)
    return litellm_mod


# ---------------------------------------------------------------------------
# OpenAILLMProvider
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestOpenAILLMProvider:
    def test_is_llm_provider(self) -> None:
        client = _mock_openai_llm_client()
        provider = OpenAILLMProvider(client=client)
        assert isinstance(provider, LLMProvider)

    def test_model_name_default(self) -> None:
        provider = OpenAILLMProvider(client=MagicMock())
        assert provider.model_name == "gpt-4o-mini"

    def test_model_name_custom(self) -> None:
        provider = OpenAILLMProvider(model="gpt-4o", client=MagicMock())
        assert provider.model_name == "gpt-4o"

    @pytest.mark.asyncio
    async def test_complete_returns_llm_response(self) -> None:
        client = _mock_openai_llm_client(
            content='{"score": 90}', input_tokens=400, output_tokens=100
        )
        provider = OpenAILLMProvider(client=client)
        resp = await provider.complete([{"role": "user", "content": "hi"}])
        assert isinstance(resp, LLMResponse)
        assert resp.content == '{"score": 90}'
        assert resp.input_tokens == 400
        assert resp.output_tokens == 100
        assert resp.total_tokens == 500

    @pytest.mark.asyncio
    async def test_complete_calls_api_with_correct_params(self) -> None:
        client = _mock_openai_llm_client()
        provider = OpenAILLMProvider(model="gpt-4o-mini", client=client)
        messages = [{"role": "user", "content": "test"}]
        await provider.complete(messages)
        client.chat.completions.create.assert_awaited_once_with(
            model="gpt-4o-mini",
            messages=messages,
        )

    @pytest.mark.asyncio
    async def test_complete_passes_kwargs_to_api(self) -> None:
        client = _mock_openai_llm_client()
        provider = OpenAILLMProvider(client=client)
        messages = [{"role": "user", "content": "test"}]
        await provider.complete(messages, temperature=0.0, max_tokens=512)
        client.chat.completions.create.assert_awaited_once_with(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.0,
            max_tokens=512,
        )

    @pytest.mark.asyncio
    async def test_complete_handles_null_usage(self) -> None:
        """When API returns no usage info, tokens default to 0."""
        client = MagicMock()
        choice = MagicMock()
        choice.message.content = '{"score": 55}'
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage = None
        client.chat.completions.create = AsyncMock(return_value=resp)

        provider = OpenAILLMProvider(client=client)
        result = await provider.complete([{"role": "user", "content": "hi"}])
        assert result.input_tokens == 0
        assert result.output_tokens == 0

    @pytest.mark.asyncio
    async def test_complete_handles_null_content(self) -> None:
        """When choice.message.content is None, returns empty string."""
        client = MagicMock()
        choice = MagicMock()
        choice.message.content = None
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        client.chat.completions.create = AsyncMock(return_value=resp)

        provider = OpenAILLMProvider(client=client)
        result = await provider.complete([{"role": "user", "content": "hi"}])
        assert result.content == ""


# ---------------------------------------------------------------------------
# LiteLLMProvider
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestLiteLLMProvider:
    def test_is_llm_provider(self) -> None:
        provider = LiteLLMProvider()
        assert isinstance(provider, LLMProvider)

    def test_model_name_default(self) -> None:
        provider = LiteLLMProvider()
        assert provider.model_name == "gpt-4o-mini"

    def test_model_name_custom(self) -> None:
        provider = LiteLLMProvider(model="claude-3-haiku-20240307")
        assert provider.model_name == "claude-3-haiku-20240307"

    @pytest.mark.asyncio
    async def test_complete_returns_llm_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        litellm_mock = _mock_litellm(content='{"score": 80}', input_tokens=200, output_tokens=100)
        monkeypatch.setitem(__import__("sys").modules, "litellm", litellm_mock)

        provider = LiteLLMProvider(model="gpt-4o-mini")
        resp = await provider.complete([{"role": "user", "content": "test"}])
        assert isinstance(resp, LLMResponse)
        assert resp.content == '{"score": 80}'
        assert resp.input_tokens == 200
        assert resp.output_tokens == 100

    @pytest.mark.asyncio
    async def test_complete_passes_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        litellm_mock = _mock_litellm()
        monkeypatch.setitem(__import__("sys").modules, "litellm", litellm_mock)

        provider = LiteLLMProvider(model="gpt-4o-mini")
        messages = [{"role": "user", "content": "hi"}]
        await provider.complete(messages, temperature=0.5)
        litellm_mock.acompletion.assert_awaited_once_with(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.5,
        )

    @pytest.mark.asyncio
    async def test_complete_merges_init_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Kwargs from __init__ are merged with kwargs from complete()."""
        litellm_mock = _mock_litellm()
        monkeypatch.setitem(__import__("sys").modules, "litellm", litellm_mock)

        provider = LiteLLMProvider(model="gpt-4o-mini", api_key="sk-test")
        messages = [{"role": "user", "content": "hi"}]
        await provider.complete(messages, temperature=0.0)
        litellm_mock.acompletion.assert_awaited_once_with(
            model="gpt-4o-mini",
            messages=messages,
            api_key="sk-test",
            temperature=0.0,
        )

    @pytest.mark.asyncio
    async def test_complete_handles_null_usage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        choice = MagicMock()
        choice.message.content = '{"score": 70}'
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage = None

        litellm_mock = MagicMock()
        litellm_mock.acompletion = AsyncMock(return_value=resp)
        monkeypatch.setitem(__import__("sys").modules, "litellm", litellm_mock)

        provider = LiteLLMProvider()
        result = await provider.complete([{"role": "user", "content": "test"}])
        assert result.input_tokens == 0
        assert result.output_tokens == 0

    @pytest.mark.asyncio
    async def test_complete_raises_import_error_when_litellm_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When litellm is not installed, complete() raises ImportError."""
        monkeypatch.setitem(__import__("sys").modules, "litellm", None)  # type: ignore[arg-type]
        provider = LiteLLMProvider()
        with pytest.raises((ImportError, ModuleNotFoundError)):
            await provider.complete([{"role": "user", "content": "test"}])


# ---------------------------------------------------------------------------
# create_llm_provider factory
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestCreateLLMProvider:
    def test_openai_by_name(self) -> None:
        mock_client = _mock_openai_llm_client()
        provider = create_llm_provider("openai", _client=mock_client)
        assert isinstance(provider, OpenAILLMProvider)
        assert provider.model_name == "gpt-4o-mini"

    def test_openai_custom_model(self) -> None:
        mock_client = _mock_openai_llm_client()
        provider = create_llm_provider("openai", model="gpt-4o", _client=mock_client)
        assert isinstance(provider, OpenAILLMProvider)
        assert provider.model_name == "gpt-4o"

    def test_litellm_by_name(self) -> None:
        provider = create_llm_provider("litellm")
        assert isinstance(provider, LiteLLMProvider)
        assert provider.model_name == "gpt-4o-mini"

    def test_litellm_custom_model(self) -> None:
        provider = create_llm_provider("litellm", model="claude-3-haiku-20240307")
        assert isinstance(provider, LiteLLMProvider)
        assert provider.model_name == "claude-3-haiku-20240307"

    def test_case_insensitive_name(self) -> None:
        mock_client = _mock_openai_llm_client()
        provider = create_llm_provider("OPENAI", _client=mock_client)
        assert isinstance(provider, OpenAILLMProvider)

    def test_unknown_provider_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm_provider("anthropic-direct")

    @pytest.mark.asyncio
    async def test_openai_provider_works_end_to_end(self) -> None:
        payload = json.dumps({"score": 88, "rationale": "Strong match."})
        client = _mock_openai_llm_client(content=payload)
        provider = create_llm_provider("openai", _client=client)
        scorer = LLMScorer(provider=provider)

        from strata_match.models import CandidateProfile, JobDescription

        profile = CandidateProfile(
            title="Senior Engineer",
            skills=["Python", "AWS"],
            experience_summary="Backend engineer.",
        )
        job = JobDescription(
            title="Staff Engineer",
            company="Acme",
            requirements=["Python", "AWS"],
        )
        result = await scorer.score(profile, job)
        assert result.score == 88.0
        assert result.rationale == "Strong match."
        assert result.llm_scored is True


# ---------------------------------------------------------------------------
# create_matcher factory — LLM scoring_provider path (matcher.py line 222)
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestCreateMatcherWithLLMScoring:
    def test_scoring_provider_as_llm_provider_instance(self) -> None:
        from strata_match.matcher import create_matcher
        from tests.conftest import FakeEmbeddingProvider

        embed = FakeEmbeddingProvider(dimension=8)
        mock_client = _mock_openai_llm_client()
        llm_provider = OpenAILLMProvider(client=mock_client)

        matcher = create_matcher(embed, scoring_provider=llm_provider)
        assert matcher.llm_scorer is not None
        assert isinstance(matcher.llm_scorer, LLMScorer)

    def test_scoring_provider_as_string_resolves_factory(self) -> None:
        from unittest.mock import MagicMock

        from strata_match.matcher import create_matcher

        mock_emb = MagicMock()
        mock_scoring = _mock_openai_llm_client()
        matcher = create_matcher(
            "openai",
            scoring_provider="openai",
            scoring_model="gpt-4o-mini",
            _provider_client=mock_emb,
            _scoring_client=mock_scoring,
        )
        assert matcher.llm_scorer is not None

    def test_scoring_provider_as_llm_scorer_instance(self) -> None:
        from strata_match.matcher import create_matcher
        from tests.conftest import FakeEmbeddingProvider

        embed = FakeEmbeddingProvider(dimension=8)
        mock_client = _mock_openai_llm_client()
        llm_provider = OpenAILLMProvider(client=mock_client)
        scorer = LLMScorer(provider=llm_provider)

        matcher = create_matcher(embed, scoring_provider=scorer)
        assert matcher.llm_scorer is scorer
