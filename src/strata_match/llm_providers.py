"""Concrete LLM providers and provider factory.

Each provider wraps a third-party chat-completion API behind the
:class:`~strata_match.llm.LLMProvider` interface.  External dependencies are
imported lazily so only the chosen provider's package needs to be installed.

Usage::

    from strata_match.llm_providers import create_llm_provider

    provider = create_llm_provider("openai", model="gpt-4o-mini", api_key="sk-...")
    scorer = LLMScorer(provider=provider)
"""

from __future__ import annotations

from typing import Any

from strata_match.exceptions import ConfigurationError, ProviderError, ScoringError
from strata_match.llm import LLMProvider, LLMResponse

# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


class OpenAILLMProvider(LLMProvider):
    """LLM provider backed by the OpenAI chat-completions API.

    Requires the ``openai`` package (``pip install strata-match[openai]``).
    Pass a pre-built ``AsyncOpenAI`` client via *client* to skip auto-creation.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        client: Any = None,
    ) -> None:
        self._model = model
        if client is not None:
            self._client = client
        else:
            try:
                import openai
            except ImportError as exc:
                raise ProviderError(
                    "OpenAI LLM provider requires the 'openai' package. "
                    "Install with: pip install strata-match[openai]"
                ) from exc
            self._client = openai.AsyncOpenAI(api_key=api_key)

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                **kwargs,
            )
        except Exception as exc:
            raise ScoringError("OpenAI chat completion failed") from exc
        choice = resp.choices[0]
        usage = resp.usage or type("U", (), {"prompt_tokens": 0, "completion_tokens": 0})()
        return LLMResponse(
            content=choice.message.content or "",
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
        )

    @property
    def model_name(self) -> str:
        return self._model


# ---------------------------------------------------------------------------
# LiteLLM (multi-provider proxy)
# ---------------------------------------------------------------------------


class LiteLLMProvider(LLMProvider):
    """LLM provider backed by the LiteLLM library.

    LiteLLM provides a unified interface to 100+ LLM providers using
    OpenAI-compatible syntax.  Requires ``pip install strata-match[litellm]``.
    """

    def __init__(self, *, model: str = "gpt-4o-mini", **litellm_kwargs: Any) -> None:
        self._model = model
        self._kwargs = litellm_kwargs

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        try:
            import litellm
        except ImportError as exc:
            raise ProviderError(
                "LiteLLM provider requires the 'litellm' package. "
                "Install with: pip install strata-match[litellm]"
            ) from exc

        merged = {**self._kwargs, **kwargs}
        try:
            resp = await litellm.acompletion(
                model=self._model,
                messages=messages,
                **merged,
            )
        except Exception as exc:
            raise ScoringError("LiteLLM completion failed") from exc
        choice = resp.choices[0]
        usage = resp.usage or type("U", (), {"prompt_tokens": 0, "completion_tokens": 0})()
        return LLMResponse(
            content=choice.message.content or "",
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
        )

    @property
    def model_name(self) -> str:
        return self._model


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_LLM_PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "openai": {"model": "gpt-4o-mini"},
    "litellm": {"model": "gpt-4o-mini"},
}


def create_llm_provider(
    name: str,
    *,
    model: str | None = None,
    _client: Any = None,
    **config: Any,
) -> LLMProvider:
    """Create an LLM provider by name.

    Args:
        name: Provider identifier — ``"openai"`` or ``"litellm"``
              (case-insensitive).
        model: Model identifier override.  Falls back to the provider default.
        _client: Pre-built API client (for testing).
        **config: Additional keyword arguments forwarded to the provider
                  constructor (e.g. ``api_key``).

    Returns:
        A configured :class:`LLMProvider` instance.

    Raises:
        ConfigurationError: If *name* does not match a known provider.
    """
    key = str(name).lower()

    defaults = _LLM_PROVIDER_DEFAULTS.get(key)
    if defaults is None:
        raise ConfigurationError(
            f"Unknown LLM provider '{name}'. Choose from: {', '.join(_LLM_PROVIDER_DEFAULTS)}"
        )

    resolved_model = model or defaults["model"]

    if key == "openai":
        return OpenAILLMProvider(model=resolved_model, client=_client, **config)

    return LiteLLMProvider(model=resolved_model, **config)
