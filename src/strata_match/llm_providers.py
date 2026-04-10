"""Concrete LLM providers and provider factory.

Each provider wraps a third-party chat-completion API behind the
:class:`~strata_match.llm.LLMProvider` interface.  External dependencies are
imported lazily so only the chosen provider's package needs to be installed.

Usage::

    from strata_match.llm_providers import create_llm_provider

    provider = create_llm_provider("openai", model="gpt-4o-mini", api_key="sk-...")
    scorer = LLMScorer(provider=provider)

    # MLX-LM local inference (OpenAI-compatible server, endpoint-agnostic)
    provider = MLXLMProvider(
        model="mlx-community/gemma-4-E4B-it-4bit",
        primary_base_url="http://192.168.50.220:8080/v1",
        fallback_base_url="http://100.64.0.2:8080/v1",  # e.g. Tailscale
    )
"""

from __future__ import annotations

import logging
from typing import Any, cast

from strata_match.exceptions import ConfigurationError, ProviderError, ScoringError
from strata_match.llm import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

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
# MLX-LM (local OpenAI-compatible server — endpoint-agnostic)
# ---------------------------------------------------------------------------


class MLXLMProvider(LLMProvider):
    """LLM provider for a local `mlx_lm.server` (OpenAI-compatible REST API).

    The server speaks the OpenAI chat-completions protocol, so this provider
    uses the ``openai`` SDK with a custom ``base_url``.  Before each
    completion it health-probes the primary endpoint; on failure it falls
    back to *fallback_base_url* (if given).  Raises :exc:`ScoringError` only
    when **all** endpoints are unreachable or the completion itself fails.

    No hostnames, IPs, or model names are hardcoded — all are injected by
    the caller (typically the private *strata* agent reading env-vars).

    Requires the ``openai`` package (``pip install strata-match[openai]``) and
    the ``aiohttp`` package (``pip install strata-match[ollama]``) for probing.
    """

    def __init__(
        self,
        *,
        model: str,
        primary_base_url: str,
        fallback_base_url: str | None = None,
        probe_timeout: float = 3.0,
        request_timeout: float = 120.0,
    ) -> None:
        self._model = model
        self._primary_base_url = primary_base_url.rstrip("/")
        self._fallback_base_url = fallback_base_url.rstrip("/") if fallback_base_url else None
        self._probe_timeout = probe_timeout
        self._request_timeout = request_timeout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _is_healthy(self, base_url: str) -> bool:
        """Return True if the server at *base_url* responds to a /models probe."""
        try:
            import aiohttp
        except ImportError as exc:
            raise ProviderError(
                "MLXLMProvider health-probe requires the 'aiohttp' package. "
                "Install with: pip install strata-match[ollama]"
            ) from exc

        probe_url = f"{base_url}/models"
        try:
            timeout = aiohttp.ClientTimeout(total=self._probe_timeout)
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.get(probe_url) as resp,
            ):
                return bool(resp.status < 500)
        except Exception:  # noqa: BLE001
            return False

    async def _resolve_base_url(self) -> str:
        """Return the first healthy base URL or raise ScoringError."""
        if await self._is_healthy(self._primary_base_url):
            logger.debug("MLXLMProvider: using primary endpoint %s", self._primary_base_url)
            return self._primary_base_url

        logger.warning("MLXLMProvider: primary endpoint %s unhealthy", self._primary_base_url)

        if self._fallback_base_url is not None:
            if await self._is_healthy(self._fallback_base_url):
                logger.info("MLXLMProvider: falling back to %s", self._fallback_base_url)
                return self._fallback_base_url
            logger.warning(
                "MLXLMProvider: fallback endpoint %s also unhealthy", self._fallback_base_url
            )

        raise ScoringError(
            "MLXLMProvider: no reachable endpoint "
            f"(primary={self._primary_base_url!r}, "
            f"fallback={self._fallback_base_url!r})"
        )

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        """Health-probe endpoints, then run an OpenAI-compat chat completion."""
        try:
            import openai
        except ImportError as exc:
            raise ProviderError(
                "MLXLMProvider requires the 'openai' package. "
                "Install with: pip install strata-match[openai]"
            ) from exc

        base_url = await self._resolve_base_url()

        client = openai.AsyncOpenAI(
            api_key="not-needed",  # mlx_lm.server ignores the key
            base_url=base_url,
            timeout=self._request_timeout,
        )
        try:
            resp = await client.chat.completions.create(
                model=self._model,
                messages=cast("Any", messages),
                **kwargs,
            )
        except ScoringError:
            raise
        except Exception as exc:
            raise ScoringError("MLXLMProvider chat completion failed") from exc

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
        name: Provider identifier — ``"openai"``, ``"litellm"``, or
              ``"mlx"`` (case-insensitive).
        model: Model identifier override.  Falls back to the provider default
               where a default exists.  Required for ``"mlx"``.
        _client: Pre-built API client (for testing, OpenAI/LiteLLM only).
        **config: Additional keyword arguments forwarded to the provider
                  constructor (e.g. ``api_key``, ``primary_base_url``).

    Returns:
        A configured :class:`LLMProvider` instance.

    Raises:
        ConfigurationError: If *name* does not match a known provider, or if
                            required parameters are missing (e.g. ``model``
                            and ``primary_base_url`` for ``"mlx"``).
    """
    key = str(name).lower()

    if key not in (*_LLM_PROVIDER_DEFAULTS, "mlx"):
        raise ConfigurationError(
            f"Unknown LLM provider '{name}'. "
            f"Choose from: {', '.join([*_LLM_PROVIDER_DEFAULTS, 'mlx'])}"
        )

    if key == "mlx":
        if not model:
            raise ConfigurationError("'model' is required for the 'mlx' provider.")
        if "primary_base_url" not in config:
            raise ConfigurationError("'primary_base_url' is required for the 'mlx' provider.")
        return MLXLMProvider(model=model, **config)

    defaults = _LLM_PROVIDER_DEFAULTS[key]
    resolved_model = model or defaults["model"]

    if key == "openai":
        return OpenAILLMProvider(model=resolved_model, client=_client, **config)

    return LiteLLMProvider(model=resolved_model, **config)
