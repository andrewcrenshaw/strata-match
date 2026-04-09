"""Tests for GeminiEmbeddingProvider concurrency and rate-limit backoff (PCC-1790).

AC1: embed_batch with 1000 texts and concurrency=10 makes ≤10 concurrent calls at once.
AC2: Results returned in original order regardless of chunk completion order.
AC3: HTTP 429/ResourceExhausted triggers retry with backoff; succeeds on second attempt.
AC4: embed_concurrency=1 reproduces sequential behaviour (backward-compat).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from strata_match.providers import GeminiEmbeddingProvider, create_embedding_provider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 8


def _make_embedding(value: float = 0.0) -> list[float]:
    return [value] * _DIM


def _build_response(embeddings: list[list[float]]) -> MagicMock:
    resp = MagicMock()
    resp.embeddings = [MagicMock(values=e) for e in embeddings]
    return resp


def _make_client_with_responses(responses: list[Any]) -> MagicMock:
    """Client whose embed_content returns successive responses (one per call)."""
    client = MagicMock()
    client.aio.models.embed_content = AsyncMock(side_effect=responses)
    return client


# ---------------------------------------------------------------------------
# AC1 — bounded concurrency (≤10 concurrent calls at once)
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestBoundedConcurrency:
    @pytest.mark.asyncio
    async def test_1000_texts_concurrency_10_max_10_parallel(self) -> None:
        """AC1: 1000 texts / 10 concurrency → at most 10 API calls in-flight at once."""
        n_texts = 1000
        concurrency = 10

        active_calls: list[int] = [0]  # mutable counter
        max_concurrent: list[int] = [0]

        async def mock_embed(*, model: str, contents: Any) -> MagicMock:
            active_calls[0] += 1
            max_concurrent[0] = max(max_concurrent[0], active_calls[0])
            await asyncio.sleep(0)  # yield so other coroutines can run
            active_calls[0] -= 1
            return _build_response([_make_embedding(0.1)] * len(contents))

        client = MagicMock()
        client.aio.models.embed_content = mock_embed  # type: ignore[assignment]
        provider = GeminiEmbeddingProvider(
            client=client, dimension=_DIM, embed_concurrency=concurrency
        )

        texts = [str(i) for i in range(n_texts)]
        result = await provider.embed_batch(texts)

        assert len(result) == n_texts
        assert max_concurrent[0] <= concurrency, (
            f"Expected at most {concurrency} concurrent calls, got {max_concurrent[0]}"
        )

    @pytest.mark.asyncio
    async def test_10_chunks_concurrency_10_all_launch_together(self) -> None:
        """With concurrency=10 and exactly 10 chunks, all 10 should run concurrently."""
        n_texts = 1000
        concurrency = 10

        active_calls: list[int] = [0]
        max_concurrent: list[int] = [0]
        launch_times: list[float] = []

        async def mock_embed(*, model: str, contents: Any) -> MagicMock:
            active_calls[0] += 1
            max_concurrent[0] = max(max_concurrent[0], active_calls[0])
            launch_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0)
            active_calls[0] -= 1
            return _build_response([_make_embedding(0.2)] * len(contents))

        client = MagicMock()
        client.aio.models.embed_content = mock_embed  # type: ignore[assignment]
        provider = GeminiEmbeddingProvider(
            client=client, dimension=_DIM, embed_concurrency=concurrency
        )

        texts = [str(i) for i in range(n_texts)]
        result = await provider.embed_batch(texts)

        assert len(result) == n_texts
        # All 10 chunks must have started (no sequential gating with concurrency=10)
        assert max_concurrent[0] == 10

    @pytest.mark.asyncio
    async def test_default_concurrency_is_10(self) -> None:
        """Default embed_concurrency should be 10 (class constant)."""
        provider = GeminiEmbeddingProvider(client=MagicMock())
        assert provider._embed_concurrency == 10

    @pytest.mark.asyncio
    async def test_custom_concurrency_stored(self) -> None:
        """Constructor param overrides the default."""
        provider = GeminiEmbeddingProvider(client=MagicMock(), embed_concurrency=5)
        assert provider._embed_concurrency == 5


# ---------------------------------------------------------------------------
# AC2 — order preservation under parallel execution
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestOrderPreservation:
    @pytest.mark.asyncio
    async def test_order_preserved_with_concurrency_10(self) -> None:
        """AC2: Parallel chunks must return vectors in input-text order."""
        n_texts = 500
        concurrency = 10

        # Assign each chunk a distinct sentinel value to verify order
        chunk_size = 100

        # chunk i returns embeddings filled with float(i+1)
        responses: list[MagicMock] = []
        for chunk_idx, i in enumerate(range(0, n_texts, chunk_size)):
            size = min(chunk_size, n_texts - i)
            sentinel = float(chunk_idx + 1)
            responses.append(_build_response([_make_embedding(sentinel)] * size))

        client = _make_client_with_responses(responses)
        provider = GeminiEmbeddingProvider(
            client=client, dimension=_DIM, embed_concurrency=concurrency
        )

        texts = [str(i) for i in range(n_texts)]
        result = await provider.embed_batch(texts)

        assert len(result) == n_texts
        for i, vec in enumerate(result):
            expected_sentinel = float(i // chunk_size + 1)
            assert np.allclose(vec, expected_sentinel), (
                f"text[{i}] vector has wrong sentinel value; "
                f"expected {expected_sentinel}, got {vec[0]}"
            )

    @pytest.mark.asyncio
    async def test_order_preserved_150_texts(self) -> None:
        """Order preserved for partial last chunk (150 → chunks of 100 + 50)."""
        dim = _DIM
        responses = [
            _build_response([_make_embedding(1.0)] * 100),  # chunk 0
            _build_response([_make_embedding(2.0)] * 50),  # chunk 1
        ]
        client = _make_client_with_responses(responses)
        provider = GeminiEmbeddingProvider(client=client, dimension=dim, embed_concurrency=5)

        texts = [str(i) for i in range(150)]
        result = await provider.embed_batch(texts)

        assert len(result) == 150
        for vec in result[:100]:
            assert np.allclose(vec, 1.0)
        for vec in result[100:]:
            assert np.allclose(vec, 2.0)


# ---------------------------------------------------------------------------
# AC3 — 429 / ResourceExhausted retry with backoff
# ---------------------------------------------------------------------------


class _FakeResourceExhaustedError(Exception):
    """Stand-in for google.api_core.exceptions.ResourceExhausted."""


@pytest.mark.verification
class TestRateLimitRetry:
    @pytest.mark.asyncio
    async def test_429_retries_and_succeeds_on_second_attempt(self) -> None:
        """AC3: First call raises ResourceExhausted; second succeeds. Result is correct."""
        call_count = 0
        success_resp = _build_response([_make_embedding(0.5)] * 10)

        async def flaky_embed(*, model: str, contents: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _FakeResourceExhaustedError("quota exceeded")
            return success_resp

        client = MagicMock()
        client.aio.models.embed_content = flaky_embed  # type: ignore[assignment]

        provider = GeminiEmbeddingProvider(client=client, dimension=_DIM, embed_concurrency=1)

        # Patch asyncio.sleep so backoff doesn't slow tests
        with patch("strata_match.providers.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await provider.embed_batch([str(i) for i in range(10)])

        assert call_count == 2, f"Expected exactly 2 calls (1 fail + 1 retry), got {call_count}"
        assert len(result) == 10
        mock_sleep.assert_awaited_once()  # backoff sleep was called

    @pytest.mark.asyncio
    async def test_429_raises_embedding_error_after_max_retries(self) -> None:
        """If all 3 attempts fail with ResourceExhausted, EmbeddingError is raised."""
        from strata_match.exceptions import EmbeddingError

        async def always_fail(*, model: str, contents: Any) -> MagicMock:
            raise _FakeResourceExhaustedError("always quota exceeded")

        client = MagicMock()
        client.aio.models.embed_content = always_fail  # type: ignore[assignment]
        provider = GeminiEmbeddingProvider(client=client, dimension=_DIM, embed_concurrency=1)

        with (
            patch("strata_match.providers.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(EmbeddingError, match="rate limit"),
        ):
            await provider.embed_batch(["text"])

    @pytest.mark.asyncio
    async def test_non_rate_limit_exception_not_retried(self) -> None:
        """Non-ResourceExhausted errors should propagate immediately without retry."""
        from strata_match.exceptions import EmbeddingError

        call_count = 0

        async def other_error(*, model: str, contents: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            raise ValueError("unexpected API error")

        client = MagicMock()
        client.aio.models.embed_content = other_error  # type: ignore[assignment]
        provider = GeminiEmbeddingProvider(client=client, dimension=_DIM, embed_concurrency=1)

        with pytest.raises(EmbeddingError):
            await provider.embed_batch(["text"])

        assert call_count == 1, "Non-rate-limit error should not be retried"


# ---------------------------------------------------------------------------
# AC4 — concurrency=1 means sequential (backward-compat)
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestSequentialFallback:
    @pytest.mark.asyncio
    async def test_concurrency_1_calls_are_sequential(self) -> None:
        """AC4: concurrency=1 → max 1 in-flight at any time."""
        n_texts = 300
        active: list[int] = [0]
        max_active: list[int] = [0]

        async def mock_embed(*, model: str, contents: Any) -> MagicMock:
            active[0] += 1
            max_active[0] = max(max_active[0], active[0])
            await asyncio.sleep(0)
            active[0] -= 1
            return _build_response([_make_embedding(0.3)] * len(contents))

        client = MagicMock()
        client.aio.models.embed_content = mock_embed  # type: ignore[assignment]
        provider = GeminiEmbeddingProvider(client=client, dimension=_DIM, embed_concurrency=1)

        result = await provider.embed_batch([str(i) for i in range(n_texts)])

        assert len(result) == n_texts
        assert max_active[0] == 1, (
            f"With concurrency=1, max concurrent calls should be 1, got {max_active[0]}"
        )


# ---------------------------------------------------------------------------
# Factory passthrough — embed_concurrency kwarg forwarded
# ---------------------------------------------------------------------------


@pytest.mark.verification
class TestFactoryConcurrencyParam:
    def test_create_embedding_provider_forwards_embed_concurrency(self) -> None:
        """create_embedding_provider should forward embed_concurrency to GeminiEmbeddingProvider."""
        mock_client = MagicMock()
        provider = create_embedding_provider("gemini", _client=mock_client, embed_concurrency=5)
        assert isinstance(provider, GeminiEmbeddingProvider)
        assert provider._embed_concurrency == 5
