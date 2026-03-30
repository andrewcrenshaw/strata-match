# strata-match

[![Status: Pre-Alpha / development](https://img.shields.io/badge/status-pre--alpha%20(0.x)-orange.svg)](https://semver.org/spec/v2.0.0.html)
[![PyPI version](https://img.shields.io/pypi/v/strata-match.svg)](https://pypi.org/project/strata-match/)
[![Python versions](https://img.shields.io/pypi/pyversions/strata-match.svg)](https://pypi.org/project/strata-match/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Development (0.x):** This package is **incomplete** and **not production-ready**. Matching behavior, prompts, and provider integrations may change; test coverage and production validation are still in progress.

Two-stage vector + LLM job-to-profile matching engine. Computes fast vector similarity as a first pass, then uses LLM-based nuance scoring for high-potential matches.

## Features

- **Stage 1 — vector gate:** Cosine similarity on profile vs job embeddings; skips expensive calls when below threshold.
- **Stage 2 — LLM scoring:** Structured prompts for score (0–100), rationale, strengths, gaps, and confidence tier (HIGH / MEDIUM / LOW).
- **Pluggable providers:** Optional extras for OpenAI, Gemini, Ollama, and LiteLLM-backed backends.
- **Batch API:** `match_job` and `match_batch` for single-job or list scoring with shared configuration.
- **Typed models:** Pydantic models for profiles, jobs, and results.

## Installation

```bash
pip install strata-match
```

Requires **Python 3.11+**.

With OpenAI embedding support:

```bash
pip install strata-match[openai]
```

With full LLM provider support (via LiteLLM):

```bash
pip install strata-match[all]
```

## Quick Start

```python
import asyncio

from strata_match import (
    CandidateProfile,
    JobDescription,
    create_matcher,
    match_batch,
    match_job,
)


async def main() -> None:
    # Create a configured matcher (set api_key via env or argument for live APIs)
    matcher = create_matcher(
        "openai",
        vector_threshold=0.5,
    )

    profile = CandidateProfile(
        title="Senior Software Engineer",
        skills=["Python", "FastAPI", "PostgreSQL", "AWS"],
        years_of_experience=8,
        experience_summary="Full-stack engineer with distributed systems focus.",
    )

    job = JobDescription(
        title="Staff Engineer",
        company="Acme Corp",
        requirements=["Python", "System Design", "Leadership"],
        description="Lead backend platform team.",
    )

    result = await match_job(matcher, profile, job)
    print(result.score, result.confidence_tier, result.rationale)

    batch = await match_batch(matcher, profile, [job])
    for r in batch.results:
        print(r.job_title, r.score)


if __name__ == "__main__":
    asyncio.run(main())
```

Custom embedding or LLM providers use the submodule APIs, for example
`from strata_match.providers import create_embedding_provider`.

## Architecture

```
Stage 1: Vector Similarity (fast, cheap)
  Profile embedding ←→ Job embedding → cosine similarity score [0, 1]
  If score < vector_threshold → skip (no LLM call)

Stage 2: LLM Nuance Scoring (slow, rich)
  Structured prompt with profile + job → score, rationale, strengths, gaps
  Confidence tier: HIGH / MEDIUM / LOW
```

## Development

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/) (or pip).

```bash
# Install with dev dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Lint
uv run ruff check .

# Type check
uv run mypy src/ tests/
```

## License

MIT
