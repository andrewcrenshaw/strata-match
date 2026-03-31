# strata-match

[![Status: Pre-Alpha](https://img.shields.io/badge/status-pre--alpha%20(0.x)-orange.svg)](https://semver.org/spec/v2.0.0.html)
[![PyPI version](https://img.shields.io/pypi/v/strata-match.svg)](https://pypi.org/project/strata-match/)
[![Python versions](https://img.shields.io/pypi/pyversions/strata-match.svg)](https://pypi.org/project/strata-match/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Two-stage job-to-profile matching engine.** Fast vector similarity as a first pass, then LLM-powered nuance scoring for the matches that matter. Get a score, a rationale, strengths, gaps, and a confidence tier — not just a number.

## Why This Exists

Keyword matching for jobs is broken. A senior Python developer doesn't match "Staff Engineer — Backend Platform" because the words don't overlap, even though the fit is obvious. Pure embedding similarity gets closer but can't reason about career trajectory, transferable skills, or the difference between "nice to have" and "must have."

`strata-match` combines both approaches:

1. **Vector gate (fast, cheap)** — Cosine similarity on embeddings catches obvious non-matches before you spend money on LLM calls. Below the threshold? Skip it. This filters out 70-80% of listings at near-zero cost.

2. **LLM scoring (slow, rich)** — For candidates that pass the vector gate, a structured prompt sends the full profile and job description to an LLM. You get back a 0-100 score, a written rationale, specific strengths and gaps, and a confidence tier (HIGH / MEDIUM / LOW).

The result: high-quality matching at a fraction of what it would cost to run every listing through an LLM.

## Use Cases

- **Job search platforms** — Score thousands of listings against a candidate profile, surface the top matches with explanations of *why* they match
- **Recruiting and talent matching** — Flip the model: score candidates against a job description, rank by fit, use gap analysis for interview prep
- **Career coaching tools** — Show candidates where they're strong, where they have gaps, and what skills would unlock the next tier of opportunities
- **Internal mobility** — Match employees to open internal roles, identify skill adjacencies, recommend lateral moves
- **Market positioning** — Score your profile against 100 job descriptions in your target space to understand where you're competitive and where you need to grow

## Quick Start

```python
import asyncio
from strata_match import (
    CandidateProfile,
    JobDescription,
    create_matcher,
    match_job,
    match_batch,
)

async def main():
    matcher = create_matcher("openai", vector_threshold=0.5)

    profile = CandidateProfile(
        title="Senior Software Engineer",
        skills=["Python", "FastAPI", "PostgreSQL", "AWS", "System Design"],
        years_of_experience=8,
        experience_summary="Full-stack engineer specializing in distributed systems, "
            "data pipelines, and API design. Led migration from monolith to "
            "microservices serving 2M requests/day.",
    )

    jobs = [
        JobDescription(
            title="Staff Engineer — Backend Platform",
            company="Acme Corp",
            requirements=["Python", "System Design", "Technical Leadership"],
            description="Lead the backend platform team. Own the API layer, "
                "drive architecture decisions, mentor senior engineers.",
        ),
        JobDescription(
            title="Frontend Developer",
            company="Widget Inc",
            requirements=["React", "TypeScript", "CSS"],
            description="Build pixel-perfect UIs for our consumer product.",
        ),
    ]

    # Single match with full rationale
    result = await match_job(matcher, profile, jobs[0])
    print(f"Score: {result.score}/100 ({result.confidence_tier})")
    print(f"Rationale: {result.rationale}")
    print(f"Strengths: {result.strengths}")
    print(f"Gaps: {result.gaps}")

    # Batch matching — vector gate skips obvious mismatches
    batch = await match_batch(matcher, profile, jobs)
    for r in batch.results:
        print(f"{r.job_title}: {r.score} ({r.confidence_tier})")

asyncio.run(main())
```

**Example output:**

```
Score: 82/100 (HIGH)
Rationale: Strong backend systems experience directly maps to platform team needs.
  8 years of Python + system design + API architecture align well with staff-level
  expectations. Migration leadership demonstrates the technical leadership requirement.
  Gap: no explicit mention of mentoring experience, though team lead implies it.
Strengths: ['Python expertise', 'System design', 'API architecture', 'Migration leadership']
Gaps: ['Explicit mentoring/coaching experience', 'Staff-level scope communication']

Staff Engineer — Backend Platform: 82 (HIGH)
Frontend Developer: 0 (skipped by vector gate)
```

## Installation

```bash
pip install strata-match
```

Requires **Python 3.11+**.

With OpenAI embedding and scoring support:

```bash
pip install strata-match[openai]
```

With all LLM providers (OpenAI, Gemini, Ollama via LiteLLM):

```bash
pip install strata-match[all]
```

## How It Works

```
Profile + Job
     │
     ▼
┌─────────────────────────────────┐
│  Stage 1: Vector Similarity     │  Cost: ~$0.0001/comparison
│  Embed profile + job → cosine   │  Speed: <100ms
│  similarity score [0, 1]        │
│                                 │
│  Below threshold? → SKIP        │  Filters 70-80% of listings
└────────────┬────────────────────┘
             │ passes gate
             ▼
┌─────────────────────────────────┐
│  Stage 2: LLM Nuance Scoring   │  Cost: ~$0.01/comparison
│  Structured prompt with full    │  Speed: 2-5 seconds
│  profile + job description      │
│                                 │
│  Returns: score (0-100),        │
│  rationale, strengths, gaps,    │
│  confidence tier                │
└─────────────────────────────────┘
```

### Why Two Stages?

Economics. LLM calls cost 100x more than embedding comparisons. If you're scoring a profile against 500 job listings, running all 500 through an LLM costs ~$5 and takes 20 minutes. With the vector gate filtering at 0.5 threshold, you send maybe 100 to the LLM — $1 and 4 minutes. Same quality matches, 80% cost reduction.

### Confidence Tiers

| Tier | Meaning | Typical Score Range |
|------|---------|-------------------|
| **HIGH** | Strong match — profile clearly fits the role | 70-100 |
| **MEDIUM** | Partial match — transferable skills, some gaps | 40-69 |
| **LOW** | Weak match — significant gaps or career pivot needed | 0-39 |

### Pluggable Providers

Embedding and LLM scoring use a provider abstraction. Swap models without changing your matching logic:

```python
# OpenAI (default, highest quality)
matcher = create_matcher("openai")

# Google Gemini (good quality, lower cost)
matcher = create_matcher("gemini")

# Local Ollama (free, private, slower)
matcher = create_matcher("ollama", model="nomic-embed-text")

# LiteLLM (route to any supported model)
matcher = create_matcher("litellm", model="anthropic/claude-3-haiku")
```

Custom providers implement the `EmbeddingProvider` and `ScoringProvider` interfaces:

```python
from strata_match.providers import create_embedding_provider, create_scoring_provider
```

## Part of the Strata Ecosystem

`strata-match` is the scoring engine for [Strata](https://github.com/andrewcrenshaw/strata) — an autonomous AI job search platform where specialized agents collaborate to discover, evaluate, and match job opportunities. In that context, the Match Agent runs `strata-match` against every new listing that passes deduplication, stores results with confidence tiers, and routes high-confidence matches to the Apply Agent for resume tailoring.

But `strata-match` is fully standalone. It has no dependency on the Strata platform and works anywhere you need intelligent job-to-profile matching.

## Development

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/) (or pip).

```bash
git clone https://github.com/andrewcrenshaw/strata-match.git
cd strata-match

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
