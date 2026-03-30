# strata-match

Two-stage vector + LLM job-to-profile matching engine. Computes fast vector similarity as a first pass, then uses LLM-based nuance scoring for high-potential matches.

## Installation

```bash
pip install strata-match
```

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
from strata_match import create_matcher, match_job, match_batch
from strata_match.models import CandidateProfile, JobDescription

# Create a configured matcher
matcher = create_matcher(
    embedding_provider="openai",
    vector_threshold=0.5,
)

# Build a candidate profile
profile = CandidateProfile(
    title="Senior Software Engineer",
    skills=["Python", "FastAPI", "PostgreSQL", "AWS"],
    experience_years=8,
    summary="Full-stack engineer with distributed systems focus.",
)

# Build a job description
job = JobDescription(
    title="Staff Engineer",
    company="Acme Corp",
    requirements=["Python", "System Design", "Leadership"],
    description="Lead backend platform team.",
)

# Match a single job
result = await match_job(matcher, profile, job)
print(result.score, result.confidence_tier, result.rationale)

# Match a batch of jobs
results = await match_batch(matcher, profile, [job])
for r in results.results:
    print(r.job_title, r.score)
```

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

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

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
