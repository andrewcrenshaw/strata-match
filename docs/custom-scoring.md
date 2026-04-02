# Custom scoring

This guide explains how to tune **Stage 1 (vector)** and **Stage 2 (LLM)** behavior, how confidence tiers are derived, and how to extend scoring for your domain.

## Two-stage pipeline (recap)

1. **Vector gate** — Cosine similarity on embeddings produces a score in `[0, 100]` (`vector_score` on results). Jobs below `vector_threshold` (raw cosine in `[0, 1]`) skip LLM scoring in batch flows, and single-match returns a low-confidence vector-only result.
2. **LLM scoring** — For candidates above the threshold, an LLM returns a 0–100 score, rationale, strengths, gaps, and optional salary/culture fields. `tokens_used` on each result reflects usage for that call.

## Tuning thresholds

### `vector_threshold` (Stage 1 gate)

Passed to `create_matcher(..., vector_threshold=0.3)`. This is the **minimum raw cosine similarity** in `[0, 1]` before Stage 2 runs (when an LLM scorer is configured).

- **Higher (e.g. 0.5–0.65)** — Fewer jobs reach the LLM; cheaper, stricter.
- **Lower (e.g. 0.2–0.35)** — More jobs reach the LLM; better recall for “non-obvious” fits, higher cost.

Use your own data: sample a few hundred pairs, plot cosine vs. human judgment, and pick a threshold that balances cost vs. miss rate.

### `llm_confirm_threshold` (confidence tier)

Passed to `create_matcher(..., llm_confirm_threshold=70.0)`. The matcher treats an LLM score **≥ this value** as “confirming” the match when computing **confidence tier** together with the vector score.

Lowering it increases the chance of **HIGH** tier when the vector score is strong; raising it makes **HIGH** rarer.

### Confidence tiers (`ConfidenceTier`)

`classify_confidence` (used internally) combines:

- Raw vector score (0–1 scale before the ×100 display)
- Whether the LLM score ≥ `llm_confirm_threshold`

Default internal cutpoints for the vector component are **0.7** (high) and **0.5** (medium). These are **not** currently exposed as constructor parameters on `Matcher`; to change them you would fork or wrap `classify_confidence` in a custom integration layer.

## Adding or weighting “criteria”

The library does **not** ship a pluggable scoring rubric (weights per skill, etc.). Practical options:

1. **Prompt-led criteria** — Encode domain rules in the scoring prompt (see [Prompt customization](prompt-customization.md)): require explicit JSON fields, add “must-have” checks, or ask the model to penalize missing licenses.
2. **Pre-computed embeddings** — Set `embedding` on `CandidateProfile` / `JobDescription` to control exactly what text (or vectors) feed Stage 1.
3. **Custom LLM provider** — Implement `LLMProvider` and pass it to `LLMScorer`, or use `create_matcher(..., scoring_provider=...)` with a `litellm` route to your preferred model.
4. **Post-processing** — Adjust or filter `MatchResult` scores in your application after `match_job` / `match_batch`.

## Vector-only mode

To skip Stage 2 entirely (embeddings only, no LLM cost):

```python
matcher = create_matcher("openai", scoring_provider=None)
```

## Token and cost tracking

- **Single match:** `MatchResult.tokens_used` after LLM scoring.
- **Batch:** `BatchMatchResult.total_tokens` and `llm_scored_count` summarize usage.

Use these fields for budgeting, dashboards, or per-tenant billing.
