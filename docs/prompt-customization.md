# Prompt customization

Stage 2 scoring uses structured prompts in `strata_match.prompts`. The default template asks for JSON: `score`, `strengths`, `gaps`, `rationale`, `salary_match`, `culture_signals`.

## Default scoring prompt

- **Module:** `strata_match.prompts.score_job`
- **System text:** `SYSTEM_PROMPT` — role instructions and JSON schema
- **Version:** `PROMPT_VERSION` — bump when you change the template (useful for auditing stored results)

`LLMScorer` calls `build_score_prompt(profile, job)` to produce chat messages. To customize behavior for a **fork** or **internal patch**, replace or wrap these functions and keep `PROMPT_VERSION` in sync with your changes.

## Prompt caching (provider-specific)

The design splits content so the **candidate profile** can be cached across many jobs:

- `build_score_prompt_parts(profile, job)` → `(static_prefix, dynamic_suffix)`.

The **static** part is the formatted profile; the **dynamic** part is the job. For **Anthropic** prompt caching, the module docstring describes building messages with `cache_control: {"type": "ephemeral"}` on the static block so repeated evaluations for the same profile hit cache (~90% of requests at large N).

The default `build_score_prompt` concatenates system + user text in a single OpenAI-style message; advanced integrations should use `build_score_prompt_parts` and construct messages per provider.

## Domain-specific matching

1. **Tighten the system prompt** — Add domain vocabulary (e.g. medical credentials, clearance levels), required JSON fields, or hard constraints (“score ≤ 40 if license X is missing”).
2. **Enrich the profile/job models** — Put structured fields into `CandidateProfile` / `JobDescription` and extend `_format_profile` / `_format_job` in a **copy** of the prompt module so the model sees the right context.
3. **Use LiteLLM for Stage 2** — `create_matcher(..., scoring_provider="litellm", scoring_model="...")` routes to many providers while keeping the same prompt shape.

## Detailed rationale (optional)

The separate template in `strata_match.prompts.rationale` (`build_rationale_prompt`) is for longer, candidate-facing narrative after an initial `MatchResult`. It is **not** invoked automatically by `Matcher`; call it from your application when you need a second LLM pass.

## Versioning

`MatchResult.prompt_version` records the scoring template version when returned from `LLMScorer`, so you can correlate stored scores with the prompt revision used at scoring time.
