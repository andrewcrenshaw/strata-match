# strata-match documentation

Guides and API reference for the [strata-match](https://github.com/andrewcrenshaw/strata-match) library (two-stage vector + LLM job-to-profile matching).

## Guides

| Guide | Description |
|-------|-------------|
| [Custom scoring](custom-scoring.md) | Vector and LLM thresholds, confidence tiers, extending scoring behavior |
| [Embedding providers](embedding-providers.md) | OpenAI, Gemini, and Ollama embedding backends |
| [Prompt customization](prompt-customization.md) | Scoring prompts, prompt caching, domain-specific matching |

## API reference

HTML API reference is **generated from docstrings** with [pdoc](https://pdoc.dev/).

- **Browse locally:** open [`api/index.html`](api/index.html) after cloning the repo.
- **Regenerate** (requires dev dependencies):

  ```bash
  uv sync --all-extras
  uv run python scripts/generate_api_docs.py
  ```

  Or run `pdoc` directly:

  ```bash
  uv run pdoc strata_match -o docs/api
  ```

## Project links

- [README](../README.md) — overview, installation, quick start
- [Repository](https://github.com/andrewcrenshaw/strata-match)
- [PyPI](https://pypi.org/project/strata-match/)
- [Changelog](https://github.com/andrewcrenshaw/strata-match/blob/main/CHANGELOG.md)
