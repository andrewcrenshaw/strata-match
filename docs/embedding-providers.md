# Embedding providers

Stage 1 matching uses the `EmbeddingProvider` interface (`strata_match.embeddings`). The factory `create_embedding_provider` (or `create_matcher("openai" | "gemini" | "ollama", ...)`) resolves built-in backends.

Install extras as needed:

```bash
pip install strata-match[openai]   # OpenAI embeddings
pip install strata-match[gemini]   # Google Gemini embeddings
pip install strata-match[ollama]   # Ollama (local) — pulls in aiohttp
pip install strata-match[all]      # all optional backends
```

## OpenAI

- **Extra:** `strata-match[openai]`
- **Default model:** `text-embedding-3-small` (1536 dimensions)
- **Typical config:** `api_key` via argument or `OPENAI_API_KEY` environment variable (standard OpenAI client behavior)

```python
from strata_match.providers import create_embedding_provider

provider = create_embedding_provider(
    "openai",
    model="text-embedding-3-small",
    dimension=1536,
    api_key="sk-...",
)
```

Or use the matcher factory (forwards embedding kwargs to the provider):

```python
from strata_match import create_matcher

matcher = create_matcher(
    "openai",
    embedding_model="text-embedding-3-small",
    scoring_provider="openai",
    api_key="sk-...",
)
```

## Google Gemini

- **Extra:** `strata-match[gemini]` (installs `google-genai`)
- **Default model:** `text-embedding-004` (768 dimensions)

```python
provider = create_embedding_provider(
    "gemini",
    model="text-embedding-004",
    api_key="...",
)
```

```python
matcher = create_matcher(
    "gemini",
    embedding_model="text-embedding-004",
    scoring_provider="litellm",
    scoring_model="gemini/gemini-2.0-flash",
    api_key="...",
)
```

Use the **Gemini** embedding provider for Stage 1 and **LiteLLM** (or OpenAI) for Stage 2 if you want a single Gemini stack for both — see `docs/prompt-customization.md` and LLM provider docs for model strings.

## Ollama (local)

- **Extra:** `strata-match[ollama]` (installs `aiohttp`)
- **Default model:** `nomic-embed-text`
- **Default base URL:** `http://localhost:11434`

Ensure Ollama is running and the model is pulled (`ollama pull nomic-embed-text`).

```python
provider = create_embedding_provider(
    "ollama",
    model="nomic-embed-text",
    base_url="http://localhost:11434",
)
```

```python
matcher = create_matcher(
    "ollama",
    model="nomic-embed-text",
    base_url="http://localhost:11434",
    scoring_provider="litellm",
    scoring_model="ollama/llama3.2",
)
```

## Pre-computed embeddings

If you already have vectors (e.g. from a hosted index), set `embedding` on `CandidateProfile` and/or `JobDescription` to skip embedding API calls for that side. Vectors must be compatible with the scorer’s cosine similarity (same dimensionality as the other side).

## Provider factory reference

`create_embedding_provider(name, *, model=None, dimension=None, **config)` accepts:

- `"openai"`, `"gemini"`, or `"ollama"` (case-insensitive)
- Optional `model` / `dimension` overrides
- Backend-specific `**config` (e.g. `api_key`, `base_url`)

See the **API reference** (`docs/api/index.html`) for full parameter lists on each provider class.
