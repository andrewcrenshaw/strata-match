# CLAUDE.md — Strata Match

> Inherits shared workspace context from `~/Development/CLAUDE.md`. This file adds `strata-match`-specific context only.

---

## Project Identity

**strata-match** is a job matching and scoring library. Python package at `src/strata_match/`. Provides semantic similarity scoring and structured criteria matching between candidate profiles and job descriptions.

---

## Stack & Dev Commands

```bash
source .venv/bin/activate
pip install -e .             # or: uv pip install -e .
pytest tests/ -v
```

---

## This Repo's Backlog Tag

When creating tickets: `"repo": "strata-match"`

---

*Shared infrastructure (PDT API, session lifecycle, ticket creation) → `~/Development/CLAUDE.md`*
