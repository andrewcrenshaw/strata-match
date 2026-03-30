# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-03-30

### Added

- Initial public release on PyPI.
- Two-stage matching: vector similarity pre-filter, then LLM nuance scoring.
- Core types: `CandidateProfile`, `JobDescription`, `MatchResult`, `BatchMatchResult`, `ConfidenceTier`.
- `create_matcher`, `match_job`, and `match_batch` public API.
- Optional extras: `openai`, `gemini`, `ollama`, `litellm`, and `all` for provider backends.
- Development extra `[dev]` with pytest, ruff, mypy, and coverage tooling.

[0.1.0]: https://github.com/andrewcrenshaw/strata-match/releases/tag/v0.1.0
