"""strata-match: Two-stage vector + LLM job-to-profile matching.

This package exposes a small **public API** for matching job postings to
candidate profiles:

* **Factory:** :func:`create_matcher` returns a :class:`Matcher` configured
  with embedding and optional LLM scoring backends.
* **Matching:** :func:`match_job` and :func:`match_batch` score jobs against
  a profile.
* **Data types:** :class:`CandidateProfile`, :class:`JobDescription`,
  :class:`MatchResult`, :class:`BatchMatchResult`, and :class:`ConfidenceTier`.

Advanced integrations (custom embedding/LLM providers) live under
``strata_match.embeddings``, ``strata_match.llm``, and
``strata_match.providers`` — they are not re-exported from the root package.

Example::

    from strata_match import (
        CandidateProfile,
        JobDescription,
        create_matcher,
        match_job,
    )

    matcher = create_matcher("openai", vector_threshold=0.35)
    profile = CandidateProfile(title="Engineer", skills=["Python"])
    job = JobDescription(title="Backend Role", company="Acme")
    result = await match_job(matcher, profile, job)
    print(result.score, result.confidence_tier)
"""

from __future__ import annotations

from strata_match.matcher import Matcher, create_matcher, match_batch, match_job
from strata_match.models import (
    BatchMatchResult,
    CandidateProfile,
    ConfidenceTier,
    JobDescription,
    MatchResult,
)

__all__ = [
    "BatchMatchResult",
    "CandidateProfile",
    "ConfidenceTier",
    "JobDescription",
    "MatchResult",
    "Matcher",
    "create_matcher",
    "match_batch",
    "match_job",
]

__version__ = "0.1.0"
