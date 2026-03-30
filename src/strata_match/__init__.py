"""strata-match: Two-stage vector + LLM job-to-profile matching engine."""

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
