"""strata-match: Two-stage vector + LLM job-to-profile matching engine."""

from strata_match.embeddings import EmbeddingProvider, EmbeddingProviderType
from strata_match.llm import LLMProvider, LLMResponse, LLMScorer
from strata_match.matcher import Matcher, create_matcher, match_batch, match_job
from strata_match.models import (
    BatchMatchResult,
    CandidateProfile,
    ConfidenceTier,
    JobDescription,
    MatchResult,
)
from strata_match.providers import create_embedding_provider

__all__ = [
    "BatchMatchResult",
    "CandidateProfile",
    "ConfidenceTier",
    "EmbeddingProvider",
    "EmbeddingProviderType",
    "JobDescription",
    "LLMProvider",
    "LLMResponse",
    "LLMScorer",
    "MatchResult",
    "Matcher",
    "create_embedding_provider",
    "create_matcher",
    "match_batch",
    "match_job",
]

__version__ = "0.1.0"
