"""LLM prompt templates for job-to-profile matching."""

from strata_match.prompts.rationale import (
    PROMPT_VERSION as RATIONALE_PROMPT_VERSION,
)
from strata_match.prompts.rationale import (
    build_rationale_prompt,
)
from strata_match.prompts.score_job import (
    PROMPT_VERSION as SCORE_PROMPT_VERSION,
)
from strata_match.prompts.score_job import (
    build_score_prompt,
    build_score_prompt_parts,
)

__all__ = [
    "RATIONALE_PROMPT_VERSION",
    "SCORE_PROMPT_VERSION",
    "build_rationale_prompt",
    "build_score_prompt",
    "build_score_prompt_parts",
]
