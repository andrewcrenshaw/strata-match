"""LLM prompt templates for job-to-profile matching."""

from strata_match.prompts.fast_score import (
    PROMPT_VERSION as FAST_SCORE_PROMPT_VERSION,
)
from strata_match.prompts.fast_score import (
    build_fast_score_prompt,
)
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
    "FAST_SCORE_PROMPT_VERSION",
    "RATIONALE_PROMPT_VERSION",
    "SCORE_PROMPT_VERSION",
    "build_fast_score_prompt",
    "build_rationale_prompt",
    "build_score_prompt",
    "build_score_prompt_parts",
]
