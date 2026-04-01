"""Exercised by mypy via test_pytyped_packaging (PCC-1612); not collected by pytest."""

from __future__ import annotations

from strata_match import CandidateProfile


def _sample() -> CandidateProfile:
    return CandidateProfile(title="Engineer", skills=["Python"])
