"""Documentation layout and README contract (PCC-1522)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.verification

_REPO_ROOT = Path(__file__).resolve().parents[1]
_README = _REPO_ROOT / "README.md"
_GUIDES = [
    _REPO_ROOT / "docs" / "README.md",
    _REPO_ROOT / "docs" / "custom-scoring.md",
    _REPO_ROOT / "docs" / "embedding-providers.md",
    _REPO_ROOT / "docs" / "prompt-customization.md",
]
_GENERATOR = _REPO_ROOT / "scripts" / "generate_api_docs.py"


def test_guide_files_exist() -> None:
    """Guides listed in PCC-1522 are present."""
    for path in _GUIDES:
        assert path.is_file(), f"missing guide: {path.relative_to(_REPO_ROOT)}"


def test_readme_has_badges_and_install_quickstart() -> None:
    """README exposes PyPI, Python, license, CI, install, and quickstart."""
    text = _README.read_text(encoding="utf-8")
    assert "pypi.org/project/strata-match" in text
    assert "pypi/pyversions/strata-match" in text
    assert "License-MIT" in text or "MIT" in text
    assert "actions/workflows/ci.yml/badge.svg" in text
    assert "pip install strata-match" in text
    assert "match_job" in text
    assert "create_matcher" in text


def test_readme_documents_feature_list() -> None:
    """README calls out two-stage scoring, providers, caching, tokens."""
    text = _README.read_text(encoding="utf-8")
    assert "## Features" in text
    assert "two-stage" in text.lower() or "Two-stage" in text
    assert "embedding" in text.lower()
    assert "prompt" in text.lower() and "cach" in text.lower()
    assert "token" in text.lower()


def test_api_doc_generator_runs() -> None:
    """pdoc can generate docs/api from installed package."""
    assert _GENERATOR.is_file()
    proc = subprocess.run(
        [sys.executable, str(_GENERATOR)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    index = _REPO_ROOT / "docs" / "api" / "index.html"
    strata = _REPO_ROOT / "docs" / "api" / "strata_match.html"
    assert index.is_file() and strata.is_file()
    assert "strata_match" in strata.read_text(encoding="utf-8")[:5000]


def test_docs_index_links_to_guides() -> None:
    """docs/README.md links to the three guides."""
    doc_readme = _REPO_ROOT / "docs" / "README.md"
    body = doc_readme.read_text(encoding="utf-8")
    for name in ("custom-scoring.md", "embedding-providers.md", "prompt-customization.md"):
        assert name in body
