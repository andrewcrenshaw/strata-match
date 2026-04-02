"""Package metadata and pyproject contract (PCC-1520 / release hygiene)."""

from __future__ import annotations

import tomllib
from importlib import metadata
from pathlib import Path
from typing import Any, cast

import pytest

pytestmark = pytest.mark.verification

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"


def _project_table() -> dict[str, Any]:
    with _PYPROJECT.open("rb") as f:
        data = tomllib.load(f)
    return cast("dict[str, Any]", data["project"])


def test_package_version_matches_pyproject() -> None:
    """Installed distribution version matches pyproject.toml."""
    expected = _project_table()["version"]
    assert metadata.version("strata-match") == expected


def test_python_requires_is_at_least_3_11() -> None:
    """PyPI contract: support Python 3.11+."""
    req = _project_table()["requires-python"]
    assert req == ">=3.11"


def test_license_and_name() -> None:
    """Project identity for PyPI."""
    proj = _project_table()
    assert proj["name"] == "strata-match"
    assert proj["license"]["text"] == "MIT"


def test_dev_optional_dependency_group_exists() -> None:
    """[dev] extra is defined for tooling."""
    dev = _project_table()["optional-dependencies"]["dev"]
    joined = " ".join(dev)
    assert "pytest" in joined
    assert "ruff" in joined
