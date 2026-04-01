"""PEP 561 py.typed wheel contract (PCC-1612)."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.verification


def test_built_wheel_contains_py_typed() -> None:
    """Downstream tools need py.typed in the published wheel."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        dist = Path(tmp) / "dist"
        dist.mkdir(parents=True)
        r = subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stdout + r.stderr
        wheels = list(dist.glob("strata_match-*.whl"))
        assert len(wheels) == 1, f"expected one wheel, got {wheels}"
        with zipfile.ZipFile(wheels[0]) as zf:
            names = zf.namelist()
        assert any(n.endswith("strata_match/py.typed") for n in names), names


def test_mypy_strict_on_typing_consumer() -> None:
    """Editable / installed package type-checks under --strict."""
    consumer = ROOT / "tests" / "mypy_typing_consumer.py"
    assert consumer.is_file()
    r = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", str(consumer)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
