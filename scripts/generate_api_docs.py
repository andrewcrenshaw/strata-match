#!/usr/bin/env python3
"""Generate HTML API documentation from docstrings (pdoc).

Writes to docs/api/ at the repository root. Run from repo root::

    uv run python scripts/generate_api_docs.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OUTPUT = _REPO_ROOT / "docs" / "api"


def main() -> int:
    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pdoc",
        "strata_match",
        "-o",
        str(_OUTPUT),
    ]
    print("Running:", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=_REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
