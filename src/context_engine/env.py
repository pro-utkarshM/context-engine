"""Minimal .env loading without external dependencies."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path = ".env", *, override: bool = False) -> None:
    target = Path(path)
    if not target.exists():
        return

    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
