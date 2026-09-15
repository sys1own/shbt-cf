"""Calibration schema resources shipped with :mod:`shbt_cf`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_DIR = Path(__file__).parent


def load_schema(name: str) -> dict[str, Any]:
    """Load a bundled JSON schema by filename or stem."""
    path = SCHEMA_DIR / name
    if path.suffix == "":
        path = path.with_suffix(".json")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)
