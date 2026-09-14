"""Workbench package with compatibility exports for the legacy module."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

_legacy_path = Path(__file__).with_name("..") / "workbench.py"
_spec = importlib.util.spec_from_file_location("shbt_cf._legacy_workbench", _legacy_path.resolve())
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load workbench implementation from {_legacy_path}")
_legacy = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _legacy
_spec.loader.exec_module(_legacy)

for _name, _value in vars(_legacy).items():
    if not _name.startswith("_"):
        globals()[_name] = _value

__all__ = [name for name in globals() if not name.startswith("_")]
