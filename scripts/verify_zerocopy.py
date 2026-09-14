#!/usr/bin/env python3
"""Zero-copy FFI verification gate (simulator_spec.pdf Section 5).

Asserts that the NumPy array returned by `Ring.values_view()` /
`Ring.frame_matrix()` shares the *identical* base address as the raw Rust
slot slice (`Ring.data_ptr()`), i.e. `PyArray::borrow_from_array` exposed
the mmap/heap ring without copying. Exits non-zero on any mismatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import numpy as np  # noqa: E402

from shbt_cf.native import require_native  # noqa: E402


def main() -> int:
    ext = require_native()
    ring = ext.Ring(1024)
    ring.push_wave(1, 0, 8, 295.15, 100_000)

    view = np.asarray(ring.values_view())
    matrix = np.asarray(ring.frame_matrix())

    rust_ptr = ring.data_ptr()
    checks = [
        ("values_view pointer identity", view.ctypes.data == rust_ptr),
        ("frame_matrix pointer identity", matrix.ctypes.data == rust_ptr),
        ("values_view capacity", view.shape == (ring.capacity() * 8,)),
        ("frame_matrix shape", matrix.shape == (ring.capacity(), 8)),
        ("view owns no data (OWNDATA clear)", not view.flags["OWNDATA"]),
        ("writes visible to Rust", _write_through(ring, view)),
    ]

    ok = True
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
        ok &= bool(passed)
    print(f"rust slot ptr = {rust_ptr:#x}, numpy data ptr = {view.ctypes.data:#x}")
    return 0 if ok else 1


def _write_through(ring, view: np.ndarray) -> bool:
    """Write via the NumPy view and read back through the Rust consumer."""
    sentinel = 1234.5
    # Frame is 8 qwords: seq, ts, packed header, then values[0..5].
    view[8 + 3] = sentinel  # second frame, values lane 0
    fr = ring.pop()  # frame 0
    fr = ring.pop()  # frame 1
    return fr is not None and fr[3][0] == sentinel


if __name__ == "__main__":
    sys.exit(main())
