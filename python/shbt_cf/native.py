"""Import shim for the PyO3 extension built by `maturin` from
`crates/shbt-fabrication-hil` (feature `python`, module `shbt_cf_native`).

Build once:

    maturin build -m crates/shbt-fabrication-hil/Cargo.toml --features python
    pip3 install --user --no-deps target/wheels/shbt_fabrication_hil-*.whl

The wheel targets the stable ABI (`abi3-py310`) so the same artifact works on
Python 3.10 and 3.12+.
"""

from __future__ import annotations

try:  # pragma: no cover - import path depends on build host
    import shbt_cf_native as native  # type: ignore

    HAVE_NATIVE = True
except ImportError:  # pragma: no cover
    native = None  # type: ignore
    HAVE_NATIVE = False


def require_native():
    """Return the extension module or raise with build instructions."""
    if native is None:
        raise RuntimeError(
            "shbt_cf_native not built. Run: maturin build -m "
            "crates/shbt-fabrication-hil/Cargo.toml --features python && "
            "pip3 install --user --no-deps target/wheels/shbt_fabrication_hil-*.whl"
        )
    return native
