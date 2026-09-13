# shbt-cf

Zero-drift precision simulator and physical engineering workbench for the Static
Holographic Boundary Theory (SHBT) reactor. Architecture: `simulator_spec.pdf`;
physics and engineering specification: `cf.pdf`.

## Layout

```
Cargo.toml                 # workspace root (six crates, unidirectional dependency graph)
.cargo/config.toml         # target-cpu=native, opt-level=3, strict IEEE 754 (no fast-math)
crates/
  shbt-core-math           # Q64.64 fixed point, precision policy; Lie-group + Yoshida (Stage 2)
  shbt-dielectric-floquet  # inverse Floquet-Adler-Wiser dielectric tensors      -> core-math
  shbt-rcwa-optics         # 2D/3D RCWA, S-matrix recursion                      -> core-math
  shbt-fea-structural      # Belleville (DIN 2092/2093), Stoney, Coffin-Manson   -> core-math
  shbt-metrology-gum       # GUM Supplement 1 uncertainty / Monte Carlo          -> core-math
  shbt-fabrication-hil     # HIL telemetry, ROM residual monitoring              -> all of the above
python/shbt_cf             # Python orchestration wrapper (stdlib only)
scripts/check_dep_graph.py # CI gate: workspace graph is a DAG matching the spec topology
```

## Commands

```sh
cargo check --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
python3 scripts/check_dep_graph.py          # cyclic dependency validation
python3 -m unittest discover -s python      # wrapper self-test

PYTHONPATH=python python3 -m shbt_cf topology   # print crate graph and build order
PYTHONPATH=python python3 -m shbt_cf verify     # compare on-disk workspace to the spec
PYTHONPATH=python python3 -m shbt_cf test shbt-core-math
```

`pip install -e python` installs the `shbt-cf` console script.

## Determinism policy

All `f64` code is compiled with fast-math disabled
(`-C llvm-args=-enable-no-nans-fp-math=false -C llvm-args=-enable-no-signed-zeros-fp-math=false`);
the Python wrapper re-asserts these flags via `CARGO_BUILD_RUSTFLAGS` on every cargo call.
Global accumulators use `shbt_core_math::fixed::Q64x64`; ill-conditioned linear algebra selects an
MPFR mantissa width via `shbt_core_math::precision::PrecisionMode::arbitrary_for_condition_number`.
