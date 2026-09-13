# shbt-cf

Zero-drift precision simulator and physical engineering workbench for the Static
Holographic Boundary Theory (SHBT) reactor. Architecture: `simulator_spec.pdf`;
physics and engineering specification: `cf.pdf`.

## Layout

```
Cargo.toml                 # workspace root (six crates, unidirectional dependency graph)
.cargo/config.toml         # target-cpu=native, opt-level=3, strict IEEE 754 (no fast-math)
crates/
  shbt-core-math           # Q64.64, MPFR adaptive precision, SO(3)/SE(3) exp maps, Yoshida-6, DAZ/FTZ
  shbt-dielectric-floquet  # inverse Floquet-Adler-Wiser dielectric tensors      -> core-math
  shbt-rcwa-optics         # 2D/3D RCWA, S-matrix recursion                      -> core-math
  shbt-fea-structural      # Belleville (DIN 2092/2093), Stoney, Coffin-Manson   -> core-math
  shbt-metrology-gum       # GUM Supplement 1 uncertainty / Monte Carlo          -> core-math
  shbt-fabrication-hil     # HIL telemetry, ROM residual monitoring              -> all of the above
python/shbt_cf             # Python orchestration wrapper (stdlib only)
scripts/check_dep_graph.py # CI gate: workspace graph is a DAG matching the spec topology
```

## Commands

`shbt-core-math` links GMP/MPFR through `rug`; on Debian/Ubuntu install
`m4 libgmp-dev libmpfr-dev` first.

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

## Core-math foundation (Stage 2)

| Module | Contents |
|---|---|
| `fixed` | `Q64x64` 128-bit signed fixed point, `2^-64` resolution, checked arithmetic |
| `precision` | `PrecisionMode`, `bits_lost_to_conditioning(κ)` → mantissa width in [128, 512] |
| `mp` | `MpMatrix` (rug/MPFR) LU solve/inverse, `κ₁(A)`, `solve_adaptive` precision escalation |
| `lie` | `Rotation` (SO(3)) and `Pose` (SE(3)) `exp`/`log`/`advance`; no quaternions or Euler angles |
| `symplectic` | `Yoshida6` (7-stage, order 6) on compensated `PhaseState`; `RigidBodyYoshida6` Lie–Poisson splitting |
| `fpenv` | `enable_flush_to_zero` / `FlushToZeroGuard`: MXCSR DAZ+FTZ (x86_64), FPCR FZ (aarch64) |

Verification gate (`cargo test -p shbt-core-math duffing`): a periodically forced double-well
Duffing oscillator, integrated for `10^7` Yoshida-6 steps in extended phase space, keeps
`|ΔH/H₀| < 10⁻¹²` (measured ≈ 6·10⁻¹⁴) while a 1-ulp shadow trajectory separates to O(1).
