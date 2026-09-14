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
## Domain solvers (Stage 3)

| Crate | Contents |
|---|---|
| `shbt-dielectric-floquet` | `FloquetModel`/`Transition` inputs (bands, occupations, `M_{ℓ,G}^{ab}` as supplied data); assembles `Π_{GG'}^{mn}(q,ω)`, `ε = δ − v_G Π` over composite `(G,m)` indices; κ₁-checked inverse via faer; `converge_unit_cell` drives a refinement series to `‖ε⁻¹_{k+1} − ε⁻¹_k‖∞ < 10⁻¹⁴` |
| `shbt-rcwa-optics` | Full-vector RCWA: Li-factorised convolution matrices (`⟦ε⟧`, `⟦1/ε⟧⁻¹` for lamellar and rectangular unit cells), eigenmodes of `Ω² = PQ` (cf.pdf Eq. 190 in the invariant-y TM limit), Redheffer star-product S-matrix recursion, per-order efficiencies, interface/internal field reconstruction. `option_b` models Λ = 960.80 nm, d = 42.50 nm, t_b = 7.50 nm, t_Ti = 10 nm at 65° silica internal incidence for both pumps |
| `shbt-fea-structural` | DIN 2092 Almen–László disc springs (`force`, `stiffness`, DIN stresses, Group 3 contact flats, compound stacks with friction), geometrically non-linear axisymmetric conical-disc FE with `E(T)`/`α(T)` and Newton–Raphson force control, extended Stoney bilayer stress (`R_pre`/`R_post`, finite-thickness correction, `α(T)` mismatch) |

Verification:
- RCWA vs analytic Fresnel/Airy slab benchmarks < 0.1 % (`solver::tests`), energy conservation
  on lossless gratings, planar four-layer `F_z` reproduces cf.pdf Table XXII (0.0999 at 65°,
  0.094 at the 68.5° minimum).
- `DIN_2093_GROUP2` gate: analytic `DiscSpring::force` vs tabulated DIN 2093 Group 2 forces at
  `s = 0.75 h₀`, all 20 series A/B discs within ±0.5 %; the exact-rotation FE agrees with DIN to
  ≲1.5 % on the same set.
- Note: the nominal grating `F_z` targets (124.5/138.2) are *not* reproduced by the stated
  Option B inputs (silica prism at 65°): the computed `F_z` is O(1) and the tooth-edge value
  grows with truncation, consistent with the spec's own sharp-corner caveat.
## Metrology & HIL (Stage 4)

| Crate | Contents |
|---|---|
| `shbt-metrology-gum` | GUM Supplement 1 parallel Monte Carlo: `Sampler` ingests scalar marginals (`Normal`/`Uniform`/`LogNormal`) and full covariance-correlated `Correlated` blocks via Cholesky; `propagate` runs `N ≥ 10⁶` evaluations over deterministic per-worker Xoshiro256** sub-streams and returns best estimate, standard uncertainty, 95 % coverage interval and Pearson sensitivities. `models::power_net` / `coffin_manson_cycles` measurands included |
| `shbt-fabrication-hil` | POSIX `shm_open`/`mmap` regions with `#[repr(C, align(64))]` `RingHeader`; SPSC lock-free ring (`AtomicUsize` head/tail, Acquire/Release, `try_push`/`push`/`send`); Tokio drain task + lock-free `PipelineStats` (atomic counters and a log₂ latency histogram); `ResidualMonitor` computing `χ²/ν = rᵀΣ⁻¹r/ν` per frame against `ReducedOrderModel` surrogates. 64 B `Frame` for Type-N (10 kHz), FBG (10 kHz) and CCD channels |

Benchmark gate (`pipeline::tests::ten_khz_streams_zero_drops_sub_microsecond`):
61 000 frames across three synthetic channels through the real ring+task — 0
dropped, 0 mutex/mlock calls, sub-microsecond mean per-frame processing latency.
