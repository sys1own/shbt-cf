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
## Python orchestration & FFI (Stage 5)

- `shbt-fabrication-hil` builds `shbt_cf_native` (`crate-type = ["rlib","cdylib"]`,
  optional `python` feature → PyO0 extension, `abi3-py310`):
  - `src/ffi.rs` — plain `extern "C"` exports (`shbt_beat_hz`, `shbt_disc_force`,
    `shbt_chi2`, `shbt_ring_*` heap-ring handle API).
  - `src/pyo3_ffi.rs` — `Ring` pyclass plus `beat_hz`, `disc_force`, `chi2`,
    `power_net_mc`, `coffin_manson_mc`, `column_means`. `values_view()` /
    `frame_matrix()` expose the ring's slot memory through
    `PyArray::borrow_from_array` — zero-copy NumPy views of the mmap/heap region.
- `python/shbt_cf/` — `workbench.py` (sweep engine, NSGA-III driver, GUM MC
  PDFs; `TorqueWrench`/`BellevillePreload`/`HipimsStress`/`GratingAlignment`
  HUD model), `optimize.py` (pure-NumPy NSGA-III: Das–Dennis refs, vectorised
  non-dominated sort, niching, SBX + polynomial mutation), `hud_streamlit.py`
  (two-mode web UI). CLI: `python -m shbt_cf {mc,sweep,optimize,hud,hud-web,verify-zerocopy}`.
- `scripts/verify_zerocopy.py` — pointer-identity gate: asserts
  `view.ctypes.data == ring.data_ptr()` for both views, `OWNDATA` clear, and
  write-through visibility into the Rust consumer.

Build the native module once:
```
maturin build -m crates/shbt-fabrication-hil/Cargo.toml --features python --release
pip3 install --user --no-deps target/wheels/shbt_fabrication_hil-*.whl
```
## Integration & build lockdown (Stage 6)

- `tests/closure_audit.rs` (shbt-fabrication-hil) — physical closure audits vs
  cf.pdf reference values:
  - volume mapping: `A_foot·t_metal = V_metal = 2.875e-12 m³`,
    `N_D·V_D = V_domains = 2.30e-8 m³`, ratio exactly 8000 (Eqs. 29/35/123);
  - GUM calorimetry: Table XLVI `(c·u)²` terms, nominal Σ=19.370 W²
    (u_RSS=4.4011 W) with the unassigned 13.0061 W² residual made explicit
    (Eq. 287), two-term allocation u_c=4.40806 W ≤ 5.6881 W (Eq. 284), and the
    literal four-input budget P=ṁ·c_p·ΔT+P_env propagated through the parallel
    MC engine to u_c≈5.696 W < 5.70 W (Eq. 305);
  - Coffin–Manson range convention (ε′f=0.18, c=−0.62): ceiling
    Δεp=0.0001530105 at N_f=44 820, N_f(0.00098)=2242, N_f(0.000155)=43 896 —
    the nominal 0.000155 ceiling provably misses the endurance target.
- `tests/bitwise_regression.rs` — deterministic kernel digest (Xoshiro256**,
  Yoshida-6 compensated integration, Q64.64 accumulation, χ²/power measurands,
  frame layout) restricted to libm-free IEEE-754 ops; golden digest
  `0x03de00d3acded967` checked on both ISAs.
- CI: new `bitwise` matrix job on `ubuntu-24.04` (x86-64/AVX-512) and
  `ubuntu-24.04-arm` (ARM64/NEON) asserting identical digests.
