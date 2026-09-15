# SHBT Cold Fusion Reactor: Precision Simulation & Engineering Workbench

`shbt-cf` is the multi-domain physics simulation engine, hardware-in-the-loop (HIL) telemetry architecture, and numerical workbench for the Static Holographic Boundary Theory (SHBT) reactor specification. Primary physics derivations, structural bounds, and electrodynamic models are detailed in `cf.pdf`.

---

## Workspace Layout

The repository is structured as a Rust Cargo workspace containing 9 crates paired with a native Python orchestration wrapper:


```

Cargo.toml                 # Workspace root (9 crates with strict DAG topology)
.cargo/config.toml         # Target CPU optimization, opt-level 3, strict IEEE 754 rules
crates/
shbt-core-math           # Q64.64 fixed-point, MPFR, Lie algebra SO(3)/SE(3), Yoshida-6, DAZ/FTZ
shbt-dielectric-floquet  # Floquet-Adler-Wiser dielectric matrix inversions & elastodynamics
shbt-rcwa-optics         # 2D/3D vector RCWA, S-matrix recursion
shbt-rcwa                # High-level RCWA solver execution and verification
shbt-fea-structural      # Belleville disc springs, Stoney stress, fatigue, viscoplastic kinetics
shbt-contact             # EHD contact mechanics & projected damped active set (PDAS) solvers
shbt-metrology-gum       # GUM Supplement 1 Monte Carlo uncertainty engine & dual numbers
shbt-fabrication-hil     # Shared-memory HIL ring, ROM residual monitoring, CFT kinetics, transport
shbt-py-bindings         # PyO3 FFI layer exposing Rust solvers to Python
python/shbt_cf             # Python orchestration suite, control, metrology, and workbench tools
scripts/
check_dep_graph.py       # CI DAG topology validator
verify_zerocopy.py       # Pointer-identity verification for PyO3 shared memory
tests/
test_full_multiphysics_pipeline.py  # Full multiphysics end-to-end simulation runner
test_physics.py                    # Multi-domain physics validation suite
test_py_bindings.py               # FFI boundary integration tests
test_update5_modules.py           # Module integration regression tests
validate_benchmarks.py            # Automated performance benchmark validator
verification_tests.rs             # Multi-crate integration verification suite

```

---

## Prerequisites & Installation

### System Dependencies

The core math engine links against system GMP and MPFR libraries. On Debian/Ubuntu distributions:

```bash
sudo apt-get install -y m4 libgmp-dev libmpfr-dev

```

### Rust Workspace Validation

Build, lint, and test all 9 crates across the workspace:

```bash
cargo check --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

```

### Python Workspace & FFI Setup

Install the Python package in editable mode and run verification tools:

```bash
# Install Python orchestration layer
pip install -e python

# Run CI topology and self-tests
python3 scripts/check_dep_graph.py
python3 -m unittest discover -s python

# Inspect topology and run crate integration tests via shbt_cf CLI
PYTHONPATH=python python3 -m shbt_cf topology
PYTHONPATH=python python3 -m shbt_cf verify
PYTHONPATH=python python3 -m shbt_cf test shbt-core-math

```

To build and link the high-performance PyO3 native extension:

```bash
maturin build -m crates/shbt-py-bindings/Cargo.toml --features python --release
pip install --user --no-deps target/wheels/shbt_py_bindings-*.whl
python3 scripts/verify_zerocopy.py

```

---

## Determinism & Numerical Safety Policy

To prevent non-deterministic floating-point drifting across microarchitectures and compiler toolchains:

1. **Strict IEEE 754 Compliance:** Fast-math flag sets are explicitly disabled across all workspace profiles (`-C llvm-args=-enable-no-nans-fp-math=false -C llvm-args=-enable-no-signed-zeros-fp-math=false`).
2. **Fixed-Point Accumulation:** Global energy and state accumulators utilize `shbt_core_math::fixed::Q64x64` 128-bit signed fixed-point integers ($2^{-64}$ fractional resolution).
3. **Adaptive Precision Inversion:** Ill-conditioned linear algebra operations dynamically select MPFR mantissa bit-widths via `PrecisionMode::arbitrary_for_condition_number` based on matrix condition number $\kappa_1(A)$.
4. **Denormal Handling:** Hardware denormals-are-zero (DAZ) and flush-to-zero (FTZ) modes are explicitly enforced via MXCSR (`x86_64`) and FPCR (`aarch64`).

---

## Crate Architecture & Stage Breakdown

### 1. Core Mathematics Foundation (`shbt-core-math`)

| Module | Scope & Theoretical Implementation |
| --- | --- |
| `fixed` | `Q64x64` 128-bit signed fixed-point math with checked arithmetic. |
| `precision` | Conditioning analysis (`bits_lost_to_conditioning`), scaling mantissa widths between 128 and 512 bits. |
| `mp` | MPFR-backed `MpMatrix` LU decomposition, condition number estimation $\kappa_1(A)$, and adaptive precision solvers. |
| `lie` | Exponential/logarithmic maps for $SO(3)$ rotations and $SE(3)$ spatial poses without gimbal-lock or quaternion ambiguities. |
| `symplectic` | 7-stage 6th-order Yoshida integrator (`Yoshida6`) operating on compensated phase-space state vectors. |
| `fpenv` | Scoped CPU register guard (`FlushToZeroGuard`) setting DAZ/FTZ flags across threads. |

*Verification Gate:* A double-well Duffing oscillator integrated for $10^7$ `Yoshida6` steps maintains energy drift $\vert\Delta H / H_0\vert < 10^{-12}$ (measured $\approx 6 \times 10^{-14}$) while a 1-ulp shadow trajectory diverges to $\mathcal{O}(1)$.

### 2. Domain Solvers

| Crate | Scope & Theoretical Implementation |
| --- | --- |
| `shbt-dielectric-floquet` | Assembles composite polarization tensors $\Pi_{GG'}^{mn}(q, \omega)$ and dielectric matrices $\epsilon = \delta - v_G \Pi$ under periodic driving. Evaluates elastodynamics and inverse tensors with matrix conditioning checks ($\kappa_1$) until convergence ($\Vert\epsilon^{-1}_{k+1} - \epsilon^{-1}_k\Vert_\infty < 10^{-14}$). |
| `shbt-rcwa-optics` / `shbt-rcwa` | Full-vector 2D/3D RCWA with Li-factorized convolution matrices ($\llbracket\epsilon\rrbracket$, $\llbracket 1/\epsilon\rrbracket^{-1}$) for rectangular unit cells. Computes eigenmodes of $\Omega^2 = PQ$, Redheffer star-product $S$-matrix recursion, diffraction efficiency, and local electric field enhancement $F_z$. |
| `shbt-fea-structural` | Non-linear structural modeling: DIN 2092/2093 disc spring stacks with friction, non-linear axisymmetric conical-disc FEA with temperature-dependent moduli $E(T)$ and thermal expansion $\alpha(T)$, Stoney bilayer film stress evaluations, and viscoplastic kinetics. |
| `shbt-contact` | Elastohydrodynamic contact solvers and projected damped active set (PDAS) numerical models for contact mechanics under extreme thermal loading. |

*Verification Gates:*

* RCWA matches analytic Fresnel/Airy slab benchmarks within $<0.1\%$ error and satisfies lossless grating energy conservation.
* Planar four-layer field enhancement $F_z$ reproduces `cf.pdf` Table XXII targets ($0.0999$ at $65^\circ$, $0.094$ at $68.5^\circ$).
* Belleville forces match tabulated DIN 2093 Group 2 series A/B specifications within $\pm 0.5\%$.

### 3. Metrology & Hardware-in-the-Loop (HIL)

| Crate | Scope & Theoretical Implementation |
| --- | --- |
| `shbt-metrology-gum` | GUM Supplement 1 parallel Monte Carlo engine ($N \ge 10^6$ iterations) and dual-number automatic differentiation. Evaluates correlated input distributions via Cholesky decomposition across deterministic per-worker Xoshiro256** PRNG streams. Generates coverage intervals and Pearson sensitivity indices. |
| `shbt-fabrication-hil` | POSIX shared memory ring (`shm_open`/`mmap`) with lock-free atomic SPSC indexing and cache-aligned `#[repr(C, align(64))]` 64-byte frame structures. Features CFT kinetics, McNabb-Foster hydrogen transport modeling, Tokio drain task, latency profiling histograms, and reduced-order model (ROM) residual monitoring ($\chi^2 / \nu$). |

*Benchmark Gate:* The 10 kHz telemetry channel pipeline processes over 61,000 frames across 3 synthetic streams with zero frame drops, zero mutex allocation, and sub-microsecond mean processing latency.

---

## Python Orchestration & FFI Layer

The native bindings layer (`shbt-py-bindings` and `shbt-fabrication-hil`) exports C-ABI functions and PyO3 native extensions.

* **Zero-Copy Memory Access:** Telemetry ring memory is exposed directly to Python using zero-copy array views. Pointer verification guarantees zero memory copy overhead (`view.ctypes.data == ring.data_ptr()`).
* **Python Package (`python/shbt_cf`):**
* `workbench.py` & `workbench/`: Parameter sweep engines, HUD models, and Coffin-Manson Monte Carlo wrappers.
* `optimize.py`: Pure-NumPy NSGA-III multi-objective optimization (Das-Dennis reference points, non-dominated sorting, polynomial mutation).
* `hud_streamlit.py`: Interactive Streamlit engineering interface.
* `control/`: System order reduction and state-space control routines.
* `metrology/`: Python-side telemetry analysis and calibration curve parsing.
* `schemas/`: JSON schemas for anisotropic permittivity, Pt100 calibration, and surface roughness topologies.



---

## Physical Closure Audits & Regression Testing

1. **Physical Closure Audit (`crates/shbt-fabrication-hil/tests/closure_audit.rs`):**
* **Volume Mapping:** Verifies active domain mapping ratios ($V_{\text{domains}} / V_{\text{metal}} = 8000$) according to `cf.pdf` Equations 29, 35, and 123.
* **Calorimetric Uncertainty Budget:** Validates GUM calorimetry Monte Carlo propagation ($P = \dot{m} c_p \Delta T + P_{\text{env}}$), confirming standard uncertainty bounds ($u_c \approx 5.696\text{ W} < 5.70\text{ W}$).
* **Fatigue Life Ceilings:** Evaluates Coffin-Manson strain-life predictions ($\epsilon'_f = 0.18, c = -0.62$), verifying fatigue endurance bounds across cyclic plastic strain thresholds.


2. **Dual-ISA Bitwise Regression (`crates/shbt-fabrication-hil/tests/bitwise_regression.rs`):**
* Asserts exact bitwise kernel digest agreement (`0x03de00d3acded967`) across x86-64 (AVX-512) and ARM64 (NEON) architectures to prevent microarchitectural float discrepancies.


3. **Multiphysics Integration Pipeline (`tests/test_full_multiphysics_pipeline.py` & `tests/verification_tests.rs`):**
* Validates full end-to-end integration across Floquet dielectric matrix assembly, RCWA optical enhancement, structural FEA deformation, and HIL telemetry streaming.
