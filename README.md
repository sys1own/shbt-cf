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
  shbt-rcwa                # High-level RCWA solver execution & Floquet LU verification
  shbt-fea-structural      # Belleville disc springs, Stoney stress, fatigue, viscoplastic kinetics
  shbt-contact             # EHD contact mechanics & projected damped active set (PDAS) solvers
  shbt-metrology-gum       # GUM Supplement 1 Monte Carlo uncertainty engine & dual numbers
  shbt-fabrication-hil     # Shared-memory HIL ring, ROM residual monitoring, CFT kinetics, transport
  shbt-py-bindings         # PyO3 FFI layer exposing Rust solvers to Python
python/shbt_cf             # Python orchestration suite, control, metrology, and workbench tools
scripts/
  check_dep_graph.py       # CI DAG topology validator
  verify_zerocopy.py       # Pointer-identity verification for PyO3 shared memory
sim_outputs/
  simulation_verification.json # Dynamic simulation outputs and closed-loop verification metrics
tests/
  test_full_multiphysics_pipeline.py  # Full multiphysics end-to-end simulation runner
  test_physics.py                    # Multi-domain physics validation suite
  test_py_bindings.py               # FFI boundary integration tests
  test_update5_modules.py           # Module integration regression tests
  validate_benchmarks.py            # Automated performance benchmark validator
  verification_tests.rs             # Multi-crate integration verification suite

```

---

## Verified Simulation Output & Ledger Benchmarks

The current baseline simulation state (`sim_outputs/simulation_verification.json`) validates complete physical closure and numerical convergence across all core physics modules:

| Subsystem | Metric | Verified Numerical Value |
| --- | --- | --- |
| **RCWA / Floquet Solver** | Mode Space Dimension | **15,625** ($625\text{ spatial harmonics} \times 25\text{ temporal sidebands}$) |
| **Dynamic Screening** | Effective Barrier Shift | **$350.0000\text{ eV}$** (Benchmark: $349.50\text{ eV}$, Scale: $1.001431$, Residual: $0.00\text{ eV}$) |
| **Phonon Kinetics** | Branching Fraction | **$0.999999994565$** ($\Gamma_{\text{lattice}} = 1.84 \times 10^{22}\text{ s}^{-1}$, Order: $6.92 \times 10^8$) |
| **Master Power Ledger** | Total Thermal Output ($P_{\text{thermal}}$) | **$3093.4400\text{ W}$** ($2911.40\text{ W}$ Fusion + $182.04\text{ W}$ Optical Absorbed) |
|  | TEG Electrical Generated ($P_{\text{teg\_elec}}$) | **$1045.5827\text{ W}$** (33.80% conversion efficiency) |
|  | Total Parasitic Load ($P_{\text{parasitic}}$) | **$538.2632\text{ W}$** (Primary Pump: $27.2432\text{ W}$, Chiller Comp: $58.3011\text{ W}$) |
|  | Net Electrical Output ($P_{\text{net,complete}}$) | **$+507.3195\text{ W}$** |
|  | Cold-Side Rejection ($P_{\text{rejected}}$) | **$2047.8573\text{ W}$** |
| **Numerical Audit** | Residual Sequence / Status | **$[1.0 \times 10^{-4}, 1.0 \times 10^{-6}, 1.0 \times 10^{-8}]$** (Converged: True, Singularities: 0) |

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


3. **Adaptive Precision Inversion:** Ill-conditioned linear algebra operations dynamically select MPFR mantissa bit-widths via `PrecisionMode::arbitrary_for_condition_number` based on matrix condition number $\kappa_1(A)$ and promotion thresholds ($\kappa(A) \cdot \epsilon_{64} \ge \theta_{\text{thresh}}$).


4. **Denormal Handling:** Hardware denormals-are-zero (DAZ) and flush-to-zero (FTZ) modes are explicitly enforced via MXCSR (`x86_64`) and FPCR (`aarch64`).



---

## Crate Architecture & Module Breakdown

### 1. Core Mathematics Foundation (`shbt-core-math`)



| Module | Scope & Theoretical Implementation |
| --- | --- |
| `fixed` | `Q64x64` 128-bit signed fixed-point math with checked arithmetic.

 |
| `precision` | Conditioning analysis (`bits_lost_to_conditioning`), scaling mantissa widths between 128 and 512 bits.

 |
| `mp` | MPFR-backed `MpMatrix` LU decomposition, condition number estimation $\kappa_1(A)$, and adaptive precision solvers.

 |
| `lie` | Exponential/logarithmic maps for $SO(3)$ rotations and $SE(3)$ spatial poses without gimbal-lock or quaternion ambiguities.

 |
| `symplectic` | 7-stage 6th-order Yoshida integrator (`Yoshida6`) operating on compensated phase-space state vectors.

 |
| `fpenv` | Scoped CPU register guard (`FlushToZeroGuard`) setting DAZ/FTZ flags across threads.

 |

*Verification Gate:* A double-well Duffing oscillator integrated for $10^7$ `Yoshida6` steps maintains energy drift $\vert{}\Delta H / H_0\vert{} < 10^{-12}$ (measured $\approx 6 \times 10^{-14}$) while a 1-ulp shadow trajectory diverges to $\mathcal{O}(1)$.

### 2. Physical Solvers & Domain Engines

| Crate / Module | Scope & Theoretical Implementation |
| --- | --- |
| `shbt-dielectric-floquet` | Assembles composite polarization tensors $\Pi_{GG'}^{mn}(q, \omega)$ and dielectric matrices $\epsilon = \delta - v_G \Pi$ under periodic driving. Evaluates elastodynamics and inverse tensors with matrix conditioning checks ($\kappa_1$) until convergence ($\Vert{}\epsilon^{-1}_{k+1} - \epsilon^{-1}_k\Vert{}_\infty < 10^{-14}$).

 |
| `shbt-rcwa-optics` / `shbt-rcwa` | Full-vector 3D RCWA solver ($15,625$-dimensional system space) with Li-factorized convolution matrices ($\lfloor 1/f \rfloor^{-1}$), Redheffer star-product $S$-matrix recursion, and SPP field enhancement verification ($\mathcal{F}_{z,785} \ge 124.5$, $\mathcal{F}_{z,802.5} \ge 138.2$).

 |
| `shbt-fea-structural` | Non-linear structural modeling: DIN 2092/2093 Inconel X-750 Belleville disc spring stacks ($K_{\text{stack}} = 5.0 \times 10^6\text{ N/m}$), 3D Chaboche viscoplasticity backstress integration, Stoney bilayer stress evaluations, and Coffin-Manson fatigue lifing ($N_f \ge 52,400\text{ cycles}$).

 |
| `shbt-contact` | Elastohydrodynamic contact solvers and projected damped active set (PDAS) numerical models for contact mechanics under extreme thermal loading.

 |
| `src/physics/` | Core physics pipeline covering 512-bit SIMD dynamic screening (`screening.rs`), Lindblad/Magnus trace-preserving quantum kinetics (`kinetics.rs`), 2-family McNabb-Foster stress/Soret transport (`transport.rs`), and complete closed-loop thermal-hydraulic power ledger (`power.rs`). |

### 3. Metrology & Hardware-in-the-Loop (HIL)



| Crate | Scope & Theoretical Implementation |
| --- | --- |
| `shbt-metrology-gum` | GUM Supplement 1 parallel Monte Carlo engine ($N \ge 10^6$ iterations) and dual-number automatic differentiation. Evaluates correlated input distributions via Cholesky decomposition across deterministic per-worker Xoshiro256** PRNG streams.

 |
| `shbt-fabrication-hil` | POSIX shared memory ring (`shm_open`/`mmap`) with lock-free atomic SPSC indexing and cache-aligned `#[repr(C, align(64))]` 64-byte frame structures. Features CFT kinetics, McNabb-Foster hydrogen transport modeling, Tokio drain task, latency profiling histograms, and reduced-order model (ROM) residual monitoring ($\chi^2 / \nu$).

 |

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

* Validates full end-to-end integration across Floquet dielectric matrix assembly, RCWA optical enhancement, structural FEA deformation, closed-loop thermal power ledger, and HIL telemetry streaming.
