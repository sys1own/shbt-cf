# SHBT Cold Fusion Reactor: Precision Simulation & Engineering Workbench

`shbt-cf` is the multi-domain physics simulation engine, hardware-in-the-loop (HIL) telemetry architecture, and numerical workbench for the Static Holographic Boundary Theory (SHBT) reactor specification. Primary physics derivations, structural bounds, and electrodynamic models are detailed in `cf.pdf`.

---

## Workspace Layout

The repository is structured as a Rust Cargo workspace containing 9 crates paired with a native Python orchestration wrapper:

```text
Cargo.toml                   # Workspace root (9 crates with strict DAG topology)
.cargo/config.toml           # Target CPU optimization, opt-level 3, strict IEEE 754 rules
crates/
  shbt-core-math             # Q64.64 fixed-point, MPFR, Lie algebra SO(3)/SE(3), Yoshida-6, DAZ/FTZ
  shbt-dielectric-floquet    # Floquet-Adler-Wiser dielectric matrix inversions & elastodynamics
  shbt-rcwa-optics           # 2D/3D vector RCWA, S-matrix recursion
  shbt-rcwa                  # High-level RCWA solver execution & Floquet LU verification
  shbt-fea-structural        # Belleville disc springs, Stoney stress, fatigue, viscoplastic kinetics
  shbt-contact               # EHD contact mechanics & projected damped active set (PDAS) solvers
  shbt-metrology-gum         # GUM Supplement 1 Monte Carlo uncertainty engine & dual numbers
  shbt-fabrication-hil       # Shared-memory HIL ring, ROM residual monitoring, CFT kinetics, transport
  shbt-py-bindings           # PyO3 FFI layer exposing Rust solvers to Python
python/shbt_cf               # Python orchestration suite, control, metrology, and workbench tools
scripts/
  check_dep_graph.py         # CI DAG topology validator
  verify_zerocopy.py         # Pointer-identity verification for PyO3 shared memory
sim_outputs/
  simulation_verification.json # Dynamic simulation outputs and closed-loop verification metrics
tests/
  test_full_multiphysics_pipeline.py  # Full multiphysics end-to-end simulation runner
  test_physics.py                     # Multi-domain physics validation suite
  test_py_bindings.py                 # FFI boundary integration tests
  test_update5_modules.py             # Module integration regression tests
  validate_benchmarks.py              # Automated performance benchmark validator
  verification_tests.rs               # Multi-crate integration verification suite

```

---

## SHBT Ecosystem Architecture & Repository Crosswalk

`shbt-cf` functions as the canonical cold fusion and thermal-hydraulic authority across the nine-repository Static Holographic Boundary Theory (SHBT) digital twin ecosystem. It defines the baseline energy generation models, non-equilibrium dynamic screening dynamics, dual-stage thermoelectric generation (TEG) recovery routines, and closed-loop two-phase thermal rejection circuits governing both stationary fusion power grids and deep-space instrumentation platforms.

### Canonical 9-Pillar Ecosystem Topology

```markdown
| Layer | Repository | Physical Domain & Primary Scope | Architectural Interface Contract |
| :--- | :--- | :--- | :--- |
| **Foundational** | [`shbt-precision`](https://github.com/sys1own/shbt-precision) | Computational Math, Cosmology & CFT Core | 512-bit MPFR, WZW $(26, 8, 312)$, $\Delta_{\text{fr}} \equiv 0$ |
| **Grid Power** | [`shbt-power`](https://github.com/sys1own/shbt-power) | Master Fusion Twin ($8{,}750\text{ MW}_{\text{th}}$ / $7{,}832\text{ MW}_{\text{e}}$) | 70-gate baseline audit, multi-channel DEC, $S = 100/1089$ |
| **Aux Power** | [`shbt-cf`](https://github.com/sys1own/shbt-cf) | Solid-State LANR Array ($999.054\text{ kW}_{\text{net}}$) | Dual-stage TEG, $906\text{ kW}$ Landauer baseline balance |
| **Runtime** | [`shbt-qc`](https://github.com/sys1own/shbt-qc) | Photonic Quantum Processor & Microkernel | Freestanding C11 `shbt-os`, MMIO `0x70000000`, SECDED ECC |
| **Vehicle** | [`shbt-ghost`](https://github.com/sys1own/shbt-ghost) | Reactionless Traction & Metric Gravity | Sub-$2.5\text{ ns}$ PCSS crowbars, 94.20% SiC recovery |
| **Vehicle** | [`shbt-recon`](https://github.com/sys1own/shbt-recon) | Macroscopic Modular State Translocator | $V_{\text{macro}}$ Stinespring dilation, 128-byte C-ABI DMA |
| **Vehicle** | [`shbt-sglt`](https://github.com/sys1own/shbt-sglt) | Deep-Space Synthetic Lensing Array (SE-L2) | TMSV metrology ($r=2.50$), 5th-order minimum-jerk kinematics |
| **Vehicle** | [`shbt-warp`](https://github.com/sys1own/shbt-warp) | Holographic Warp Drive & Relativistic Engine | ADM 3+1 foliation ($\alpha=1.0$), $500\text{ TJ}$ $^{178\text{m2}}\text{Hf}$ graser |
| **Apex Bench** | [`shbt-exotic`](https://github.com/sys1own/shbt-exotic) | Unified Multi-Protocol Co-Simulation Twin | Full 6-protocol cross-coupling, Ford-Roman QI auditing |
```

```text
                              [shbt-precision]
                       Computational Math & Cosmology
                       (512-bit MPFR / WZW Characters)
                                     │
    ┌────────────────────────────────┼───────────────────────────────┐
    ▼                                ▼                               ▼
 [shbt-power]                     [shbt-cf]                       [shbt-qc]
Commercial Fusion Grid          1,800-Module LANR Array         Bare-Metal Microkernel &
(8,750 MW p-11B Twin)           & Thermal-Hydraulics            Photonic Quantum Bus
    │                                │                               │
    └────────────────────────┬───────┴───────────────────────────────┘
                             ▼
┌────────────────────────────────────────────────────────────────┐
│                  SPECIALIZED VEHICLE TWINS                     │
│  • shbt-ghost : Reactionless Propulsion & Local Gravity Wells  │
│  • shbt-recon : Macroscopic State Translocation Gateway        │
│  • shbt-sglt  : Synthetic Gravitational Lensing Telescope      │
│  • shbt-warp  : Holographic Warp Metric & 3+1D Flight Twin     │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                               shbt-exotic                                │
│        MULTI-PROTOCOL SPACETIME ENGINEERING CO-SIMULATION BENCH          │
│  • Cross-Protocol Field Coupling (Warp + Stasis + Translocation + Wells) │
│  • Global Energy Condition & Ford-Roman Quantum Inequality Auditing      │
│  • Dynamic 5-Stage Multi-Technology Flight Director                      │
└──────────────────────────────────────────────────────────────────────────┘
```

#### Standardized 9-Pillar Ecosystem Crosswalk Table

| Repository | Domain Role & Platform Scope | Shared Invariants & Interface Contracts |
| :--- | :--- | :--- |
| [`shbt-precision`](https://github.com/sys1own/shbt-precision) | Computational Math & Cosmological Foundation Core | 512-bit MPFR numerics, canonical WZW (26, 8, 312), Δ<sub>fr</sub> ≡ 0, Landauer debt P<sub>debt</sub> = 906.00 kW. |
| [`shbt-power`](https://github.com/sys1own/shbt-power) | Commercial p-¹¹B Aneutronic Fusion Power Plant Twin | 8,750 MW fusion / 7,832.903 MW net export, 70-gate audit, closed-loop thermal ledger, 128-byte SHBT-MMIO-POWER. |
| [`shbt-cf`](https://github.com/sys1own/shbt-cf) | LANR Cold Fusion Reactor Workbench & Thermal-Hydraulics | 1,800-module LANR starter grid (999.054 kW net DC), dual-stage CoSb<sub>3</sub>/ZrNiSn TEG, Kapitza resistance ΔT<sub>K</sub> = 3.546 K. |
| [`shbt-qc`](https://github.com/sys1own/shbt-qc) | Photonic Quantum Computer Twin & C11 Microkernel | Bare-metal C11 shbt-os microkernel, base 56-byte SHBT-MMIO-1 at 0x70000000, SECDED Hamming(72,64) ECC, AVX-512 interlocks. |
| [`shbt-ghost`](https://github.com/sys1own/shbt-ghost) | Ghost Seed Reactionless Propulsion & Metric Stabilization | Sub-2.5 ns PCSS crowbars, 94.20% SiC inductive recovery, 3+1 CCZ4/ADM stabilization (β<sup>i</sup> → 0, \|det(g)+1\| ≤ 10<sup>-12</sup>). |
| [`shbt-recon`](https://github.com/sys1own/shbt-recon) | Macroscopic State Translocation & Gateway Twin | Macroscopic Stinespring dilation (V<sub>unified</sub><sup>macro</sup>), dark ledger η<sub>D</sub> = 23/33, 128-byte C-ABI DMA streaming, 78-gate audit. |
| [`shbt-sglt`](https://github.com/sys1own/shbt-sglt) | Synthetic Gravitational Lensing Telescope (SE-L2) Stack | 2PN relativistic beam optics, TMSV heterodyne metrology (r = 2.50, 21.715 dB), 5th-order minimum-jerk flight profiles. |
| [`shbt-exotic`](https://github.com/sys1own/shbt-exotic) | Multi-Protocol Spacetime Engineering Co-Simulation | Cross-protocol metric coupling (all 6 phenomena), Ford-Roman QI dark-ledger auditing, Heegaard-Floer boundary relabeling. |
| [`shbt-warp`](https://github.com/sys1own/shbt-warp) | Holographic Warp Drive Digital Twin & 3+1D ADM Engine | Alcubierre metric foliation (α = 1.0, γ<sub>ij</sub> = δ<sub>ij</sub>), 500 TJ ¹⁷⁸ᵐ²Hf graser battery (109 TW burst), 128-gate audit, 8 Z3 proofs. |

#### `shbt-cf` Integration Surface

* **Upstream Model & Ledger Integration:** `shbt-cf` exports the 1,800-module Lattice-Assisted Nuclear Reaction (LANR) core specification (555.03 W net DC per cell, 999.054 kW net array) as continuous housekeeping power to [`sys1own/shbt-power`](https://github.com/sys1own/shbt-power) for power-plant auxiliary cold-start bootstrapping, to [`sys1own/shbt-warp`](https://github.com/sys1own/shbt-warp) to service its boundary-emitter Landauer debt, and to [`sys1own/shbt-recon`](https://github.com/sys1own/shbt-recon) as the continuous baseload rail behind its pulsed ¹⁷⁸ᵐ²Hf graser bursts. It additionally exports the LANR ledger to [`sys1own/shbt-sglt`](https://github.com/sys1own/shbt-sglt) to balance the 906.000 kW non-sheddable entropy debt of the deep-space synthetic gravitational lensing telescope array.
* **Warp Drive Energy & Cryogenic Transfer:** The 1,800-module LANR starter core (555.03 W net DC/cell, 999.054 kW net array) is additionally exported to [`sys1own/shbt-warp`](https://github.com/sys1own/shbt-warp) to balance the continuous 906.000 kW Landauer entropy debt of the boundary emitter array, maintaining a +93.054 kW raw electrical margin and +33.104 kW net continuous system surplus. `shbt-cf` further provides two-phase helium flow boiling and Kapitza thermal resistance models (ΔT<sub>K</sub> = 3.546 K) ensuring ΔT ≥ 11.790 K cryogenic headroom below superconducting quench limits.
* **Kinetics & Fast Recovery Transfers:** The McNabb–Foster hydrogen transport and isotope trapping solvers (`McnabbFosterSolver`) developed in `shbt-cf` are imported by [`sys1own/shbt-ghost`](https://github.com/sys1own/shbt-ghost) (`ghost-lanr-interface`) for long-term fuel retention, while `shbt-cf` integrates the sub-2.5 ns Photoconductive Semiconductor Switch (PCSS) optical trigger logic and 94.20% SiC inductive recovery shunts engineered in [`sys1own/shbt-ghost`](https://github.com/sys1own/shbt-ghost) to harvest stored magnetic coil energy.
* **Mathematical Foundations & Microkernel Contracts:** Foundational MPFR arbitrary-precision arithmetic, condition-number tracking, and symplectic integrators (`Yoshida6`) are anchored in the canonical WZW affine branch models from [`sys1own/shbt-precision`](https://github.com/sys1own/shbt-precision). Low-level register definitions and SECDED ECC standards derive from the bare-metal C11 `shbt-os` runtime in [`sys1own/shbt-qc`](https://github.com/sys1own/shbt-qc), while boundary Conformal Field Theory state-vector formulations and the invariant rational dark ledger capacity partitioning (η<sub>D</sub> = 23/33, η<sub>A</sub> = 10/33) originate from [`sys1own/shbt-exotic`](https://github.com/sys1own/shbt-exotic). Finally, zero-copy lock-free POSIX SPSC circular ring buffers and 128-byte dual-cacheline C-ABI mapping conventions are shared with [`sys1own/shbt-recon`](https://github.com/sys1own/shbt-recon).

| Repository | Ecosystem Domain Role | Bidirectional Technology Transfer & Direct Integration with `shbt-cf` |
| --- | --- | --- |
| [`sys1own/shbt-cf`](https://github.com/sys1own/shbt-cf) | Cold Fusion Authority & Engineering Workbench | **Canonical repository.** Provides the primary 1,800-module LANR reactor core (555.03 W net DC/cell, 999.054 kW array), dynamic Floquet-Adler-Wiser dielectric screening (U<sub>eff</sub> = 352.48 eV), dual-stage TEG enthalpy models, and 50-gate numerical verification harness (`GATE-01`–`GATE-50`). |
| [`sys1own/shbt-power`](https://github.com/sys1own/shbt-power) | Master Commercial Fusion Digital Twin | Directly integrates `shbt-cf`'s 1,800-module LANR starter array into `crates/shbt-power-grid` to drive a 7.51-minute cold-start bootstrap sequence, and imports `shbt-cf`'s dual-stage (CoSb <sub>3</sub> / ZrNiSn) TEG enthalpy recovery (447.903 MW) and 3D Eulerian-Eulerian helium coolant models. |
| [`sys1own/shbt-ghost`](https://github.com/sys1own/shbt-ghost) | Fast Optical Interlocks & Metric Stabilization | Supplies sub-2.5 ns PCSS crowbars, 94.20% SiC inductive recovery shunts recovering 16.62 mJ/cycle for `shbt-cf`'s magnetic coils, and ADM 3+1 metric stabilization (β<sup>i</sup> → 0, |det(g)+1 |≤ 10<sup>-12</sup>); imports `shbt-cf`'s McNabb–Foster deuterium kinetics (`ghost-lanr-interface`) for mobile-fuel retention. |
| [`sys1own/shbt-exotic`](https://github.com/sys1own/shbt-exotic) | Boundary CFT Foundations & Dark Ledger | Provides the Boundary Conformal Field Theory state-vector formulations, Heegaard–Floer symplectic boundary relabeling (T<sup>∂</sup><sub>ij</sub>), and invariant rational dark ledger capacity partitioning (η<sub>D</sub> = 23/33, η<sub>A</sub> = 10/33) underlying `shbt-cf`'s coherent lattice branching fraction (B<sub>lat</sub> ≥ 0.999999994). |
| [`sys1own/shbt-precision`](https://github.com/sys1own/shbt-precision) | Computational Mathematics Core & Precision Audits | Supplies the 512-bit arbitrary-precision hybrid numeric framework (`rug`/MPFR), canonical WZW affine branch (26, 8, 312) arithmetic, and zero-allocation execution primitives supporting `shbt-core-math`'s MPFR-backed solvers, conditioning estimators (κ<sub>1</sub>), and symplectic Yoshida-6 integrators. |
| [`sys1own/shbt-sglt`](https://github.com/sys1own/shbt-sglt) | SGLT Telescope Platform & Relativistic Optics | Imports `shbt-cf`'s 1,800-module LANR power ledger (`crates/sglt-lanr-power`) for its deep-space satellite bus, satisfying its 906.000 kW non-sheddable entropy debt with 999.054 kW of generated power (+93.054 kW margin). |
| [`sys1own/shbt-warp`](https://github.com/sys1own/shbt-warp) | Holographic Warp Drive & Spacetime Engine | Imports `shbt-cf`'s 1,800-module LANR power ledger (999.054 kW net DC) to balance its 906.000 kW boundary-emitter Landauer debt (+93.054 kW raw margin, +33.104 kW net surplus), and its two-phase helium/Kapitza cryo models (ΔT<sub>K</sub> = 3.546 K) certifying ≥ 11.790 K junction headroom. |
| [`sys1own/shbt-qc`](https://github.com/sys1own/shbt-qc) | Bare-Metal Microkernel Runtime (`shbt-os`) & HIL | Supplies the freestanding C11 `shbt-os` runtime, SECDED Hamming(72,64) ECC scrubbing, and normative base 56-byte `SHBT-MMIO-1` register layout standard at `0x70000000`, which `shbt-cf` extends into its 128-byte dual-cacheline aligned `shbt_mmio_control_t` HIL register structure. |
| [`sys1own/shbt-recon`](https://github.com/sys1own/shbt-recon) | Macroscopic State Tracking & Telemetry Rings | Supplies macroscopic Stinespring dilation (V<sub>unified</sub><sup>macro</sup> over N ~ 10<sup>20</sup> particles), 128-byte dual-cacheline C-ABI mapping standards, and high-throughput lock-free POSIX SPSC circular shared-memory telemetry rings implemented in `shbt-fabrication-hil`. |

---

## Verified Simulation Output & Ledger Benchmarks

The current baseline simulation state (`sim_outputs/simulation_verification.json`) validates complete physical closure and numerical convergence across all core physics modules, and audits the full **50-gate system verification matrix** (`GATE-01` – `GATE-50`, all PASS) defined by the cf3 engineering specification.

### Master Power Ledger (cf2)

$$P_{\text{net}} = P_{\text{TEG}} - P_{\text{drive,net}} - P_{\text{aux}} = 1045.58 - 368.45 - 122.10 = +555.03\text{ W} > +550.00\text{ W}$$

| Subsystem | Metric | Verified Numerical Value |
| --- | --- | --- |
| **RCWA / Floquet Solver** | Mode Space Dimension | **15,625** (625 spatial harmonics × 25 temporal sidebands) |
| Dynamic Screening | Effective Screening Potential U<sub>eff</sub> | 352.48 eV (≥ 350.00 eV bound; recalculated at 100 Hz from Δ x(r, t)) |
|  | Coherent Lattice Branching Fraction B<sub>lat</sub> | **0.999999996** (≥ 0.999999994 bound) |
|  | Nominal Deuterium Loading x<sub>0</sub> | **0.9132** stoichiometric (dynamic range 0.8850 ≤ x ≤ 0.9450) |
| **3D Thermal-Hydraulics** | Peak Surge Load (P<sub>thermal</sub>) | **3093.44 W** continuous peak |
|  | Hot-Side / Coolant Return Temp | **618.42 K / 301.88 K** (limits 623.15 K / 303.15 K) |
|  | Peak Void Fraction α<sub>v</sub> | **0.162** (limit ≤ 0.185) |
|  | Core Pressure Drop / φ<sub>lo</sub><sup>2</sup> | **42.8 kPa / 1.34** (limits 50.0 kPa / 1.50) |
|  | CHF Operating Margin | **q<sup>′′</sup>/q<sup>′′</sup><sub>CHF</sub> = 0.412** (2.42× margin, limit ≤ 0.50) |
| **Magnetic Excitation** | RF Drive / Coil | **f<sub>rf</sub> = 68.5 kHz, thin-film MgB <sub>2</sub>** (T<sub>c</sub> = 39.0 K, B<sub>peak</sub> = 1.42 T) |
|  | Stored Inductive Energy | **E<sub>m</sub> = 16.62 mJ/cycle** (P<sub>reactive</sub> = 1138.47 VAR) |
|  | SiC Crowbar Recovery | **η<sub>SiC</sub> = 94.20%** (≥ 92.00%) |
|  | Parasitic Drive Power | **368.45 W** (recovered 88.45 W to 400 V bus; limit < 380.00 W) |
|  | Gross TEG / Auxiliary | **1045.58 W / 122.10 W** |
|  | **Net Electrical Output P<sub>net</sub>** | **+555.03 W** (> +550.00 W) |
| **GST Self-Healing Optics** | Buffer / Pulse | **Ge <sub>2</sub> Sb <sub>2</sub> Te <sub>5</sub>** layer, F<sub>pulse</sub> = 27.9 mJ/cm <sup>2</sup>, t<sub>pulse</sub> = 50 ns |
|  | Post-Healing Roughness / A / R<sub>grating</sub> | **0.62 nm / 98.74% / 99.94%** (30-year service life) |
| **C-ABI MMIO Map** | `shbt_mmio_control_t` | **128-byte dual-cacheline record (64-byte aligned)**, `matrix_dim = 15625` at `0x0040`, GPUDirect pointer at `0x0048` |
| **Numerical Audit** | Residual Sequence / Status | **[1.0 × 10<sup>-4</sup>, 1.0 × 10<sup>-6</sup>, 1.0 × 10<sup>-8</sup>]** (Converged: True, Singularities: 0, Gates: 50/50 PASS) |

### Physics Upgrades

* **3D Eulerian–Eulerian Thermal-Hydraulics** (`src/physics/thermal_hydraulics.rs`): two-phase liquid/vapor conservation with Ishii–Zuber drag, Tomiyama lift, Antal–Frank wall lubrication, Burns turbulent dispersion, and RPI wall heat-flux partitioning with Hibiki–Ishii site density across 64 parallel OFHC-Cu micro-channels (D<sub>h</sub> = 250 μm):

$$q''_{\mathrm{tot}} = q''_{1\varphi} + q''_{q} + q''_{e}$$

* **Dynamic Non-Equilibrium Deuterium Screening** (`src/physics/screening.rs`): Fick–Soret transport with D<sub>0</sub> = 2.85×10<sup>-7</sup> m<sup>2</sup>/s, E<sub>a</sub> = 0.224 eV, Q<sup>*</sup> = 0.048 eV, driving 100 Hz recalculation of the 15,625 × 15,625 Floquet–Adler–Wiser dielectric matrix:

$$\partial_t x = \nabla \cdot \left[ D_D(T,x)\left(\nabla x + \frac{x(1-x)\,Q^{*}}{k_B T^{2}}\nabla T\right) \right] + \dot{S}_{\mathrm{phase}}$$

$$\varepsilon_{G,G'} = \delta_{G,G'} - v(q+G)\,\chi^{0}_{G,G'}(x)$$

* **High-T<sub>c</sub> Superconducting Coils + SiC Crowbar** (`src/physics/power.rs`): thin-film MgB<sub>2</sub> micro-coils (T<sub>c</sub> = 39.0 K) at 68.5 kHz with Bean critical-state and flux-flow losses (P<sub>coil</sub> = 12.35 W); the SiC crowbar harvests the 16.62 mJ/cycle inductive energy at 94.20% efficiency, cutting parasitic drive from 422.22 W to 368.45 W.
* **GST Optical Self-Healing** (`src/physics/optics_healing.rs`): chalcogenide Ge<sub>2</sub>Sb<sub>2</sub>Te<sub>5</sub> buffer in the Pd<sub>0.9132</sub>Ir<sub>0.0868</sub> grating stack; 50 ns electro-thermal pulses (27.9 mJ/cm<sup>2</sup>) trigger melt-quench recrystallization restoring R<sub>a</sub> < 0.8 nm, A ≥ 98.40%, R<sub>grating</sub> > 99.9%.
* **C-ABI MMIO Map** (`crates/shbt-fabrication-hil/include/shbt_floquet_mmio.h`): zero-copy 64-byte aligned `shbt_mmio_control_t` register map with 100 Hz HIL frame kernel (`shbt_floquet_mmio_execute_frame`).
* **Active-Alloy McNabb–Foster Kinetics** (`src/physics/transport.rs`): two-family diffusion and trapping model for Pd<sub>0.9132</sub>Ir<sub>0.0868</sub>D<sub>x</sub> at x = 0.9132 with Soret thermophoresis and partial-molar-volume stress drift:

$$\mathbf{J}_D = -D_D(T,x)\,\nabla C_L + \frac{V_H^{\ast}}{RT} D_D\, C_L \nabla \sigma_h + \frac{Q^{\ast}}{RT^{2}} D_D\, C_L \nabla T$$

  with D<sub>0</sub> = 2.85×10<sup>-7</sup> m<sup>2</sup>/s, E<sub>a</sub> = 0.224 eV, Q<sup>*</sup> = 0.048 eV, V<sub>H</sub><sup>*</sup> = 1.70×10<sup>-6</sup> m<sup>3</sup>/mol, dislocation traps N<sub>1</sub> = 1.50×10<sup>24</sup> m<sup>-3</sup> (E<sub>t,1</sub> = 0.23 eV) and grain-boundary traps N<sub>2</sub> = 5.00×10<sup>23</sup> m<sup>-3</sup> (E<sub>t,2</sub> = 0.15 eV).

* **3D Chaboche Thermoviscoplasticity & Joint FEA** (`src/physics/mechanics.rs`, `crates/shbt-fea-structural`): two-term nonlinear kinematic hardening (C<sub>1</sub> = 45.2 GPa, γ<sub>1</sub> = 410, C<sub>2</sub> = 8.5 GPa, γ<sub>2</sub> = 62 at 298.15 K, linearly interpolated to the 623.15 K set) plus Voce isotropic hardening (R<sub>∞</sub> = 85 MPa, b = 12.46); VCCT energy release rates on the 3.5 μm Ni–Cu–Sn TLP bondline (G<sub>IC</sub> = 25, G<sub>IIC</sub> = 65, G<sub>IIIC</sub> = 60 J/m<sup>2</sup>, delamination factor f = 0.484); Morrow strain-life fatigue N<sub>f</sub> ≈ 65,474 ≥ 52,400 cycles.
* **Plant Stability Maps** (`src/physics/thermal_hydraulics.rs`): system pump head H<sub>pump</sub>(Q) = (65.0 − 1.25Q − 0.62Q<sup>2</sup>) kPa coupled to the two-phase core; Ledinegg excursive margin and Ishii–Zuber DWO ratio N<sub>pch,crit</sub> = 1.45·N<sub>sub</sub> + 2.5 evaluated across startup, nominal (ΔP = 42.8 kPa @ 4.85 L/min), surge, and low-flow regimes with Q < 3.95 L/min interlock.
* **3D Radiation Transport** (`src/physics/radiation.rs`): 2.45 MeV DD neutrons plus secondary capture gammas (478 keV from <sup>10</sup>B(n,α)<sup>7</sup>Li<sup>*</sup>, 2.223 MeV from H(n,γ)D) through the H<sub>2</sub>O / 316L / 5 wt% B-PE / Pb shielding stack; accessible surface dose (0.38 ± 0.012) μSv/h < 0.50 μSv/h at Q<sub>N</sub> ≤ 10<sup>6</sup> n/s.
* **GUM Covariance Metrology** (`src/physics/metrology.rs`, `crates/shbt-metrology-gum`): ISO/IEC 98-3 multi-variable input covariance Σ<sub>X</sub> (8×8, correlations r<sub>T<sub>in</sub>,T<sub>out</sub></sub> = 0.85, r<sub>Q<sub>t</sub>,Q<sub>T</sub></sub> = 0.92, r<sub>Q,UA</sub> = 0.30) with analytic/dual-number Jacobian J, Σ<sub>Y</sub> = JΣ<sub>X</sub>J<sup>T</sup>, N = 10<sup>6</sup> MC cross-check; propagated bounds u(P<sub>thermal</sub>) ≈ 11.8 W, u(P<sub><sup>4</sup>He</sub>) ≈ 2.5×10<sup>-9</sup> mbar, u(FOM) ≈ 0.018 securing P<sub>net</sub> = +555.03 W at 3σ.
* **128-byte C-ABI MMIO Record** (`crates/shbt-fabrication-hil/include/shbt_floquet_mmio.h`): `shbt_mmio_control_t` widened to a dual-cacheline packed record — `plastic_strain_max` @ 0x0028, `dose_surface_usv` @ 0x0030, `gpu_frame_count` @ 0x0038, `matrix_dim` = 15625 @ 0x0040, `d_floquet_ptr` @ 0x0048, `pad[16]` @ 0x0050 — 64-byte aligned.

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
maturin build -m crates/shbt-py-bindings/Cargo.toml --release
maturin build -m crates/shbt-fabrication-hil/Cargo.toml --features python --release
pip install --user --no-deps target/wheels/shbt_py_bindings-*.whl
pip install --user --no-deps target/wheels/shbt_fabrication_hil-*.whl
python3 scripts/verify_zerocopy.py

```

---

## Determinism & Numerical Safety Policy

To prevent non-deterministic floating-point drifting across microarchitectures and compiler toolchains:

1. **Strict IEEE 754 Compliance:** Fast-math flag sets are explicitly disabled across all workspace profiles (`-C llvm-args=-enable-no-nans-fp-math=false -C llvm-args=-enable-no-signed-zeros-fp-math=false`).
2. **Fixed-Point Accumulation:** Global energy and state accumulators utilize `shbt_core_math::fixed::Q64x64` 128-bit signed fixed-point integers (2<sup>-64</sup> fractional resolution).
3. **Adaptive Precision Inversion:** Ill-conditioned linear algebra operations dynamically select MPFR mantissa bit-widths via `PrecisionMode::arbitrary_for_condition_number` based on matrix condition number κ<sub>1</sub>(A) and promotion thresholds (κ(A) · ε<sub>64</sub> ≥ θ<sub>thresh</sub>).
4. **Denormal Handling:** Hardware denormals-are-zero (DAZ) and flush-to-zero (FTZ) modes are explicitly enforced via MXCSR (`x86_64`) and FPCR (`aarch64`).

---

## Crate Architecture & Module Breakdown

### 1. Core Mathematics Foundation (`shbt-core-math`)

| Module | Scope & Theoretical Implementation |
| --- | --- |
| `fixed` | `Q64x64` 128-bit signed fixed-point math with checked arithmetic. |
| `precision` | Conditioning analysis (`bits_lost_to_conditioning`), scaling mantissa widths between 128 and 512 bits. |
| `mp` | MPFR-backed `MpMatrix` LU decomposition, condition number estimation κ<sub>1</sub>(A), and adaptive precision solvers. |
| `lie` | Exponential/logarithmic maps for SO(3) rotations and SE(3) spatial poses without gimbal-lock or quaternion ambiguities. |
| `symplectic` | 7-stage 6th-order Yoshida integrator (`Yoshida6`) operating on compensated phase-space state vectors. |
| `fpenv` | Scoped CPU register guard (`FlushToZeroGuard`) setting DAZ/FTZ flags across threads. |

*Verification Gate:* A double-well Duffing oscillator integrated for 10<sup>7</sup> `Yoshida6` steps maintains energy drift |Δ H / H<sub>0</sub> |< 10<sup>-12</sup> (measured ≈ 6 × 10<sup>-14</sup>) while a 1-ulp shadow trajectory diverges to O(1).

### 2. Physical Solvers & Domain Engines

| Crate / Module | Scope & Theoretical Implementation |
| --- | --- |
| `shbt-dielectric-floquet` | Assembles composite polarization tensors Π<sub>GG<sup>′</sup></sub><sup>mn</sup>(q, ω) and dielectric matrices ε = I − v<sub>G</sub> Π under periodic driving. Evaluates elastodynamics and inverse tensors with matrix conditioning checks (κ<sub>1</sub>) until convergence (‖ε<sup>-1</sup><sub>k+1</sub> − ε<sup>-1</sup><sub>k</sub>‖<sub>∞</sub> < 10<sup>-10</sup>). |
| `shbt-rcwa-optics` / `shbt-rcwa` | Full-vector 3D RCWA solver (15,625-dimensional system space) with Li-factorized convolution matrices (⌊1/f⌋<sup>-1</sup>), Redheffer star-product S-matrix recursion, and SPP field enhancement verification (F<sub>z,785</sub> ≥ 124.5, F<sub>z,802.5</sub> ≥ 138.2). |
| `shbt-fea-structural` | Non-linear structural modeling: DIN 2092/2093 Inconel X-750 Belleville disc spring stacks (K<sub>stack</sub> = 5.0 × 10<sup>6</sup> N/m), 3D Chaboche viscoplasticity backstress integration, Stoney bilayer stress evaluations, and Coffin-Manson fatigue lifing (N<sub>f</sub> ≥ 52,400 cycles). |
| `shbt-contact` | Elastohydrodynamic contact solvers and projected damped active set (PDAS) numerical models for contact mechanics under extreme thermal loading. |
| `src/physics/` | Core physics pipeline covering 512-bit SIMD dynamic screening (`screening.rs`), Lindblad/Magnus trace-preserving quantum kinetics (`kinetics.rs`), 2-family McNabb-Foster stress/Soret transport (`transport.rs`), and complete closed-loop thermal-hydraulic power ledger (`power.rs`). |

### 3. Metrology & Hardware-in-the-Loop (HIL)

| Crate | Scope & Theoretical Implementation |
| --- | --- |
| `shbt-metrology-gum` | GUM Supplement 1 parallel Monte Carlo engine (N ≥ 10<sup>6</sup> iterations) and dual-number automatic differentiation. Evaluates correlated input distributions via Cholesky decomposition across deterministic per-worker Xoshiro256** PRNG streams. |
| `shbt-fabrication-hil` | POSIX shared memory ring (`shm_open`/`mmap`) with lock-free atomic SPSC indexing and cache-aligned `#[repr(C, align(64))]` 64-byte frame structures. Features CFT kinetics, McNabb-Foster hydrogen transport modeling, Tokio drain task, latency profiling histograms, and reduced-order model (ROM) residual monitoring (χ<sup>2</sup> / ν). |

*Benchmark Gate:* The 10 kHz telemetry channel pipeline processes over 61,000 frames across 3 synthetic streams with zero frame drops, zero mutex allocation, and sub-microsecond mean processing latency.

---

## Python Orchestration & FFI Layer

The native bindings layer (`shbt-py-bindings` and `shbt-fabrication-hil`) exports C-ABI functions and PyO3 native extensions:

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
* **Volume Mapping:** Verifies active domain mapping ratios (V<sub>domains</sub> / V<sub>metal</sub> = 8000) according to `cf.pdf` Equations 29, 35, and 123.
* **Calorimetric Uncertainty Budget:** Validates GUM calorimetry Monte Carlo propagation (P = ṁ c<sub>p</sub> Δ T + P<sub>env</sub>), confirming standard uncertainty bounds (u<sub>c</sub> ≈ 5.696 W < 5.70 W).
* **Fatigue Life Ceilings:** Evaluates Coffin-Manson strain-life predictions (ε'<sub>f</sub> = 0.18, c = -0.62), verifying fatigue endurance bounds across cyclic plastic strain thresholds.


2. **Dual-ISA Bitwise Regression (`crates/shbt-fabrication-hil/tests/bitwise_regression.rs`):**
* Asserts exact bitwise kernel digest agreement (`0x03de00d3acded967`) across x86-64 (AVX-512) and ARM64 (NEON) architectures to prevent microarchitectural float discrepancies.


3. **Multiphysics Integration Pipeline (`tests/test_full_multiphysics_pipeline.py` & `tests/verification_tests.rs`):**
* Validates full end-to-end integration across Floquet dielectric matrix assembly, RCWA optical enhancement, structural FEA deformation, closed-loop thermal power ledger, and HIL telemetry streaming.
