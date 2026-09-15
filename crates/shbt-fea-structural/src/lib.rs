//! Non-linear thermomechanical structural models for the SHBT reactor frame
//! (simulator_spec.pdf §3, §4; cf.pdf §C "Four-Disc Belleville Compliance"):
//!
//! * [`belleville`] — DIN 2092 conical disc springs (force, stiffness, DIN
//!   stresses, Group 3 contact flats, compound stacks with friction) and the
//!   DIN 2093 Group 2 reference table used as the verification gate.
//! * [`disc_fe`] — geometrically non-linear axisymmetric finite-element disc
//!   model with `E(T)` / `α(T)` thermal loading and Newton–Raphson force
//!   control, 25 °C – 350 °C.
//! * [`stoney`] — extended Stoney thin-film stress with `R_pre`/`R_post`
//!   curvature radii, finite-thickness correction and `α(T)` mismatch.
//! * [`material`] — `E(T)`, `α(T)` laws and reference alloys.
//!
//! Depends exclusively on `shbt-core-math`.

#![forbid(unsafe_code)]

pub mod belleville;
pub mod disc_fe;
pub mod kinetics;
pub mod material;
pub mod quadrature;
pub mod stoney;
pub mod variational;
pub mod viscoplastic;

pub use belleville::{DiscSpring, DiscStack, StackLayout};
pub use disc_fe::ConicalDiscFe;
pub use material::{
    solve_ab_initio_material_tensors, validate_structural_design, AbInitioPhase, FeaOutput,
    MaterialTensors, A_CONTACT, DL_NET, D_CORE, F_MAX, F_MIN, K_STACK, PLASTIC_STRAIN_CEILING,
};
pub use stoney::FilmOnSubstrate;

use shbt_core_math::precision::PrecisionMode;

/// Canonical crate identifier used by the workbench for topology reporting.
pub const CRATE_NAME: &str = "shbt-fea-structural";

/// Real-time dynamic FEA solves run in strict binary64 (spec §1 precision table).
pub const DYNAMIC_SOLVE_PRECISION: PrecisionMode = PrecisionMode::StrictF64;
