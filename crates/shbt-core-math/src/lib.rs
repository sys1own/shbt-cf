//! Mathematical foundations for the SHBT zero-drift precision simulator.
//!
//! This crate is the root of the workspace dependency graph and depends on no
//! other workspace crate. It hosts the primitives every domain solver builds on:
//!
//! * [`fixed`] — 128-bit signed fixed-point arithmetic (Q64.64) for drift-free
//!   global accumulators (phase angles, path lengths, grid coordinates).
//! * [`precision`] — the precision-policy model (strict `f64`, Q64.64,
//!   arbitrary precision) and the `κ(A)`-driven mantissa-width selection.
//! * [`mp`] — `rug`/MPFR-backed dense linear algebra with adaptive 128–512-bit
//!   precision escalation.
//! * [`lie`] — SO(3)/SE(3) exponential and logarithm maps; all spatial updates
//!   are group-manifold moves, never quaternion or Euler-angle integration.
//! * [`symplectic`] — Störmer–Verlet and 6th-order Yoshida integrators for
//!   separable Hamiltonians, plus the Lie-algebraic free-rigid-body scheme.
//! * [`fpenv`] — DAZ/FTZ control-register setup (MXCSR / FPCR).
//!
//! `unsafe` is confined to [`fpenv`], where it is required to touch the CPU
//! control registers.

#![deny(unsafe_code)]

pub mod fixed;
pub mod fpenv;
pub mod lie;
pub mod mp;
pub mod precision;
pub mod symplectic;

/// Canonical crate identifier used by the workbench for topology reporting.
pub const CRATE_NAME: &str = "shbt-core-math";
