//! Mathematical foundations for the SHBT zero-drift precision simulator.
//!
//! This crate is the root of the workspace dependency graph and depends on no
//! other workspace crate. It hosts the primitives every domain solver builds on:
//!
//! * [`fixed`] — 128-bit signed fixed-point arithmetic (Q64.64) for drift-free
//!   global accumulators (phase angles, path lengths, grid coordinates).
//! * [`precision`] — the precision-policy model (strict `f64`, Q64.64,
//!   arbitrary precision) used to select a numeric backend per subsystem.
//!
//! Lie-group (SE(3)) operators, Yoshida 6th-order symplectic integrators and the
//! `rug`/MPFR arbitrary-precision wrappers are implemented in Stage 2 of the
//! blueprint and will live alongside these modules.

#![forbid(unsafe_code)]
#![cfg_attr(not(test), no_std)]

pub mod fixed;
pub mod precision;

/// Canonical crate identifier used by the workbench for topology reporting.
pub const CRATE_NAME: &str = "shbt-core-math";
