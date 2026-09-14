//! Real-time hardware-in-the-loop diagnostics: POSIX shared-memory telemetry
//! rings, async Tokio ingestion, and residual monitoring against reduced-order
//! surrogates (simulator_spec.pdf §2, §4).
//!
//! * [`telemetry`] — 64-byte `#[repr(C, align(64))]` sensor frames (Type-N
//!   thermocouples, FBG strain, CCD spectrometer).
//! * [`ring`] — SPSC lock-free ring over raw memory (`AtomicUsize` head/tail,
//!   Acquire/Release).
//! * [`shm`] — `shm_open`/`ftruncate`/`mmap` region allocation.
//! * [`pipeline`] — Tokio drain task + lock-free stats.
//! * [`rom`] — `χ²(t) = rᵀΣ⁻¹r` residual scoring against reduced-order models.
//!
//! This is the integration crate and depends on every other workspace crate.

// `shm`/`ring` use raw shared-memory pointers; unsafe is confined to them.
#![allow(unsafe_code)]
#![allow(clippy::missing_safety_doc)]

pub mod ffi;
pub mod optics;
pub mod pipeline;
pub mod physics;
pub mod ring;
pub mod rom;
pub mod transport;
#[cfg(unix)]
pub mod shm;
pub mod telemetry;

#[cfg(feature = "python")]
mod pyo3_ffi;

pub use ring::{Consumer, Producer, Region, SpscRing};
pub use rom::{chi2, reduced_chi2, ReducedOrderModel, ResidualMonitor};
pub use telemetry::{Channel, Frame, FRAME_VALUES};

/// Canonical crate identifier used by the workbench for topology reporting.
pub const CRATE_NAME: &str = "shbt-fabrication-hil";

/// Identifiers of every upstream workspace crate this integration layer links.
pub const UPSTREAM_CRATES: [&str; 5] = [
    shbt_core_math::CRATE_NAME,
    shbt_dielectric_floquet::CRATE_NAME,
    shbt_rcwa_optics::CRATE_NAME,
    shbt_fea_structural::CRATE_NAME,
    shbt_metrology_gum::CRATE_NAME,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn links_all_upstream_crates() {
        assert_eq!(
            UPSTREAM_CRATES,
            [
                "shbt-core-math",
                "shbt-dielectric-floquet",
                "shbt-rcwa-optics",
                "shbt-fea-structural",
                "shbt-metrology-gum",
            ]
        );
    }
}
