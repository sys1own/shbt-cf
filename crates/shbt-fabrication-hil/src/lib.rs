//! Real-time hardware-in-the-loop diagnostics: sensor telemetry ingestion,
//! reduced-order surrogate evaluation and residual monitoring against the
//! theoretical models (simulator_spec.pdf §2, §4).
//!
//! This is the integration crate and depends on every other workspace crate.
//! Async ingestion, shared-memory ring buffers and PyO3 bindings are
//! implemented in Stages 4–5 of the blueprint.

#![forbid(unsafe_code)]

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
