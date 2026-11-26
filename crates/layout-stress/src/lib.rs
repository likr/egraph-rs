//! Stress majorization, Kamada-Kawai, and maxent-stress graph layout solvers for `egraph-rs`.

pub mod kamada_kawai;
mod manifold_utils;
#[cfg(feature = "baselines")]
pub mod maxent;
pub mod stress_majorization;

pub use kamada_kawai::{KamadaKawaiConfig, KamadaKawaiSolver};
#[cfg(feature = "baselines")]
pub use maxent::{MaxentStressConfig, MaxentStressSolver};
pub use stress_majorization::{PcgWorkspace, StressMajorizationConfig, StressMajorizationSolver};

/// Returns the crate version.
pub fn crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::crate_version;

    #[test]
    fn layout_stress_crate_version_non_empty() {
        assert!(!crate_version().is_empty());
    }
}
