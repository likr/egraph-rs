//! Algebraic multigrid preconditioned conjugate gradient solvers and effective resistance for `egraph-rs`.

#![warn(missing_docs)]

pub mod amg_pcg;
pub mod resistance_curvature;

pub use amg_pcg::{
    AmgPcgBatchResult, AmgPcgConfig, AmgPcgSolveResult, AmgPcgSolver, AmgPcgWorkspace,
};
pub use resistance_curvature::{
    EffectiveResistanceConfig, EffectiveResistanceResult, ResistanceCurvatureConfig,
    ResistanceCurvatureFlowResult, ResistanceCurvatureStep, SuggestedGeometry,
    classify_geometry_from_curvature, compute_effective_resistances, resistance_curvature_flow,
};

/// Returns the crate version string.
pub fn crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::crate_version;

    #[test]
    fn linalg_amg_crate_version_non_empty() {
        assert!(!crate_version().is_empty());
    }
}
