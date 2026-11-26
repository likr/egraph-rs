//! Stochastic gradient descent and coarse-backbone graph layout solvers for `egraph-rs`.

pub mod coarse_backbone;
pub mod sgd;

pub use coarse_backbone::{
    CoarseBackboneConfig, CoarseBackboneResult, CoarseBackboneStats, CoarseDistanceBackend,
    CoarseEdgeLengthMode, coarse_backbone_init,
};
pub use sgd::{
    BarrierSymmetry, DrGraphConfig, GraphCneConfig, GraphCneObjective, LargeVisConfig,
    OrtmannPivotStrategy, OrtmannSparseStressConfig, PacmapGraphConfig, Sgd2Config,
    SgdScheduleConfig, SgdScheduleKind, SgdScheduleWindowLayout, SgdSolver, SnapTFdpConfig,
    SparseBarrierConfig, SparseBarrierStats, StreamingNegativeConfig,
};

/// Returns the crate version.
pub fn crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::crate_version;

    #[test]
    fn layout_sgd_crate_version_non_empty() {
        assert!(!crate_version().is_empty());
    }
}
