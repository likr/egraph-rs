//! Single-threaded, pure-Rust graph layout solvers and spatial refiners for `egraph-rs`.
//!
//! `egraph-layout` is a thin umbrella over `egraph-layout-constraints`, `egraph-layout-force`,
//! `egraph-layout-multilevel`, `egraph-layout-neighbor`, `egraph-layout-sgd`, and
//! `egraph-layout-stress`, providing cooperative `step()`/`run_steps(k)` layout execution across
//! Native Rust, Python (PyO3), and WebAssembly (`wasm32-unknown-unknown`) targets.

pub use egraph_layout_constraints::{constraints, layering, prism, twin_declutter};
#[cfg(feature = "baselines")]
pub use egraph_layout_constraints::{forbid, gtree, pfs_prime};
pub use egraph_layout_force::{continuous_fdp, fdeb};
#[cfg(feature = "baselines")]
pub use egraph_layout_force::{fr, gem, linlog};
#[cfg(feature = "baselines")]
pub use egraph_layout_multilevel::{backbone, cluster_refine, grip, group_in_a_box, sel, sfc};
pub use egraph_layout_multilevel::{density_field, refiner};
pub use egraph_layout_neighbor::{field_layout, gumap, ts_net};
pub use egraph_layout_sgd::{coarse_backbone, sgd};
#[cfg(feature = "baselines")]
pub use egraph_layout_stress::maxent;
pub use egraph_layout_stress::{kamada_kawai, stress_majorization};

#[cfg(feature = "baselines")]
pub use backbone::{
    BackboneConfig, BackboneLayoutMode, BackboneResult, BackboneSolver,
    extract_quadrilateral_backbone,
};
#[cfg(feature = "baselines")]
pub use cluster_refine::{
    ClusterRefineConfig, ClusterRefineMethod, ClusterRefineSolver, refine_cluster_layout,
};
pub use coarse_backbone::{
    CoarseBackboneConfig, CoarseBackboneResult, CoarseBackboneStats, CoarseDistanceBackend,
    CoarseEdgeLengthMode, coarse_backbone_init,
};
pub use constraints::{
    ConstraintProjector, SeparationConstraint, generate_layered_constraints,
    project_clustered_rectangle_no_overlap_2d, project_rectangle_no_overlap_2d,
};
pub use continuous_fdp::{
    ContinuousFdp, ForceAtlas2Config, ForceAtlas2Solver, YifanHuConfig, YifanHuSolver,
};
pub use density_field::{ScreenedPoissonRefiner, ScreenedPoissonRefinerConfig};
pub use fdeb::{FdebConfig, FdebSolver};
pub use field_layout::{
    DEFAULT_MAX_2HOP_PER_NODE, FieldFftStats, FieldLayout, FieldLayoutSolver,
    FieldRepulsionBackend, IbUmapConfig, RhoSchedule, TFdpConfig,
};
#[cfg(feature = "baselines")]
pub use forbid::{ForbidConfig, ForbidOverlapRemoval, ForbidVariant};
#[cfg(feature = "baselines")]
pub use fr::{FrBackend, FruchtermanReingoldConfig, FruchtermanReingoldSolver};
#[cfg(feature = "baselines")]
pub use gem::{GemConfig, GemSolver};
#[cfg(feature = "baselines")]
pub use grip::{GripConfig, GripSolver};
#[cfg(feature = "baselines")]
pub use group_in_a_box::{
    GibBox, GibBoxLayoutMode, GibIntraLayoutMode, GroupInABoxConfig, GroupInABoxSolver,
    squarified_treemap,
};
#[cfg(feature = "baselines")]
pub use gtree::{GTreeConfig, GTreeOverlapRemoval};
pub use gumap::{GuMapConfig, GuMapSolver};
pub use kamada_kawai::{KamadaKawaiConfig, KamadaKawaiSolver};
pub use layering::{
    LayeringMethod, assign_layers, cycle_edges, remove_cycles, transitive_reduction,
};
#[cfg(feature = "baselines")]
pub use linlog::{LinLogConfig, LinLogRepulsion, LinLogSolver};
#[cfg(feature = "baselines")]
pub use maxent::{MaxentStressConfig, MaxentStressSolver};
#[cfg(feature = "baselines")]
pub use pfs_prime::{PfsPrimeConfig, PfsPrimeOverlapRemoval};
pub use prism::{PrismConfig, PrismOverlapRemoval, SparsePcgWorkspace};
pub use refiner::{
    SpatialIntruderRefiner, SpatialRefinerConfig, SpatialRefinerRoundStats, SpatialRefinerStats,
};
#[cfg(feature = "baselines")]
pub use sel::{SelConfig, SelLayoutBackend, SelLayoutSolver, SelWeightMapping};
#[cfg(feature = "baselines")]
pub use sfc::{SfcConfig, SfcCurveKind, SfcLayoutSolver};
pub use sgd::{
    BarrierSymmetry, DrGraphConfig, GraphCneConfig, GraphCneObjective, LargeVisConfig,
    OrtmannPivotStrategy, OrtmannSparseStressConfig, PacmapGraphConfig, Sgd2Config,
    SgdScheduleConfig, SgdScheduleKind, SgdSolver, SnapTFdpConfig, SparseBarrierConfig,
    SparseBarrierStats, StreamingNegativeConfig,
};
pub use stress_majorization::{PcgWorkspace, StressMajorizationConfig, StressMajorizationSolver};
pub use ts_net::{TsNetBackend, TsNetConfig, TsNetSolver, TsNetStage};
pub use twin_declutter::{TwinDeclutterConfig, TwinDeclutterer, declutter_twins_2d};

/// Returns the crate version string.
pub fn crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::crate_version;

    #[test]
    fn layout_crate_version_non_empty() {
        assert!(!crate_version().is_empty());
    }
}
