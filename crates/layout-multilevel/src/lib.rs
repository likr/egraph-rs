//! Multilevel, hierarchical, backbone, and spatial-intruder refinement layout solvers for `egraph-rs`.

#[cfg(feature = "baselines")]
pub mod backbone;
#[cfg(feature = "baselines")]
pub mod cluster_refine;
pub mod density_field;
#[cfg(feature = "baselines")]
pub mod grip;
#[cfg(feature = "baselines")]
pub mod group_in_a_box;
pub mod refiner;
#[cfg(feature = "baselines")]
pub mod sel;
#[cfg(feature = "baselines")]
pub mod sfc;

#[cfg(feature = "baselines")]
pub use backbone::{
    BackboneConfig, BackboneLayoutMode, BackboneResult, BackboneSolver,
    extract_quadrilateral_backbone,
};
#[cfg(feature = "baselines")]
pub use cluster_refine::{
    ClusterRefineConfig, ClusterRefineMethod, ClusterRefineSolver, refine_cluster_layout,
};
pub use density_field::{ScreenedPoissonRefiner, ScreenedPoissonRefinerConfig};
#[cfg(feature = "baselines")]
pub use grip::{GripConfig, GripSolver};
#[cfg(feature = "baselines")]
pub use group_in_a_box::{
    GibBox, GibBoxLayoutMode, GibIntraLayoutMode, GroupInABoxConfig, GroupInABoxSolver,
    squarified_treemap,
};
pub use refiner::{
    SpatialIntruderRefiner, SpatialRefinerConfig, SpatialRefinerRoundStats, SpatialRefinerStats,
};
#[cfg(feature = "baselines")]
pub use sel::{SelConfig, SelLayoutBackend, SelLayoutSolver, SelWeightMapping};
#[cfg(feature = "baselines")]
pub use sfc::{SfcConfig, SfcCurveKind, SfcLayoutSolver};

/// Returns the crate version.
pub fn crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::crate_version;

    #[test]
    fn layout_multilevel_crate_version_non_empty() {
        assert!(!crate_version().is_empty());
    }
}
