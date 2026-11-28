//! Single-threaded, pure-Rust composite multi-solver graph layout algorithms for `egraph-rs`.

#![warn(missing_docs)]

#[cfg(feature = "baselines")]
mod backbone;
#[cfg(feature = "baselines")]
mod group_in_a_box;
#[cfg(feature = "baselines")]
mod sel;

#[cfg(feature = "baselines")]
pub use backbone::{
    BackboneConfig, BackboneLayoutMode, BackboneResult, BackboneSolver,
    extract_quadrilateral_backbone, extract_quadrilateral_backbone_unweighted,
};
#[cfg(feature = "baselines")]
pub use egraph_spatial_trees::squarified_treemap;
#[cfg(feature = "baselines")]
pub use group_in_a_box::{
    GibBox, GibBoxLayoutMode, GibIntraLayoutMode, GroupInABoxConfig, GroupInABoxSolver,
};
#[cfg(feature = "baselines")]
pub use sel::{SelConfig, SelLayoutBackend, SelLayoutSolver, SelWeightMapping};

#[cfg(all(test, not(feature = "baselines")))]
mod tests {
    #[test]
    fn baselines_disabled_by_default() {
        const { assert!(!cfg!(feature = "baselines")) };
    }
}
