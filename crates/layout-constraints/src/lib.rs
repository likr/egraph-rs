//! Separation constraints, layering, overlap removal, and proximity layout solvers for `egraph-rs`.

pub mod constraints;
#[cfg(feature = "baselines")]
pub mod forbid;
#[cfg(feature = "baselines")]
pub mod gtree;
pub mod layering;
#[cfg(feature = "baselines")]
pub mod pfs_prime;
pub mod prism;

pub use constraints::{
    ConstraintProjector, SeparationConstraint, generate_layered_constraints,
    project_clustered_rectangle_no_overlap_2d, project_rectangle_no_overlap_2d,
};
#[cfg(feature = "baselines")]
pub use forbid::{ForbidConfig, ForbidOverlapRemoval, ForbidVariant};
#[cfg(feature = "baselines")]
pub use gtree::{GTreeConfig, GTreeOverlapRemoval};
pub use layering::{LayeringMethod, assign_layers, cycle_edges, remove_cycles};
#[cfg(feature = "baselines")]
pub use pfs_prime::{PfsPrimeConfig, PfsPrimeOverlapRemoval};
pub use prism::{PrismConfig, PrismOverlapRemoval, SparsePcgWorkspace};

/// Returns the crate version.
pub fn crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::crate_version;

    #[test]
    fn layout_constraints_crate_version_non_empty() {
        assert!(!crate_version().is_empty());
    }
}
