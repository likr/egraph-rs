//! Separation constraints, layering, overlap removal, and proximity layout solvers for `egraph-rs`.

#![cfg_attr(docsrs, feature(doc_cfg))]

mod constraints;
#[cfg(feature = "baselines")]
mod forbid;
#[cfg(feature = "baselines")]
mod gtree;
mod layering;
#[cfg(feature = "baselines")]
mod pfs_prime;
mod prism;
mod twin_declutter;

pub use constraints::{
    ConstraintProjector, SeparationConstraint, generate_layered_constraints,
    project_clustered_rectangle_no_overlap_2d, project_rectangle_no_overlap_2d,
};
#[cfg(feature = "baselines")]
#[cfg_attr(docsrs, doc(cfg(feature = "baselines")))]
pub use forbid::{ForbidConfig, ForbidOverlapRemoval, ForbidVariant};
#[cfg(feature = "baselines")]
#[cfg_attr(docsrs, doc(cfg(feature = "baselines")))]
pub use gtree::{GTreeConfig, GTreeOverlapRemoval};
pub use layering::{
    LayeringMethod, assign_layers, cycle_edges, remove_cycles, transitive_reduction,
    transitive_reduction_weighted,
};
#[cfg(feature = "baselines")]
#[cfg_attr(docsrs, doc(cfg(feature = "baselines")))]
pub use pfs_prime::{PfsPrimeConfig, PfsPrimeOverlapRemoval};
pub use prism::{PrismConfig, PrismOverlapRemoval, SparsePcgWorkspace};
pub use twin_declutter::{TwinDeclutterConfig, TwinDeclutterer, declutter_twins_2d};
