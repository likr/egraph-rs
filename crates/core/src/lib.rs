//! Core zero-copy, single-threaded data structures and foundational traits for `egraph-rs`.
//!
//! `egraph-core` provides the shared substrate across Native Rust, Python (PyO3), and
//! WebAssembly (`wasm-bindgen`) targets without host bindings, `petgraph`, or threading
//! dependencies.

pub mod csr;
pub mod error;
pub mod layout;
pub mod manifold;
pub mod rng;
pub mod scalar;
pub mod solver;
pub mod visit_stamp;

pub use csr::CsrGraph;
pub use error::{CoreError, CoreResult};
pub use layout::{Layout2d, LayoutNd};
pub use manifold::{
    ManifoldKind, ManifoldLayout2d, cartesian_to_spherical, clamp_poincare_point,
    poincare_distance, poincare_einstein_midpoint, poincare_exp_map, poincare_log_map,
    poincare_mobius_add, project_spherical_coords, spherical_distance, spherical_exp_map,
    spherical_log_map, spherical_to_cartesian, torus_delta, torus_distance, wrap_torus_coord,
    wrap_torus_point,
};
pub use rng::DeterministicRng;
pub use scalar::Scalar;
pub use solver::{IterativeSolver, SolverStatus};
pub use visit_stamp::VisitStamp;
