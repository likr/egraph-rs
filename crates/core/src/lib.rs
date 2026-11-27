//! Core zero-copy, single-threaded data structures and foundational traits for `egraph-rs`.
//!
//! `egraph-core` provides the shared substrate across Native Rust, Python (PyO3), and
//! WebAssembly (`wasm-bindgen`) targets without host bindings, `petgraph`, or threading
//! dependencies.

mod csr;
mod error;
mod extents;
mod geometry;
mod layout;
mod manifold;
mod pin_mask;
mod rng;
mod scalar;
mod solver;
mod visit_stamp;

pub use csr::{CsrGraph, CsrGraphBuilder};
pub use error::{CoreError, CoreResult};
pub use extents::{NodeExtents2d, NodeShape2d};
pub use geometry::{
    EuclideanLayout, LayoutStorage, MetricSpace, RepulsionBackend, RiemannianLayout,
};
#[doc(hidden)]
pub use geometry::{EuclideanStorageMut, EuclideanStorageRef};
pub use layout::{EdgePolylines, Layout2d, Layout3d, LayoutNd};
pub use manifold::{
    ManifoldKind, ManifoldLayout2d, ManifoldMetric, POINCARE_EPS_F32, POINCARE_EPS_F64,
    cartesian_to_spherical, clamp_poincare_point, lorentz_distance, lorentz_einstein_midpoint,
    lorentz_exp_map, lorentz_log_map, lorentz_to_poincare_point, poincare_distance,
    poincare_einstein_midpoint, poincare_exp_map, poincare_log_map, poincare_mobius_add,
    project_spherical_coords, spherical_distance, spherical_exp_map, spherical_log_map,
    spherical_to_cartesian, torus_delta, torus_distance, wrap_torus_coord, wrap_torus_point,
};
pub use pin_mask::PinMask;
pub use rng::DeterministicRng;
pub use scalar::Scalar;
pub use solver::{IterativeSolver, LayoutGenerator, LayoutTransform, SolverStatus};
pub use visit_stamp::VisitStamp;

#[cfg(all(test, feature = "serde"))]
mod serde_tests {
    use super::*;

    fn assert_serde<T: serde::Serialize + serde::de::DeserializeOwned>() {}

    #[test]
    fn test_optional_serde_bounds() {
        assert_serde::<CsrGraph<f64>>();
        assert_serde::<CsrGraph<()>>();
        assert_serde::<Layout2d<f64>>();
        assert_serde::<Layout3d<f32>>();
        assert_serde::<LayoutNd<f64>>();
        assert_serde::<EdgePolylines<Layout2d<f64>>>();
        assert_serde::<EdgePolylines<Layout3d<f32>>>();
        assert_serde::<ManifoldKind>();
        assert_serde::<ManifoldMetric>();
        assert_serde::<ManifoldLayout2d<f64>>();
        assert_serde::<SolverStatus>();
        assert_serde::<NodeShape2d<f64>>();
        assert_serde::<PinMask>();
    }
}
