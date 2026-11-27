//! Non-Euclidean 2D manifold geometries: Hyperbolic Poincaré disk, Hyperbolic Lorentz
//! hyperboloid, Unit Sphere, and 2D Torus.

mod layout;
mod lorentz;
mod lorentz_midpoint;
mod metric;
mod poincare;
mod spectral;
mod spectral_eigen;
mod spectral_jacobi;
mod spherical;
mod torus;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_lorentz_and_metric;
#[cfg(test)]
mod tests_torus;

/// Supported 2D non-Euclidean manifold geometries.
#[non_exhaustive]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ManifoldKind {
    /// 2D Poincaré unit disk model of hyperbolic space (`x^2 + y^2 < 1`).
    HyperbolicPoincare,
    /// Unconstrained 2D Gans / spatial hyperboloid coordinates `(x, y) in R^2` with `t = sqrt(1 + x^2 + y^2)`.
    HyperbolicLorentz,
    /// 2D unit sphere parameterized by `(longitude, colatitude)` in `[0, 2π) × [0, π]`.
    Spherical,
    /// 2D flat torus `[0, 1)^2` with periodic wrap-around boundaries and metric extents `(Lx, Ly)`.
    Torus2d,
}

pub use layout::ManifoldLayout2d;
pub use lorentz::{lorentz_distance, lorentz_exp_map, lorentz_log_map, lorentz_to_poincare_point};
pub use lorentz_midpoint::lorentz_einstein_midpoint;
pub use metric::ManifoldMetric;
pub use poincare::{
    POINCARE_EPS_F32, POINCARE_EPS_F64, clamp_poincare_point, poincare_distance,
    poincare_einstein_midpoint, poincare_exp_map, poincare_log_map, poincare_mobius_add,
};
pub use spherical::{
    cartesian_to_spherical, project_spherical_coords, spherical_distance, spherical_exp_map,
    spherical_log_map, spherical_to_cartesian,
};
pub use torus::{torus_delta, torus_distance, wrap_torus_coord, wrap_torus_point};
