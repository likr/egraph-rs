//! Non-Euclidean 2D manifold geometries: Hyperbolic Poincaré disk, Unit Sphere, and 2D Torus.

mod layout;
mod poincare;
mod spherical;
mod torus;

#[cfg(test)]
mod tests;

/// Supported 2D non-Euclidean manifold geometries.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ManifoldKind {
    /// 2D Poincaré unit disk model of hyperbolic space (`x^2 + y^2 < 1`).
    HyperbolicPoincare,
    /// 2D unit sphere parameterized by `(longitude, colatitude)` in `[0, 2π) × [0, π]`.
    Spherical,
    /// 2D flat unit torus `[0, 1)^2` with periodic wrap-around boundaries.
    Torus2d,
}

pub use layout::ManifoldLayout2d;
pub use poincare::{
    clamp_poincare_point, poincare_distance, poincare_einstein_midpoint, poincare_exp_map,
    poincare_log_map, poincare_mobius_add,
};
pub use spherical::{
    cartesian_to_spherical, project_spherical_coords, spherical_distance, spherical_exp_map,
    spherical_log_map, spherical_to_cartesian,
};
pub use torus::{torus_delta, torus_distance, wrap_torus_coord, wrap_torus_point};
