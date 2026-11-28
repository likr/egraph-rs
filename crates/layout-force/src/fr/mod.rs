//! Fruchterman–Reingold (1991) force-directed graph layout solver.
//!
//! Implements the classic Fruchterman & Reingold (1991) algorithm with optimal distance
//! $k = \sqrt{\text{area}/N}$, quadratic edge attraction $f_a(d) = d^2/k$, inverse-distance
//! repulsion $f_r(d) = k^2/d$, temperature clamping with linear cooling, and three selectable
//! spatial repulsion backends:
//! - [`FrBackend::Exact`]: $O(N^2)$ all-pairs repulsion across any [`RiemannianLayout`](egraph_core::RiemannianLayout).
//! - [`FrBackend::BarnesHut`]: $O(N \log N)$ [`ArenaQuadtree2d`] (`Layout2d`) or
//!   [`ArenaOctree3d`](egraph_spatial_trees::ArenaOctree3d) (`Layout3d`) Coulomb approximation.
//! - [`FrBackend::Grid`]: Fruchterman–Reingold (1991) Fig. 2 $2k$ spatial grid with
//!   strict Euclidean cutoff $d < 2k$ ($f_r(d) = 0$ for $d \ge 2k$) on `Layout2d` and `Layout3d`.

mod config;
mod dim3;
mod helpers;
mod manifold;
mod nd;
mod riemannian;
mod solver;
mod step2d;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_manifold;
#[cfg(test)]
mod tests_nd;

#[cfg(doc)]
use egraph_spatial_trees::ArenaQuadtree2d;

pub use config::{FrBackend, FruchtermanReingoldConfig};
pub use solver::FruchtermanReingoldSolver;
