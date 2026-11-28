//! Fruchterman–Reingold (1991) force-directed graph layout solver.
//!
//! Implements the classic Fruchterman & Reingold (1991) algorithm with optimal distance
//! $k = \sqrt{\text{area}/N}$, quadratic edge attraction $f_a(d) = d^2/k$, inverse-distance
//! repulsion $f_r(d) = k^2/d$, temperature clamping with linear cooling, and compile-time
//! type-safe spatial repulsion backends:
//! - [`Exact`](crate::Exact): $O(N^2)$ all-pairs repulsion across any
//!   [`RiemannianLayout`](egraph_core::RiemannianLayout).
//! - [`BarnesHut`](crate::BarnesHut): $O(N \log N)$ quadtree (`Layout2d`) or octree
//!   (`Layout3d`) Coulomb approximation.
//! - [`Grid`](crate::Grid): Fruchterman–Reingold (1991) Fig. 2 $2k$ spatial grid with
//!   strict Euclidean cutoff $d < 2k$ ($f_r(d) = 0$ for $d \ge 2k$) on `Layout2d` and `Layout3d`.

mod barnes_hut;
mod config;
mod grid;
mod helpers;
mod repulsion;
mod solver;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_manifold;
#[cfg(test)]
mod tests_nd;

pub use config::FruchtermanReingoldConfig;
pub use repulsion::FrRepulsion;
pub use solver::FruchtermanReingoldSolver;
