//! Nachmanson et al. (2016) GTree proximity-MST node overlap removal solver.
//!
//! Implements GTree ("Node Overlap Removal by Growing a Tree", Nachmanson, Nocaj, Bereg,
//! Zhang & Holroyd, *Graph Drawing* 2016) using [`Delaunay2d`](egraph_spatial_trees::Delaunay2d),
//! combined with Prim's minimum spanning tree and $O(N \log N + |\mathcal{O}|)$
//! `RectangleOverlapSweep` Stage-2 cleanup to guarantee strictly `0` overlapping rectangles.

mod config;
mod mst;
mod proximity;
mod solver;
#[cfg(test)]
mod tests;

pub use config::GTreeConfig;
pub use solver::GTreeOverlapRemoval;
