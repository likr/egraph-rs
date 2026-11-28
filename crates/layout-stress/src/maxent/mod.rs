//! Gansner et al. (2013) Maxent-Stress graph layout solver.
//!
//! Implements the force-augmented stress majorization model from
//! *"A Maxent-Stress Model for Graph Layout"* (Gansner, Hu & North, IEEE TVCG 2013):
//!
//! $$L_w x^{(t+1)} = L_{w,d}(x^{(t)}) x^{(t)} + \alpha\, b(x^{(t)})$$
//!
//! where $L_w$ is the weighted graph Laplacian over 1-hop edges $S = E$ ($w_{ij} = d_{ij}^{-2}$),
//! $b(x)$ is the generalized $q$-power repulsive force over non-adjacent vertex pairs
//! ($\{i,j\} \notin S$) evaluated via [`RepulsionBackend`](egraph_core::RepulsionBackend)
//! ([`ArenaQuadtree2d`](egraph_spatial_trees::ArenaQuadtree2d) on [`Layout2d`](egraph_core::Layout2d),
//! [`ArenaOctree3d`](egraph_spatial_trees::ArenaOctree3d) on [`Layout3d`](egraph_core::Layout3d),
//! or exact pairwise summation), and $L_w x = \mathrm{rhs}$ is solved per coordinate via
//! [`AmgPcgSolver`](egraph_linalg_amg::AmgPcgSolver).
//!
//! On disconnected or isolated-vertex graphs, $b(x)$ is zeroed on isolated vertices and centered
//! per connected component so the right-hand side lies strictly in $\mathrm{range}(L_w)$.

mod config;
mod graph_prep;
mod init;
mod repulsion;
mod solver;
mod step;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_nd;

pub use config::MaxentStressConfig;
pub use solver::MaxentStressSolver;
