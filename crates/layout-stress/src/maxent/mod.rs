//! Gansner et al. (2013) Maxent-Stress graph layout solver.
//!
//! Implements the force-augmented stress majorization model from
//! *"A Maxent-Stress Model for Graph Layout"* (Gansner, Hu & North, IEEE TVCG 2013):
//!
//! $$L_w x^{(t+1)} = L_{w,d}(x^{(t)}) x^{(t)} + \alpha\, b(x^{(t)})$$
//!
//! where $L_w$ is the weighted graph Laplacian over 1-hop edges $S = E$ ($w_{ij} = d_{ij}^{-2}$),
//! $b(x)$ is the generalized $q$-power repulsive force over non-adjacent vertex pairs
//! ($\{i,j\} \notin S$) evaluated via [`ArenaQuadtree2d::compute_maxent_repulsion`](egraph_spatial_trees::ArenaQuadtree2d::compute_maxent_repulsion)
//! (or exact pairwise summation), and $L_w x = \mathrm{rhs}$ is solved per coordinate via
//! [`AmgPcgSolver`](egraph_linalg_amg::AmgPcgSolver).
//!
//! On disconnected or isolated-vertex graphs, $b(x)$ is zeroed on isolated vertices and centered
//! per connected component so the right-hand side lies strictly in $\mathrm{range}(L_w)$.

mod config;
mod graph_prep;
mod nd;
mod solver;
mod step;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_nd;

pub use config::MaxentStressConfig;
pub use solver::MaxentStressSolver;
