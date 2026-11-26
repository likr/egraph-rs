//! Pure-Rust GuMap (Graph UMAP) layout solver with `dist_sq.powf(b)` gradient evaluation
//! and cooperative [`IterativeSolver`](egraph_core::IterativeSolver) stepping.

mod config;
mod manifold;
mod nd;
mod solver;
mod step;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_manifold;
#[cfg(test)]
mod tests_nd;

pub use config::GuMapConfig;
pub use solver::GuMapSolver;
