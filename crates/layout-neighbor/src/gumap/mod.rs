//! Pure-Rust GuMap (Graph UMAP) layout solver with `dist_sq.powf(b)` gradient evaluation
//! and cooperative [`IterativeSolver`](egraph_core::IterativeSolver) stepping.

mod config;
mod solver;
mod step;
#[cfg(test)]
mod tests;

pub use config::GuMapConfig;
pub use solver::GuMapSolver;
