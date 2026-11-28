//! Pure-Rust unified tsNET / BH-tsNET / FIt-tsNET / L-tsNET layout solver.
//!
//! Consolidates exact $O(N^2)$ tsNET, Barnes-Hut spatial tree repulsion, and fused 2D FFT
//! interpolation (`FitField2d`) into a single allocation-free cooperative
//! [`IterativeSolver`](egraph_core::IterativeSolver) with a validated Pure 1+2-Hop adaptive
//! perplexity preset ([`TsNetConfig::pure_2hop`]).

mod config;
mod dim3;
mod dispatch;
mod manifold;
mod nd;
mod riemannian;
mod solver;
mod step2d;
#[cfg(test)]
mod tests;

pub use config::{TsNetBackend, TsNetConfig, TsNetStage};
pub use solver::TsNetSolver;
