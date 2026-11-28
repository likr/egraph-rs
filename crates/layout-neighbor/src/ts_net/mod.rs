//! Pure-Rust unified tsNET / BH-tsNET / FIt-tsNET / L-tsNET layout solver.
//!
//! Consolidates exact $O(N^2)$ tsNET, Barnes-Hut spatial tree repulsion, and fused 2D FFT
//! interpolation (`FitField2d`) into a single allocation-free cooperative
//! [`IterativeSolver`](egraph_core::IterativeSolver) with a validated Pure 1+2-Hop adaptive
//! perplexity preset ([`TsNetConfig::pure_2hop`]) and compile-time typed repulsion backends.

mod config;
mod dim3;
mod manifold;
mod nd;
mod repulsion;
mod repulsion_fft;
mod repulsion_impls;
mod repulsion_wrappers;
mod riemannian;
mod solver;
mod solver_with_backend;
mod step2d;
#[cfg(test)]
mod tests;

pub use config::{TsNetConfig, TsNetStage};
pub use repulsion::{DefaultTsNetBackend, TsNetRepulsion};
#[doc(hidden)]
pub use repulsion::{TsNetCore, TsNetWorkspace};
#[doc(hidden)]
pub use repulsion_fft::FitFft2dState;
#[doc(hidden)]
pub use repulsion_impls::{BarnesHut2dState, BarnesHut3dState, ExactTsNetState};
#[doc(hidden)]
pub use repulsion_wrappers::{Auto2dState, Auto3dState, Backend2dState, Backend3dState};
pub use solver::TsNetSolver;
