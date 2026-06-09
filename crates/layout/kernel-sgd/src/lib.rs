//! Kernel-based SGD layout algorithm using diffusion kernel approximation.
//!
//! This crate implements a graph layout algorithm that uses the diffusion kernel
//! exp(-tL) to compute ideal distances between nodes, where L is the graph Laplacian.

mod kernel_sgd;

pub use kernel_sgd::KernelSgd;
pub use petgraph_linalg_diffusion_kernel::{
    DiffusionDistanceMatrix, DiffusionKernel, HutchinsonEstimator,
};
