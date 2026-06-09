//! Diffusion kernel matrix computation and distance representation.

mod chebyshev;
mod diffusion_kernel;
mod hutchinson;
mod power_method;

pub use diffusion_kernel::{DiffusionDistanceMatrix, DiffusionKernel};
pub use hutchinson::HutchinsonEstimator;
