//! Diffusion kernel matrix computation and distance representation.

mod bicgstab;
mod chebyshev;
mod diffusion_kernel;
mod hutchinson;
mod multiscale;
mod power_method;

pub use diffusion_kernel::{
    DiffusionDistanceMatrix, DiffusionKernel, PivotDiffusionDistanceMatrix,
};
pub use hutchinson::HutchinsonEstimator;
pub use multiscale::{MultiscaleDiffusionDistanceMatrix, MultiscaleDiffusionKernel};
