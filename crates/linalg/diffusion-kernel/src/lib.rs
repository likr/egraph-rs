//! Diffusion kernel matrix computation and distance representation.

pub mod bicgstab;
pub mod chebyshev;
mod diffusion_kernel;
pub mod hutchinson;
pub mod multiscale;
pub mod power_method;

pub use diffusion_kernel::{
    DiffusionDistanceMatrix, DiffusionKernel, PivotDiffusionDistanceMatrix,
};
pub use hutchinson::HutchinsonEstimator;
pub use multiscale::{MultiscaleDiffusionDistanceMatrix, MultiscaleDiffusionKernel};
