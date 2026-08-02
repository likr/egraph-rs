//! Diffusion kernel matrix computation and distance representation.

pub mod bicgstab;
pub mod chebyshev;
mod diffusion_kernel;
pub mod exact;
pub mod heat_kernel;
pub mod hutchinson;
pub mod low_rank;
pub mod multiscale;
pub mod power_method;

pub use diffusion_kernel::{
    DiffusionDistanceMatrix, DiffusionKernel, PivotDiffusionDistanceMatrix,
};
pub use exact::ExactDiffusionKernel;
pub use heat_kernel::HeatKernel;
pub use hutchinson::HutchinsonEstimator;
pub use low_rank::{
    HeatGeodesicDistanceMatrix, LowRankDiffusionDistanceMatrix, LowRankDiffusionKernel,
};
pub use multiscale::{MultiscaleDiffusionDistanceMatrix, MultiscaleDiffusionKernel};
