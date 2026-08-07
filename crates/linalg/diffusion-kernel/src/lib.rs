//! Diffusion kernel matrix computation and distance representation.

pub mod bicgstab;
pub mod chebyshev;
pub mod diffusion_kernel;
pub mod distance;
pub mod low_rank;
pub mod multiscale;
pub mod power_method;
pub mod traits;

pub use diffusion_kernel::{
    DiffusionKernel, DiffusionKernelBuilder, PivotedDiffusionKernel, PivotedDiffusionKernelBuilder,
};
pub use distance::{
    NegLogDistance, NegLogDistanceBuilder, NegLogSimDistance, NegLogSimDistanceBuilder,
    PivotedNegLogDistance, PivotedNegLogDistanceBuilder,
};
pub use low_rank::{
    LowRankDiffusionKernel, LowRankDiffusionKernelBuilder, LowRankMultiscaleDiffusionKernel,
    LowRankMultiscaleDiffusionKernelBuilder,
};
pub use multiscale::{
    MultiscaleDiffusionKernel, MultiscaleDiffusionKernelBuilder, PivotedMultiscaleDiffusionKernel,
    PivotedMultiscaleDiffusionKernelBuilder,
};
pub use traits::PivotedKernel;
