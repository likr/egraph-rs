//! Compressed Sparse Row (`CsrMatrix`) representation, graph Laplacians, and $O(|E|)$ spectral moments.

mod anisotropic;
mod laplacians;
mod matrix;
mod moments;
mod spectrum;
mod spectrum_operator;

pub use anisotropic::{
    AnisotropicLaplacianConfig, AnisotropicLaplacianResult, anisotropic_laplacian,
};
pub use laplacians::{
    LaplacianKind, combinatorial_laplacian, laplacian, random_walk_laplacian,
    shifted_normalized_laplacian, symmetric_normalized_laplacian,
};
pub use matrix::CsrMatrix;
pub use moments::{SpectralMomentStats, spectral_moment_stats};
pub use spectrum::{
    DEFAULT_SPECTRUM_LANCZOS_SAFETY_FACTOR, DEFAULT_SPECTRUM_LANCZOS_STEPS,
    DEFAULT_SPECTRUM_POWER_ITERS, DEFAULT_SPECTRUM_SAFETY_FACTOR,
};
pub use spectrum_operator::estimate_lambda_max_operator;

#[cfg(test)]
mod tests;
