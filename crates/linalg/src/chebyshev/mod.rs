//! Zero-allocation Chebyshev spectral filtering, Miller Bessel $e^{-t}I_k(t)$ evaluation,
//! DCT-II spectral square root, R1/R2 regularized DoG terms, and JL embeddings.

mod coefficients;
mod dog_builder;
mod dog_kernel;
mod dynamic;
mod jl;
mod operator;
mod workspace;
mod workspace_kernels;

pub use coefficients::{
    adaptive_chebyshev_degree, adaptive_chebyshev_degree_scaled, build_dog_terms,
    build_regularized_dog_terms, compute_chebyshev_bandpass_coefficients,
    compute_chebyshev_sqrt_filter_coefficients, effective_chebyshev_len, exp_scaled_bessel_i,
    exp_scaled_bessel_i_all, project_psd_spectral_filter,
};
pub use dog_kernel::{ChebyshevDogConfig, ChebyshevDogKernel};
pub use workspace::ChebyshevWorkspace;

#[cfg(test)]
mod tests_filters;
#[cfg(test)]
mod tests_tiled;
