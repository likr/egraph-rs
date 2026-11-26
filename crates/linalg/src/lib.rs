//! Single-threaded, pure-Rust linear algebra, spectral diffusion, and pivot primitives for `egraph-rs`.
//!
//! `egraph-linalg` provides flat `CsrGraph`-based workspaces targeting Native Rust, Python (PyO3),
//! and WebAssembly (`wasm32-unknown-unknown`) without BLAS/LAPACK FFI or threading dependencies:
//! - [`csr_matrix`]: `f64` [`CsrMatrix`] with SpMV/SpMM, normalized/combinatorial/random-walk
//!   Laplacians, shifted Laplacians, and exact $O(|E|)$ [`SpectralMomentStats`].
//! - [`shortest_path`]: Zero-per-call-alloc [`BfsWorkspace`] and [`DijkstraWorkspace`], plus
//!   Core-First MaxMin shortest-path pivot selection [`select_maxmin_sp_pivots`].
//! - [`chebyshev`]: Overflow-safe Miller Bessel $e^{-t}I_k(t)$, DCT-II $\sqrt{h}$ coefficients,
//!   R1/R2 PSD-projected regularized DoG filtering, zero-alloc [`ChebyshevWorkspace`], and
//!   [`ChebyshevDogKernel`] with deterministic JL embeddings.
//! - [`eigen`]: Cyclic [`symmetric_eigen_jacobi`] and SpMV-only thick-restart [`LanczosEigen`]
//!   with two-pass MGS reorthogonalization via [`smallest_sym_laplacian_eigenpairs`].

pub mod chebyshev;
pub mod csr_matrix;
pub mod eigen;
pub mod shortest_path;

pub use chebyshev::{
    ChebyshevDogConfig, ChebyshevDogKernel, ChebyshevWorkspace, adaptive_chebyshev_degree,
    adaptive_chebyshev_degree_scaled, build_dog_terms, build_regularized_dog_terms,
    compute_chebyshev_bandpass_coefficients, compute_chebyshev_sqrt_filter_coefficients,
    exp_scaled_bessel_i, exp_scaled_bessel_i_all, project_psd_spectral_filter,
};
pub use csr_matrix::{
    AnisotropicLaplacianConfig, AnisotropicLaplacianResult, CsrMatrix,
    DEFAULT_SPECTRUM_LANCZOS_SAFETY_FACTOR, DEFAULT_SPECTRUM_LANCZOS_STEPS,
    DEFAULT_SPECTRUM_POWER_ITERS, DEFAULT_SPECTRUM_SAFETY_FACTOR, LaplacianKind,
    SpectralMomentStats, anisotropic_laplacian, combinatorial_laplacian, laplacian,
    random_walk_laplacian, shifted_normalized_laplacian, spectral_moment_stats,
    symmetric_normalized_laplacian,
};
pub use eigen::{
    EigenResult, LanczosEigen, SubspaceEigen, smallest_sym_laplacian_eigenpairs,
    symmetric_eigen_jacobi,
};
pub use shortest_path::{BfsWorkspace, DijkstraWorkspace, select_maxmin_sp_pivots};

/// Returns the crate version string.
pub fn crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
