//! Single-threaded, pure-Rust linear algebra, spectral diffusion, and pivot primitives for `egraph-rs`.
//!
//! `egraph-linalg` provides flat `CsrGraph`-based workspaces and matrix-free [`LinearOperator`]
//! abstractions targeting Native Rust, Python (PyO3), and WebAssembly (`wasm32-unknown-unknown`)
//! without BLAS/LAPACK FFI or threading dependencies:
//! - Sparse CSR matrices & Laplacians: `f64` [`CsrMatrix`] with SpMV/SpMM, normalized,
//!   combinatorial, random-walk, and anisotropic Laplacians, shifted Laplacians, and exact
//!   $O(|E|)$ [`SpectralMomentStats`].
//! - Shortest-path workspaces & pivots: zero-per-call-alloc [`BfsWorkspace`] and
//!   [`DijkstraWorkspace`], plus Core-First MaxMin shortest-path pivot selection
//!   [`select_maxmin_sp_pivots`].
//! - Chebyshev spectral filtering & DoG kernels: overflow-safe Miller Bessel $e^{-t}I_k(t)$,
//!   DCT-II $\sqrt{h}$ coefficients, R1/R2 PSD-projected regularized DoG filtering, zero-alloc
//!   [`ChebyshevWorkspace`], and [`ChebyshevDogKernel`] with deterministic JL embeddings.
//! - Eigensolvers: cyclic [`symmetric_eigen_jacobi`] and SpMV-only thick-restart [`LanczosEigen`]
//!   with two-pass MGS reorthogonalization via [`LinearOperator`] and
//!   [`smallest_sym_laplacian_eigenpairs`].

mod chebyshev;
mod csr_matrix;
mod eigen;
mod operator;
mod shortest_path;

pub use chebyshev::{
    ChebyshevDogConfig, ChebyshevDogKernel, ChebyshevWorkspace, adaptive_chebyshev_degree,
    adaptive_chebyshev_degree_scaled, build_dog_terms, build_regularized_dog_terms,
    compute_chebyshev_bandpass_coefficients, compute_chebyshev_sqrt_filter_coefficients,
    effective_chebyshev_len, exp_scaled_bessel_i, exp_scaled_bessel_i_all,
    project_psd_spectral_filter,
};
pub use csr_matrix::{
    AnisotropicLaplacianConfig, AnisotropicLaplacianResult, CsrMatrix,
    DEFAULT_SPECTRUM_LANCZOS_SAFETY_FACTOR, DEFAULT_SPECTRUM_LANCZOS_STEPS,
    DEFAULT_SPECTRUM_POWER_ITERS, DEFAULT_SPECTRUM_SAFETY_FACTOR, LaplacianKind,
    SpectralMomentStats, anisotropic_laplacian, combinatorial_laplacian,
    estimate_lambda_max_operator, laplacian, random_walk_laplacian, shifted_normalized_laplacian,
    spectral_moment_stats, symmetric_normalized_laplacian,
};
pub use eigen::{
    EigenResult, LanczosEigen, SubspaceEigen, smallest_eigenpairs_from_inverse_operator,
    smallest_sym_laplacian_eigenpairs, smallest_sym_laplacian_eigenpairs_with_config,
    symmetric_eigen_jacobi,
};
pub use operator::{FnOperator, LinearOperator};
pub use shortest_path::{
    BfsWorkspace, DijkstraWorkspace, all_pairs_dijkstra, all_pairs_dijkstra_into,
    dijkstra_single_source, select_maxmin_sp_pivots,
};
