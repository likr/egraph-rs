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
//! - [`mds`]: Isotropic [`normalize_layout_2d`], normalized [`pivot_mds`], Koren spectral
//!   initialization [`koren_spectral_init`], zero-guarded [`RdMds`], and [`classical_mds`].

pub mod chebyshev;
pub mod csr_matrix;
pub mod eigen;
pub mod mds;
pub mod shortest_path;

pub use chebyshev::{
    ChebyshevDogConfig, ChebyshevDogKernel, ChebyshevWorkspace, adaptive_chebyshev_degree,
    build_dog_terms, build_regularized_dog_terms, compute_chebyshev_bandpass_coefficients,
    compute_chebyshev_sqrt_filter_coefficients, exp_scaled_bessel_i, exp_scaled_bessel_i_all,
    project_psd_spectral_filter,
};
pub use csr_matrix::{
    CsrMatrix, LaplacianKind, SpectralMomentStats, combinatorial_laplacian, laplacian,
    random_walk_laplacian, shifted_normalized_laplacian, spectral_moment_stats,
    symmetric_normalized_laplacian,
};
pub use eigen::{
    EigenResult, LanczosEigen, SubspaceEigen, smallest_sym_laplacian_eigenpairs,
    symmetric_eigen_jacobi,
};
pub use mds::{
    PivotMdsConfig, RdMds, classical_mds, classical_mds_2d, classical_mds_from_distances,
    koren_spectral_init, normalize_layout_2d, pivot_mds, pivot_mds_from_distances,
    pivot_mds_with_pivots, rd_mds, representative_mds,
};
pub use shortest_path::{BfsWorkspace, DijkstraWorkspace, select_maxmin_sp_pivots};

/// Returns the crate version string.
pub fn crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
