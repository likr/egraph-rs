//! Pure-Rust dense symmetric Jacobi eigensolver and SpMV-only block thick-restart Lanczos
//! with two-pass Modified Gram-Schmidt reorthogonalization.

mod jacobi;
mod lanczos;
mod laplacian_eigen;
mod subspace;

pub use jacobi::symmetric_eigen_jacobi;
pub use lanczos::LanczosEigen;
pub use laplacian_eigen::{
    EigenResult, smallest_eigenpairs_from_inverse_operator, smallest_sym_laplacian_eigenpairs,
    smallest_sym_laplacian_eigenpairs_with_config,
};
pub use subspace::SubspaceEigen;

#[cfg(test)]
mod tests;
