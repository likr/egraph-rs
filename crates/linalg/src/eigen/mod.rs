//! Pure-Rust dense symmetric Jacobi eigensolver and SpMV-only block thick-restart Lanczos
//! with two-pass Modified Gram-Schmidt reorthogonalization.

mod jacobi;
mod lanczos;
mod subspace;

pub use jacobi::symmetric_eigen_jacobi;
pub use lanczos::{EigenResult, LanczosEigen, smallest_sym_laplacian_eigenpairs};
pub use subspace::SubspaceEigen;

#[cfg(test)]
mod tests;
