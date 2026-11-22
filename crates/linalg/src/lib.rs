//! Single-threaded, pure-Rust linear algebra, spectral diffusion, and pivot primitives for `egraph-rs`.
//!
//! `egraph-linalg` provides flat `CsrGraph`-based workspaces targeting Native Rust, Python (PyO3),
//! and WebAssembly (`wasm32-unknown-unknown`) without BLAS/LAPACK FFI or threading dependencies.

/// Returns the crate version string.
pub fn crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
