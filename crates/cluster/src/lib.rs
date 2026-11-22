//! Single-threaded, pure-Rust graph clustering and coarsening primitives for `egraph-rs`.
//!
//! `egraph-clustering` provides deterministic clustering algorithms (`KMeans++`, `Spectral`,
//! `Louvain`, `LabelPropagation`, and coarsening) over flat `CsrGraph` and `egraph-linalg`
//! primitives across Native Rust, Python (PyO3), and WebAssembly targets.

/// Returns the crate version string.
pub fn crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::crate_version;

    #[test]
    fn m3_scaffold_ok() {
        assert!(!crate_version().is_empty());
    }
}
