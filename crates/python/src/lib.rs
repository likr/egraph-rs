#![warn(missing_docs)]

//! Python bindings for the egraph-rs library.
//!
//! This module provides a Python interface to the egraph-rs library, which is a collection
//! of graph visualization and layout algorithms implemented in Rust. The library is designed
//! to provide efficient implementations of graph algorithms that can be used from Python.
//!
//! # Module Organization
//!
//! - `csr_graph`: `CsrGraph` CSR graph bindings backed by `egraph-core` and `egraph-ffi`
//! - `layout_buffers`: `Layout2d` interleaved zero-copy coordinate buffer bindings
//! - `layout_nd`: `LayoutNd` arbitrary-dimensional coordinate buffer bindings
//! - `metrics`: Layout quality evaluation metrics backed by `egraph-metrics`
//! - `spatial_dataset`: Delaunay triangulation, rectangle overlap, and dataset loader bindings
//! - `linalg`: Sparse matrix, spectral initialization, Chebyshev diffusion, and AMG-PCG bindings
//! - `neighborhood`: Neighborhood graph construction, perplexity, probability, and DoG bindings
//! - `ffi_convert` / `array_extract` / `solver_common`: Shared FFI conversion and dual-precision solver helpers
//! - `graph`: Graph data structures (Graph, DiGraph)
//! - `drawing`: Drawing spaces (Euclidean, Hyperbolic, Spherical, Torus)
//! - `distance_matrix`: Distance matrix operations
//! - `layout`: Layout algorithms (SGD, MDS, Kamada-Kawai, etc.)
//! - `algorithm`: Graph algorithms (shortest path)
//! - `clustering`: Community detection algorithms (Louvain, Label Propagation, etc.)
//! - `quality_metrics`: Layout quality evaluation metrics
//! - `rng`: Random number generation utilities

use pyo3::prelude::*;

mod algorithm;
mod array;
mod array_extract;
mod clustering;
mod csr_graph;
mod distance_matrix;
mod drawing;
mod ffi_convert;
mod graph;
mod layout;
mod layout_buffers;
mod layout_nd;
mod linalg;
mod metrics;
mod neighborhood;
mod quality_metrics;
mod rng;
mod solver_common;
mod spatial_dataset;
// Temporary module declarations retained until solver, community, and baseline modules are split.
mod v6_m3;
mod v6_m4;
mod v6_m5;
mod v6_w1;
mod v6_w2;

/// Standard floating-point precision type used in python bindings.
pub type FloatType = f64;

/// Converts a Cargo SemVer string (e.g. `6.0.0-alpha.6`) into PEP 440 format (`6.0.0a6`).
fn pep440_version(semver: &str) -> String {
    let base = semver.split('+').next().unwrap_or(semver);
    if let Some((core, pre)) = base.split_once('-') {
        let converted = if let Some(rest) = pre.strip_prefix("alpha.") {
            format!("a{rest}")
        } else if let Some(rest) = pre.strip_prefix("beta.") {
            format!("b{rest}")
        } else if let Some(rest) = pre.strip_prefix("rc.") {
            format!("rc{rest}")
        } else if let Some(rest) = pre.strip_prefix("dev.") {
            format!(".dev{rest}")
        } else {
            format!(".{pre}")
        };
        format!("{core}{converted}")
    } else {
        base.to_owned()
    }
}

/// Creates and initializes the `egraph._core` Python extension module.
///
/// Registers core `CsrGraph`, `Layout2d`, `LayoutNd`, metrics, spatial, linear algebra,
/// neighborhood, clustering, and layout solver bindings alongside legacy graph drawing modules.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", pep440_version(env!("CARGO_PKG_VERSION")))?;
    csr_graph::register(m)?;
    layout_buffers::register(m)?;
    layout_nd::register(m)?;
    metrics::register(m)?;
    spatial_dataset::register(m)?;
    linalg::register(m)?;
    neighborhood::register(m)?;
    v6_m3::register_m3(m)?;
    v6_m4::register_m4(m)?;
    v6_m5::register_m5(m)?;
    v6_w1::register_w1(m)?;
    v6_w2::register_w2(m)?;
    graph::register(m)?;
    drawing::register(m)?;
    distance_matrix::register(m)?;
    rng::register(m)?;
    layout::register(m)?;
    algorithm::register(m)?;
    quality_metrics::register(m)?;
    clustering::register(m)?;
    array::register(m)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::pep440_version;

    #[test]
    fn test_pep440_version_conversion() {
        assert_eq!(pep440_version("6.0.0-alpha.6"), "6.0.0a6");
        assert_eq!(pep440_version("6.0.0-alpha.7"), "6.0.0a7");
        assert_eq!(pep440_version("6.0.0-beta.2"), "6.0.0b2");
        assert_eq!(pep440_version("6.0.0-rc.1"), "6.0.0rc1");
        assert_eq!(pep440_version("6.0.0-dev.3"), "6.0.0.dev3");
        assert_eq!(pep440_version("6.0.0"), "6.0.0");
        assert_eq!(pep440_version("6.0.0+build42"), "6.0.0");
    }
}
