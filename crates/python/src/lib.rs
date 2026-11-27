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
//! - `linalg`: Sparse matrix, spectral initialization, Chebyshev diffusion, AMG-PCG, and HDE bindings
//! - `neighborhood`: Neighborhood graph construction, perplexity, probability, and DoG bindings
//! - `community`: Clustering, community detection, coarsening, and centroid prolongation bindings
//! - `manifold`: Non-Euclidean `ManifoldLayout2d` and geodesic metric bindings
//! - `solvers`: Cooperative layout solvers, overlap removal, edge bundling, and density field bindings
//! - `constraints_layering`: Separation constraint, 1D/2D overlap projection, and layering bindings
//! - `ffi_convert` / `array_extract` / `solver_common`: Shared FFI conversion and dual-precision solver helpers

use pyo3::prelude::*;

mod array_extract;
mod community;
mod constraints_declutter;
mod constraints_layering;
mod csr_graph;
mod csr_helpers;
mod dataset;
mod dynamic_kdtree;
mod ffi_convert;
mod layout_buffers;
mod layout_nd;
mod linalg;
mod manifold;
mod metrics;
mod neighborhood;
mod solver_common;
mod solvers;
mod spatial_dataset;
mod spatial_trees;

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
/// Registers core `CsrGraph`, `Layout2d`, `LayoutNd`, `ManifoldLayout2d`, metrics, spatial,
/// linear algebra, neighborhood, clustering, and layout solver bindings.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", pep440_version(env!("CARGO_PKG_VERSION")))?;
    csr_graph::register(m)?;
    layout_buffers::register(m)?;
    layout_nd::register(m)?;
    metrics::register(m)?;
    spatial_dataset::register(m)?;
    spatial_trees::register(m)?;
    dynamic_kdtree::register(m)?;
    dataset::register(m)?;
    linalg::register(m)?;
    neighborhood::register(m)?;
    community::register(m)?;
    manifold::register(m)?;
    solvers::register(m)?;
    constraints_layering::register(m)?;
    constraints_declutter::register(m)?;
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
