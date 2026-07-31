#![warn(missing_docs)]

//! Python bindings for the egraph-rs library.
//!
//! This module provides a Python interface to the egraph-rs library, which is a collection
//! of graph visualization and layout algorithms implemented in Rust. The library is designed
//! to provide efficient implementations of graph algorithms that can be used from Python.
//!
//! # Module Organization
//!
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
mod clustering;
mod distance_matrix;
mod drawing;
mod graph;
mod layout;
mod quality_metrics;
mod rng;

/// Standard floating-point precision type used in python bindings.
pub type FloatType = f64;

/// Creates and initializes the egraph Python module.
///
/// This function is the main entry point for the Python module. It registers all submodules
/// and their associated classes and functions.
#[pymodule]
fn egraph(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
