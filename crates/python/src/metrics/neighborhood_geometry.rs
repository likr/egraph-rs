//! Neighborhood preservation and proximity graph metric bindings across 2D, 3D, N-D, and manifolds.

use egraph::metrics as egraph_metrics;
use numpy::PyArray2;
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{StrictU64, StrictUsize, pairs_to_u32_array2};

use super::nd_manifold_helpers::{MetricLayoutInput, dispatch_owned_metric};

/// Compute 1-hop neighborhood preservation (`NP1`) in `[0.0, 1.0]`.
#[pyfunction]
#[pyo3(signature = (graph, layout), text_signature = "(graph, layout)")]
fn neighborhood_preservation_1hop(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: MetricLayoutInput<'_>,
) -> PyResult<f64> {
    let owned = layout.extract_owned()?;
    let g = graph.inner();
    dispatch_owned_metric!(py, owned, |l| {
        egraph_metrics::neighborhood_preservation_1hop(&*g, l)
    })
}

/// Compute 2-hop neighborhood preservation (`NP2`) in `[0.0, 1.0]`.
#[pyfunction]
#[pyo3(signature = (graph, layout), text_signature = "(graph, layout)")]
fn neighborhood_preservation_2hop(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: MetricLayoutInput<'_>,
) -> PyResult<f64> {
    let owned = layout.extract_owned()?;
    let g = graph.inner();
    dispatch_owned_metric!(py, owned, |l| {
        egraph_metrics::neighborhood_preservation_2hop(&*g, l)
    })
}

/// Estimate 1-hop neighborhood preservation (`NP1`) in `[0.0, 1.0]` via deterministic vertex sampling.
#[pyfunction]
#[pyo3(
    signature = (graph, layout, *, num_samples = StrictUsize(4096), seed = StrictU64(42)),
    text_signature = "(graph, layout, *, num_samples=4096, seed=42)"
)]
fn sampled_np1(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: MetricLayoutInput<'_>,
    num_samples: StrictUsize,
    seed: StrictU64,
) -> PyResult<f64> {
    let owned = layout.extract_owned()?;
    let g = graph.inner();
    let (samples, seed_val) = (num_samples.0, seed.0);
    dispatch_owned_metric!(py, owned, |l| {
        egraph_metrics::sampled_neighborhood_preservation_1hop(&*g, l, samples, seed_val)
    })
}

/// Estimate 2-hop neighborhood preservation (`NP2`) in `[0.0, 1.0]` via deterministic vertex sampling.
#[pyfunction]
#[pyo3(
    signature = (graph, layout, *, num_samples = StrictUsize(4096), seed = StrictU64(42)),
    text_signature = "(graph, layout, *, num_samples=4096, seed=42)"
)]
fn sampled_np2(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: MetricLayoutInput<'_>,
    num_samples: StrictUsize,
    seed: StrictU64,
) -> PyResult<f64> {
    let owned = layout.extract_owned()?;
    let g = graph.inner();
    let (samples, seed_val) = (num_samples.0, seed.0);
    dispatch_owned_metric!(py, owned, |l| {
        egraph_metrics::sampled_neighborhood_preservation_2hop(&*g, l, samples, seed_val)
    })
}

/// Compute Eades et al. (2017) unweighted shape-based Relative Neighborhood Graph preservation (`S_RNG`).
#[pyfunction]
#[pyo3(signature = (graph, layout), text_signature = "(graph, layout)")]
fn drng_preservation(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: MetricLayoutInput<'_>,
) -> PyResult<f64> {
    let owned = layout.extract_owned()?;
    let g = graph.inner();
    dispatch_owned_metric!(py, owned, |l| {
        egraph_metrics::relative_neighborhood_preservation(&*g, l)
    })
}

/// Compute Eades et al. (2017) unweighted shape-based Gabriel Graph preservation (`S_GG`).
#[pyfunction]
#[pyo3(signature = (graph, layout), text_signature = "(graph, layout)")]
fn dgg_preservation(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: MetricLayoutInput<'_>,
) -> PyResult<f64> {
    let owned = layout.extract_owned()?;
    let g = graph.inner();
    dispatch_owned_metric!(py, owned, |l| {
        egraph_metrics::gabriel_graph_preservation(&*g, l)
    })
}

/// Compute Gabriel graph property violation sum over `graph` and `layout`.
#[pyfunction]
#[pyo3(signature = (graph, layout), text_signature = "(graph, layout)")]
fn gabriel_graph_property(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: MetricLayoutInput<'_>,
) -> PyResult<f64> {
    let owned = layout.extract_owned()?;
    let g = graph.inner();
    dispatch_owned_metric!(py, owned, |l| egraph_metrics::gabriel_graph_property(
        &*g, l
    ))
}

/// Compute hinge node-resolution violation penalty for `layout`.
#[pyfunction]
#[pyo3(signature = (layout), text_signature = "(layout)")]
fn node_resolution(py: Python<'_>, layout: MetricLayoutInput<'_>) -> PyResult<f64> {
    let owned = layout.extract_owned()?;
    dispatch_owned_metric!(py, owned, |l| egraph_metrics::node_resolution(l))
}

/// Extract exact undirected Gabriel Graph edges `(u, v)` (`u < v`) as a `(E, 2)` `uint32` NumPy array.
#[pyfunction]
#[pyo3(signature = (layout), text_signature = "(layout)")]
fn gabriel_graph_edges<'py>(
    py: Python<'py>,
    layout: MetricLayoutInput<'py>,
) -> PyResult<Bound<'py, PyArray2<u32>>> {
    let owned = layout.extract_owned()?;
    let edges = dispatch_owned_metric!(py, owned, |l| egraph_metrics::gabriel_graph_edges(l))?;
    pairs_to_u32_array2(py, edges)
}

/// Extract exact undirected Relative Neighborhood Graph edges `(u, v)` (`u < v`) as a `(E, 2)` `uint32` NumPy array.
#[pyfunction]
#[pyo3(signature = (layout), text_signature = "(layout)")]
fn relative_neighborhood_edges<'py>(
    py: Python<'py>,
    layout: MetricLayoutInput<'py>,
) -> PyResult<Bound<'py, PyArray2<u32>>> {
    let owned = layout.extract_owned()?;
    let edges = dispatch_owned_metric!(py, owned, |l| {
        egraph_metrics::relative_neighborhood_edges(l)
    })?;
    pairs_to_u32_array2(py, edges)
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(neighborhood_preservation_1hop, m)?)?;
    m.add_function(wrap_pyfunction!(neighborhood_preservation_2hop, m)?)?;
    m.add_function(wrap_pyfunction!(sampled_np1, m)?)?;
    m.add_function(wrap_pyfunction!(sampled_np2, m)?)?;
    m.add_function(wrap_pyfunction!(drng_preservation, m)?)?;
    m.add_function(wrap_pyfunction!(dgg_preservation, m)?)?;
    m.add_function(wrap_pyfunction!(gabriel_graph_property, m)?)?;
    m.add_function(wrap_pyfunction!(node_resolution, m)?)?;
    m.add_function(wrap_pyfunction!(gabriel_graph_edges, m)?)?;
    m.add_function(wrap_pyfunction!(relative_neighborhood_edges, m)?)?;
    Ok(())
}
