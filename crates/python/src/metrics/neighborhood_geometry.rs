//! Neighborhood preservation and proximity graph metric bindings across 2D, N-D, and manifolds.

use egraph::metrics as egraph_metrics;
use numpy::PyArray2;
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{StrictU64, StrictUsize, pairs_to_u32_array2};

use super::nd_manifold_helpers::{MetricLayoutInput, dispatch_owned_metric_layout};

/// Compute 1-hop neighborhood preservation (`NP1`) in `[0.0, 1.0]`.
///
/// Supports `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (graph, layout), text_signature = "(graph, layout)")]
fn neighborhood_preservation_1hop(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: MetricLayoutInput<'_>,
) -> PyResult<f64> {
    let owned = layout.extract_owned()?;
    let g = graph.inner();
    dispatch_owned_metric_layout(
        py,
        owned,
        |l| egraph_metrics::neighborhood_preservation_1hop(&*g, l),
        |l| egraph_metrics::neighborhood_preservation_1hop(&*g, l),
        |l| egraph_metrics::neighborhood_preservation_1hop_nd(&*g, l),
        |l| egraph_metrics::neighborhood_preservation_1hop_nd(&*g, l),
        |l| egraph_metrics::neighborhood_preservation_1hop_manifold(&*g, l),
        |l| egraph_metrics::neighborhood_preservation_1hop_manifold(&*g, l),
    )
}

/// Compute 2-hop neighborhood preservation (`NP2`) in `[0.0, 1.0]`.
///
/// Supports `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (graph, layout), text_signature = "(graph, layout)")]
fn neighborhood_preservation_2hop(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: MetricLayoutInput<'_>,
) -> PyResult<f64> {
    let owned = layout.extract_owned()?;
    let g = graph.inner();
    dispatch_owned_metric_layout(
        py,
        owned,
        |l| egraph_metrics::neighborhood_preservation_2hop(&*g, l),
        |l| egraph_metrics::neighborhood_preservation_2hop(&*g, l),
        |l| egraph_metrics::neighborhood_preservation_2hop_nd(&*g, l),
        |l| egraph_metrics::neighborhood_preservation_2hop_nd(&*g, l),
        |l| egraph_metrics::neighborhood_preservation_2hop_manifold(&*g, l),
        |l| egraph_metrics::neighborhood_preservation_2hop_manifold(&*g, l),
    )
}

/// Estimate 1-hop neighborhood preservation (`NP1`) in `[0.0, 1.0]` via deterministic vertex sampling.
///
/// Supports `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`. Releases the Python GIL during computation.
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
    let samples = num_samples.0;
    let seed_val = seed.0;
    dispatch_owned_metric_layout(
        py,
        owned,
        |l| egraph_metrics::sampled_neighborhood_preservation_1hop(&*g, l, samples, seed_val),
        |l| egraph_metrics::sampled_neighborhood_preservation_1hop(&*g, l, samples, seed_val),
        |l| egraph_metrics::sampled_neighborhood_preservation_1hop_nd(&*g, l, samples, seed_val),
        |l| egraph_metrics::sampled_neighborhood_preservation_1hop_nd(&*g, l, samples, seed_val),
        |l| {
            egraph_metrics::sampled_neighborhood_preservation_1hop_manifold(
                &*g, l, samples, seed_val,
            )
        },
        |l| {
            egraph_metrics::sampled_neighborhood_preservation_1hop_manifold(
                &*g, l, samples, seed_val,
            )
        },
    )
}

/// Estimate 2-hop neighborhood preservation (`NP2`) in `[0.0, 1.0]` via deterministic vertex sampling.
///
/// Supports `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`. Releases the Python GIL during computation.
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
    let samples = num_samples.0;
    let seed_val = seed.0;
    dispatch_owned_metric_layout(
        py,
        owned,
        |l| egraph_metrics::sampled_neighborhood_preservation_2hop(&*g, l, samples, seed_val),
        |l| egraph_metrics::sampled_neighborhood_preservation_2hop(&*g, l, samples, seed_val),
        |l| egraph_metrics::sampled_neighborhood_preservation_2hop_nd(&*g, l, samples, seed_val),
        |l| egraph_metrics::sampled_neighborhood_preservation_2hop_nd(&*g, l, samples, seed_val),
        |l| {
            egraph_metrics::sampled_neighborhood_preservation_2hop_manifold(
                &*g, l, samples, seed_val,
            )
        },
        |l| {
            egraph_metrics::sampled_neighborhood_preservation_2hop_manifold(
                &*g, l, samples, seed_val,
            )
        },
    )
}

/// Compute Eades et al. (2017) unweighted shape-based Relative Neighborhood Graph preservation (`S_RNG`) in `[0.0, 1.0]`.
///
/// Supports `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (graph, layout), text_signature = "(graph, layout)")]
fn drng_preservation(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: MetricLayoutInput<'_>,
) -> PyResult<f64> {
    let owned = layout.extract_owned()?;
    let g = graph.inner();
    dispatch_owned_metric_layout(
        py,
        owned,
        |l| egraph_metrics::relative_neighborhood_preservation(&*g, l),
        |l| egraph_metrics::relative_neighborhood_preservation(&*g, l),
        |l| egraph_metrics::relative_neighborhood_preservation_nd(&*g, l),
        |l| egraph_metrics::relative_neighborhood_preservation_nd(&*g, l),
        |l| egraph_metrics::relative_neighborhood_preservation_manifold(&*g, l),
        |l| egraph_metrics::relative_neighborhood_preservation_manifold(&*g, l),
    )
}

/// Compute Eades et al. (2017) unweighted shape-based Gabriel Graph preservation (`S_GG`) in `[0.0, 1.0]`.
///
/// Supports `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (graph, layout), text_signature = "(graph, layout)")]
fn dgg_preservation(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: MetricLayoutInput<'_>,
) -> PyResult<f64> {
    let owned = layout.extract_owned()?;
    let g = graph.inner();
    dispatch_owned_metric_layout(
        py,
        owned,
        |l| egraph_metrics::gabriel_graph_preservation(&*g, l),
        |l| egraph_metrics::gabriel_graph_preservation(&*g, l),
        |l| egraph_metrics::gabriel_graph_preservation_nd(&*g, l),
        |l| egraph_metrics::gabriel_graph_preservation_nd(&*g, l),
        |l| egraph_metrics::gabriel_graph_preservation_manifold(&*g, l),
        |l| egraph_metrics::gabriel_graph_preservation_manifold(&*g, l),
    )
}

/// Compute Gabriel graph property violation sum over `graph` and `layout`.
///
/// Supports `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (graph, layout), text_signature = "(graph, layout)")]
fn gabriel_graph_property(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: MetricLayoutInput<'_>,
) -> PyResult<f64> {
    let owned = layout.extract_owned()?;
    let g = graph.inner();
    dispatch_owned_metric_layout(
        py,
        owned,
        |l| egraph_metrics::gabriel_graph_property(&*g, l),
        |l| egraph_metrics::gabriel_graph_property(&*g, l),
        |l| egraph_metrics::gabriel_graph_property_nd(&*g, l),
        |l| egraph_metrics::gabriel_graph_property_nd(&*g, l),
        |l| egraph_metrics::gabriel_graph_property_manifold(&*g, l),
        |l| egraph_metrics::gabriel_graph_property_manifold(&*g, l),
    )
}

/// Compute hinge node-resolution violation penalty for `layout`.
///
/// Supports `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (layout), text_signature = "(layout)")]
fn node_resolution(py: Python<'_>, layout: MetricLayoutInput<'_>) -> PyResult<f64> {
    let owned = layout.extract_owned()?;
    dispatch_owned_metric_layout(
        py,
        owned,
        egraph_metrics::node_resolution,
        egraph_metrics::node_resolution,
        egraph_metrics::node_resolution_nd,
        egraph_metrics::node_resolution_nd,
        egraph_metrics::node_resolution_manifold,
        egraph_metrics::node_resolution_manifold,
    )
}

/// Extract exact undirected Gabriel Graph edges `(u, v)` (`u < v`) as a `(E, 2)` `uint32` NumPy array.
///
/// Supports `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (layout), text_signature = "(layout)")]
fn gabriel_graph_edges<'py>(
    py: Python<'py>,
    layout: MetricLayoutInput<'py>,
) -> PyResult<Bound<'py, PyArray2<u32>>> {
    let owned = layout.extract_owned()?;
    let edges = dispatch_owned_metric_layout(
        py,
        owned,
        egraph_metrics::gabriel_graph_edges,
        egraph_metrics::gabriel_graph_edges,
        egraph_metrics::gabriel_graph_edges_nd,
        egraph_metrics::gabriel_graph_edges_nd,
        egraph_metrics::gabriel_graph_edges_manifold,
        egraph_metrics::gabriel_graph_edges_manifold,
    )?;
    pairs_to_u32_array2(py, edges)
}

/// Extract exact undirected Relative Neighborhood Graph edges `(u, v)` (`u < v`) as a `(E, 2)` `uint32` NumPy array.
///
/// Supports `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (layout), text_signature = "(layout)")]
fn relative_neighborhood_edges<'py>(
    py: Python<'py>,
    layout: MetricLayoutInput<'py>,
) -> PyResult<Bound<'py, PyArray2<u32>>> {
    let owned = layout.extract_owned()?;
    let edges = dispatch_owned_metric_layout(
        py,
        owned,
        egraph_metrics::relative_neighborhood_edges,
        egraph_metrics::relative_neighborhood_edges,
        egraph_metrics::relative_neighborhood_edges_nd,
        egraph_metrics::relative_neighborhood_edges_nd,
        egraph_metrics::relative_neighborhood_edges_manifold,
        egraph_metrics::relative_neighborhood_edges_manifold,
    )?;
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
