//! Neighborhood preservation, proximity graphs, and geometric aesthetic metric bindings.

use egraph::metrics as egraph_metrics;
use numpy::PyArray2;
use pyo3::prelude::*;

use crate::csr_graph::{PyCsrGraph, extract_flat_edges};
use crate::ffi_convert::{
    StrictF64, StrictU64, StrictUsize, core_error_to_py, pairs_to_u32_array2,
};
use crate::layout_buffers::{Layout2dInner, PyLayout2d, dispatch_layout2d, layout_f32_to_f64};

pub(crate) fn csr_neighborhood_preservation_1hop_impl(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
) -> PyResult<f64> {
    let g = graph.inner();
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::neighborhood_preservation_1hop(g, &soa)
    })
}

pub(crate) fn csr_neighborhood_preservation_2hop_impl(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
) -> PyResult<f64> {
    let g = graph.inner();
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::neighborhood_preservation_2hop(g, &soa)
    })
}

pub(crate) fn csr_gabriel_graph_property_impl(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
) -> PyResult<f64> {
    let g = graph.inner();
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::gabriel_graph_property(g, &soa)
    })
}

pub(crate) fn csr_node_resolution_impl(py: Python<'_>, layout: &PyLayout2d) -> PyResult<f64> {
    dispatch_layout2d!(py, layout, |soa| egraph_metrics::node_resolution(&soa))
}

pub(crate) fn csr_aspect_ratio_impl(py: Python<'_>, layout: &PyLayout2d) -> PyResult<f64> {
    dispatch_layout2d!(py, layout, |soa| egraph_metrics::aspect_ratio(&soa))
}

pub(crate) fn csr_angular_resolution_impl(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
) -> PyResult<f64> {
    let g = graph.inner();
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::angular_resolution(g, &soa)
    })
}

pub(crate) fn csr_ideal_edge_lengths_impl(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
) -> PyResult<f64> {
    let g = graph.inner();
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::ideal_edge_lengths(g, &soa)
    })
}

/// Compute 1-hop neighborhood preservation (`NP1`) in `[0.0, 1.0]` backed by `StaticKdTree`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
fn neighborhood_preservation_1hop(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
) -> PyResult<f64> {
    csr_neighborhood_preservation_1hop_impl(py, graph, layout)
}

/// Compute 2-hop neighborhood preservation (`NP2`) in `[0.0, 1.0]` backed by `StaticKdTree`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
fn neighborhood_preservation_2hop_v6(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
) -> PyResult<f64> {
    csr_neighborhood_preservation_2hop_impl(py, graph, layout)
}

/// Estimate 1-hop neighborhood preservation (`NP1`) in `[0.0, 1.0]` via deterministic vertex sampling.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (graph, layout, *, num_samples = StrictUsize(4096), seed = StrictU64(42)),
    text_signature = "(graph, layout, *, num_samples=4096, seed=42)"
)]
fn sampled_np1(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
    num_samples: StrictUsize,
    seed: StrictU64,
) -> PyResult<f64> {
    let g = graph.inner();
    let samples = num_samples.0;
    let seed_val = seed.0;
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::sampled_neighborhood_preservation_1hop(g, &soa, samples, seed_val)
    })
}

/// Estimate 2-hop neighborhood preservation (`NP2`) in `[0.0, 1.0]` via deterministic vertex sampling.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (graph, layout, *, num_samples = StrictUsize(4096), seed = StrictU64(42)),
    text_signature = "(graph, layout, *, num_samples=4096, seed=42)"
)]
fn sampled_np2(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
    num_samples: StrictUsize,
    seed: StrictU64,
) -> PyResult<f64> {
    let g = graph.inner();
    let samples = num_samples.0;
    let seed_val = seed.0;
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::sampled_neighborhood_preservation_2hop(g, &soa, samples, seed_val)
    })
}

/// Compute the undirected angle in `[0.0, pi]` radians between 2D vectors `(x1, y1)` and `(x2, y2)`.
///
/// Returns `None` if either vector is non-finite or has zero length.
#[pyfunction]
fn edge_angle(x1: StrictF64, y1: StrictF64, x2: StrictF64, y2: StrictF64) -> Option<f64> {
    egraph_metrics::edge_angle(x1.0, y1.0, x2.0, y2.0)
}

/// Compute PRISM layout-preservation distortion `(sigma_dist, sigma_disp)` between
/// `original_layout` and `adjusted_layout`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
fn prism_distortion(
    py: Python<'_>,
    original_layout: &PyLayout2d,
    adjusted_layout: &PyLayout2d,
) -> PyResult<(f64, f64)> {
    let res = py
        .detach(
            || match (original_layout.inner(), adjusted_layout.inner()) {
                (Layout2dInner::F64(orig), Layout2dInner::F64(adj)) => {
                    let o = orig.to_layout2d();
                    let a = adj.to_layout2d();
                    egraph_metrics::prism_distortion(&o, &a)
                }
                (Layout2dInner::F32(orig), Layout2dInner::F32(adj)) => {
                    let o = orig.to_layout2d();
                    let a = adj.to_layout2d();
                    egraph_metrics::prism_distortion(&o, &a)
                }
                (Layout2dInner::F64(orig), Layout2dInner::F32(adj)) => {
                    let o = orig.to_layout2d();
                    let a = layout_f32_to_f64(&adj.to_layout2d());
                    egraph_metrics::prism_distortion(&o, &a)
                }
                (Layout2dInner::F32(orig), Layout2dInner::F64(adj)) => {
                    let o = layout_f32_to_f64(&orig.to_layout2d());
                    let a = adj.to_layout2d();
                    egraph_metrics::prism_distortion(&o, &a)
                }
            },
        )
        .map_err(core_error_to_py)?;
    Ok((res.sigma_dist, res.sigma_disp))
}

/// Compute Eades et al. (2017) unweighted shape-based Relative Neighborhood Graph preservation (`S_RNG`) in `[0.0, 1.0]`.
///
/// Exposed as `drng_preservation` for backwards compatibility; distinct from Hong et al. (2022)
/// degree-sensitive `dRNG` and Fang et al. (2025) degree-constrained `dcRNG` (deferred pending
/// primary specification PDFs). Releases the Python GIL during computation.
#[pyfunction]
fn drng_preservation(py: Python<'_>, graph: &PyCsrGraph, layout: &PyLayout2d) -> PyResult<f64> {
    let g = graph.inner();
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::relative_neighborhood_preservation(g, &soa)
    })
}

/// Compute Eades et al. (2017) unweighted shape-based Gabriel Graph preservation (`S_GG`) in `[0.0, 1.0]`.
///
/// Exposed as `dgg_preservation` for backwards compatibility; distinct from Hong et al. (2022)
/// degree-sensitive `dGG` and Fang et al. (2025) degree-constrained `dcGG` (deferred pending
/// primary specification PDFs). Releases the Python GIL during computation.
#[pyfunction]
fn dgg_preservation(py: Python<'_>, graph: &PyCsrGraph, layout: &PyLayout2d) -> PyResult<f64> {
    let g = graph.inner();
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::gabriel_graph_preservation(g, &soa)
    })
}

/// Compute Gabriel graph property violation sum over `graph` and `layout`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
fn gabriel_graph_property_v6(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
) -> PyResult<f64> {
    csr_gabriel_graph_property_impl(py, graph, layout)
}

/// Compute hinge node-resolution violation penalty for `layout`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
fn node_resolution_v6(py: Python<'_>, layout: &PyLayout2d) -> PyResult<f64> {
    csr_node_resolution_impl(py, layout)
}

/// Compute PCA singular-value aspect ratio `sigma_2 / sigma_1` in `[0.0, 1.0]` for `layout`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
fn aspect_ratio_v6(py: Python<'_>, layout: &PyLayout2d) -> PyResult<f64> {
    csr_aspect_ratio_impl(py, layout)
}

/// Compute exponential angular-resolution penalty across incident edge pairs at each vertex.
///
/// Releases the Python GIL during computation.
#[pyfunction]
fn angular_resolution_v6(py: Python<'_>, graph: &PyCsrGraph, layout: &PyLayout2d) -> PyResult<f64> {
    csr_angular_resolution_impl(py, graph, layout)
}

/// Compute relative squared ideal edge-length error using edge weights (or `1.0` if unweighted).
///
/// Releases the Python GIL during computation.
#[pyfunction]
fn ideal_edge_lengths_v6(py: Python<'_>, graph: &PyCsrGraph, layout: &PyLayout2d) -> PyResult<f64> {
    csr_ideal_edge_lengths_impl(py, graph, layout)
}

/// Evaluate relative squared ideal edge-length error against uniform `unit_edge_length`.
#[pyfunction]
#[pyo3(
    signature = (graph, layout, unit_edge_length = StrictF64(1.0)),
    text_signature = "(graph, layout, unit_edge_length=1.0)"
)]
fn ideal_edge_lengths_with_unit(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
    unit_edge_length: StrictF64,
) -> PyResult<f64> {
    let g = graph.inner();
    let unit = unit_edge_length.0;
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::ideal_edge_lengths_with_unit(g, &soa, unit)
    })
}

/// Extract exact undirected Gabriel Graph edges `(u, v)` (`u < v`) as a `(E, 2)` `uint32` NumPy array.
#[pyfunction]
#[pyo3(signature = (layout), text_signature = "(layout)")]
fn gabriel_graph_edges<'py>(
    py: Python<'py>,
    layout: &PyLayout2d,
) -> PyResult<Bound<'py, PyArray2<u32>>> {
    let edges = dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::gabriel_graph_edges(&soa)
    })?;
    pairs_to_u32_array2(py, edges)
}

/// Extract exact undirected Relative Neighborhood Graph edges `(u, v)` (`u < v`) as a `(E, 2)` `uint32` NumPy array.
#[pyfunction]
#[pyo3(signature = (layout), text_signature = "(layout)")]
fn relative_neighborhood_edges<'py>(
    py: Python<'py>,
    layout: &PyLayout2d,
) -> PyResult<Bound<'py, PyArray2<u32>>> {
    let edges = dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::relative_neighborhood_edges(&soa)
    })?;
    pairs_to_u32_array2(py, edges)
}

/// Compute PRISM layout-preservation distortion `(sigma_dist, sigma_disp)` using explicit proximity edges.
#[pyfunction]
#[pyo3(
    signature = (original_layout, adjusted_layout, edges),
    text_signature = "(original_layout, adjusted_layout, edges)"
)]
fn prism_distortion_with_edges(
    py: Python<'_>,
    original_layout: &PyLayout2d,
    adjusted_layout: &PyLayout2d,
    edges: &Bound<'_, PyAny>,
) -> PyResult<(f64, f64)> {
    let flat = extract_flat_edges(edges)?;
    let (pair_chunks, _) = flat.as_chunks::<2>();
    let pairs: Vec<(u32, u32)> = pair_chunks.iter().map(|&[u, v]| (u, v)).collect();
    let res = py
        .detach(
            || match (original_layout.inner(), adjusted_layout.inner()) {
                (Layout2dInner::F64(orig), Layout2dInner::F64(adj)) => {
                    let o = orig.to_layout2d();
                    let a = adj.to_layout2d();
                    egraph_metrics::prism_distortion_with_edges(&o, &a, &pairs)
                }
                (Layout2dInner::F32(orig), Layout2dInner::F32(adj)) => {
                    let o = orig.to_layout2d();
                    let a = adj.to_layout2d();
                    egraph_metrics::prism_distortion_with_edges(&o, &a, &pairs)
                }
                (Layout2dInner::F64(orig), Layout2dInner::F32(adj)) => {
                    let o = orig.to_layout2d();
                    let a = layout_f32_to_f64(&adj.to_layout2d());
                    egraph_metrics::prism_distortion_with_edges(&o, &a, &pairs)
                }
                (Layout2dInner::F32(orig), Layout2dInner::F64(adj)) => {
                    let o = layout_f32_to_f64(&orig.to_layout2d());
                    let a = adj.to_layout2d();
                    egraph_metrics::prism_distortion_with_edges(&o, &a, &pairs)
                }
            },
        )
        .map_err(core_error_to_py)?;
    Ok((res.sigma_dist, res.sigma_disp))
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(neighborhood_preservation_1hop, m)?)?;
    m.add_function(wrap_pyfunction!(neighborhood_preservation_2hop_v6, m)?)?;
    m.add_function(wrap_pyfunction!(sampled_np1, m)?)?;
    m.add_function(wrap_pyfunction!(sampled_np2, m)?)?;
    m.add_function(wrap_pyfunction!(edge_angle, m)?)?;
    m.add_function(wrap_pyfunction!(prism_distortion, m)?)?;
    m.add_function(wrap_pyfunction!(prism_distortion_with_edges, m)?)?;
    m.add_function(wrap_pyfunction!(drng_preservation, m)?)?;
    m.add_function(wrap_pyfunction!(dgg_preservation, m)?)?;
    m.add_function(wrap_pyfunction!(gabriel_graph_property_v6, m)?)?;
    m.add_function(wrap_pyfunction!(node_resolution_v6, m)?)?;
    m.add_function(wrap_pyfunction!(aspect_ratio_v6, m)?)?;
    m.add_function(wrap_pyfunction!(angular_resolution_v6, m)?)?;
    m.add_function(wrap_pyfunction!(ideal_edge_lengths_v6, m)?)?;
    m.add_function(wrap_pyfunction!(ideal_edge_lengths_with_unit, m)?)?;
    m.add_function(wrap_pyfunction!(gabriel_graph_edges, m)?)?;
    m.add_function(wrap_pyfunction!(relative_neighborhood_edges, m)?)?;
    Ok(())
}
