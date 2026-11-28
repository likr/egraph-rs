//! Geometric aesthetic, PRISM distortion, and ideal edge-length metric bindings.

use egraph::metrics as egraph_metrics;
use pyo3::prelude::*;

use crate::array_extract::extract_f64_2d_raw;
use crate::csr_graph::{PyCsrGraph, extract_flat_edges};
use crate::errors::{core_error_to_py, dimension_mismatch_err};
use crate::ffi_convert::StrictF64;
use crate::layout_buffers::{Layout2dInner, PyLayout2d, dispatch_layout2d, layout_f32_to_f64};

use super::nd_manifold_helpers::{MetricLayoutInput, dispatch_owned_metric};

/// Compute the undirected angle in `[0.0, pi]` radians between 2D vectors `(x1, y1)` and `(x2, y2)`.
///
/// Returns `None` if either vector is non-finite or has zero length.
#[pyfunction]
#[pyo3(signature = (x1, y1, x2, y2), text_signature = "(x1, y1, x2, y2)")]
fn edge_angle(x1: StrictF64, y1: StrictF64, x2: StrictF64, y2: StrictF64) -> Option<f64> {
    egraph_metrics::edge_angle(x1.0, y1.0, x2.0, y2.0)
}

/// Compute PCA singular-value aspect ratio `sigma_2 / sigma_1` in `[0.0, 1.0]` for `layout`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (layout), text_signature = "(layout)")]
fn aspect_ratio(py: Python<'_>, layout: &PyLayout2d) -> PyResult<f64> {
    dispatch_layout2d!(py, layout, |soa| egraph_metrics::aspect_ratio(&soa))
}

/// Compute exponential angular-resolution penalty across incident edge pairs at each vertex.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (graph, layout), text_signature = "(graph, layout)")]
fn angular_resolution(py: Python<'_>, graph: &PyCsrGraph, layout: &PyLayout2d) -> PyResult<f64> {
    let g = graph.inner();
    dispatch_layout2d!(py, layout, |soa| egraph_metrics::angular_resolution(
        &*g, &soa
    ))
}

/// Compute PRISM layout-preservation distortion `(sigma_dist, sigma_disp)` between
/// `original_layout` and `adjusted_layout`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (original_layout, adjusted_layout),
    text_signature = "(original_layout, adjusted_layout)"
)]
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

/// Compute relative squared ideal edge-length error using edge weights (or `1.0` if unweighted).
///
/// Supports `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (graph, layout), text_signature = "(graph, layout)")]
fn ideal_edge_lengths(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: MetricLayoutInput<'_>,
) -> PyResult<f64> {
    let owned = layout.extract_owned()?;
    let g = graph.inner();
    dispatch_owned_metric!(py, owned, |l| egraph_metrics::ideal_edge_lengths(&*g, l))
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
    layout: MetricLayoutInput<'_>,
    unit_edge_length: StrictF64,
) -> PyResult<f64> {
    let owned = layout.extract_owned()?;
    let g = graph.inner();
    let unit = unit_edge_length.0;
    dispatch_owned_metric!(py, owned, |l| {
        egraph_metrics::ideal_edge_lengths_with_unit(&*g, l, unit)
    })
}

/// Evaluate ideal edge-length error scaled by `unit_edge_length` (using `weights * unit_edge_length`).
#[pyfunction]
#[pyo3(
    signature = (graph, layout, unit_edge_length = StrictF64(1.0)),
    text_signature = "(graph, layout, unit_edge_length=1.0)"
)]
fn ideal_edge_lengths_scaled(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: MetricLayoutInput<'_>,
    unit_edge_length: StrictF64,
) -> PyResult<f64> {
    let owned = layout.extract_owned()?;
    let g = graph.inner();
    let unit = unit_edge_length.0;
    dispatch_owned_metric!(py, owned, |l| {
        egraph_metrics::ideal_edge_lengths_scaled(&*g, l, unit)
    })
}

/// Evaluate relative squared ideal edge-length error using a `(N, N)` target distance matrix.
#[pyfunction]
#[pyo3(
    signature = (graph, layout, distances),
    text_signature = "(graph, layout, distances)"
)]
fn ideal_edge_lengths_from_distances(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: MetricLayoutInput<'_>,
    distances: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let (dists_flat, d_rows, d_cols) = extract_f64_2d_raw(distances, "distances")?;
    if d_rows != d_cols {
        return Err(dimension_mismatch_err(format!(
            "distances must be square (N, N), got ({d_rows}, {d_cols})"
        )));
    }
    let owned = layout.extract_owned()?;
    let g = graph.inner();
    dispatch_owned_metric!(py, owned, |l| {
        egraph_metrics::ideal_edge_lengths_from_distances(&*g, l, &dists_flat)
    })
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(edge_angle, m)?)?;
    m.add_function(wrap_pyfunction!(aspect_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(angular_resolution, m)?)?;
    m.add_function(wrap_pyfunction!(prism_distortion, m)?)?;
    m.add_function(wrap_pyfunction!(prism_distortion_with_edges, m)?)?;
    m.add_function(wrap_pyfunction!(ideal_edge_lengths, m)?)?;
    m.add_function(wrap_pyfunction!(ideal_edge_lengths_with_unit, m)?)?;
    m.add_function(wrap_pyfunction!(ideal_edge_lengths_scaled, m)?)?;
    m.add_function(wrap_pyfunction!(ideal_edge_lengths_from_distances, m)?)?;
    Ok(())
}
