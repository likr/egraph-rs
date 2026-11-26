//! Stress, edge-crossing, minimum-angle, degeneracy, and BFS shortest-path metric bindings.

use egraph::core::CoreResult;
use egraph::metrics as egraph_metrics;
use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict};

use crate::array_extract::extract_f64_2d_raw;
use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    StrictF64, StrictU64, StrictUsize, core_error_to_py, is_non_sequence_container,
};
use crate::layout_buffers::{PyLayout2d, dispatch_layout2d};

/// Count exact non-adjacent edge crossings in `graph` under `layout`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (graph, layout), text_signature = "(graph, layout)")]
fn crossing_number(py: Python<'_>, graph: &PyCsrGraph, layout: &PyLayout2d) -> PyResult<f64> {
    let g = graph.inner();
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::crossing_number(g, &soa).map(|c| c as f64)
    })
}

/// Compute the crossing-angle penalty `sum cos^2(theta)` over all crossing edge pairs.
///
/// Lower values indicate crossings closer to perpendicular (`0.0` = all perpendicular crossings).
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (graph, layout), text_signature = "(graph, layout)")]
fn crossing_angle(py: Python<'_>, graph: &PyCsrGraph, layout: &PyLayout2d) -> PyResult<f64> {
    let g = graph.inner();
    dispatch_layout2d!(py, layout, |soa| egraph_metrics::crossing_angle(g, &soa))
}

/// Return all non-adjacent crossing edge pairs in `graph` under `layout` as a list of dicts.
///
/// Each dictionary contains `"edge_a"`, `"edge_b"`, `"u1"`, `"v1"`, `"u2"`, `"v2"`,
/// `"angle_rad"`, and `"cos_sq"`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (graph, layout), text_signature = "(graph, layout)")]
fn crossing_edges(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
) -> PyResult<Vec<Py<PyDict>>> {
    let g = graph.inner();
    let pairs = dispatch_layout2d!(py, layout, |soa| egraph_metrics::crossing_edges(g, &soa))?;
    let mut out = Vec::with_capacity(pairs.len());
    for pair in pairs {
        let dict = PyDict::new(py);
        dict.set_item("edge_a", pair.edge_a)?;
        dict.set_item("edge_b", pair.edge_b)?;
        dict.set_item("u1", pair.u1)?;
        dict.set_item("v1", pair.v1)?;
        dict.set_item("u2", pair.u2)?;
        dict.set_item("v2", pair.v2)?;
        dict.set_item("angle_rad", pair.angle_rad)?;
        dict.set_item("cos_sq", pair.cos_sq)?;
        out.push(dict.unbind());
    }
    Ok(out)
}

fn extract_crossing_cos_sq(pairs: &Bound<'_, PyAny>, fn_name: &str) -> PyResult<Vec<f64>> {
    if is_non_sequence_container(pairs) {
        return Err(PyTypeError::new_err(format!(
            "{fn_name}: pairs must be a sequence or iterable of crossing-edge dicts"
        )));
    }
    let iter = pairs.try_iter().map_err(|_| {
        PyTypeError::new_err(format!(
            "{fn_name}: pairs must be a sequence or iterable of crossing-edge dicts"
        ))
    })?;
    let mut cos_sqs = Vec::new();
    for item_res in iter {
        let item = item_res?;
        let cos_sq_obj = item.get_item("cos_sq").map_err(|_| {
            PyTypeError::new_err(format!(
                "{fn_name}: each pair must be a mapping containing 'cos_sq'"
            ))
        })?;
        if cos_sq_obj.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(format!(
                "{fn_name}: 'cos_sq' must be float, not bool"
            )));
        }
        let val = cos_sq_obj
            .extract::<f64>()
            .map_err(|_| PyTypeError::new_err(format!("{fn_name}: 'cos_sq' must be numeric")))?;
        if !val.is_finite() {
            return Err(PyValueError::new_err(format!(
                "{fn_name}: 'cos_sq' must be finite"
            )));
        }
        cos_sqs.push(val);
    }
    Ok(cos_sqs)
}

/// Compute the edge crossing count directly from precomputed `crossing_edges(graph, layout)` pairs.
#[pyfunction]
#[pyo3(signature = (pairs), text_signature = "(pairs)")]
fn crossing_number_from_pairs(pairs: &Bound<'_, PyAny>) -> PyResult<f64> {
    let cos_sqs = extract_crossing_cos_sq(pairs, "crossing_number_from_pairs")?;
    Ok(cos_sqs.len() as f64)
}

/// Compute the crossing-angle penalty `sum cos^2(theta)` directly from precomputed `crossing_edges(graph, layout)` pairs.
///
/// Lower values indicate crossings closer to perpendicular (`0.0` = all perpendicular crossings).
#[pyfunction]
#[pyo3(signature = (pairs), text_signature = "(pairs)")]
fn crossing_angle_from_pairs(pairs: &Bound<'_, PyAny>) -> PyResult<f64> {
    let cos_sqs = extract_crossing_cos_sq(pairs, "crossing_angle_from_pairs")?;
    Ok(cos_sqs.into_iter().sum())
}

/// Compute closed-form Scale-Normalized Stress (`SNS`) in `[0.0, 1.0]` over `graph` and `layout`.
///
/// Uses unweighted BFS hop distances with inverse-squared weights `w_ij = 1 / d_ij^2` and
/// analytical optimal coordinate rescaling so uniform scaling leaves the metric invariant.
/// Releases the Python GIL during computation.
#[pyfunction]
fn scale_normalized_stress(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
) -> PyResult<f64> {
    let g = graph.inner();
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::scale_normalized_stress(g, &soa)
    })
}

/// Compute O(kN) degree-stratified pivot-sampled Scale-Normalized Stress (`SNS`) from `num_pivots` deterministic BFS sources.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (graph, layout, *, num_pivots = StrictUsize(64), seed = StrictU64(42)),
    text_signature = "(graph, layout, *, num_pivots=64, seed=42)"
)]
fn sampled_scale_normalized_stress(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
    num_pivots: StrictUsize,
    seed: StrictU64,
) -> PyResult<f64> {
    let g = graph.inner();
    let pivots = num_pivots.0;
    let seed_val = seed.0;
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::sampled_scale_normalized_stress(g, &soa, pivots, seed_val)
    })
}

/// Compute exact DRGraph Eq. (17) crosslessness `1 - sqrt(c / c_max)` in `[0.0, 1.0]`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (graph, layout), text_signature = "(graph, layout)")]
fn crosslessness(py: Python<'_>, graph: &PyCsrGraph, layout: &PyLayout2d) -> PyResult<f64> {
    let g = graph.inner();
    dispatch_layout2d!(py, layout, |soa| egraph_metrics::crosslessness(g, &soa))
}

/// Estimate DRGraph Eq. (17) crosslessness `1 - sqrt(c_hat / c_max)` in `[0.0, 1.0]` via edge-pair sampling.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (graph, layout, *, num_samples = StrictUsize(4096), seed = StrictU64(42)),
    text_signature = "(graph, layout, *, num_samples=4096, seed=42)"
)]
fn sampled_crosslessness(
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
        egraph_metrics::sampled_crosslessness(g, &soa, samples, seed_val)
    })
}

/// Compute DRGraph Eq. (19) strict minimum-angle resolution (`A2`) in `[0.0, 1.0]`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (graph, layout), text_signature = "(graph, layout)")]
fn minimum_angle(py: Python<'_>, graph: &PyCsrGraph, layout: &PyLayout2d) -> PyResult<f64> {
    let g = graph.inner();
    dispatch_layout2d!(py, layout, |soa| egraph_metrics::minimum_angle(g, &soa))
}

/// Compute DRGraph Eq. (19) minimum-angle resolution summary `{"a1": ..., "a2": ...}`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (graph, layout), text_signature = "(graph, layout)")]
fn minimum_angle_summary(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
) -> PyResult<Py<PyDict>> {
    let g = graph.inner();
    let summary = dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::minimum_angle_summary(g, &soa)
    })?;
    let dict = PyDict::new(py);
    dict.set_item("a1", summary.a1)?;
    dict.set_item("a2", summary.a2)?;
    Ok(dict.unbind())
}

/// Evaluate coordinate-collapse (`f32`/`f64` duplicate nodes), zero-length edge, and bbox degeneracies.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (graph, layout), text_signature = "(graph, layout)")]
fn layout_degeneracy(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
) -> PyResult<Py<PyDict>> {
    let g = graph.inner();
    let deg = dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::layout_degeneracy(g, &soa)
    })?;
    let dict = PyDict::new(py);
    dict.set_item("num_nodes", deg.num_nodes)?;
    dict.set_item("num_edges", deg.num_edges)?;
    dict.set_item("duplicate_nodes_f32", deg.duplicate_nodes_f32)?;
    dict.set_item("duplicate_ratio_f32", deg.duplicate_ratio_f32)?;
    dict.set_item("duplicate_nodes_f64", deg.duplicate_nodes_f64)?;
    dict.set_item("duplicate_ratio_f64", deg.duplicate_ratio_f64)?;
    dict.set_item("zero_length_edges", deg.zero_length_edges)?;
    dict.set_item("zero_length_edge_ratio", deg.zero_length_edge_ratio)?;
    dict.set_item("bbox_width", deg.bbox_width)?;
    dict.set_item("bbox_height", deg.bbox_height)?;
    dict.set_item("is_bbox_degenerate", deg.is_bbox_degenerate)?;
    Ok(dict.unbind())
}

/// Count exact non-adjacent edge crossings using the ``O((|E| + K) log |E|)`` sweep-line detector.
///
/// Releases the Python GIL during computation.
#[pyfunction]
fn sweep_crossing_number(py: Python<'_>, graph: &PyCsrGraph, layout: &PyLayout2d) -> PyResult<f64> {
    crossing_number(py, graph, layout)
}

/// Compute the sweep-line crossing-angle penalty `sum cos^2(theta)` over all crossing edge pairs.
///
/// Releases the Python GIL during computation.
#[pyfunction]
fn sweep_crossing_angle(py: Python<'_>, graph: &PyCsrGraph, layout: &PyLayout2d) -> PyResult<f64> {
    crossing_angle(py, graph, layout)
}

/// Estimate total edge crossings in O(num_samples) time via deterministic edge-pair sampling.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (graph, layout, *, num_samples = StrictUsize(4096), seed = StrictU64(42)),
    text_signature = "(graph, layout, *, num_samples=4096, seed=42)"
)]
fn sampled_crossing_number(
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
        egraph_metrics::sampled_crossing_number(g, &soa, samples, seed_val)
    })
}

/// Count crossing edge sub-segments on the periodic unit torus `[0, 1)^2`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
fn torus_crossing_number(py: Python<'_>, graph: &PyCsrGraph, layout: &PyLayout2d) -> PyResult<f64> {
    let g = graph.inner();
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::crossing_edges_torus(g, &soa).map(|pairs| pairs.len() as f64)
    })
}

/// Find all crossing edge sub-segment pairs for `graph` laid out on the unit torus `[0, 1)^2`.
#[pyfunction]
#[pyo3(signature = (graph, layout), text_signature = "(graph, layout)")]
fn crossing_edges_torus(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
) -> PyResult<Vec<Py<PyDict>>> {
    let g = graph.inner();
    let pairs = dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::crossing_edges_torus(g, &soa)
    })?;
    let mut out = Vec::with_capacity(pairs.len());
    for pair in pairs {
        let dict = PyDict::new(py);
        dict.set_item("edge_a", pair.edge_a)?;
        dict.set_item("edge_b", pair.edge_b)?;
        dict.set_item("seg_a", pair.seg_a.to_vec())?;
        dict.set_item("seg_b", pair.seg_b.to_vec())?;
        out.push(dict.unbind());
    }
    Ok(out)
}

/// Compute crossing count and crossing-angle summary statistics in a single sweep pass.
#[pyfunction]
#[pyo3(signature = (graph, layout), text_signature = "(graph, layout)")]
fn crossing_summary(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
) -> PyResult<Py<PyDict>> {
    let g = graph.inner();
    let summary = dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::crossing_summary(g, &soa)
    })?;
    let dict = PyDict::new(py);
    dict.set_item("count", summary.count)?;
    dict.set_item("angle_cos_sq_sum", summary.angle_cos_sq_sum)?;
    dict.set_item("min_angle_rad", summary.min_angle_rad)?;
    Ok(dict.unbind())
}

/// Compute an all-pairs shortest-path `(N, N)` `float64` distance matrix via BFS.
#[pyfunction]
#[pyo3(
    signature = (graph, unit_edge_length = StrictF64(1.0)),
    text_signature = "(graph, unit_edge_length=1.0)"
)]
fn all_pairs_bfs<'py>(
    py: Python<'py>,
    graph: &PyCsrGraph,
    unit_edge_length: StrictF64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let g = graph.inner();
    let n = g.num_nodes();
    let unit = unit_edge_length.0;
    let dists = py
        .detach(|| egraph_metrics::all_pairs_bfs(g, unit))
        .map_err(core_error_to_py)?;
    let arr =
        Array2::from_shape_vec((n, n), dists).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

/// Compute single-source BFS shortest-path distances from `source` as a `(N,)` `float64` NumPy array.
#[pyfunction]
#[pyo3(
    signature = (graph, source, unit_edge_length = StrictF64(1.0)),
    text_signature = "(graph, source, unit_edge_length=1.0)"
)]
fn bfs_single_source<'py>(
    py: Python<'py>,
    graph: &PyCsrGraph,
    source: StrictUsize,
    unit_edge_length: StrictF64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let g = graph.inner();
    let src = source.0;
    let unit = unit_edge_length.0;
    let dists = py
        .detach(|| -> CoreResult<Vec<f64>> {
            let mut scratch = egraph_metrics::BfsScratch::new(g.num_nodes());
            let slice = egraph_metrics::bfs_single_source(g, src, unit, &mut scratch)?;
            Ok(slice.to_vec())
        })
        .map_err(core_error_to_py)?;
    Ok(PyArray1::from_vec(py, dists))
}

/// Compute closed-form Scale-Normalized Stress (`SNS`) from an `(N, N)` distance matrix and optional weights.
#[pyfunction]
#[pyo3(
    signature = (layout, distances, weights = None),
    text_signature = "(layout, distances, weights=None)"
)]
fn scale_normalized_stress_from_distances(
    py: Python<'_>,
    layout: &PyLayout2d,
    distances: &Bound<'_, PyAny>,
    weights: Option<&Bound<'_, PyAny>>,
) -> PyResult<f64> {
    let (dists_flat, d_rows, d_cols) = extract_f64_2d_raw(distances, "distances")?;
    if d_rows != d_cols {
        return Err(PyValueError::new_err(format!(
            "distances must be square (N, N), got ({d_rows}, {d_cols})"
        )));
    }
    let weights_flat = match weights {
        Some(w_obj) if !w_obj.is_none() => {
            let (wf, w_rows, w_cols) = extract_f64_2d_raw(w_obj, "weights")?;
            if w_rows != w_cols || w_rows != d_rows {
                return Err(PyValueError::new_err(format!(
                    "weights shape ({w_rows}, {w_cols}) does not match distances shape ({d_rows}, {d_cols})"
                )));
            }
            Some(wf)
        }
        _ => None,
    };
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::scale_normalized_stress_from_distances(
            &soa,
            &dists_flat,
            weights_flat.as_deref(),
        )
    })
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(scale_normalized_stress, m)?)?;
    m.add_function(wrap_pyfunction!(sampled_scale_normalized_stress, m)?)?;
    m.add_function(wrap_pyfunction!(crossing_number, m)?)?;
    m.add_function(wrap_pyfunction!(crossing_angle, m)?)?;
    m.add_function(wrap_pyfunction!(crossing_edges, m)?)?;
    m.add_function(wrap_pyfunction!(crossing_number_from_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(crossing_angle_from_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(sweep_crossing_number, m)?)?;
    m.add_function(wrap_pyfunction!(sweep_crossing_angle, m)?)?;
    m.add_function(wrap_pyfunction!(sampled_crossing_number, m)?)?;
    m.add_function(wrap_pyfunction!(crosslessness, m)?)?;
    m.add_function(wrap_pyfunction!(sampled_crosslessness, m)?)?;
    m.add_function(wrap_pyfunction!(torus_crossing_number, m)?)?;
    m.add_function(wrap_pyfunction!(minimum_angle, m)?)?;
    m.add_function(wrap_pyfunction!(minimum_angle_summary, m)?)?;
    m.add_function(wrap_pyfunction!(layout_degeneracy, m)?)?;
    m.add_function(wrap_pyfunction!(crossing_edges_torus, m)?)?;
    m.add_function(wrap_pyfunction!(crossing_summary, m)?)?;
    m.add_function(wrap_pyfunction!(all_pairs_bfs, m)?)?;
    m.add_function(wrap_pyfunction!(bfs_single_source, m)?)?;
    m.add_function(wrap_pyfunction!(scale_normalized_stress_from_distances, m)?)?;
    Ok(())
}
