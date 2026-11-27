//! Scale-Normalized Stress (`SNS`) and manifold geodesic distortion bindings.

use egraph::metrics as egraph_metrics;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict};

use crate::array_extract::extract_f64_2d_raw;
use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{StrictU64, StrictUsize, core_error_to_py, is_non_sequence_container};
use crate::manifold::{ManifoldLayout2dInner, PyManifoldLayout2d, build_core_f32, build_core_f64};

use super::nd_manifold_helpers::{
    MetricLayoutInput, dispatch_owned_metric_layout, stress_evaluation_to_py_dict,
};

fn extract_u32_item(obj: &Bound<'_, PyAny>, fn_name: &str, field: &str) -> PyResult<u32> {
    if obj.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(format!(
            "{fn_name}: {field} must be int, not bool"
        )));
    }
    let val = obj.extract::<i64>().map_err(|_| {
        PyTypeError::new_err(format!("{fn_name}: {field} must be a non-negative integer"))
    })?;
    u32::try_from(val).map_err(|_| {
        PyValueError::new_err(format!(
            "{fn_name}: {field} value {val} out of range for uint32"
        ))
    })
}

fn extract_f64_item(obj: &Bound<'_, PyAny>, fn_name: &str, field: &str) -> PyResult<f64> {
    if obj.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(format!(
            "{fn_name}: {field} must be float, not bool"
        )));
    }
    obj.extract::<f64>()
        .map_err(|_| PyTypeError::new_err(format!("{fn_name}: {field} must be numeric")))
}

fn extract_stress_pairs(pairs: &Bound<'_, PyAny>) -> PyResult<Vec<(u32, u32, f64, f64)>> {
    if is_non_sequence_container(pairs) {
        return Err(PyTypeError::new_err(
            "pairs must be a (M, 4) array or sequence of (u, v, dist, weight) tuples",
        ));
    }
    if pairs.cast::<numpy::PyUntypedArray>().is_ok() {
        let (flat, rows, cols) = extract_f64_2d_raw(pairs, "pairs")?;
        if cols != 4 {
            return Err(PyValueError::new_err(format!(
                "pairs array must have shape (M, 4), got ({rows}, {cols})"
            )));
        }
        let mut out = Vec::with_capacity(rows);
        for chunk in flat.as_chunks::<4>().0 {
            let (uf, vf, d, w) = (chunk[0], chunk[1], chunk[2], chunk[3]);
            if !uf.is_finite() || uf < 0.0 || uf.fract() != 0.0 || uf > u32::MAX as f64 {
                return Err(PyValueError::new_err(format!(
                    "pairs endpoint u={uf} must be a non-negative integer"
                )));
            }
            if !vf.is_finite() || vf < 0.0 || vf.fract() != 0.0 || vf > u32::MAX as f64 {
                return Err(PyValueError::new_err(format!(
                    "pairs endpoint v={vf} must be a non-negative integer"
                )));
            }
            out.push((uf as u32, vf as u32, d, w));
        }
        return Ok(out);
    }

    let iter = pairs.try_iter().map_err(|_| {
        PyTypeError::new_err(
            "pairs must be a (M, 4) array or sequence of (u, v, dist, weight) tuples",
        )
    })?;
    let mut out = Vec::new();
    for item_res in iter {
        let item = item_res?;
        if is_non_sequence_container(&item) {
            return Err(PyTypeError::new_err(
                "each pair entry must be a 4-element (u, v, dist, weight) sequence",
            ));
        }
        let sub_iter = item.try_iter().map_err(|_| {
            PyTypeError::new_err("each pair entry must be a 4-element (u, v, dist, weight) tuple")
        })?;
        let mut elems = Vec::with_capacity(4);
        for sub_res in sub_iter {
            elems.push(sub_res?);
        }
        if elems.len() != 4 {
            return Err(PyValueError::new_err(format!(
                "each pair entry must have 4 elements (u, v, dist, weight), got {}",
                elems.len()
            )));
        }
        let u = extract_u32_item(&elems[0], "scale_normalized_stress_from_pairs", "u")?;
        let v = extract_u32_item(&elems[1], "scale_normalized_stress_from_pairs", "v")?;
        let d = extract_f64_item(&elems[2], "scale_normalized_stress_from_pairs", "dist")?;
        let w = extract_f64_item(&elems[3], "scale_normalized_stress_from_pairs", "weight")?;
        out.push((u, v, d, w));
    }
    Ok(out)
}

/// Compute closed-form Scale-Normalized Stress (`SNS`) in `[0.0, 1.0]` over `graph` and `layout`.
///
/// Supports `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (graph, layout), text_signature = "(graph, layout)")]
fn scale_normalized_stress(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: MetricLayoutInput<'_>,
) -> PyResult<f64> {
    let owned = layout.extract_owned()?;
    let g = graph.inner();
    dispatch_owned_metric_layout(
        py,
        owned,
        |l| egraph_metrics::scale_normalized_stress(&*g, l),
        |l| egraph_metrics::scale_normalized_stress(&*g, l),
        |l| egraph_metrics::scale_normalized_stress_nd(&*g, l),
        |l| egraph_metrics::scale_normalized_stress_nd(&*g, l),
        |l| egraph_metrics::scale_normalized_stress_manifold(&*g, l),
        |l| egraph_metrics::scale_normalized_stress_manifold(&*g, l),
    )
}

/// Compute detailed closed-form Scale-Normalized Stress (`SNS`) diagnostics.
///
/// Supports `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (graph, layout), text_signature = "(graph, layout)")]
fn scale_normalized_stress_detail(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: MetricLayoutInput<'_>,
) -> PyResult<Py<PyDict>> {
    let owned = layout.extract_owned()?;
    let g = graph.inner();
    let eval = dispatch_owned_metric_layout(
        py,
        owned,
        |l| egraph_metrics::scale_normalized_stress_detail(&*g, l),
        |l| egraph_metrics::scale_normalized_stress_detail(&*g, l),
        |l| egraph_metrics::scale_normalized_stress_detail_nd(&*g, l),
        |l| egraph_metrics::scale_normalized_stress_detail_nd(&*g, l),
        |l| egraph_metrics::scale_normalized_stress_detail_manifold(&*g, l),
        |l| egraph_metrics::scale_normalized_stress_detail_manifold(&*g, l),
    )?;
    stress_evaluation_to_py_dict(py, eval)
}

/// Compute O(kN) degree-stratified pivot-sampled Scale-Normalized Stress (`SNS`).
///
/// Supports `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (graph, layout, *, num_pivots = StrictUsize(64), seed = StrictU64(42)),
    text_signature = "(graph, layout, *, num_pivots=64, seed=42)"
)]
fn sampled_scale_normalized_stress(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: MetricLayoutInput<'_>,
    num_pivots: StrictUsize,
    seed: StrictU64,
) -> PyResult<f64> {
    let owned = layout.extract_owned()?;
    let g = graph.inner();
    let pivots = num_pivots.0;
    let seed_val = seed.0;
    dispatch_owned_metric_layout(
        py,
        owned,
        |l| egraph_metrics::sampled_scale_normalized_stress(&*g, l, pivots, seed_val),
        |l| egraph_metrics::sampled_scale_normalized_stress(&*g, l, pivots, seed_val),
        |l| egraph_metrics::sampled_scale_normalized_stress_nd(&*g, l, pivots, seed_val),
        |l| egraph_metrics::sampled_scale_normalized_stress_nd(&*g, l, pivots, seed_val),
        |l| egraph_metrics::sampled_scale_normalized_stress_manifold(&*g, l, pivots, seed_val),
        |l| egraph_metrics::sampled_scale_normalized_stress_manifold(&*g, l, pivots, seed_val),
    )
}

/// Compute detailed pivot-sampled Scale-Normalized Stress (`SNS`) diagnostics.
///
/// Supports `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (graph, layout, *, num_pivots = StrictUsize(64), seed = StrictU64(42)),
    text_signature = "(graph, layout, *, num_pivots=64, seed=42)"
)]
fn sampled_scale_normalized_stress_detail(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: MetricLayoutInput<'_>,
    num_pivots: StrictUsize,
    seed: StrictU64,
) -> PyResult<Py<PyDict>> {
    let owned = layout.extract_owned()?;
    let g = graph.inner();
    let pivots = num_pivots.0;
    let seed_val = seed.0;
    let eval = dispatch_owned_metric_layout(
        py,
        owned,
        |l| egraph_metrics::sampled_scale_normalized_stress_detail(&*g, l, pivots, seed_val),
        |l| egraph_metrics::sampled_scale_normalized_stress_detail(&*g, l, pivots, seed_val),
        |l| egraph_metrics::sampled_scale_normalized_stress_detail_nd(&*g, l, pivots, seed_val),
        |l| egraph_metrics::sampled_scale_normalized_stress_detail_nd(&*g, l, pivots, seed_val),
        |l| {
            egraph_metrics::sampled_scale_normalized_stress_detail_manifold(
                &*g, l, pivots, seed_val,
            )
        },
        |l| {
            egraph_metrics::sampled_scale_normalized_stress_detail_manifold(
                &*g, l, pivots, seed_val,
            )
        },
    )?;
    stress_evaluation_to_py_dict(py, eval)
}

fn extract_distance_weight_matrices(
    distances: &Bound<'_, PyAny>,
    weights: Option<&Bound<'_, PyAny>>,
) -> PyResult<(Vec<f64>, Option<Vec<f64>>)> {
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
    Ok((dists_flat, weights_flat))
}

/// Compute closed-form Scale-Normalized Stress (`SNS`) from an `(N, N)` distance matrix and optional weights.
///
/// Supports `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, distances, weights = None),
    text_signature = "(layout, distances, weights=None)"
)]
fn scale_normalized_stress_from_distances(
    py: Python<'_>,
    layout: MetricLayoutInput<'_>,
    distances: &Bound<'_, PyAny>,
    weights: Option<&Bound<'_, PyAny>>,
) -> PyResult<f64> {
    let (dists_flat, weights_flat) = extract_distance_weight_matrices(distances, weights)?;
    let owned = layout.extract_owned()?;
    dispatch_owned_metric_layout(
        py,
        owned,
        |l| {
            egraph_metrics::scale_normalized_stress_from_distances(
                l,
                &dists_flat,
                weights_flat.as_deref(),
            )
        },
        |l| {
            egraph_metrics::scale_normalized_stress_from_distances(
                l,
                &dists_flat,
                weights_flat.as_deref(),
            )
        },
        |l| {
            egraph_metrics::scale_normalized_stress_from_distances_nd(
                l,
                &dists_flat,
                weights_flat.as_deref(),
            )
        },
        |l| {
            egraph_metrics::scale_normalized_stress_from_distances_nd(
                l,
                &dists_flat,
                weights_flat.as_deref(),
            )
        },
        |l| {
            egraph_metrics::scale_normalized_stress_from_distances_manifold(
                l,
                &dists_flat,
                weights_flat.as_deref(),
            )
        },
        |l| {
            egraph_metrics::scale_normalized_stress_from_distances_manifold(
                l,
                &dists_flat,
                weights_flat.as_deref(),
            )
        },
    )
}

/// Compute detailed Scale-Normalized Stress (`SNS`) from an `(N, N)` distance matrix and optional weights.
///
/// Supports `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (layout, distances, weights = None),
    text_signature = "(layout, distances, weights=None)"
)]
fn scale_normalized_stress_from_distances_detail(
    py: Python<'_>,
    layout: MetricLayoutInput<'_>,
    distances: &Bound<'_, PyAny>,
    weights: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyDict>> {
    let (dists_flat, weights_flat) = extract_distance_weight_matrices(distances, weights)?;
    let owned = layout.extract_owned()?;
    let eval = dispatch_owned_metric_layout(
        py,
        owned,
        |l| {
            egraph_metrics::scale_normalized_stress_from_distances_detail(
                l,
                &dists_flat,
                weights_flat.as_deref(),
            )
        },
        |l| {
            egraph_metrics::scale_normalized_stress_from_distances_detail(
                l,
                &dists_flat,
                weights_flat.as_deref(),
            )
        },
        |l| {
            egraph_metrics::scale_normalized_stress_from_distances_detail_nd(
                l,
                &dists_flat,
                weights_flat.as_deref(),
            )
        },
        |l| {
            egraph_metrics::scale_normalized_stress_from_distances_detail_nd(
                l,
                &dists_flat,
                weights_flat.as_deref(),
            )
        },
        |l| {
            egraph_metrics::scale_normalized_stress_from_distances_detail_manifold(
                l,
                &dists_flat,
                weights_flat.as_deref(),
            )
        },
        |l| {
            egraph_metrics::scale_normalized_stress_from_distances_detail_manifold(
                l,
                &dists_flat,
                weights_flat.as_deref(),
            )
        },
    )?;
    stress_evaluation_to_py_dict(py, eval)
}

/// Compute detailed Scale-Normalized Stress (`SNS`) from sparse `(u, v, dist, weight)` pairs.
///
/// Supports `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (layout, pairs), text_signature = "(layout, pairs)")]
fn scale_normalized_stress_from_pairs(
    py: Python<'_>,
    layout: MetricLayoutInput<'_>,
    pairs: &Bound<'_, PyAny>,
) -> PyResult<Py<PyDict>> {
    let pair_vec = extract_stress_pairs(pairs)?;
    let owned = layout.extract_owned()?;
    let eval = dispatch_owned_metric_layout(
        py,
        owned,
        |l| egraph_metrics::scale_normalized_stress_from_pairs(l, &pair_vec),
        |l| egraph_metrics::scale_normalized_stress_from_pairs(l, &pair_vec),
        |l| egraph_metrics::scale_normalized_stress_from_pairs_nd(l, &pair_vec),
        |l| egraph_metrics::scale_normalized_stress_from_pairs_nd(l, &pair_vec),
        |l| egraph_metrics::scale_normalized_stress_from_pairs_manifold(l, &pair_vec),
        |l| egraph_metrics::scale_normalized_stress_from_pairs_manifold(l, &pair_vec),
    )?;
    stress_evaluation_to_py_dict(py, eval)
}

/// Compute Miller et al. (2024) Geodesic Distortion without post-hoc rescaling on `ManifoldLayout2d`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (graph, layout), text_signature = "(graph, layout)")]
fn geodesic_distortion(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &Bound<'_, PyManifoldLayout2d>,
) -> PyResult<f64> {
    let borrowed = layout.try_borrow()?;
    let owned = match borrowed.inner() {
        ManifoldLayout2dInner::F64 { kind, metric, buf } => {
            super::nd_manifold_helpers::OwnedMetricLayout::F64Manifold(
                build_core_f64(*kind, *metric, buf).map_err(core_error_to_py)?,
            )
        }
        ManifoldLayout2dInner::F32 { kind, metric, buf } => {
            super::nd_manifold_helpers::OwnedMetricLayout::F32Manifold(
                build_core_f32(*kind, *metric, buf).map_err(core_error_to_py)?,
            )
        }
    };
    let g = graph.inner();
    py.detach(|| match &owned {
        super::nd_manifold_helpers::OwnedMetricLayout::F64Manifold(m) => {
            egraph_metrics::geodesic_distortion(&*g, m)
        }
        super::nd_manifold_helpers::OwnedMetricLayout::F32Manifold(m) => {
            egraph_metrics::geodesic_distortion(&*g, m)
        }
        _ => unreachable!(),
    })
    .map_err(core_error_to_py)
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(scale_normalized_stress, m)?)?;
    m.add_function(wrap_pyfunction!(scale_normalized_stress_detail, m)?)?;
    m.add_function(wrap_pyfunction!(sampled_scale_normalized_stress, m)?)?;
    m.add_function(wrap_pyfunction!(sampled_scale_normalized_stress_detail, m)?)?;
    m.add_function(wrap_pyfunction!(scale_normalized_stress_from_distances, m)?)?;
    m.add_function(wrap_pyfunction!(
        scale_normalized_stress_from_distances_detail,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(scale_normalized_stress_from_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(geodesic_distortion, m)?)?;
    Ok(())
}
