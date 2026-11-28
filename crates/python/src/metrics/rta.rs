//! Landmark-Stratified Random Triplet Accuracy (`landmark_stratified_rta*`) bindings.

use egraph::metrics as egraph_metrics;
use numpy::{PyReadonlyArrayDyn, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyBool;

use crate::csr_graph::PyCsrGraph;
use crate::errors::{dimension_mismatch_err, invalid_input_err, non_finite_value_err};
use crate::ffi_convert::{StrictU64, StrictUsize, extract_u32_1d, is_non_sequence_container};

use super::nd_manifold_helpers::{MetricLayoutInput, dispatch_owned_metric};
use super::result_types::PyRtaEvaluation;

fn float_to_landmark_u32(v: f64) -> PyResult<u32> {
    if v.is_nan() {
        return Err(non_finite_value_err("landmark_dists contains NaN"));
    }
    if v == f64::INFINITY {
        return Ok(u32::MAX);
    }
    if v < 0.0 {
        return Err(invalid_input_err(format!(
            "landmark_dists entry {v} must be non-negative"
        )));
    }
    if v.fract() != 0.0 || v > u32::MAX as f64 {
        return Err(invalid_input_err(format!(
            "landmark_dists float entry {v} must be a non-negative integer or +inf"
        )));
    }
    Ok(v as u32)
}

fn is_numpy_bool_scalar(obj: &Bound<'_, PyAny>) -> bool {
    obj.getattr("dtype")
        .and_then(|dt| dt.str())
        .and_then(|s| s.to_str().map(|v| v == "bool"))
        .unwrap_or(false)
}

fn extract_landmark_item_u32(item: &Bound<'_, PyAny>) -> PyResult<u32> {
    if item.is_instance_of::<PyBool>() || is_numpy_bool_scalar(item) {
        return Err(PyTypeError::new_err(
            "landmark_dists entries must be integers or floats, not bool",
        ));
    }
    if let Ok(iv) = item.extract::<i128>() {
        return u32::try_from(iv).map_err(|_| {
            invalid_input_err(format!("landmark_dists entry {iv} out of range for u32"))
        });
    }
    if let Ok(fv) = item.extract::<f64>() {
        return float_to_landmark_u32(fv);
    }
    Err(PyTypeError::new_err(
        "landmark_dists entries must be integers or floats",
    ))
}

fn extract_landmark_dists_flat(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
    if let Ok(untyped) = obj.cast::<PyUntypedArray>() {
        let ndim = untyped.ndim();
        if ndim != 1 && ndim != 2 {
            return Err(dimension_mismatch_err(format!(
                "landmark_dists must be a 1D or 2D array, got {ndim}D"
            )));
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArrayDyn<'_, f64>>() {
            return arr
                .as_array()
                .iter()
                .map(|&v| float_to_landmark_u32(v))
                .collect();
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArrayDyn<'_, f32>>() {
            return arr
                .as_array()
                .iter()
                .map(|&v| float_to_landmark_u32(f64::from(v)))
                .collect();
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArrayDyn<'_, u32>>() {
            return Ok(arr.as_array().iter().copied().collect());
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArrayDyn<'_, u16>>() {
            return Ok(arr
                .as_array()
                .iter()
                .map(|&v| {
                    if v == u16::MAX {
                        u32::MAX
                    } else {
                        u32::from(v)
                    }
                })
                .collect());
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArrayDyn<'_, u8>>() {
            return Ok(arr.as_array().iter().map(|&v| u32::from(v)).collect());
        }
        macro_rules! try_int_array {
            ($ty:ty) => {
                if let Ok(arr) = obj.extract::<PyReadonlyArrayDyn<'_, $ty>>() {
                    return arr
                        .as_array()
                        .iter()
                        .map(|&v| {
                            u32::try_from(v).map_err(|_| {
                                invalid_input_err(format!(
                                    "landmark_dists entry {v} out of range for u32"
                                ))
                            })
                        })
                        .collect();
                }
            };
        }
        try_int_array!(i64);
        try_int_array!(i32);
        try_int_array!(i16);
        try_int_array!(i8);
        try_int_array!(u64);
        return Err(PyTypeError::new_err(
            "landmark_dists NumPy array must have integer or float dtype",
        ));
    }

    if is_non_sequence_container(obj) {
        return Err(PyTypeError::new_err(
            "landmark_dists must be a 1D or 2D array or sequence",
        ));
    }
    let iter = obj
        .try_iter()
        .map_err(|_| PyTypeError::new_err("landmark_dists must be a 1D or 2D array or sequence"))?;
    let mut items = Vec::new();
    for item_res in iter {
        items.push(item_res?);
    }
    let Some(first) = items.first() else {
        return Ok(Vec::new());
    };
    if !is_non_sequence_container(first) && first.try_iter().is_ok() {
        let mut flat = Vec::new();
        let mut expected_cols: Option<usize> = None;
        for (r_idx, row_obj) in items.iter().enumerate() {
            if is_non_sequence_container(row_obj) {
                return Err(PyTypeError::new_err(
                    "landmark_dists rows must be sequences",
                ));
            }
            let row_iter = row_obj
                .try_iter()
                .map_err(|_| PyTypeError::new_err("landmark_dists rows must be sequences"))?;
            let mut row_len = 0usize;
            for elem_res in row_iter {
                flat.push(extract_landmark_item_u32(&elem_res?)?);
                row_len += 1;
            }
            match expected_cols {
                None => expected_cols = Some(row_len),
                Some(cols) if cols != row_len => {
                    return Err(dimension_mismatch_err(format!(
                        "landmark_dists has ragged row {r_idx}: expected {cols} columns, got {row_len}"
                    )));
                }
                Some(_) => {}
            }
        }
        return Ok(flat);
    }

    items.iter().map(extract_landmark_item_u32).collect()
}

/// Compute Landmark-Stratified Random Triplet Accuracy (`L-Stratified-RTA`) in `[0.0, 1.0]`.
#[pyfunction]
#[pyo3(
    signature = (graph, layout, *, num_landmarks = StrictUsize(64), num_triplets = StrictUsize(1_000_000), min_degree = StrictUsize(2), seed = StrictU64(42)),
    text_signature = "(graph, layout, *, num_landmarks=64, num_triplets=1000000, min_degree=2, seed=42)"
)]
fn landmark_stratified_rta(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: MetricLayoutInput<'_>,
    num_landmarks: StrictUsize,
    num_triplets: StrictUsize,
    min_degree: StrictUsize,
    seed: StrictU64,
) -> PyResult<f64> {
    let owned = layout.extract_owned()?;
    let g = graph.inner();
    let (nl, nt, md, s) = (num_landmarks.0, num_triplets.0, min_degree.0, seed.0);
    dispatch_owned_metric!(py, owned, |l| {
        egraph_metrics::landmark_stratified_rta(&*g, l, nl, nt, md, s)
    })
}

/// Compute Landmark-Stratified Random Triplet Accuracy (`L-Stratified-RTA`) with diagnostics.
#[pyfunction]
#[pyo3(
    signature = (graph, layout, *, num_landmarks = StrictUsize(64), num_triplets = StrictUsize(1_000_000), min_degree = StrictUsize(2), seed = StrictU64(42)),
    text_signature = "(graph, layout, *, num_landmarks=64, num_triplets=1000000, min_degree=2, seed=42)"
)]
fn landmark_stratified_rta_detail(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: MetricLayoutInput<'_>,
    num_landmarks: StrictUsize,
    num_triplets: StrictUsize,
    min_degree: StrictUsize,
    seed: StrictU64,
) -> PyResult<PyRtaEvaluation> {
    let owned = layout.extract_owned()?;
    let g = graph.inner();
    let (nl, nt, md, s) = (num_landmarks.0, num_triplets.0, min_degree.0, seed.0);
    let inner = dispatch_owned_metric!(py, owned, |l| {
        egraph_metrics::landmark_stratified_rta_detail(&*g, l, nl, nt, md, s)
    })?;
    Ok(PyRtaEvaluation { inner })
}

/// Compute Landmark-Stratified RTA from a precomputed `(P_L, N)` hop-distance table.
#[pyfunction]
#[pyo3(
    signature = (layout, landmark_ids, landmark_dists, *, num_triplets = StrictUsize(1_000_000), seed = StrictU64(42)),
    text_signature = "(layout, landmark_ids, landmark_dists, *, num_triplets=1000000, seed=42)"
)]
fn landmark_stratified_rta_from_landmarks(
    py: Python<'_>,
    layout: MetricLayoutInput<'_>,
    landmark_ids: &Bound<'_, PyAny>,
    landmark_dists: &Bound<'_, PyAny>,
    num_triplets: StrictUsize,
    seed: StrictU64,
) -> PyResult<PyRtaEvaluation> {
    let l_ids = extract_u32_1d(landmark_ids, "landmark_ids")?;
    let l_dists = extract_landmark_dists_flat(landmark_dists)?;
    let owned = layout.extract_owned()?;
    let (nt, s) = (num_triplets.0, seed.0);
    let inner = dispatch_owned_metric!(py, owned, |l| {
        egraph_metrics::landmark_stratified_rta_from_landmarks(l, &l_ids, &l_dists, nt, s)
    })?;
    Ok(PyRtaEvaluation { inner })
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(landmark_stratified_rta, m)?)?;
    m.add_function(wrap_pyfunction!(landmark_stratified_rta_detail, m)?)?;
    m.add_function(wrap_pyfunction!(landmark_stratified_rta_from_landmarks, m)?)?;
    Ok(())
}
