//! Extraction helpers for constructing `Layout3dInner` from Python arrays and sequences.

use egraph_ffi::InterleavedLayout3d;
use numpy::{PyReadonlyArray2, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyBool;

use crate::errors::{dimension_mismatch_err, ffi_error_to_py, non_finite_value_err};
use crate::ffi_convert::is_non_sequence_container;
use crate::layout3d::Layout3dInner;

fn extract_interleaved_f64_from_view(
    view: numpy::ndarray::ArrayView2<'_, f64>,
) -> PyResult<Vec<f64>> {
    let [n, cols] = [view.nrows(), view.ncols()];
    if cols != 3 {
        return Err(dimension_mismatch_err(format!(
            "coords must have shape (N, 3), got ({n}, {cols})"
        )));
    }
    let mut flat = Vec::with_capacity(n * 3);
    for (idx, row) in view.rows().into_iter().enumerate() {
        let (x, y, z) = (row[0], row[1], row[2]);
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
            return Err(non_finite_value_err(format!(
                "coords contains non-finite value at row {idx}"
            )));
        }
        flat.push(x);
        flat.push(y);
        flat.push(z);
    }
    Ok(flat)
}

fn extract_interleaved_f32_from_view(
    view: numpy::ndarray::ArrayView2<'_, f32>,
) -> PyResult<Vec<f32>> {
    let [n, cols] = [view.nrows(), view.ncols()];
    if cols != 3 {
        return Err(dimension_mismatch_err(format!(
            "coords must have shape (N, 3), got ({n}, {cols})"
        )));
    }
    let mut flat = Vec::with_capacity(n * 3);
    for (idx, row) in view.rows().into_iter().enumerate() {
        let (x, y, z) = (row[0], row[1], row[2]);
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
            return Err(non_finite_value_err(format!(
                "coords contains non-finite value at row {idx}"
            )));
        }
        flat.push(x);
        flat.push(y);
        flat.push(z);
    }
    Ok(flat)
}

pub(crate) fn extract_layout3d_inner(coords: &Bound<'_, PyAny>) -> PyResult<Layout3dInner> {
    if let Ok(untyped) = coords.cast::<PyUntypedArray>() {
        if untyped.ndim() != 2 {
            return Err(dimension_mismatch_err(format!(
                "coords must be a 2D array of shape (N, 3), got {}D",
                untyped.ndim()
            )));
        }
        let shape = untyped.shape();
        if shape[1] != 3 {
            return Err(dimension_mismatch_err(format!(
                "coords must have shape (N, 3), got ({}, {})",
                shape[0], shape[1]
            )));
        }
        if let Ok(arr) = coords.extract::<PyReadonlyArray2<'_, f64>>() {
            let flat = extract_interleaved_f64_from_view(arr.as_array())?;
            let buf = InterleavedLayout3d::from_interleaved(&flat).map_err(ffi_error_to_py)?;
            return Ok(Layout3dInner::F64(buf));
        }
        if let Ok(arr) = coords.extract::<PyReadonlyArray2<'_, f32>>() {
            let flat = extract_interleaved_f32_from_view(arr.as_array())?;
            let buf = InterleavedLayout3d::from_interleaved(&flat).map_err(ffi_error_to_py)?;
            return Ok(Layout3dInner::F32(buf));
        }
        return Err(PyTypeError::new_err(
            "coords NumPy array must have dtype float64 or float32",
        ));
    }

    if is_non_sequence_container(coords) {
        return Err(PyTypeError::new_err(
            "coords must be a 2D float64/float32 array or sequence of (x, y, z) triples",
        ));
    }
    let outer = coords.try_iter().map_err(|_| {
        PyTypeError::new_err(
            "coords must be a 2D float64/float32 array or sequence of (x, y, z) triples",
        )
    })?;
    let mut flat = Vec::new();
    for (row_idx, row_res) in outer.enumerate() {
        let row = row_res?;
        if is_non_sequence_container(&row) {
            return Err(PyTypeError::new_err(
                "each coordinate row must be a 3-element sequence (x, y, z)",
            ));
        }
        let inner = row.try_iter().map_err(|_| {
            PyTypeError::new_err("each coordinate row must be a 3-element sequence (x, y, z)")
        })?;
        let mut xyz = [0.0_f64; 3];
        let mut count = 0usize;
        for elem_res in inner {
            let elem = elem_res?;
            if elem.is_instance_of::<PyBool>() {
                return Err(PyTypeError::new_err(
                    "coords entries must be floats, not bool",
                ));
            }
            if count < 3 {
                let val = elem
                    .extract::<f64>()
                    .map_err(|_| PyTypeError::new_err("coords entries must be finite floats"))?;
                if !val.is_finite() {
                    return Err(non_finite_value_err(format!(
                        "coords contains non-finite value at row {row_idx}"
                    )));
                }
                xyz[count] = val;
            }
            count += 1;
        }
        if count != 3 {
            return Err(dimension_mismatch_err(format!(
                "each coordinate row must contain exactly 3 elements, got {count}"
            )));
        }
        flat.extend_from_slice(&xyz);
    }
    let buf = InterleavedLayout3d::from_interleaved(&flat).map_err(ffi_error_to_py)?;
    Ok(Layout3dInner::F64(buf))
}
