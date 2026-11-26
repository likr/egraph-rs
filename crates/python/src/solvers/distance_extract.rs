//! Dense 2D square distance matrix extraction helper for stress and Kamada-Kawai solvers.

use numpy::ndarray::ArrayView2;
use numpy::{PyReadonlyArray2, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBool;

use crate::ffi_convert::is_non_sequence_container;

fn convert_distance_view2_to_f64<T, F>(
    view: ArrayView2<'_, T>,
    mut to_f64: F,
) -> PyResult<(Vec<f64>, usize)>
where
    T: Copy,
    F: FnMut(T) -> f64,
{
    let nrows = view.nrows();
    let ncols = view.ncols();
    if nrows != ncols {
        return Err(PyValueError::new_err(format!(
            "distance_matrix must be square (N, N), got ({nrows}, {ncols})"
        )));
    }
    let mut flat = Vec::with_capacity(nrows.saturating_mul(ncols));
    for row in view.rows() {
        for &val in row.iter() {
            flat.push(to_f64(val));
        }
    }
    Ok((flat, nrows))
}

pub(super) fn extract_distance_matrix_2d(obj: &Bound<'_, PyAny>) -> PyResult<(Vec<f64>, usize)> {
    if let Ok(untyped) = obj.cast::<PyUntypedArray>() {
        if untyped.ndim() != 2 {
            return Err(PyValueError::new_err(format!(
                "distance_matrix must be a 2D square numeric array, got {}D",
                untyped.ndim()
            )));
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, f64>>() {
            return convert_distance_view2_to_f64(arr.as_array(), |v| v);
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, f32>>() {
            return convert_distance_view2_to_f64(arr.as_array(), f64::from);
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, i64>>() {
            return convert_distance_view2_to_f64(arr.as_array(), |v| v as f64);
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, i32>>() {
            return convert_distance_view2_to_f64(arr.as_array(), f64::from);
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, u64>>() {
            return convert_distance_view2_to_f64(arr.as_array(), |v| v as f64);
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, u32>>() {
            return convert_distance_view2_to_f64(arr.as_array(), f64::from);
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, isize>>() {
            return convert_distance_view2_to_f64(arr.as_array(), |v| v as f64);
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, usize>>() {
            return convert_distance_view2_to_f64(arr.as_array(), |v| v as f64);
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, i16>>() {
            return convert_distance_view2_to_f64(arr.as_array(), f64::from);
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, u16>>() {
            return convert_distance_view2_to_f64(arr.as_array(), f64::from);
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, i8>>() {
            return convert_distance_view2_to_f64(arr.as_array(), f64::from);
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, u8>>() {
            return convert_distance_view2_to_f64(arr.as_array(), f64::from);
        }
        return Err(PyTypeError::new_err(
            "distance_matrix NumPy array must have a numeric dtype",
        ));
    }

    if is_non_sequence_container(obj) {
        return Err(PyTypeError::new_err(
            "distance_matrix must be a 2D square numeric array or sequence of sequences",
        ));
    }
    let outer = obj.try_iter().map_err(|_| {
        PyTypeError::new_err(
            "distance_matrix must be a 2D square numeric array or sequence of sequences",
        )
    })?;
    let mut flat = Vec::new();
    let mut nrows = 0usize;
    let mut ncols: Option<usize> = None;

    for row_res in outer {
        let row = row_res?;
        if is_non_sequence_container(&row) {
            return Err(PyTypeError::new_err(
                "distance_matrix rows must be numeric sequences",
            ));
        }
        let inner = row
            .try_iter()
            .map_err(|_| PyTypeError::new_err("distance_matrix rows must be numeric sequences"))?;
        let mut row_len = 0usize;
        for elem_res in inner {
            let elem = elem_res?;
            if elem.is_instance_of::<PyBool>() {
                return Err(PyTypeError::new_err(
                    "distance_matrix entries must be numeric, not bool",
                ));
            }
            let val = elem
                .extract::<f64>()
                .map_err(|_| PyTypeError::new_err("distance_matrix entries must be numeric"))?;
            flat.push(val);
            row_len += 1;
        }
        match ncols {
            None => ncols = Some(row_len),
            Some(expected) if row_len != expected => {
                return Err(PyValueError::new_err(format!(
                    "distance_matrix has ragged row {nrows}: expected {expected} columns, got {row_len}"
                )));
            }
            Some(_) => {}
        }
        nrows += 1;
    }

    let c = ncols.unwrap_or(0);
    if nrows != c {
        return Err(PyValueError::new_err(format!(
            "distance_matrix must be square (N, N), got ({nrows}, {c})"
        )));
    }
    Ok((flat, nrows))
}
