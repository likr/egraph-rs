//! Numeric 1D and 2D NumPy/sequence extraction helpers for Python bindings.

use numpy::ndarray::{ArrayView1, ArrayView2};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBool;

use crate::ffi_convert::{
    extract_non_negative_usize, extract_optional_weights_1d, is_non_sequence_container,
};

fn convert_int_slice_to_usize<T, F>(view: ArrayView1<'_, T>, mut convert: F) -> PyResult<Vec<usize>>
where
    T: Copy,
    F: FnMut(T) -> PyResult<usize>,
{
    let mut out = Vec::with_capacity(view.len());
    for &val in view.iter() {
        out.push(convert(val)?);
    }
    Ok(out)
}

pub(crate) fn extract_usize_1d(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<usize>> {
    if let Ok(untyped) = obj.cast::<PyUntypedArray>() {
        if untyped.ndim() != 1 {
            return Err(PyValueError::new_err(format!(
                "{name} must be a 1D array, got {}D",
                untyped.ndim()
            )));
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, usize>>() {
            return Ok(match arr.as_slice() {
                Ok(slice) => slice.to_vec(),
                Err(_) => arr.as_array().iter().copied().collect(),
            });
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, u64>>() {
            return convert_int_slice_to_usize(arr.as_array(), |v| {
                usize::try_from(v).map_err(|_| {
                    PyValueError::new_err(format!("{name} entry {v} out of range for usize"))
                })
            });
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, i64>>() {
            return convert_int_slice_to_usize(arr.as_array(), |v| {
                usize::try_from(v).map_err(|_| {
                    PyValueError::new_err(format!("{name} entry {v} out of range for usize"))
                })
            });
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, u32>>() {
            return Ok(arr.as_array().iter().map(|&v| v as usize).collect());
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, i32>>() {
            return convert_int_slice_to_usize(arr.as_array(), |v| {
                usize::try_from(v).map_err(|_| {
                    PyValueError::new_err(format!("{name} entry {v} out of range for usize"))
                })
            });
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, isize>>() {
            return convert_int_slice_to_usize(arr.as_array(), |v| {
                usize::try_from(v).map_err(|_| {
                    PyValueError::new_err(format!("{name} entry {v} out of range for usize"))
                })
            });
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, u16>>() {
            return Ok(arr.as_array().iter().map(|&v| usize::from(v)).collect());
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, u8>>() {
            return Ok(arr.as_array().iter().map(|&v| usize::from(v)).collect());
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, i16>>() {
            return convert_int_slice_to_usize(arr.as_array(), |v| {
                usize::try_from(v).map_err(|_| {
                    PyValueError::new_err(format!("{name} entry {v} out of range for usize"))
                })
            });
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, i8>>() {
            return convert_int_slice_to_usize(arr.as_array(), |v| {
                usize::try_from(v).map_err(|_| {
                    PyValueError::new_err(format!("{name} entry {v} out of range for usize"))
                })
            });
        }
        return Err(PyTypeError::new_err(format!(
            "{name} must be a 1D integer NumPy array"
        )));
    }

    if is_non_sequence_container(obj) {
        return Err(PyTypeError::new_err(format!(
            "{name} must be a 1D integer array or sequence"
        )));
    }
    let iter = obj.try_iter().map_err(|_| {
        PyTypeError::new_err(format!("{name} must be a 1D integer array or sequence"))
    })?;
    let mut out = Vec::new();
    for item_res in iter {
        let item = item_res?;
        out.push(extract_non_negative_usize(&item, name)?);
    }
    Ok(out)
}

pub(crate) fn extract_f64_1d(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<f64>> {
    if obj.is_none() {
        return Err(PyTypeError::new_err(format!(
            "{name} must be a 1D numeric array or sequence, not None"
        )));
    }
    match extract_optional_weights_1d(Some(obj), name)? {
        Some(v) => Ok(v),
        None => Err(PyTypeError::new_err(format!(
            "{name} must be a 1D numeric array or sequence"
        ))),
    }
}

pub(crate) fn extract_perplexities(obj: &Bound<'_, PyAny>, num_nodes: usize) -> PyResult<Vec<f64>> {
    if obj.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(
            "perplexities must be a 1D numeric array or scalar float, not bool",
        ));
    }
    if obj.cast::<PyUntypedArray>().is_err()
        && let Ok(scalar) = obj.extract::<f64>()
    {
        if !scalar.is_finite() {
            return Err(PyValueError::new_err(
                "non-finite floating-point value in perplexities",
            ));
        }
        if scalar <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "perplexities must be > 0, got {scalar}"
            )));
        }
        return Ok(vec![scalar; num_nodes]);
    }
    extract_f64_1d(obj, "perplexities")
}

fn convert_numeric_view2_to_f64<T, F>(
    view: ArrayView2<'_, T>,
    name: &str,
    mut to_f64: F,
) -> PyResult<(Vec<f64>, usize, usize)>
where
    T: Copy,
    F: FnMut(T) -> f64,
{
    let nrows = view.nrows();
    let ncols = view.ncols();
    let mut flat = Vec::with_capacity(nrows.saturating_mul(ncols));
    for (r_idx, row) in view.rows().into_iter().enumerate() {
        for &val in row.iter() {
            let f = to_f64(val);
            if !f.is_finite() {
                return Err(PyValueError::new_err(format!(
                    "{name} contains non-finite value at row {r_idx}"
                )));
            }
            flat.push(f);
        }
    }
    Ok((flat, nrows, ncols))
}

fn extract_numeric_ndarray_2d(
    obj: &Bound<'_, PyAny>,
    untyped: &Bound<'_, PyUntypedArray>,
    name: &str,
) -> PyResult<(Vec<f64>, usize, usize)> {
    if untyped.ndim() != 2 {
        return Err(PyValueError::new_err(format!(
            "{name} must be a 2D numeric array, got {}D",
            untyped.ndim()
        )));
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, f64>>() {
        return convert_numeric_view2_to_f64(arr.as_array(), name, |v| v);
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, f32>>() {
        return convert_numeric_view2_to_f64(arr.as_array(), name, f64::from);
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, i64>>() {
        return convert_numeric_view2_to_f64(arr.as_array(), name, |v| v as f64);
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, i32>>() {
        return convert_numeric_view2_to_f64(arr.as_array(), name, f64::from);
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, u64>>() {
        return convert_numeric_view2_to_f64(arr.as_array(), name, |v| v as f64);
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, u32>>() {
        return convert_numeric_view2_to_f64(arr.as_array(), name, f64::from);
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, isize>>() {
        return convert_numeric_view2_to_f64(arr.as_array(), name, |v| v as f64);
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, usize>>() {
        return convert_numeric_view2_to_f64(arr.as_array(), name, |v| v as f64);
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, i16>>() {
        return convert_numeric_view2_to_f64(arr.as_array(), name, f64::from);
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, u16>>() {
        return convert_numeric_view2_to_f64(arr.as_array(), name, f64::from);
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, i8>>() {
        return convert_numeric_view2_to_f64(arr.as_array(), name, f64::from);
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, u8>>() {
        return convert_numeric_view2_to_f64(arr.as_array(), name, f64::from);
    }
    Err(PyTypeError::new_err(format!(
        "{name} NumPy array must have a numeric dtype"
    )))
}

pub(crate) fn extract_f64_2d(
    obj: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<(Vec<f64>, usize, usize)> {
    if let Ok(untyped) = obj.cast::<PyUntypedArray>() {
        return extract_numeric_ndarray_2d(obj, untyped, name);
    }
    if is_non_sequence_container(obj) {
        return Err(PyTypeError::new_err(format!(
            "{name} must be a 2D numeric array or sequence of sequences"
        )));
    }
    let outer = obj.try_iter().map_err(|_| {
        PyTypeError::new_err(format!(
            "{name} must be a 2D numeric array or sequence of sequences"
        ))
    })?;
    let mut flat = Vec::new();
    let mut nrows = 0usize;
    let mut ncols: Option<usize> = None;

    for row_res in outer {
        let row = row_res?;
        if is_non_sequence_container(&row) {
            return Err(PyTypeError::new_err(format!(
                "{name} rows must be numeric sequences"
            )));
        }
        let inner = row
            .try_iter()
            .map_err(|_| PyTypeError::new_err(format!("{name} rows must be numeric sequences")))?;
        let mut row_len = 0usize;
        for elem_res in inner {
            let elem = elem_res?;
            if elem.is_instance_of::<PyBool>() {
                return Err(PyTypeError::new_err(format!(
                    "{name} entries must be numeric, not bool"
                )));
            }
            let val = elem
                .extract::<f64>()
                .map_err(|_| PyTypeError::new_err(format!("{name} entries must be numeric")))?;
            if !val.is_finite() {
                return Err(PyValueError::new_err(format!(
                    "{name} contains non-finite value at row {nrows}"
                )));
            }
            flat.push(val);
            row_len += 1;
        }
        match ncols {
            None => ncols = Some(row_len),
            Some(expected) if row_len != expected => {
                return Err(PyValueError::new_err(format!(
                    "{name} has ragged row {nrows}: expected {expected} columns, got {row_len}"
                )));
            }
            Some(_) => {}
        }
        nrows += 1;
    }

    Ok((flat, nrows, ncols.unwrap_or(0)))
}

fn convert_numeric_view2_raw<T, F>(
    view: ArrayView2<'_, T>,
    mut to_f64: F,
) -> (Vec<f64>, usize, usize)
where
    T: Copy,
    F: FnMut(T) -> f64,
{
    let nrows = view.nrows();
    let ncols = view.ncols();
    let mut flat = Vec::with_capacity(nrows.saturating_mul(ncols));
    for row in view.rows() {
        for &val in row.iter() {
            flat.push(to_f64(val));
        }
    }
    (flat, nrows, ncols)
}

pub(crate) fn extract_f64_2d_raw(
    obj: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<(Vec<f64>, usize, usize)> {
    if let Ok(untyped) = obj.cast::<PyUntypedArray>() {
        if untyped.ndim() != 2 {
            return Err(PyValueError::new_err(format!(
                "{name} must be a 2D numeric array, got {}D",
                untyped.ndim()
            )));
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, f64>>() {
            return Ok(convert_numeric_view2_raw(arr.as_array(), |v| v));
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, f32>>() {
            return Ok(convert_numeric_view2_raw(arr.as_array(), f64::from));
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, i64>>() {
            return Ok(convert_numeric_view2_raw(arr.as_array(), |v| v as f64));
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, i32>>() {
            return Ok(convert_numeric_view2_raw(arr.as_array(), f64::from));
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, u64>>() {
            return Ok(convert_numeric_view2_raw(arr.as_array(), |v| v as f64));
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, u32>>() {
            return Ok(convert_numeric_view2_raw(arr.as_array(), f64::from));
        }
        return Err(PyTypeError::new_err(format!(
            "{name} NumPy array must have a numeric dtype"
        )));
    }
    if is_non_sequence_container(obj) {
        return Err(PyTypeError::new_err(format!(
            "{name} must be a 2D numeric array or sequence of sequences"
        )));
    }
    let outer = obj.try_iter().map_err(|_| {
        PyTypeError::new_err(format!(
            "{name} must be a 2D numeric array or sequence of sequences"
        ))
    })?;
    let mut flat = Vec::new();
    let mut nrows = 0usize;
    let mut ncols: Option<usize> = None;

    for row_res in outer {
        let row = row_res?;
        if is_non_sequence_container(&row) {
            return Err(PyTypeError::new_err(format!(
                "{name} rows must be numeric sequences"
            )));
        }
        let inner = row
            .try_iter()
            .map_err(|_| PyTypeError::new_err(format!("{name} rows must be numeric sequences")))?;
        let mut row_len = 0usize;
        for elem_res in inner {
            let elem = elem_res?;
            if elem.is_instance_of::<PyBool>() {
                return Err(PyTypeError::new_err(format!(
                    "{name} entries must be numeric, not bool"
                )));
            }
            let val = elem
                .extract::<f64>()
                .map_err(|_| PyTypeError::new_err(format!("{name} entries must be numeric")))?;
            flat.push(val);
            row_len += 1;
        }
        match ncols {
            None => ncols = Some(row_len),
            Some(expected) if row_len != expected => {
                return Err(PyValueError::new_err(format!(
                    "{name} has ragged row {nrows}: expected {expected} columns, got {row_len}"
                )));
            }
            Some(_) => {}
        }
        nrows += 1;
    }

    Ok((flat, nrows, ncols.unwrap_or(0)))
}

pub(crate) fn extract_f64_1d_or_2d(
    obj: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<(Vec<f64>, usize, usize, bool)> {
    if let Ok(untyped) = obj.cast::<PyUntypedArray>() {
        return match untyped.ndim() {
            1 => {
                let vec = extract_f64_1d(obj, name)?;
                for (idx, &val) in vec.iter().enumerate() {
                    if !val.is_finite() {
                        return Err(PyValueError::new_err(format!(
                            "{name} contains non-finite value at index {idx}"
                        )));
                    }
                }
                let n = vec.len();
                Ok((vec, n, 1, false))
            }
            2 => {
                let (flat, nrows, ncols) = extract_numeric_ndarray_2d(obj, untyped, name)?;
                Ok((flat, nrows, ncols, true))
            }
            other => Err(PyValueError::new_err(format!(
                "{name} must be a 1D (N,) or 2D (N, C) array, got {other}D"
            ))),
        };
    }

    if is_non_sequence_container(obj) {
        return Err(PyTypeError::new_err(format!(
            "{name} must be a 1D or 2D numeric array or sequence"
        )));
    }
    let iter = obj.try_iter().map_err(|_| {
        PyTypeError::new_err(format!(
            "{name} must be a 1D or 2D numeric array or sequence"
        ))
    })?;
    let mut items = Vec::new();
    for item_res in iter {
        items.push(item_res?);
    }
    let Some(first) = items.first() else {
        return Ok((Vec::new(), 0, 1, false));
    };
    if !is_non_sequence_container(first) && first.try_iter().is_ok() {
        let (flat, nrows, ncols) = extract_f64_2d(obj, name)?;
        return Ok((flat, nrows, ncols, true));
    }

    let mut out = Vec::with_capacity(items.len());
    for (idx, item) in items.iter().enumerate() {
        if item.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(format!(
                "{name} entries must be numeric, not bool"
            )));
        }
        let val = item
            .extract::<f64>()
            .map_err(|_| PyTypeError::new_err(format!("{name} entries must be numeric")))?;
        if !val.is_finite() {
            return Err(PyValueError::new_err(format!(
                "{name} contains non-finite value at index {idx}"
            )));
        }
        out.push(val);
    }
    let n = out.len();
    Ok((out, n, 1, false))
}
