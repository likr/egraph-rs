//! Shared NumPy array extraction and validation helpers for k-NN and CSR pair inputs.

use crate::FloatType;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::prelude::*;
use std::borrow::Cow;

fn build_knn_rows_from_view<'a, T, F>(
    indices: numpy::ndarray::ArrayView2<'a, T>,
    distances: numpy::ndarray::ArrayView2<'a, FloatType>,
    n: usize,
    mut convert_idx: F,
) -> PyResult<Vec<Vec<(usize, FloatType)>>>
where
    T: Copy,
    F: FnMut(T) -> PyResult<usize>,
{
    let (rows, cols) = indices.dim();
    if rows != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "knn_indices row count ({}) must match drawing node count ({})",
            rows, n
        )));
    }
    if distances.dim() != (rows, cols) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "knn_distances shape {:?} must match knn_indices shape {:?}",
            distances.dim(),
            (rows, cols)
        )));
    }

    let mut knn = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(cols);
        for j_pos in 0..cols {
            let idx = convert_idx(indices[[i, j_pos]])?;
            if idx >= n {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "knn_indices[{}, {}] = {} is out of range for drawing with {} nodes",
                    i, j_pos, idx, n
                )));
            }
            let dist = distances[[i, j_pos]];
            if !dist.is_finite() || dist < 0.0 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "knn_distances[{}, {}] = {} must be finite and non-negative",
                    i, j_pos, dist
                )));
            }
            row.push((idx, dist));
        }
        knn.push(row);
    }
    Ok(knn)
}

pub(crate) fn check_2d_array(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<()> {
    if let Ok(untyped) = obj.cast::<PyUntypedArray>() {
        if untyped.ndim() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{} must be a 2D array, got {}D",
                name,
                untyped.ndim()
            )));
        }
    }
    Ok(())
}

pub(crate) fn extract_knn_rows(
    n: usize,
    knn_indices: &Bound<'_, PyAny>,
    knn_distances: &Bound<'_, PyAny>,
) -> PyResult<Vec<Vec<(usize, FloatType)>>> {
    check_2d_array(knn_indices, "knn_indices")?;
    check_2d_array(knn_distances, "knn_distances")?;

    let dist_arr = knn_distances
        .extract::<PyReadonlyArray2<'_, FloatType>>()
        .map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("knn_distances must be a 2D float64 numpy array")
        })?;
    let dist_view = dist_arr.as_array();

    if let Ok(arr) = knn_indices.extract::<PyReadonlyArray2<'_, i64>>() {
        return build_knn_rows_from_view(arr.as_array(), dist_view, n, |v| {
            usize::try_from(v).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "knn_indices must contain non-negative integers, got {}",
                    v
                ))
            })
        });
    }
    if let Ok(arr) = knn_indices.extract::<PyReadonlyArray2<'_, i32>>() {
        return build_knn_rows_from_view(arr.as_array(), dist_view, n, |v| {
            usize::try_from(v).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "knn_indices must contain non-negative integers, got {}",
                    v
                ))
            })
        });
    }
    if let Ok(arr) = knn_indices.extract::<PyReadonlyArray2<'_, usize>>() {
        return build_knn_rows_from_view(arr.as_array(), dist_view, n, Ok);
    }
    if let Ok(arr) = knn_indices.extract::<PyReadonlyArray2<'_, isize>>() {
        return build_knn_rows_from_view(arr.as_array(), dist_view, n, |v| {
            usize::try_from(v).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "knn_indices must contain non-negative integers, got {}",
                    v
                ))
            })
        });
    }
    if let Ok(arr) = knn_indices.extract::<PyReadonlyArray2<'_, u64>>() {
        return build_knn_rows_from_view(arr.as_array(), dist_view, n, |v| {
            usize::try_from(v).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "knn_indices value {} overflows usize",
                    v
                ))
            })
        });
    }
    if let Ok(arr) = knn_indices.extract::<PyReadonlyArray2<'_, u32>>() {
        return build_knn_rows_from_view(arr.as_array(), dist_view, n, |v| Ok(v as usize));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "knn_indices must be a 2D numpy integer array (int64, int32, or uintp/usize)",
    ))
}

pub(crate) fn check_1d_array(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<()> {
    if let Ok(untyped) = obj.cast::<PyUntypedArray>() {
        if untyped.ndim() != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{} must be a 1D array, got {}D",
                name,
                untyped.ndim()
            )));
        }
    }
    Ok(())
}

fn convert_index_view_1d<T, F>(
    view: numpy::ndarray::ArrayView1<'_, T>,
    mut convert: F,
) -> PyResult<Vec<usize>>
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

pub(crate) fn extract_index_array_1d_with_isize(
    obj: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<Vec<usize>> {
    // Check ndim before dtype extraction so wrong-dimension arrays consistently raise PyValueError.
    check_1d_array(obj, name)?;

    if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, i64>>() {
        return convert_index_view_1d(arr.as_array(), |v| {
            usize::try_from(v).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "{} must contain non-negative integers, got {}",
                    name, v
                ))
            })
        });
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, i32>>() {
        return convert_index_view_1d(arr.as_array(), |v| {
            usize::try_from(v).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "{} must contain non-negative integers, got {}",
                    name, v
                ))
            })
        });
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, usize>>() {
        return convert_index_view_1d(arr.as_array(), Ok);
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, isize>>() {
        return convert_index_view_1d(arr.as_array(), |v| {
            usize::try_from(v).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "{} must contain non-negative integers, got {}",
                    name, v
                ))
            })
        });
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, u64>>() {
        return convert_index_view_1d(arr.as_array(), |v| {
            usize::try_from(v).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "{} value {} overflows usize",
                    name, v
                ))
            })
        });
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, u32>>() {
        return convert_index_view_1d(arr.as_array(), |v| Ok(v as usize));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "{} must be a 1D numpy integer array (int64, int32, uintp/usize, intp/isize, uint64, or uint32)",
        name
    )))
}

pub(crate) fn extract_f64_array_1d<'py>(
    obj: &'py Bound<'py, PyAny>,
    name: &str,
) -> PyResult<PyReadonlyArray1<'py, FloatType>> {
    check_1d_array(obj, name)?;
    obj.extract::<PyReadonlyArray1<'py, FloatType>>()
        .map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "{} must be a 1D float64 numpy array",
                name
            ))
        })
}

pub(crate) fn extract_f64_array_2d<'py>(
    obj: &'py Bound<'py, PyAny>,
    name: &str,
) -> PyResult<PyReadonlyArray2<'py, FloatType>> {
    check_2d_array(obj, name)?;
    obj.extract::<PyReadonlyArray2<'py, FloatType>>()
        .map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "{} must be a 2D float64 numpy array",
                name
            ))
        })
}

pub(crate) fn f64_array_as_cow<'a, 'py>(
    arr: &'a PyReadonlyArray1<'py, FloatType>,
) -> Cow<'a, [FloatType]> {
    match arr.as_slice() {
        Ok(slice) => Cow::Borrowed(slice),
        Err(_) => Cow::Owned(arr.as_array().iter().copied().collect()),
    }
}
