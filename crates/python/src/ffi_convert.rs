//! Shared FFI error translation, scalar type extractors, and enum normalization helpers.

use egraph::core::{CoreError, SolverStatus};
use egraph_ffi::{ErrorKind, FfiError};
use numpy::ndarray::Array2;
use numpy::{
    Element, PyArray2, PyArrayDescr, PyArrayDescrMethods, PyReadonlyArray1, PyUntypedArray,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyByteArray, PyBytes, PyDict, PyString, PyType};

pub(crate) fn ffi_error_to_py(err: FfiError) -> PyErr {
    match err.kind {
        ErrorKind::IndexOutOfBounds => PyIndexError::new_err(err.message),
        ErrorKind::InvalidInput
        | ErrorKind::DimensionMismatch
        | ErrorKind::InvalidCsr
        | ErrorKind::NonFiniteValue
        | ErrorKind::UnsupportedOperation
        | ErrorKind::NumericalBreakdown => PyValueError::new_err(err.message),
    }
}

pub(crate) fn core_error_to_py(err: CoreError) -> PyErr {
    match err {
        CoreError::IndexOutOfBounds { .. } => PyIndexError::new_err(err.to_string()),
        CoreError::InvalidInput(_)
        | CoreError::DimensionMismatch { .. }
        | CoreError::InvalidCsr(_)
        | CoreError::NonFiniteValue { .. }
        | CoreError::UnsupportedOperation(_)
        | CoreError::NumericalBreakdown(_) => PyValueError::new_err(err.to_string()),
    }
}

pub(crate) fn is_non_sequence_container(obj: &Bound<'_, PyAny>) -> bool {
    obj.is_instance_of::<PyString>()
        || obj.is_instance_of::<PyBytes>()
        || obj.is_instance_of::<PyByteArray>()
        || obj.is_instance_of::<PyDict>()
}

pub(crate) fn normalize_enum_key(raw: &str) -> String {
    raw.trim()
        .chars()
        .filter(|c| *c != '_' && *c != '-' && *c != ' ')
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

pub(crate) fn solver_status_str(status: SolverStatus) -> &'static str {
    match status {
        SolverStatus::Initialized => "initialized",
        SolverStatus::Running => "running",
        SolverStatus::Converged => "converged",
        SolverStatus::Completed => "completed",
    }
}

pub(crate) fn extract_u32_item(item: &Bound<'_, PyAny>, name: &str) -> PyResult<u32> {
    if item.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(format!(
            "{name} entries must be integers, not bool"
        )));
    }
    let val = item
        .extract::<i128>()
        .map_err(|_| PyTypeError::new_err(format!("{name} entries must be integers")))?;
    if val < 0 || val > i128::from(u32::MAX) {
        return Err(PyValueError::new_err(format!(
            "{name} entry {val} out of range for u32"
        )));
    }
    Ok(val as u32)
}

fn convert_int_slice_to_u32<T, F>(
    view: numpy::ndarray::ArrayView1<'_, T>,
    name: &str,
    mut convert: F,
) -> PyResult<Vec<u32>>
where
    T: Copy,
    F: FnMut(T) -> PyResult<u32>,
{
    let mut out = Vec::with_capacity(view.len());
    for &val in view.iter() {
        out.push(convert(val)?);
    }
    let _ = name;
    Ok(out)
}

pub(crate) fn extract_u32_1d(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<u32>> {
    if let Ok(untyped) = obj.cast::<PyUntypedArray>() {
        if untyped.ndim() != 1 {
            return Err(PyValueError::new_err(format!(
                "{name} must be a 1D array, got {}D",
                untyped.ndim()
            )));
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, u32>>() {
            return Ok(match arr.as_slice() {
                Ok(slice) => slice.to_vec(),
                Err(_) => arr.as_array().iter().copied().collect(),
            });
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, i64>>() {
            return convert_int_slice_to_u32(arr.as_array(), name, |v| {
                u32::try_from(v).map_err(|_| {
                    PyValueError::new_err(format!("{name} entry {v} out of range for u32"))
                })
            });
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, u64>>() {
            return convert_int_slice_to_u32(arr.as_array(), name, |v| {
                u32::try_from(v).map_err(|_| {
                    PyValueError::new_err(format!("{name} entry {v} out of range for u32"))
                })
            });
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, i32>>() {
            return convert_int_slice_to_u32(arr.as_array(), name, |v| {
                u32::try_from(v).map_err(|_| {
                    PyValueError::new_err(format!("{name} entry {v} out of range for u32"))
                })
            });
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, usize>>() {
            return convert_int_slice_to_u32(arr.as_array(), name, |v| {
                u32::try_from(v).map_err(|_| {
                    PyValueError::new_err(format!("{name} entry {v} out of range for u32"))
                })
            });
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, isize>>() {
            return convert_int_slice_to_u32(arr.as_array(), name, |v| {
                u32::try_from(v).map_err(|_| {
                    PyValueError::new_err(format!("{name} entry {v} out of range for u32"))
                })
            });
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, u16>>() {
            return Ok(arr.as_array().iter().map(|&v| u32::from(v)).collect());
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, u8>>() {
            return Ok(arr.as_array().iter().map(|&v| u32::from(v)).collect());
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, i16>>() {
            return convert_int_slice_to_u32(arr.as_array(), name, |v| {
                u32::try_from(v).map_err(|_| {
                    PyValueError::new_err(format!("{name} entry {v} out of range for u32"))
                })
            });
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, i8>>() {
            return convert_int_slice_to_u32(arr.as_array(), name, |v| {
                u32::try_from(v).map_err(|_| {
                    PyValueError::new_err(format!("{name} entry {v} out of range for u32"))
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
        out.push(extract_u32_item(&item, name)?);
    }
    Ok(out)
}

pub(crate) fn extract_optional_weights_1d(
    weights: Option<&Bound<'_, PyAny>>,
    name: &str,
) -> PyResult<Option<Vec<f64>>> {
    let Some(obj) = weights else {
        return Ok(None);
    };
    if obj.is_none() {
        return Ok(None);
    }
    if let Ok(untyped) = obj.cast::<PyUntypedArray>() {
        if untyped.ndim() != 1 {
            return Err(PyValueError::new_err(format!(
                "{name} must be a 1D array, got {}D",
                untyped.ndim()
            )));
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, f64>>() {
            return Ok(Some(match arr.as_slice() {
                Ok(slice) => slice.to_vec(),
                Err(_) => arr.as_array().iter().copied().collect(),
            }));
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, f32>>() {
            return Ok(Some(arr.as_array().iter().map(|&v| f64::from(v)).collect()));
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, i64>>() {
            return Ok(Some(arr.as_array().iter().map(|&v| v as f64).collect()));
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, i32>>() {
            return Ok(Some(arr.as_array().iter().map(|&v| f64::from(v)).collect()));
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, u64>>() {
            return Ok(Some(arr.as_array().iter().map(|&v| v as f64).collect()));
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, u32>>() {
            return Ok(Some(arr.as_array().iter().map(|&v| f64::from(v)).collect()));
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, i16>>() {
            return Ok(Some(arr.as_array().iter().map(|&v| f64::from(v)).collect()));
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, i8>>() {
            return Ok(Some(arr.as_array().iter().map(|&v| f64::from(v)).collect()));
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, u16>>() {
            return Ok(Some(arr.as_array().iter().map(|&v| f64::from(v)).collect()));
        }
        if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, u8>>() {
            return Ok(Some(arr.as_array().iter().map(|&v| f64::from(v)).collect()));
        }
        return Err(PyTypeError::new_err(format!(
            "{name} must be a 1D numeric NumPy array"
        )));
    }

    if is_non_sequence_container(obj) {
        return Err(PyTypeError::new_err(format!(
            "{name} must be a 1D numeric array or sequence"
        )));
    }
    let iter = obj.try_iter().map_err(|_| {
        PyTypeError::new_err(format!("{name} must be a 1D numeric array or sequence"))
    })?;
    let mut out = Vec::new();
    for item_res in iter {
        let item = item_res?;
        if item.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(format!(
                "{name} entries must be numeric, not bool"
            )));
        }
        let val = item
            .extract::<f64>()
            .map_err(|_| PyTypeError::new_err(format!("{name} entries must be numeric")))?;
        out.push(val);
    }
    Ok(Some(out))
}

pub(crate) fn extract_non_negative_usize(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<usize> {
    if obj.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(format!(
            "{name} must be an integer, not bool"
        )));
    }
    let val = obj
        .extract::<i128>()
        .map_err(|_| PyTypeError::new_err(format!("{name} must be an integer")))?;
    if val < 0 {
        return Err(PyValueError::new_err(format!(
            "{name} must be non-negative, got {val}"
        )));
    }
    usize::try_from(val)
        .map_err(|_| PyValueError::new_err(format!("{name} value {val} exceeds usize capacity")))
}

#[derive(Clone, Copy)]
pub(crate) struct StrictUsize(pub(crate) usize);

impl<'a, 'py> FromPyObject<'a, 'py> for StrictUsize {
    type Error = PyErr;

    fn extract(ob: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        extract_non_negative_usize(&ob, "integer argument").map(Self)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct StrictU64(pub(crate) u64);

impl<'a, 'py> FromPyObject<'a, 'py> for StrictU64 {
    type Error = PyErr;

    fn extract(ob: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if ob.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err("seed must be an integer, not bool"));
        }
        let val = ob
            .extract::<i128>()
            .map_err(|_| PyTypeError::new_err("seed must be an integer"))?;
        if val < 0 {
            return Err(PyValueError::new_err(format!(
                "seed must be non-negative, got {val}"
            )));
        }
        u64::try_from(val)
            .map(Self)
            .map_err(|_| PyValueError::new_err(format!("seed value {val} exceeds u64 capacity")))
    }
}

#[derive(Clone, Copy)]
pub(crate) struct StrictF64(pub(crate) f64);

impl<'a, 'py> FromPyObject<'a, 'py> for StrictF64 {
    type Error = PyErr;

    fn extract(ob: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if ob.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err("coordinate must be float, not bool"));
        }
        ob.extract::<f64>()
            .map(Self)
            .map_err(|_| PyTypeError::new_err("coordinate must be numeric"))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScalarDtype {
    F64,
    F32,
}

impl ScalarDtype {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "float64" | "f64" => Some(Self::F64),
            "float32" | "f32" => Some(Self::F32),
            _ => None,
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::F64 => "float64",
            Self::F32 => "float32",
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for ScalarDtype {
    type Error = PyErr;

    fn extract(ob: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let py = ob.py();
        if ob.is_none() || ob.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(
                "dtype must be 'float64'/'f64' or 'float32'/'f32'",
            ));
        }
        if let Ok(s) = ob.extract::<String>() {
            return Self::from_name(&s).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "unsupported dtype {s:?}; expected 'float64'/'f64' or 'float32'/'f32'"
                ))
            });
        }
        if let Ok(descr) = ob.cast::<PyArrayDescr>() {
            if descr.is_equiv_to(&f64::get_dtype(py)) {
                return Ok(Self::F64);
            }
            if descr.is_equiv_to(&f32::get_dtype(py)) {
                return Ok(Self::F32);
            }
            return Err(PyValueError::new_err(format!(
                "unsupported NumPy dtype {descr:?}; expected float64 or float32"
            )));
        }
        if ob.is_instance_of::<PyType>() {
            let np = PyModule::import(py, "numpy")?;
            let descr_any = np.getattr("dtype")?.call1((&*ob,))?;
            let descr = descr_any.cast::<PyArrayDescr>()?;
            if descr.is_equiv_to(&f64::get_dtype(py)) {
                return Ok(Self::F64);
            }
            if descr.is_equiv_to(&f32::get_dtype(py)) {
                return Ok(Self::F32);
            }
            return Err(PyValueError::new_err(format!(
                "unsupported dtype {ob:?}; expected float64 or float32"
            )));
        }
        Err(PyTypeError::new_err(
            "dtype must be a string ('float64'/'f64' or 'float32'/'f32') or NumPy dtype",
        ))
    }
}

pub(crate) fn pairs_to_u32_array2<'py>(
    py: Python<'py>,
    pairs: Vec<(u32, u32)>,
) -> PyResult<Bound<'py, PyArray2<u32>>> {
    let m = pairs.len();
    let mut flat = Vec::with_capacity(m * 2);
    for (u, v) in pairs {
        flat.push(u);
        flat.push(v);
    }
    let arr =
        Array2::from_shape_vec((m, 2), flat).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

#[cfg(test)]
mod tests {
    use super::ScalarDtype;

    #[test]
    fn test_scalar_dtype_parsing() {
        assert!(ScalarDtype::from_name("float64") == Some(ScalarDtype::F64));
        assert!(ScalarDtype::from_name("f64") == Some(ScalarDtype::F64));
        assert!(ScalarDtype::from_name("float32") == Some(ScalarDtype::F32));
        assert!(ScalarDtype::from_name("f32") == Some(ScalarDtype::F32));
        assert!(ScalarDtype::from_name("int32").is_none());
        assert_eq!(ScalarDtype::F64.as_str(), "float64");
        assert_eq!(ScalarDtype::F32.as_str(), "float32");
    }
}
