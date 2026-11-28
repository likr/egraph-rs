//! Typed exception hierarchy and CoreError/FfiError translation for `egraph._core`.

use egraph::core::CoreError;
use egraph_ffi::{ErrorKind, FfiError};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyTuple, PyType};

pyo3::create_exception!(
    egraph._core,
    EgraphError,
    PyValueError,
    "Base exception for domain and validation errors raised by egraph."
);
pyo3::create_exception!(
    egraph._core,
    InvalidInputError,
    EgraphError,
    "Raised when an argument or configuration parameter has an invalid value."
);
pyo3::create_exception!(
    egraph._core,
    DimensionMismatchError,
    EgraphError,
    "Raised when array shapes, layout dimensions, or node counts do not match."
);
pyo3::create_exception!(
    egraph._core,
    InvalidCsrError,
    EgraphError,
    "Raised when CSR indptr/indices/weights fail structural validation."
);
pyo3::create_exception!(
    egraph._core,
    NonFiniteValueError,
    EgraphError,
    "Raised when coordinates, weights, or parameters contain NaN or Inf."
);
pyo3::create_exception!(
    egraph._core,
    UnsupportedOperationError,
    EgraphError,
    "Raised when a requested target, backend, or operation is unsupported."
);
pyo3::create_exception!(
    egraph._core,
    NumericalBreakdownError,
    EgraphError,
    "Raised when a solver or linear algebra routine encounters numerical breakdown."
);
pyo3::create_exception!(
    egraph._core,
    BackendUnavailableError,
    EgraphError,
    "Raised when a requested compute or acceleration backend is unavailable."
);
pyo3::create_exception!(
    egraph._core,
    DeviceLimitExceededError,
    EgraphError,
    "Raised when a workload or buffer allocation exceeds backend device limits."
);

static INDEX_OUT_OF_BOUNDS_TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();

pub(crate) fn index_out_of_bounds_type(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    INDEX_OUT_OF_BOUNDS_TYPE
        .get_or_try_init(py, || {
            let bases = PyTuple::new(
                py,
                [
                    py.get_type::<EgraphError>().into_any(),
                    py.get_type::<PyIndexError>().into_any(),
                ],
            )?;
            let name = c"egraph._core.IndexOutOfBoundsError";
            let doc = c"Raised when a node, edge, or buffer index is out of bounds.";
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("__doc__", doc.to_str()?)?;
            // Safety: `name` is NUL-terminated, `bases` is a valid tuple of exception types,
            // and `dict` is a valid Python dict.
            let raw = unsafe {
                pyo3::ffi::PyErr_NewException(name.as_ptr(), bases.as_ptr(), dict.as_ptr())
            };
            if raw.is_null() {
                return Err(PyErr::fetch(py));
            }
            let obj = unsafe { Bound::from_owned_ptr(py, raw) };
            obj.cast_into::<PyType>()
                .map(Bound::unbind)
                .map_err(Into::into)
        })
        .map(|ty| ty.bind(py))
}

pub(crate) fn invalid_input_err(msg: impl Into<String>) -> PyErr {
    InvalidInputError::new_err(msg.into())
}

pub(crate) fn dimension_mismatch_err(msg: impl Into<String>) -> PyErr {
    DimensionMismatchError::new_err(msg.into())
}

pub(crate) fn index_out_of_bounds_err(msg: impl Into<String>) -> PyErr {
    let msg = msg.into();
    Python::attach(|py| match index_out_of_bounds_type(py) {
        Ok(ty) => PyErr::from_type(ty.clone(), (msg,)),
        Err(_) => PyIndexError::new_err(msg),
    })
}

pub(crate) fn invalid_csr_err(msg: impl Into<String>) -> PyErr {
    InvalidCsrError::new_err(msg.into())
}

pub(crate) fn non_finite_value_err(msg: impl Into<String>) -> PyErr {
    NonFiniteValueError::new_err(msg.into())
}

pub(crate) fn unsupported_operation_err(msg: impl Into<String>) -> PyErr {
    UnsupportedOperationError::new_err(msg.into())
}

pub(crate) fn numerical_breakdown_err(msg: impl Into<String>) -> PyErr {
    NumericalBreakdownError::new_err(msg.into())
}

pub(crate) fn ffi_error_to_py(err: FfiError) -> PyErr {
    match err.kind {
        ErrorKind::IndexOutOfBounds => index_out_of_bounds_err(err.message),
        ErrorKind::InvalidInput => invalid_input_err(err.message),
        ErrorKind::DimensionMismatch => dimension_mismatch_err(err.message),
        ErrorKind::InvalidCsr => invalid_csr_err(err.message),
        ErrorKind::NonFiniteValue => non_finite_value_err(err.message),
        ErrorKind::UnsupportedOperation => unsupported_operation_err(err.message),
        ErrorKind::NumericalBreakdown => numerical_breakdown_err(err.message),
        ErrorKind::BackendUnavailable => BackendUnavailableError::new_err(err.message),
        ErrorKind::DeviceLimitExceeded => DeviceLimitExceededError::new_err(err.message),
        _ => EgraphError::new_err(err.message),
    }
}

pub(crate) fn core_error_to_py(err: CoreError) -> PyErr {
    let msg = err.to_string();
    match err {
        CoreError::IndexOutOfBounds { .. } => index_out_of_bounds_err(msg),
        CoreError::InvalidInput(_) => invalid_input_err(msg),
        CoreError::DimensionMismatch { .. } => dimension_mismatch_err(msg),
        CoreError::InvalidCsr(_) => invalid_csr_err(msg),
        CoreError::NonFiniteValue { .. } => non_finite_value_err(msg),
        CoreError::UnsupportedOperation(_) => unsupported_operation_err(msg),
        CoreError::NumericalBreakdown(_) => numerical_breakdown_err(msg),
        CoreError::BackendUnavailable(_) => BackendUnavailableError::new_err(msg),
        CoreError::DeviceLimitExceeded(_) => DeviceLimitExceededError::new_err(msg),
        _ => EgraphError::new_err(msg),
    }
}

/// Register the `EgraphError` exception hierarchy on `egraph._core`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("EgraphError", py.get_type::<EgraphError>())?;
    m.add("InvalidInputError", py.get_type::<InvalidInputError>())?;
    m.add(
        "DimensionMismatchError",
        py.get_type::<DimensionMismatchError>(),
    )?;
    m.add("IndexOutOfBoundsError", index_out_of_bounds_type(py)?)?;
    m.add("InvalidCsrError", py.get_type::<InvalidCsrError>())?;
    m.add("NonFiniteValueError", py.get_type::<NonFiniteValueError>())?;
    m.add(
        "UnsupportedOperationError",
        py.get_type::<UnsupportedOperationError>(),
    )?;
    m.add(
        "NumericalBreakdownError",
        py.get_type::<NumericalBreakdownError>(),
    )?;
    m.add(
        "BackendUnavailableError",
        py.get_type::<BackendUnavailableError>(),
    )?;
    m.add(
        "DeviceLimitExceededError",
        py.get_type::<DeviceLimitExceededError>(),
    )?;
    Ok(())
}
