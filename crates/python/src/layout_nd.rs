//! Arbitrary-dimensional coordinate buffer (`LayoutNd`) Python bindings.

use egraph::core::LayoutNd;
use numpy::ndarray::{Array2, ArrayViewMut2};
use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::array_extract::extract_f64_2d;
use crate::ffi_convert::{ScalarDtype, StrictUsize, core_error_to_py};

pub(crate) enum LayoutNdInner {
    F64(LayoutNd<f64>),
    F32(LayoutNd<f32>),
}

/// Arbitrary-dimensional node coordinate buffer backed by `LayoutNd` with zero-copy NumPy views.
#[pyclass(name = "LayoutNd", module = "egraph._core")]
pub struct PyLayoutNd {
    inner: LayoutNdInner,
}

impl PyLayoutNd {
    pub(crate) fn from_f64_nd(nd: LayoutNd<f64>) -> Self {
        Self {
            inner: LayoutNdInner::F64(nd),
        }
    }

    pub(crate) fn from_f32_nd(nd: LayoutNd<f32>) -> Self {
        Self {
            inner: LayoutNdInner::F32(nd),
        }
    }
}

fn extract_layout_nd_inner(coords: &Bound<'_, PyAny>) -> PyResult<LayoutNdInner> {
    if let Ok(untyped) = coords.cast::<PyUntypedArray>() {
        if untyped.ndim() != 2 {
            return Err(PyValueError::new_err(format!(
                "coords must be a 2D array of shape (N, D), got {}D",
                untyped.ndim()
            )));
        }
        let shape = untyped.shape();
        let n = shape[0];
        let dim = shape[1];
        if dim == 0 {
            return Err(PyValueError::new_err("LayoutNd dims must be >= 1"));
        }
        if let Ok(arr) = coords.extract::<PyReadonlyArray2<'_, f64>>() {
            let mut flat = Vec::with_capacity(n.saturating_mul(dim));
            for (r_idx, row) in arr.as_array().rows().into_iter().enumerate() {
                for &v in row.iter() {
                    if !v.is_finite() {
                        return Err(PyValueError::new_err(format!(
                            "coords contains non-finite value at row {r_idx}"
                        )));
                    }
                    flat.push(v);
                }
            }
            return LayoutNd::from_flat(n, dim, flat)
                .map(LayoutNdInner::F64)
                .map_err(core_error_to_py);
        }
        if let Ok(arr) = coords.extract::<PyReadonlyArray2<'_, f32>>() {
            let mut flat = Vec::with_capacity(n.saturating_mul(dim));
            for (r_idx, row) in arr.as_array().rows().into_iter().enumerate() {
                for &v in row.iter() {
                    if !v.is_finite() {
                        return Err(PyValueError::new_err(format!(
                            "coords contains non-finite value at row {r_idx}"
                        )));
                    }
                    flat.push(v);
                }
            }
            return LayoutNd::from_flat(n, dim, flat)
                .map(LayoutNdInner::F32)
                .map_err(core_error_to_py);
        }
        return Err(PyTypeError::new_err(
            "coords NumPy array must have dtype float64 or float32",
        ));
    }

    let (flat, nrows, ncols) = extract_f64_2d(coords, "coords")?;
    LayoutNd::from_flat(nrows, ncols, flat)
        .map(LayoutNdInner::F64)
        .map_err(core_error_to_py)
}

#[pymethods]
impl PyLayoutNd {
    /// Create a zero-initialized `LayoutNd` for `n` vertices in `dim` dimensions.
    #[staticmethod]
    #[pyo3(
        signature = (n, dim, dtype = ScalarDtype::F64),
        text_signature = "(n, dim, dtype='float64')"
    )]
    fn zeros(n: StrictUsize, dim: StrictUsize, dtype: ScalarDtype) -> PyResult<Self> {
        let inner = match dtype {
            ScalarDtype::F64 => LayoutNd::<f64>::zeros(n.0, dim.0)
                .map(LayoutNdInner::F64)
                .map_err(core_error_to_py)?,
            ScalarDtype::F32 => LayoutNd::<f32>::zeros(n.0, dim.0)
                .map(LayoutNdInner::F32)
                .map_err(core_error_to_py)?,
        };
        Ok(Self { inner })
    }

    /// Create a `LayoutNd` from an `(N, D)` coordinate array or nested sequence.
    #[staticmethod]
    #[pyo3(signature = (coords), text_signature = "(coords)")]
    fn from_coords(coords: &Bound<'_, PyAny>) -> PyResult<Self> {
        extract_layout_nd_inner(coords).map(|inner| Self { inner })
    }

    /// Number of vertices in the layout.
    #[getter]
    fn num_nodes(&self) -> usize {
        match &self.inner {
            LayoutNdInner::F64(nd) => nd.num_nodes(),
            LayoutNdInner::F32(nd) => nd.num_nodes(),
        }
    }

    /// Embedding dimensionality `D`.
    #[getter]
    fn dim(&self) -> usize {
        match &self.inner {
            LayoutNdInner::F64(nd) => nd.dims(),
            LayoutNdInner::F32(nd) => nd.dims(),
        }
    }

    /// Scalar coordinate precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        match &self.inner {
            LayoutNdInner::F64(_) => ScalarDtype::F64.as_str(),
            LayoutNdInner::F32(_) => ScalarDtype::F32.as_str(),
        }
    }

    /// Return the number of vertices in the layout.
    fn __len__(&self) -> usize {
        self.num_nodes()
    }

    /// Return `True` if the layout contains zero vertices.
    fn is_empty(&self) -> bool {
        self.num_nodes() == 0
    }

    /// Return `True` if every coordinate in the layout is finite.
    fn is_all_finite(&self) -> bool {
        match &self.inner {
            LayoutNdInner::F64(nd) => nd.is_all_finite(),
            LayoutNdInner::F32(nd) => nd.is_all_finite(),
        }
    }

    /// Return an `(N, D)` NumPy array of coordinates (zero-copy view when `copy=False`).
    #[pyo3(signature = (copy = false), text_signature = "($self, copy=False)")]
    fn coords<'py>(slf: Bound<'py, Self>, copy: bool) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        if copy {
            let borrowed = slf.borrow();
            return match &borrowed.inner {
                LayoutNdInner::F64(nd) => {
                    let arr =
                        Array2::from_shape_vec((nd.num_nodes(), nd.dims()), nd.coords().to_vec())
                            .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    Ok(PyArray2::from_owned_array(py, arr).into_any())
                }
                LayoutNdInner::F32(nd) => {
                    let arr =
                        Array2::from_shape_vec((nd.num_nodes(), nd.dims()), nd.coords().to_vec())
                            .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    Ok(PyArray2::from_owned_array(py, arr).into_any())
                }
            };
        }

        let container = slf.clone().into_any();
        let mut borrowed = slf.borrow_mut();
        match &mut borrowed.inner {
            LayoutNdInner::F64(nd) => {
                let n = nd.num_nodes();
                let d = nd.dims();
                let view = ArrayViewMut2::from_shape((n, d), nd.coords_mut())
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                // SAFETY: `container` is `slf` (`PyLayoutNd`), registered as the NumPy array's
                // `base` object so the backing `LayoutNd` buffer outlives any view, and
                // `LayoutNd` never reallocates its coordinate `Vec` after creation.
                Ok(unsafe { PyArray2::borrow_from_array(&view, container) }.into_any())
            }
            LayoutNdInner::F32(nd) => {
                let n = nd.num_nodes();
                let d = nd.dims();
                let view = ArrayViewMut2::from_shape((n, d), nd.coords_mut())
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                // SAFETY: `container` is `slf` (`PyLayoutNd`), registered as the NumPy array's
                // `base` object so the backing `LayoutNd` buffer outlives any view, and
                // `LayoutNd` never reallocates its coordinate `Vec` after creation.
                Ok(unsafe { PyArray2::borrow_from_array(&view, container) }.into_any())
            }
        }
    }

    /// Return a concise string representation of the `LayoutNd`.
    fn __repr__(&self) -> String {
        format!(
            "LayoutNd(num_nodes={}, dim={}, dtype='{}')",
            self.num_nodes(),
            self.dim(),
            self.dtype()
        )
    }
}

/// Register `LayoutNd` on the Python extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLayoutNd>()?;
    Ok(())
}
