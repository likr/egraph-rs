//! Arbitrary-dimensional coordinate buffer (`LayoutNd`) Python bindings.

use egraph::core::LayoutNd;
use numpy::ndarray::{Array2, ArrayViewMut2};
use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::array_extract::extract_f64_2d;
use crate::errors::{dimension_mismatch_err, invalid_input_err, non_finite_value_err};
use crate::ffi_convert::{ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py};

#[derive(Clone)]
pub(crate) enum LayoutNdInner {
    F64(LayoutNd<f64>),
    F32(LayoutNd<f32>),
}

/// Arbitrary-dimensional node coordinate buffer backed by `LayoutNd` with zero-copy NumPy views.
#[pyclass(name = "LayoutNd", module = "egraph._core", from_py_object)]
#[derive(Clone)]
pub struct PyLayoutNd {
    inner: LayoutNdInner,
}

impl PyLayoutNd {
    pub(crate) fn inner(&self) -> &LayoutNdInner {
        &self.inner
    }

    pub(crate) fn inner_mut(&mut self) -> &mut LayoutNdInner {
        &mut self.inner
    }

    pub(crate) fn dtype_enum(&self) -> ScalarDtype {
        match &self.inner {
            LayoutNdInner::F64(_) => ScalarDtype::F64,
            LayoutNdInner::F32(_) => ScalarDtype::F32,
        }
    }

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
            return Err(dimension_mismatch_err(format!(
                "coords must be a 2D array of shape (N, D), got {}D",
                untyped.ndim()
            )));
        }
        let shape = untyped.shape();
        let n = shape[0];
        let dims = shape[1];
        if dims == 0 {
            return Err(invalid_input_err("LayoutNd dims must be >= 1"));
        }
        if let Ok(arr) = coords.extract::<PyReadonlyArray2<'_, f64>>() {
            let mut flat = Vec::with_capacity(n.saturating_mul(dims));
            for (r_idx, row) in arr.as_array().rows().into_iter().enumerate() {
                for &v in row.iter() {
                    if !v.is_finite() {
                        return Err(non_finite_value_err(format!(
                            "coords contains non-finite value at row {r_idx}"
                        )));
                    }
                    flat.push(v);
                }
            }
            return LayoutNd::from_flat_vec(n, dims, flat)
                .map(LayoutNdInner::F64)
                .map_err(core_error_to_py);
        }
        if let Ok(arr) = coords.extract::<PyReadonlyArray2<'_, f32>>() {
            let mut flat = Vec::with_capacity(n.saturating_mul(dims));
            for (r_idx, row) in arr.as_array().rows().into_iter().enumerate() {
                for &v in row.iter() {
                    if !v.is_finite() {
                        return Err(non_finite_value_err(format!(
                            "coords contains non-finite value at row {r_idx}"
                        )));
                    }
                    flat.push(v);
                }
            }
            return LayoutNd::from_flat_vec(n, dims, flat)
                .map(LayoutNdInner::F32)
                .map_err(core_error_to_py);
        }
        return Err(PyTypeError::new_err(
            "coords NumPy array must have dtype float64 or float32",
        ));
    }

    let (flat, nrows, ncols) = extract_f64_2d(coords, "coords")?;
    LayoutNd::from_flat_vec(nrows, ncols, flat)
        .map(LayoutNdInner::F64)
        .map_err(core_error_to_py)
}

#[pymethods]
impl PyLayoutNd {
    /// Create a zero-initialized `LayoutNd` for `n` vertices in `dims` dimensions.
    #[staticmethod]
    #[pyo3(
        signature = (n, dims, dtype = ScalarDtype::F64),
        text_signature = "(n, dims, dtype='float64')"
    )]
    fn zeros(n: StrictUsize, dims: StrictUsize, dtype: ScalarDtype) -> PyResult<Self> {
        let inner = match dtype {
            ScalarDtype::F64 => LayoutNd::<f64>::zeros(n.0, dims.0)
                .map(LayoutNdInner::F64)
                .map_err(core_error_to_py)?,
            ScalarDtype::F32 => LayoutNd::<f32>::zeros(n.0, dims.0)
                .map(LayoutNdInner::F32)
                .map_err(core_error_to_py)?,
        };
        Ok(Self { inner })
    }

    /// Create a deterministically initialized `LayoutNd` for `n` vertices in `dims` dimensions.
    #[staticmethod]
    #[pyo3(
        signature = (n, dims, *, seed = StrictU64(42), scale = StrictF64(1.0), dtype = ScalarDtype::F64),
        text_signature = "(n, dims, *, seed=42, scale=1.0, dtype='float64')"
    )]
    fn init(
        n: StrictUsize,
        dims: StrictUsize,
        seed: StrictU64,
        scale: StrictF64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let inner = match dtype {
            ScalarDtype::F64 => LayoutNd::<f64>::init(n.0, dims.0, seed.0, scale.0)
                .map(LayoutNdInner::F64)
                .map_err(core_error_to_py)?,
            ScalarDtype::F32 => LayoutNd::<f32>::init(n.0, dims.0, seed.0, scale.0)
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

    /// Return a deep copy of this `LayoutNd` preserving `dtype`.
    fn copy(&self) -> Self {
        self.clone()
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
    fn dims(&self) -> usize {
        match &self.inner {
            LayoutNdInner::F64(nd) => nd.dims(),
            LayoutNdInner::F32(nd) => nd.dims(),
        }
    }

    /// Scalar coordinate precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.dtype_enum().as_str()
    }

    /// Return the number of vertices in the layout.
    fn __len__(&self) -> usize {
        self.num_nodes()
    }

    /// Return `True` if the layout contains zero vertices.
    fn is_empty(&self) -> bool {
        self.num_nodes() == 0
    }

    /// Return coordinate `(i, d)` as `float`.
    #[pyo3(signature = (i, d), text_signature = "($self, i, d)")]
    fn get(&self, i: StrictUsize, d: StrictUsize) -> PyResult<f64> {
        match &self.inner {
            LayoutNdInner::F64(nd) => nd.get(i.0, d.0).map_err(core_error_to_py),
            LayoutNdInner::F32(nd) => nd.get(i.0, d.0).map(f64::from).map_err(core_error_to_py),
        }
    }

    /// Set coordinate `(i, d)` in place after validating bounds and finiteness.
    #[pyo3(signature = (i, d, value), text_signature = "($self, i, d, value)")]
    fn set(&mut self, i: StrictUsize, d: StrictUsize, value: StrictF64) -> PyResult<()> {
        if !value.0.is_finite() {
            return Err(non_finite_value_err("LayoutNd.set value must be finite"));
        }
        match &mut self.inner {
            LayoutNdInner::F64(nd) => nd.set(i.0, d.0, value.0).map_err(core_error_to_py),
            LayoutNdInner::F32(nd) => {
                let vf = value.0 as f32;
                if !vf.is_finite() {
                    return Err(non_finite_value_err(format!(
                        "LayoutNd.set value {} overflows float32",
                        value.0
                    )));
                }
                nd.set(i.0, d.0, vf).map_err(core_error_to_py)
            }
        }
    }

    /// Center coordinates at zero mean per axis and scale isotropically in place to `target_std`.
    #[pyo3(signature = (target_std = StrictF64(1.0)), text_signature = "($self, target_std=1.0)")]
    fn normalize_std(&mut self, target_std: StrictF64) -> PyResult<()> {
        match &mut self.inner {
            LayoutNdInner::F64(nd) => nd
                .center_and_normalize_to_std(target_std.0)
                .map_err(core_error_to_py),
            LayoutNdInner::F32(nd) => {
                if !target_std.0.is_finite() || target_std.0 <= 0.0 {
                    return Err(invalid_input_err(
                        "target_std must be finite and strictly positive",
                    ));
                }
                let t32 = target_std.0 as f32;
                if !t32.is_finite() || t32 <= 0.0 {
                    return Err(invalid_input_err(format!(
                        "target_std {} is out of range for float32",
                        target_std.0
                    )));
                }
                nd.center_and_normalize_to_std(f64::from(t32))
                    .map_err(core_error_to_py)
            }
        }
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
                            .map_err(|e| invalid_input_err(e.to_string()))?;
                    Ok(PyArray2::from_owned_array(py, arr).into_any())
                }
                LayoutNdInner::F32(nd) => {
                    let arr =
                        Array2::from_shape_vec((nd.num_nodes(), nd.dims()), nd.coords().to_vec())
                            .map_err(|e| invalid_input_err(e.to_string()))?;
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
                    .map_err(|e| invalid_input_err(e.to_string()))?;
                // SAFETY: `container` is `slf` (`PyLayoutNd`), registered as the NumPy array's
                // `base` object so the backing `LayoutNd` buffer outlives any view, and
                // `LayoutNd` never reallocates its coordinate `Vec` after creation.
                Ok(unsafe { PyArray2::borrow_from_array(&view, container) }.into_any())
            }
            LayoutNdInner::F32(nd) => {
                let n = nd.num_nodes();
                let d = nd.dims();
                let view = ArrayViewMut2::from_shape((n, d), nd.coords_mut())
                    .map_err(|e| invalid_input_err(e.to_string()))?;
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
            "LayoutNd(num_nodes={}, dims={}, dtype='{}')",
            self.num_nodes(),
            self.dims(),
            self.dtype()
        )
    }
}

/// Register `LayoutNd` on the Python extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLayoutNd>()?;
    Ok(())
}
