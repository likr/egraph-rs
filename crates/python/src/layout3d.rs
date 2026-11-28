//! 3D interleaved coordinate buffer (`Layout3d`) Python bindings and GIL-free dispatch macro.

use egraph::core::Layout3d;
use egraph_ffi::InterleavedLayout3d;
use numpy::PyArray2;
use numpy::ndarray::{Array2, ArrayViewMut2};
use pyo3::prelude::*;

use crate::errors::{
    core_error_to_py, dimension_mismatch_err, ffi_error_to_py, invalid_input_err,
    non_finite_value_err,
};
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, extract_non_negative_usize,
    extract_optional_weights_1d,
};
use crate::layout3d_extract::extract_layout3d_inner;

#[derive(Clone)]
pub(crate) enum Layout3dInner {
    F64(InterleavedLayout3d<f64>),
    F32(InterleavedLayout3d<f32>),
}

/// 3D node coordinate buffer backed by `InterleavedLayout3d` for zero-copy NumPy views.
#[pyclass(name = "Layout3d", module = "egraph._core", from_py_object)]
#[derive(Clone)]
pub struct PyLayout3d {
    inner: Layout3dInner,
}

impl PyLayout3d {
    pub(crate) fn inner(&self) -> &Layout3dInner {
        &self.inner
    }

    pub(crate) fn inner_mut(&mut self) -> &mut Layout3dInner {
        &mut self.inner
    }

    pub(crate) fn from_f64_layout(layout: &Layout3d<f64>) -> Self {
        Self {
            inner: Layout3dInner::F64(InterleavedLayout3d::from_layout3d(layout)),
        }
    }

    pub(crate) fn from_f32_layout(layout: &Layout3d<f32>) -> Self {
        Self {
            inner: Layout3dInner::F32(InterleavedLayout3d::from_layout3d(layout)),
        }
    }
}

#[pymethods]
impl PyLayout3d {
    /// Create a zero-initialized `Layout3d` for `n` vertices.
    #[staticmethod]
    #[pyo3(signature = (n, dtype = ScalarDtype::F64), text_signature = "(n, dtype='float64')")]
    fn zeros(n: &Bound<'_, PyAny>, dtype: ScalarDtype) -> PyResult<Self> {
        let n_usize = extract_non_negative_usize(n, "n")?;
        let inner = match dtype {
            ScalarDtype::F64 => Layout3dInner::F64(
                InterleavedLayout3d::<f64>::try_zeros(n_usize).map_err(ffi_error_to_py)?,
            ),
            ScalarDtype::F32 => Layout3dInner::F32(
                InterleavedLayout3d::<f32>::try_zeros(n_usize).map_err(ffi_error_to_py)?,
            ),
        };
        Ok(Self { inner })
    }

    /// Create a deterministically initialized `Layout3d` for `n` vertices using `seed` and `scale`.
    #[staticmethod]
    #[pyo3(
        signature = (n, *, seed = StrictU64(42), scale = StrictF64(1.0), dtype = ScalarDtype::F64),
        text_signature = "(n, *, seed=42, scale=1.0, dtype='float64')"
    )]
    fn init(
        n: &Bound<'_, PyAny>,
        seed: StrictU64,
        scale: StrictF64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let n_usize = extract_non_negative_usize(n, "n")?;
        let inner = match dtype {
            ScalarDtype::F64 => {
                let soa =
                    Layout3d::<f64>::init(n_usize, seed.0, scale.0).map_err(core_error_to_py)?;
                Layout3dInner::F64(InterleavedLayout3d::from_layout3d(&soa))
            }
            ScalarDtype::F32 => {
                let soa =
                    Layout3d::<f32>::init(n_usize, seed.0, scale.0).map_err(core_error_to_py)?;
                Layout3dInner::F32(InterleavedLayout3d::from_layout3d(&soa))
            }
        };
        Ok(Self { inner })
    }

    /// Return a deep copy of this `Layout3d` preserving `dtype`.
    fn copy(&self) -> Self {
        self.clone()
    }

    /// Create a `Layout3d` from an `(N, 3)` coordinate array or sequence of `(x, y, z)` triples.
    #[staticmethod]
    fn from_coords(coords: &Bound<'_, PyAny>) -> PyResult<Self> {
        extract_layout3d_inner(coords).map(|inner| Self { inner })
    }

    /// Create a `Layout3d` from separate 1D coordinate arrays `x`, `y`, and `z`.
    #[staticmethod]
    #[pyo3(
        signature = (x, y, z, *, dtype = ScalarDtype::F64),
        text_signature = "(x, y, z, *, dtype='float64')"
    )]
    fn from_xyz(
        x: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
        z: &Bound<'_, PyAny>,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let xv = extract_optional_weights_1d(Some(x), "x")?.unwrap_or_default();
        let yv = extract_optional_weights_1d(Some(y), "y")?.unwrap_or_default();
        let zv = extract_optional_weights_1d(Some(z), "z")?.unwrap_or_default();
        if xv.len() != yv.len() || xv.len() != zv.len() {
            return Err(dimension_mismatch_err(format!(
                "x, y, z must have equal length, got ({}, {}, {})",
                xv.len(),
                yv.len(),
                zv.len()
            )));
        }
        if !xv.iter().all(|v| v.is_finite())
            || !yv.iter().all(|v| v.is_finite())
            || !zv.iter().all(|v| v.is_finite())
        {
            return Err(non_finite_value_err(
                "Layout3d.from_xyz requires finite coordinates",
            ));
        }
        let inner = match dtype {
            ScalarDtype::F64 => {
                let soa = Layout3d::from_xyz(xv, yv, zv).map_err(core_error_to_py)?;
                Layout3dInner::F64(InterleavedLayout3d::from_layout3d(&soa))
            }
            ScalarDtype::F32 => {
                let xf: Vec<f32> = xv.into_iter().map(|v| v as f32).collect();
                let yf: Vec<f32> = yv.into_iter().map(|v| v as f32).collect();
                let zf: Vec<f32> = zv.into_iter().map(|v| v as f32).collect();
                if !xf.iter().all(|v| v.is_finite())
                    || !yf.iter().all(|v| v.is_finite())
                    || !zf.iter().all(|v| v.is_finite())
                {
                    return Err(non_finite_value_err(
                        "Layout3d.from_xyz coordinates overflow float32",
                    ));
                }
                let soa = Layout3d::from_xyz(xf, yf, zf).map_err(core_error_to_py)?;
                Layout3dInner::F32(InterleavedLayout3d::from_layout3d(&soa))
            }
        };
        Ok(Self { inner })
    }

    fn __len__(&self) -> usize {
        match &self.inner {
            Layout3dInner::F64(buf) => buf.len(),
            Layout3dInner::F32(buf) => buf.len(),
        }
    }

    #[getter]
    fn num_nodes(&self) -> usize {
        self.__len__()
    }

    fn is_empty(&self) -> bool {
        self.__len__() == 0
    }

    #[pyo3(signature = (i), text_signature = "($self, i)")]
    fn x(&self, i: StrictUsize) -> PyResult<f64> {
        match &self.inner {
            Layout3dInner::F64(buf) => buf.x(i.0).map_err(ffi_error_to_py),
            Layout3dInner::F32(buf) => buf.x(i.0).map(f64::from).map_err(ffi_error_to_py),
        }
    }

    #[pyo3(signature = (i), text_signature = "($self, i)")]
    fn y(&self, i: StrictUsize) -> PyResult<f64> {
        match &self.inner {
            Layout3dInner::F64(buf) => buf.y(i.0).map_err(ffi_error_to_py),
            Layout3dInner::F32(buf) => buf.y(i.0).map(f64::from).map_err(ffi_error_to_py),
        }
    }

    #[pyo3(signature = (i), text_signature = "($self, i)")]
    fn z(&self, i: StrictUsize) -> PyResult<f64> {
        match &self.inner {
            Layout3dInner::F64(buf) => buf.z(i.0).map_err(ffi_error_to_py),
            Layout3dInner::F32(buf) => buf.z(i.0).map(f64::from).map_err(ffi_error_to_py),
        }
    }

    #[pyo3(signature = (i, v), text_signature = "($self, i, v)")]
    fn set_x(&mut self, i: StrictUsize, v: StrictF64) -> PyResult<()> {
        if !v.0.is_finite() {
            return Err(non_finite_value_err("Layout3d.set_x value must be finite"));
        }
        match &mut self.inner {
            Layout3dInner::F64(buf) => buf.set_x(i.0, v.0).map_err(ffi_error_to_py),
            Layout3dInner::F32(buf) => {
                let vf = v.0 as f32;
                if !vf.is_finite() {
                    return Err(non_finite_value_err(format!(
                        "Layout3d.set_x value {} overflows float32",
                        v.0
                    )));
                }
                buf.set_x(i.0, vf).map_err(ffi_error_to_py)
            }
        }
    }

    #[pyo3(signature = (i, v), text_signature = "($self, i, v)")]
    fn set_y(&mut self, i: StrictUsize, v: StrictF64) -> PyResult<()> {
        if !v.0.is_finite() {
            return Err(non_finite_value_err("Layout3d.set_y value must be finite"));
        }
        match &mut self.inner {
            Layout3dInner::F64(buf) => buf.set_y(i.0, v.0).map_err(ffi_error_to_py),
            Layout3dInner::F32(buf) => {
                let vf = v.0 as f32;
                if !vf.is_finite() {
                    return Err(non_finite_value_err(format!(
                        "Layout3d.set_y value {} overflows float32",
                        v.0
                    )));
                }
                buf.set_y(i.0, vf).map_err(ffi_error_to_py)
            }
        }
    }

    #[pyo3(signature = (i, v), text_signature = "($self, i, v)")]
    fn set_z(&mut self, i: StrictUsize, v: StrictF64) -> PyResult<()> {
        if !v.0.is_finite() {
            return Err(non_finite_value_err("Layout3d.set_z value must be finite"));
        }
        match &mut self.inner {
            Layout3dInner::F64(buf) => buf.set_z(i.0, v.0).map_err(ffi_error_to_py),
            Layout3dInner::F32(buf) => {
                let vf = v.0 as f32;
                if !vf.is_finite() {
                    return Err(non_finite_value_err(format!(
                        "Layout3d.set_z value {} overflows float32",
                        v.0
                    )));
                }
                buf.set_z(i.0, vf).map_err(ffi_error_to_py)
            }
        }
    }

    #[pyo3(signature = (target_std = StrictF64(1.0)), text_signature = "($self, target_std=1.0)")]
    fn normalize_std(&mut self, target_std: StrictF64) -> PyResult<()> {
        match &mut self.inner {
            Layout3dInner::F64(buf) => {
                let mut soa = buf.to_layout3d();
                soa.center_and_normalize_to_std(target_std.0)
                    .map_err(core_error_to_py)?;
                buf.copy_from_layout3d(&soa).map_err(ffi_error_to_py)
            }
            Layout3dInner::F32(buf) => {
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
                let mut soa = buf.to_layout3d();
                soa.center_and_normalize_to_std(f64::from(t32))
                    .map_err(core_error_to_py)?;
                buf.copy_from_layout3d(&soa).map_err(ffi_error_to_py)
            }
        }
    }

    #[getter]
    pub(crate) fn dtype(&self) -> &'static str {
        match &self.inner {
            Layout3dInner::F64(_) => ScalarDtype::F64.as_str(),
            Layout3dInner::F32(_) => ScalarDtype::F32.as_str(),
        }
    }

    fn is_all_finite(&self) -> bool {
        match &self.inner {
            Layout3dInner::F64(buf) => buf.is_all_finite(),
            Layout3dInner::F32(buf) => buf.is_all_finite(),
        }
    }

    fn copy_coords<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            Layout3dInner::F64(buf) => {
                let arr = Array2::from_shape_vec((buf.len(), 3), buf.as_slice().to_vec())
                    .map_err(|e| dimension_mismatch_err(e.to_string()))?;
                Ok(PyArray2::from_owned_array(py, arr).into_any())
            }
            Layout3dInner::F32(buf) => {
                let arr = Array2::from_shape_vec((buf.len(), 3), buf.as_slice().to_vec())
                    .map_err(|e| dimension_mismatch_err(e.to_string()))?;
                Ok(PyArray2::from_owned_array(py, arr).into_any())
            }
        }
    }

    #[pyo3(signature = (copy = false))]
    fn coords<'py>(slf: Bound<'py, Self>, copy: bool) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        if copy {
            return slf.borrow().copy_coords(py);
        }
        let container = slf.clone().into_any();
        let mut borrowed = slf.borrow_mut();
        match &mut borrowed.inner {
            Layout3dInner::F64(buf) => {
                let n = buf.len();
                let view = ArrayViewMut2::from_shape((n, 3), buf.as_mut_slice())
                    .map_err(|e| dimension_mismatch_err(e.to_string()))?;
                // SAFETY: `container` is `slf` (`PyLayout3d`), registered as the NumPy array's
                // `base` object so the backing `InterleavedLayout3d` buffer outlives any view,
                // and `InterleavedLayout3d` never reallocates its coordinate `Vec` after creation.
                Ok(unsafe { PyArray2::borrow_from_array(&view, container) }.into_any())
            }
            Layout3dInner::F32(buf) => {
                let n = buf.len();
                let view = ArrayViewMut2::from_shape((n, 3), buf.as_mut_slice())
                    .map_err(|e| dimension_mismatch_err(e.to_string()))?;
                // SAFETY: `container` is `slf` (`PyLayout3d`), registered as the NumPy array's
                // `base` object so the backing `InterleavedLayout3d` buffer outlives any view,
                // and `InterleavedLayout3d` never reallocates its coordinate `Vec` after creation.
                Ok(unsafe { PyArray2::borrow_from_array(&view, container) }.into_any())
            }
        }
    }

    fn __repr__(&self) -> String {
        format!("Layout3d(len={}, dtype='{}')", self.__len__(), self.dtype())
    }
}

/// Register `Layout3d` on `egraph._core`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLayout3d>()?;
    Ok(())
}
