//! 2D interleaved coordinate buffer (`Layout2d`) Python bindings and GIL-free dispatch macro.

use egraph::core::{DeterministicRng, Layout2d};
use egraph_ffi::InterleavedLayout2d;
use numpy::ndarray::{Array2, ArrayViewMut2};
use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBool;

use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py, extract_non_negative_usize,
    ffi_error_to_py, is_non_sequence_container,
};

macro_rules! dispatch_layout2d {
    ($py:expr, $layout:expr, |$soa:ident| $body:expr) => {
        $py.detach(|| match $layout.inner() {
            $crate::layout_buffers::Layout2dInner::F64(buf) => {
                let $soa = buf.to_layout2d();
                $body
            }
            $crate::layout_buffers::Layout2dInner::F32(buf) => {
                let $soa = buf.to_layout2d();
                $body
            }
        })
        .map_err($crate::ffi_convert::core_error_to_py)
    };
}

pub(crate) use dispatch_layout2d;

pub(crate) enum Layout2dInner {
    F64(InterleavedLayout2d<f64>),
    F32(InterleavedLayout2d<f32>),
}

impl PyLayout2d {
    pub(crate) fn inner(&self) -> &Layout2dInner {
        &self.inner
    }

    pub(crate) fn inner_mut(&mut self) -> &mut Layout2dInner {
        &mut self.inner
    }

    pub(crate) fn from_f64_layout(layout: &Layout2d<f64>) -> Self {
        Self {
            inner: Layout2dInner::F64(InterleavedLayout2d::from_layout2d(layout)),
        }
    }

    pub(crate) fn from_f32_layout(layout: &Layout2d<f32>) -> Self {
        Self {
            inner: Layout2dInner::F32(InterleavedLayout2d::from_layout2d(layout)),
        }
    }
}

pub(crate) fn layout_f32_to_f64(layout: &Layout2d<f32>) -> Layout2d<f64> {
    let x: Vec<f64> = layout.x().iter().map(|&v| f64::from(v)).collect();
    let y: Vec<f64> = layout.y().iter().map(|&v| f64::from(v)).collect();
    // Lengths from an existing `Layout2d<f32>` are guaranteed equal.
    Layout2d::from_xy(x, y).unwrap_or_else(|_| Layout2d::zeros(0))
}

fn extract_interleaved_f64_from_view(
    view: numpy::ndarray::ArrayView2<'_, f64>,
) -> PyResult<Vec<f64>> {
    let [n, cols] = [view.nrows(), view.ncols()];
    if cols != 2 {
        return Err(PyValueError::new_err(format!(
            "coords must have shape (N, 2), got ({n}, {cols})"
        )));
    }
    let mut flat = Vec::with_capacity(n * 2);
    for (idx, row) in view.rows().into_iter().enumerate() {
        let x = row[0];
        let y = row[1];
        if !x.is_finite() || !y.is_finite() {
            return Err(PyValueError::new_err(format!(
                "coords contains non-finite value at row {idx}"
            )));
        }
        flat.push(x);
        flat.push(y);
    }
    Ok(flat)
}

fn extract_interleaved_f32_from_view(
    view: numpy::ndarray::ArrayView2<'_, f32>,
) -> PyResult<Vec<f32>> {
    let [n, cols] = [view.nrows(), view.ncols()];
    if cols != 2 {
        return Err(PyValueError::new_err(format!(
            "coords must have shape (N, 2), got ({n}, {cols})"
        )));
    }
    let mut flat = Vec::with_capacity(n * 2);
    for (idx, row) in view.rows().into_iter().enumerate() {
        let x = row[0];
        let y = row[1];
        if !x.is_finite() || !y.is_finite() {
            return Err(PyValueError::new_err(format!(
                "coords contains non-finite value at row {idx}"
            )));
        }
        flat.push(x);
        flat.push(y);
    }
    Ok(flat)
}

/// 2D node coordinate buffer backed by `InterleavedLayout2d` for zero-copy NumPy views.
///
/// Supports both `"float64"` (`f64`) and `"float32"` (`f32`) precision and exposes
/// zero-copy writable `(N, 2)` NumPy array views via `coords(copy=False)`.
#[pyclass(name = "Layout2d", module = "egraph._core")]
pub struct PyLayout2d {
    inner: Layout2dInner,
}

pub(crate) fn extract_layout_inner(coords: &Bound<'_, PyAny>) -> PyResult<Layout2dInner> {
    if let Ok(untyped) = coords.cast::<PyUntypedArray>() {
        if untyped.ndim() != 2 {
            return Err(PyValueError::new_err(format!(
                "coords must be a 2D array of shape (N, 2), got {}D",
                untyped.ndim()
            )));
        }
        let shape = untyped.shape();
        if shape[1] != 2 {
            return Err(PyValueError::new_err(format!(
                "coords must have shape (N, 2), got ({}, {})",
                shape[0], shape[1]
            )));
        }
        if let Ok(arr) = coords.extract::<PyReadonlyArray2<'_, f64>>() {
            let flat = extract_interleaved_f64_from_view(arr.as_array())?;
            let buf = InterleavedLayout2d::from_interleaved(&flat).map_err(ffi_error_to_py)?;
            return Ok(Layout2dInner::F64(buf));
        }
        if let Ok(arr) = coords.extract::<PyReadonlyArray2<'_, f32>>() {
            let flat = extract_interleaved_f32_from_view(arr.as_array())?;
            let buf = InterleavedLayout2d::from_interleaved(&flat).map_err(ffi_error_to_py)?;
            return Ok(Layout2dInner::F32(buf));
        }
        return Err(PyTypeError::new_err(
            "coords NumPy array must have dtype float64 or float32",
        ));
    }

    if is_non_sequence_container(coords) {
        return Err(PyTypeError::new_err(
            "coords must be a 2D float64/float32 array or sequence of (x, y) pairs",
        ));
    }
    let outer = coords.try_iter().map_err(|_| {
        PyTypeError::new_err(
            "coords must be a 2D float64/float32 array or sequence of (x, y) pairs",
        )
    })?;
    let mut flat = Vec::new();
    for (row_idx, row_res) in outer.enumerate() {
        let row = row_res?;
        if is_non_sequence_container(&row) {
            return Err(PyTypeError::new_err(
                "each coordinate row must be a 2-element sequence (x, y)",
            ));
        }
        let inner = row.try_iter().map_err(|_| {
            PyTypeError::new_err("each coordinate row must be a 2-element sequence (x, y)")
        })?;
        let mut xy = [0.0_f64; 2];
        let mut count = 0usize;
        for elem_res in inner {
            let elem = elem_res?;
            if elem.is_instance_of::<PyBool>() {
                return Err(PyTypeError::new_err(
                    "coords entries must be floats, not bool",
                ));
            }
            if count < 2 {
                let val = elem
                    .extract::<f64>()
                    .map_err(|_| PyTypeError::new_err("coords entries must be finite floats"))?;
                if !val.is_finite() {
                    return Err(PyValueError::new_err(format!(
                        "coords contains non-finite value at row {row_idx}"
                    )));
                }
                xy[count] = val;
            }
            count += 1;
        }
        if count != 2 {
            return Err(PyValueError::new_err(format!(
                "each coordinate row must contain exactly 2 elements, got {count}"
            )));
        }
        flat.push(xy[0]);
        flat.push(xy[1]);
    }
    let buf = InterleavedLayout2d::from_interleaved(&flat).map_err(ffi_error_to_py)?;
    Ok(Layout2dInner::F64(buf))
}

#[pymethods]
impl PyLayout2d {
    /// Create a zero-initialized `Layout2d` for `n` vertices.
    ///
    /// Parameters
    /// ----------
    /// n : int
    ///     Number of vertices (`>= 0`).
    /// dtype : str or numpy.dtype, default "float64"
    ///     Floating-point precision: `"float64"`/`"f64"` or `"float32"`/`"f32"`.
    #[staticmethod]
    #[pyo3(signature = (n, dtype = ScalarDtype::F64), text_signature = "(n, dtype='float64')")]
    fn zeros(n: &Bound<'_, PyAny>, dtype: ScalarDtype) -> PyResult<Self> {
        let n_usize = extract_non_negative_usize(n, "n")?;
        let inner = match dtype {
            ScalarDtype::F64 => Layout2dInner::F64(
                InterleavedLayout2d::<f64>::try_zeros(n_usize).map_err(ffi_error_to_py)?,
            ),
            ScalarDtype::F32 => Layout2dInner::F32(
                InterleavedLayout2d::<f32>::try_zeros(n_usize).map_err(ffi_error_to_py)?,
            ),
        };
        Ok(Self { inner })
    }

    /// Create a deterministically initialized `Layout2d` for `n` vertices using `seed` (default `42`).
    #[staticmethod]
    #[pyo3(
        signature = (n, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(n, seed=42, dtype='float64')"
    )]
    fn init(n: &Bound<'_, PyAny>, seed: StrictU64, dtype: ScalarDtype) -> PyResult<Self> {
        let n_usize = extract_non_negative_usize(n, "n")?;
        let mut rng = DeterministicRng::from_seed_u64(seed.0);
        let inner = match dtype {
            ScalarDtype::F64 => {
                let mut coords = Vec::with_capacity(n_usize.saturating_mul(2));
                for _ in 0..n_usize {
                    coords.push(rng.next_f64() - 0.5);
                    coords.push(rng.next_f64() - 0.5);
                }
                Layout2dInner::F64(
                    InterleavedLayout2d::<f64>::from_interleaved(&coords)
                        .map_err(ffi_error_to_py)?,
                )
            }
            ScalarDtype::F32 => {
                let mut coords = Vec::with_capacity(n_usize.saturating_mul(2));
                for _ in 0..n_usize {
                    coords.push((rng.next_f64() - 0.5) as f32);
                    coords.push((rng.next_f64() - 0.5) as f32);
                }
                Layout2dInner::F32(
                    InterleavedLayout2d::<f32>::from_interleaved(&coords)
                        .map_err(ffi_error_to_py)?,
                )
            }
        };
        Ok(Self { inner })
    }

    /// Create a `Layout2d` from an `(N, 2)` coordinate array or sequence of `(x, y)` pairs.
    ///
    /// Preserves `float32` precision when passed a `float32` NumPy array, and uses `float64`
    /// for `float64` NumPy arrays and Python sequences. Rejects non-finite (`NaN`/`Inf`) entries.
    #[staticmethod]
    fn from_coords(coords: &Bound<'_, PyAny>) -> PyResult<Self> {
        extract_layout_inner(coords).map(|inner| Self { inner })
    }

    /// Return the number of vertices in the layout.
    fn __len__(&self) -> usize {
        match &self.inner {
            Layout2dInner::F64(buf) => buf.len(),
            Layout2dInner::F32(buf) => buf.len(),
        }
    }

    /// Number of vertices in the layout.
    #[getter]
    fn num_nodes(&self) -> usize {
        self.__len__()
    }

    /// Return `True` if the layout contains zero vertices.
    fn is_empty(&self) -> bool {
        self.__len__() == 0
    }

    /// Return the `x` coordinate of vertex `i`.
    #[pyo3(signature = (i), text_signature = "($self, i)")]
    fn x(&self, i: StrictUsize) -> PyResult<f64> {
        match &self.inner {
            Layout2dInner::F64(buf) => buf.x(i.0).map_err(ffi_error_to_py),
            Layout2dInner::F32(buf) => buf.x(i.0).map(f64::from).map_err(ffi_error_to_py),
        }
    }

    /// Return the `y` coordinate of vertex `i`.
    #[pyo3(signature = (i), text_signature = "($self, i)")]
    fn y(&self, i: StrictUsize) -> PyResult<f64> {
        match &self.inner {
            Layout2dInner::F64(buf) => buf.y(i.0).map_err(ffi_error_to_py),
            Layout2dInner::F32(buf) => buf.y(i.0).map(f64::from).map_err(ffi_error_to_py),
        }
    }

    /// Set the `x` coordinate of vertex `i`.
    #[pyo3(signature = (i, v), text_signature = "($self, i, v)")]
    fn set_x(&mut self, i: StrictUsize, v: StrictF64) -> PyResult<()> {
        if !v.0.is_finite() {
            return Err(PyValueError::new_err("Layout2d.set_x value must be finite"));
        }
        match &mut self.inner {
            Layout2dInner::F64(buf) => buf.set_x(i.0, v.0).map_err(ffi_error_to_py),
            Layout2dInner::F32(buf) => {
                let vf = v.0 as f32;
                if !vf.is_finite() {
                    return Err(PyValueError::new_err(format!(
                        "Layout2d.set_x value {} overflows float32",
                        v.0
                    )));
                }
                buf.set_x(i.0, vf).map_err(ffi_error_to_py)
            }
        }
    }

    /// Set the `y` coordinate of vertex `i`.
    #[pyo3(signature = (i, v), text_signature = "($self, i, v)")]
    fn set_y(&mut self, i: StrictUsize, v: StrictF64) -> PyResult<()> {
        if !v.0.is_finite() {
            return Err(PyValueError::new_err("Layout2d.set_y value must be finite"));
        }
        match &mut self.inner {
            Layout2dInner::F64(buf) => buf.set_y(i.0, v.0).map_err(ffi_error_to_py),
            Layout2dInner::F32(buf) => {
                let vf = v.0 as f32;
                if !vf.is_finite() {
                    return Err(PyValueError::new_err(format!(
                        "Layout2d.set_y value {} overflows float32",
                        v.0
                    )));
                }
                buf.set_y(i.0, vf).map_err(ffi_error_to_py)
            }
        }
    }

    /// Center coordinates at zero mean and scale isotropically in place to `target_std`.
    #[pyo3(signature = (target_std = StrictF64(1.0)), text_signature = "($self, target_std=1.0)")]
    fn normalize_std(&mut self, target_std: StrictF64) -> PyResult<()> {
        match &mut self.inner {
            Layout2dInner::F64(buf) => {
                let mut soa = buf.to_layout2d();
                soa.center_and_normalize_to_std(target_std.0)
                    .map_err(core_error_to_py)?;
                buf.copy_from_layout2d(&soa).map_err(ffi_error_to_py)
            }
            Layout2dInner::F32(buf) => {
                if !target_std.0.is_finite() || target_std.0 <= 0.0 {
                    return Err(PyValueError::new_err(
                        "target_std must be finite and strictly positive",
                    ));
                }
                let t32 = target_std.0 as f32;
                if !t32.is_finite() || t32 <= 0.0 {
                    return Err(PyValueError::new_err(format!(
                        "target_std {} is out of range for float32",
                        target_std.0
                    )));
                }
                let mut soa = buf.to_layout2d();
                soa.center_and_normalize_to_std(t32)
                    .map_err(core_error_to_py)?;
                buf.copy_from_layout2d(&soa).map_err(ffi_error_to_py)
            }
        }
    }

    /// Scalar coordinate precision (`"float64"` or `"float32"`).
    #[getter]
    pub(crate) fn dtype(&self) -> &'static str {
        match &self.inner {
            Layout2dInner::F64(_) => ScalarDtype::F64.as_str(),
            Layout2dInner::F32(_) => ScalarDtype::F32.as_str(),
        }
    }

    /// Return `True` if every coordinate in the layout is finite (`not NaN` and `not Inf`).
    fn is_all_finite(&self) -> bool {
        match &self.inner {
            Layout2dInner::F64(buf) => buf.is_all_finite(),
            Layout2dInner::F32(buf) => buf.is_all_finite(),
        }
    }

    /// Return a freshly allocated `(N, 2)` NumPy array copy of coordinates.
    fn copy_coords<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner {
            Layout2dInner::F64(buf) => {
                let arr = Array2::from_shape_vec((buf.len(), 2), buf.as_slice().to_vec())
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyArray2::from_owned_array(py, arr).into_any())
            }
            Layout2dInner::F32(buf) => {
                let arr = Array2::from_shape_vec((buf.len(), 2), buf.as_slice().to_vec())
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyArray2::from_owned_array(py, arr).into_any())
            }
        }
    }

    /// Return an `(N, 2)` NumPy array of coordinates.
    ///
    /// When `copy=False` (the default), returns a zero-copy writable NumPy view backed by
    /// the layout's internal `InterleavedLayout2d` buffer. Mutations to the returned array
    /// immediately update the layout, and the NumPy array holds a reference to `self` so
    /// the backing buffer stays alive as long as the view does. When `copy=True`, returns a
    /// freshly allocated NumPy array copy.
    #[pyo3(signature = (copy = false))]
    fn coords<'py>(slf: Bound<'py, Self>, copy: bool) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        if copy {
            let borrowed = slf.borrow();
            return match &borrowed.inner {
                Layout2dInner::F64(buf) => {
                    let arr = Array2::from_shape_vec((buf.len(), 2), buf.as_slice().to_vec())
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    Ok(PyArray2::from_owned_array(py, arr).into_any())
                }
                Layout2dInner::F32(buf) => {
                    let arr = Array2::from_shape_vec((buf.len(), 2), buf.as_slice().to_vec())
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    Ok(PyArray2::from_owned_array(py, arr).into_any())
                }
            };
        }

        let container = slf.clone().into_any();
        let mut borrowed = slf.borrow_mut();
        match &mut borrowed.inner {
            Layout2dInner::F64(buf) => {
                let n = buf.len();
                let view = ArrayViewMut2::from_shape((n, 2), buf.as_mut_slice())
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                // SAFETY: `container` is `slf` (`PyLayout2d`), registered as the NumPy array's
                // `base` object so the backing `InterleavedLayout2d` buffer outlives any view,
                // and `InterleavedLayout2d` never reallocates its coordinate `Vec` after creation.
                Ok(unsafe { PyArray2::borrow_from_array(&view, container) }.into_any())
            }
            Layout2dInner::F32(buf) => {
                let n = buf.len();
                let view = ArrayViewMut2::from_shape((n, 2), buf.as_mut_slice())
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                // SAFETY: `container` is `slf` (`PyLayout2d`), registered as the NumPy array's
                // `base` object so the backing `InterleavedLayout2d` buffer outlives any view,
                // and `InterleavedLayout2d` never reallocates its coordinate `Vec` after creation.
                Ok(unsafe { PyArray2::borrow_from_array(&view, container) }.into_any())
            }
        }
    }

    /// Return a concise string representation of the `Layout2d`.
    fn __repr__(&self) -> String {
        format!("Layout2d(len={}, dtype='{}')", self.__len__(), self.dtype())
    }
}

/// Register `Layout2d` on the Python extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLayout2d>()?;
    Ok(())
}
