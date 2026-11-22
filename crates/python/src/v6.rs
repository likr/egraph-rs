//! v6 zero-copy `CsrGraph` and `Layout2d` Python bindings backed by `egraph-core` and `egraph-ffi`.

use egraph_core::CsrGraph;
use egraph_ffi::{ErrorKind, FfiError, InterleavedLayout2d, build_csr_f64, build_edges_f64};
use numpy::ndarray::{Array2, ArrayViewMut2};
use numpy::{
    Element, PyArray1, PyArray2, PyArrayDescr, PyArrayDescrMethods, PyReadonlyArray1,
    PyReadonlyArray2, PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyByteArray, PyBytes, PyDict, PyString, PyType};

fn ffi_error_to_py(err: FfiError) -> PyErr {
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

fn is_non_sequence_container(obj: &Bound<'_, PyAny>) -> bool {
    obj.is_instance_of::<PyString>()
        || obj.is_instance_of::<PyBytes>()
        || obj.is_instance_of::<PyByteArray>()
        || obj.is_instance_of::<PyDict>()
}

fn extract_u32_item(item: &Bound<'_, PyAny>, name: &str) -> PyResult<u32> {
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

fn extract_u32_1d(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<u32>> {
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

fn extract_optional_weights_1d(
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

fn convert_edges_2d_to_flat<T, F>(
    view: numpy::ndarray::ArrayView2<'_, T>,
    mut convert: F,
) -> PyResult<Vec<u32>>
where
    T: Copy,
    F: FnMut(T) -> PyResult<u32>,
{
    let [m, cols] = [view.nrows(), view.ncols()];
    if cols != 2 {
        return Err(PyValueError::new_err(format!(
            "edges must have shape (M, 2), got ({m}, {cols})"
        )));
    }
    let mut out = Vec::with_capacity(m * 2);
    for row in view.rows() {
        out.push(convert(row[0])?);
        out.push(convert(row[1])?);
    }
    Ok(out)
}

fn extract_flat_edges(edges: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
    if let Ok(untyped) = edges.cast::<PyUntypedArray>() {
        if untyped.ndim() != 2 {
            return Err(PyValueError::new_err(format!(
                "edges must be a 2D array of shape (M, 2), got {}D",
                untyped.ndim()
            )));
        }
        let shape = untyped.shape();
        if shape[1] != 2 {
            return Err(PyValueError::new_err(format!(
                "edges must have shape (M, 2), got ({}, {})",
                shape[0], shape[1]
            )));
        }
        if let Ok(arr) = edges.extract::<PyReadonlyArray2<'_, u32>>() {
            return convert_edges_2d_to_flat(arr.as_array(), Ok);
        }
        if let Ok(arr) = edges.extract::<PyReadonlyArray2<'_, i64>>() {
            return convert_edges_2d_to_flat(arr.as_array(), |v| {
                u32::try_from(v).map_err(|_| {
                    PyValueError::new_err(format!("edge endpoint {v} out of range for u32"))
                })
            });
        }
        if let Ok(arr) = edges.extract::<PyReadonlyArray2<'_, u64>>() {
            return convert_edges_2d_to_flat(arr.as_array(), |v| {
                u32::try_from(v).map_err(|_| {
                    PyValueError::new_err(format!("edge endpoint {v} out of range for u32"))
                })
            });
        }
        if let Ok(arr) = edges.extract::<PyReadonlyArray2<'_, i32>>() {
            return convert_edges_2d_to_flat(arr.as_array(), |v| {
                u32::try_from(v).map_err(|_| {
                    PyValueError::new_err(format!("edge endpoint {v} out of range for u32"))
                })
            });
        }
        if let Ok(arr) = edges.extract::<PyReadonlyArray2<'_, usize>>() {
            return convert_edges_2d_to_flat(arr.as_array(), |v| {
                u32::try_from(v).map_err(|_| {
                    PyValueError::new_err(format!("edge endpoint {v} out of range for u32"))
                })
            });
        }
        if let Ok(arr) = edges.extract::<PyReadonlyArray2<'_, isize>>() {
            return convert_edges_2d_to_flat(arr.as_array(), |v| {
                u32::try_from(v).map_err(|_| {
                    PyValueError::new_err(format!("edge endpoint {v} out of range for u32"))
                })
            });
        }
        if let Ok(arr) = edges.extract::<PyReadonlyArray2<'_, u16>>() {
            return convert_edges_2d_to_flat(arr.as_array(), |v| Ok(u32::from(v)));
        }
        if let Ok(arr) = edges.extract::<PyReadonlyArray2<'_, u8>>() {
            return convert_edges_2d_to_flat(arr.as_array(), |v| Ok(u32::from(v)));
        }
        if let Ok(arr) = edges.extract::<PyReadonlyArray2<'_, i16>>() {
            return convert_edges_2d_to_flat(arr.as_array(), |v| {
                u32::try_from(v).map_err(|_| {
                    PyValueError::new_err(format!("edge endpoint {v} out of range for u32"))
                })
            });
        }
        if let Ok(arr) = edges.extract::<PyReadonlyArray2<'_, i8>>() {
            return convert_edges_2d_to_flat(arr.as_array(), |v| {
                u32::try_from(v).map_err(|_| {
                    PyValueError::new_err(format!("edge endpoint {v} out of range for u32"))
                })
            });
        }
        return Err(PyTypeError::new_err(
            "edges must be a 2D integer NumPy array of shape (M, 2)",
        ));
    }

    if is_non_sequence_container(edges) {
        return Err(PyTypeError::new_err(
            "edges must be a 2D integer array or sequence of (u, v) pairs",
        ));
    }
    let outer = edges.try_iter().map_err(|_| {
        PyTypeError::new_err("edges must be a 2D integer array or sequence of (u, v) pairs")
    })?;
    let mut flat = Vec::new();
    for pair_res in outer {
        let pair = pair_res?;
        if is_non_sequence_container(&pair) {
            return Err(PyTypeError::new_err("each edge must be a 2-tuple (u, v)"));
        }
        let inner = pair
            .try_iter()
            .map_err(|_| PyTypeError::new_err("each edge must be a 2-element sequence (u, v)"))?;
        let mut endpoints = [0u32; 2];
        let mut count = 0usize;
        for elem_res in inner {
            let elem = elem_res?;
            if count < 2 {
                endpoints[count] = extract_u32_item(&elem, "edges")?;
            }
            count += 1;
        }
        if count != 2 {
            return Err(PyValueError::new_err(format!(
                "each edge must contain exactly 2 endpoints, got {count}"
            )));
        }
        flat.push(endpoints[0]);
        flat.push(endpoints[1]);
    }
    Ok(flat)
}

fn extract_non_negative_usize(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<usize> {
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

/// Compact `u32`-indexed Compressed Sparse Row (`CsrGraph`) graph representation.
///
/// Stores vertex neighborhoods in flat CSR arrays with optional `float64` edge weights,
/// supporting both undirected and directed graphs as well as largest connected component
/// extraction without `petgraph` or Python object overhead.
#[pyclass(name = "CsrGraph", module = "egraph._core", from_py_object)]
#[derive(Clone)]
pub struct PyCsrGraph {
    inner: CsrGraph<f64>,
}

#[pymethods]
impl PyCsrGraph {
    /// Construct a `CsrGraph` from CSR `indptr`, `indices`, and optional `weights`.
    ///
    /// Parameters
    /// ----------
    /// indptr : array_like
    ///     1D non-decreasing integer array of length `num_nodes + 1` with `indptr[0] == 0`.
    /// indices : array_like
    ///     1D integer array of neighbor vertex indices in `0..num_nodes`.
    /// weights : array_like, optional
    ///     Optional 1D finite numeric array of edge weights aligned with `indices`.
    /// directed : bool, keyword-only, default False
    ///     Whether the graph is directed.
    /// sanitize : bool, keyword-only, default True
    ///     If True, removes self-loops, sorts and deduplicates parallel edges (retaining
    ///     maximum weight), and symmetrizes undirected graphs. If False, strictly validates
    ///     all invariants.
    #[staticmethod]
    #[pyo3(signature = (indptr, indices, weights = None, *, directed = false, sanitize = true))]
    fn from_csr(
        py: Python<'_>,
        indptr: &Bound<'_, PyAny>,
        indices: &Bound<'_, PyAny>,
        weights: Option<&Bound<'_, PyAny>>,
        directed: bool,
        sanitize: bool,
    ) -> PyResult<Self> {
        let indptr_vec = extract_u32_1d(indptr, "indptr")?;
        let indices_vec = extract_u32_1d(indices, "indices")?;
        let weights_vec = extract_optional_weights_1d(weights, "weights")?;

        let inner = py
            .detach(|| {
                build_csr_f64(
                    &indptr_vec,
                    &indices_vec,
                    weights_vec.as_deref(),
                    directed,
                    sanitize,
                )
            })
            .map_err(ffi_error_to_py)?;

        Ok(Self { inner })
    }

    /// Construct a `CsrGraph` from `num_nodes` and an `(M, 2)` edge list with optional `weights`.
    ///
    /// Parameters
    /// ----------
    /// num_nodes : int
    ///     Number of vertices (`0..num_nodes`).
    /// edges : array_like
    ///     2D integer array of shape `(M, 2)` or sequence of `(u, v)` index pairs.
    /// weights : array_like, optional
    ///     Optional 1D finite numeric array of edge weights of length `M`.
    /// directed : bool, keyword-only, default False
    ///     Whether edges are directed.
    /// sanitize : bool, keyword-only, default True
    ///     If True, drops self-loops and deduplicates parallel edges by keeping maximum weight.
    #[staticmethod]
    #[pyo3(signature = (num_nodes, edges, weights = None, *, directed = false, sanitize = true))]
    fn from_edges(
        py: Python<'_>,
        num_nodes: &Bound<'_, PyAny>,
        edges: &Bound<'_, PyAny>,
        weights: Option<&Bound<'_, PyAny>>,
        directed: bool,
        sanitize: bool,
    ) -> PyResult<Self> {
        let num_nodes_usize = extract_non_negative_usize(num_nodes, "num_nodes")?;
        let flat_edges = extract_flat_edges(edges)?;
        let weights_vec = extract_optional_weights_1d(weights, "weights")?;

        let inner = py
            .detach(|| {
                build_edges_f64(
                    num_nodes_usize,
                    &flat_edges,
                    weights_vec.as_deref(),
                    directed,
                    sanitize,
                )
            })
            .map_err(ffi_error_to_py)?;

        Ok(Self { inner })
    }

    /// Number of vertices in the graph.
    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Number of logical edges in the graph (`nnz` if directed, `nnz // 2` if undirected).
    #[getter]
    fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }

    /// Whether the graph is directed.
    #[getter]
    fn is_directed(&self) -> bool {
        self.inner.is_directed()
    }

    /// Return a `uint32` NumPy copy of the CSR row pointer array of length `num_nodes + 1`.
    fn indptr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_slice(py, self.inner.indptr())
    }

    /// Return a `uint32` NumPy copy of the CSR neighbor index array.
    fn indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_slice(py, self.inner.indices())
    }

    /// Return a `float64` NumPy copy of edge weights aligned with `indices()`, or `None`.
    fn weights<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.weights().map(|w| PyArray1::from_slice(py, w))
    }

    /// Extract the largest (weakly) connected component as `(subgraph, original_node_ids)`.
    ///
    /// Returns the induced `CsrGraph` on compact `0..k` vertex IDs together with a sorted
    /// `uint32` NumPy array mapping each vertex in `subgraph` back to its original vertex ID.
    fn largest_connected_component<'py>(
        &self,
        py: Python<'py>,
    ) -> (Self, Bound<'py, PyArray1<u32>>) {
        let (subgraph, original_node_ids) = py.detach(|| self.inner.largest_connected_component());
        let ids_array = PyArray1::from_slice(py, &original_node_ids);
        (Self { inner: subgraph }, ids_array)
    }

    /// Return a concise string representation of the `CsrGraph`.
    fn __repr__(&self) -> String {
        format!(
            "CsrGraph(num_nodes={}, num_edges={}, directed={})",
            self.inner.num_nodes(),
            self.inner.num_edges(),
            if self.inner.is_directed() {
                "True"
            } else {
                "False"
            }
        )
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ScalarDtype {
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

    fn as_str(self) -> &'static str {
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

enum Layout2dInner {
    F64(InterleavedLayout2d<f64>),
    F32(InterleavedLayout2d<f32>),
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

    /// Create a `Layout2d` from an `(N, 2)` coordinate array or sequence of `(x, y)` pairs.
    ///
    /// Preserves `float32` precision when passed a `float32` NumPy array, and uses `float64`
    /// for `float64` NumPy arrays and Python sequences. Rejects non-finite (`NaN`/`Inf`) entries.
    #[staticmethod]
    fn from_coords(coords: &Bound<'_, PyAny>) -> PyResult<Self> {
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
                return Ok(Self {
                    inner: Layout2dInner::F64(buf),
                });
            }
            if let Ok(arr) = coords.extract::<PyReadonlyArray2<'_, f32>>() {
                let flat = extract_interleaved_f32_from_view(arr.as_array())?;
                let buf = InterleavedLayout2d::from_interleaved(&flat).map_err(ffi_error_to_py)?;
                return Ok(Self {
                    inner: Layout2dInner::F32(buf),
                });
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
                    let val = elem.extract::<f64>().map_err(|_| {
                        PyTypeError::new_err("coords entries must be finite floats")
                    })?;
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
        Ok(Self {
            inner: Layout2dInner::F64(buf),
        })
    }

    /// Return the number of vertices in the layout.
    fn __len__(&self) -> usize {
        match &self.inner {
            Layout2dInner::F64(buf) => buf.len(),
            Layout2dInner::F32(buf) => buf.len(),
        }
    }

    /// Scalar coordinate precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
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

/// Register the v6 zero-copy `CsrGraph` and `Layout2d` classes on the Python extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCsrGraph>()?;
    m.add_class::<PyLayout2d>()?;
    Ok(())
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
