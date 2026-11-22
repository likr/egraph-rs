//! v6 zero-copy `CsrGraph`, `Layout2d`, metrics, spatial, linear algebra, and
//! neighborhood bindings backed by `egraph-core`, `egraph-ffi`, `egraph-linalg`,
//! `egraph-metrics`, `egraph-neighborhood`, and `egraph-spatial`.

use std::collections::BTreeMap;

use egraph::{
    core as egraph_core, dataset as egraph_dataset, metrics as egraph_metrics,
    spatial as egraph_spatial,
};
use egraph_core::{CoreError, CoreResult, CsrGraph, DeterministicRng, Layout2d, LayoutNd};
use egraph_ffi::{ErrorKind, FfiError, InterleavedLayout2d, build_csr_f64, build_edges_f64};
use egraph_metrics::{MetricConfig, MetricTarget};
use egraph_spatial::{Delaunay2d, RectangleOverlapSweep};
use numpy::ndarray::{Array2, ArrayViewMut2};
use numpy::{
    Element, PyArray1, PyArray2, PyArrayDescr, PyArrayDescrMethods, PyReadonlyArray1,
    PyReadonlyArray2, PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyByteArray, PyBytes, PyDict, PyString, PyType};

#[path = "v6_m2.rs"]
mod v6_m2;
#[path = "v6_m3.rs"]
mod v6_m3;
#[path = "v6_m4.rs"]
mod v6_m4;

pub(crate) use v6_m4::{
    v6_cycle_edges_impl, v6_generate_layered_constraints_impl, v6_project_1d_impl,
};

pub(super) fn ffi_error_to_py(err: FfiError) -> PyErr {
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

pub(super) fn core_error_to_py(err: CoreError) -> PyErr {
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

pub(super) fn is_non_sequence_container(obj: &Bound<'_, PyAny>) -> bool {
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

pub(super) fn extract_u32_1d(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<u32>> {
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

pub(super) fn extract_optional_weights_1d(
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

pub(super) fn extract_non_negative_usize(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<usize> {
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
pub(super) struct StrictUsize(pub(super) usize);

impl<'a, 'py> FromPyObject<'a, 'py> for StrictUsize {
    type Error = PyErr;

    fn extract(ob: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        extract_non_negative_usize(&ob, "integer argument").map(Self)
    }
}

#[derive(Clone, Copy)]
pub(super) struct StrictU64(pub(super) u64);

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

impl PyCsrGraph {
    pub(crate) fn inner(&self) -> &CsrGraph<f64> {
        &self.inner
    }

    pub(super) fn from_csr_graph(inner: CsrGraph<f64>) -> Self {
        Self { inner }
    }
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

    /// Alias for :meth:`CsrGraph.from_edges` matching WASM `CsrGraph.fromEdgeList`.
    #[staticmethod]
    #[pyo3(signature = (num_nodes, edges, weights = None, *, directed = false, sanitize = true))]
    fn from_edge_list(
        py: Python<'_>,
        num_nodes: &Bound<'_, PyAny>,
        edges: &Bound<'_, PyAny>,
        weights: Option<&Bound<'_, PyAny>>,
        directed: bool,
        sanitize: bool,
    ) -> PyResult<Self> {
        Self::from_edges(py, num_nodes, edges, weights, directed, sanitize)
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

    /// Number of stored CSR half-edge entries (`indices().len()`).
    #[getter]
    fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    /// Whether the graph is directed.
    #[getter]
    fn is_directed(&self) -> bool {
        self.inner.is_directed()
    }

    /// Whether explicit edge weights are stored.
    #[getter]
    fn is_weighted(&self) -> bool {
        self.inner.is_weighted()
    }

    /// Return the degree (outgoing degree for directed graphs) of vertex `u`.
    #[pyo3(signature = (u), text_signature = "($self, u)")]
    fn degree(&self, u: StrictUsize) -> PyResult<usize> {
        self.inner.try_degree(u.0).map_err(core_error_to_py)
    }

    /// Return a `uint32` NumPy copy of neighbor vertex IDs adjacent to `u`.
    #[pyo3(signature = (u), text_signature = "($self, u)")]
    fn neighbors<'py>(
        &self,
        py: Python<'py>,
        u: StrictUsize,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let slice = self.inner.try_neighbors(u.0).map_err(core_error_to_py)?;
        Ok(PyArray1::from_slice(py, slice))
    }

    /// Extract the induced subgraph on `nodes` (with compact `0..len(nodes)` vertex IDs).
    #[pyo3(signature = (nodes), text_signature = "($self, nodes)")]
    fn induced_subgraph(&self, py: Python<'_>, nodes: &Bound<'_, PyAny>) -> PyResult<Self> {
        let nodes_vec = extract_u32_1d(nodes, "nodes")?;
        let sub = py
            .detach(|| self.inner.induced_subgraph(&nodes_vec))
            .map_err(core_error_to_py)?;
        Ok(Self { inner: sub })
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
pub(super) enum ScalarDtype {
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

    pub(super) fn as_str(self) -> &'static str {
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

    pub(super) fn from_f64_layout(layout: &Layout2d<f64>) -> Self {
        Self {
            inner: Layout2dInner::F64(InterleavedLayout2d::from_layout2d(layout)),
        }
    }

    pub(super) fn from_f32_layout(layout: &Layout2d<f32>) -> Self {
        Self {
            inner: Layout2dInner::F32(InterleavedLayout2d::from_layout2d(layout)),
        }
    }
}

macro_rules! dispatch_layout2d {
    ($py:expr, $layout:expr, |$soa:ident| $body:expr) => {
        $py.detach(|| match $layout.inner() {
            Layout2dInner::F64(buf) => {
                let $soa = buf.to_layout2d();
                $body
            }
            Layout2dInner::F32(buf) => {
                let $soa = buf.to_layout2d();
                $body
            }
        })
        .map_err(core_error_to_py)
    };
}

fn layout_f32_to_f64(layout: &Layout2d<f32>) -> Layout2d<f64> {
    let x: Vec<f64> = layout.x().iter().map(|&v| f64::from(v)).collect();
    let y: Vec<f64> = layout.y().iter().map(|&v| f64::from(v)).collect();
    // Lengths from an existing `Layout2d<f32>` are guaranteed equal.
    Layout2d::from_xy(x, y).unwrap_or_else(|_| Layout2d::zeros(0))
}

fn parse_metric_target(raw: &str) -> Result<MetricTarget, String> {
    let normalized: String = raw
        .trim()
        .chars()
        .filter(|c| *c != '_' && *c != '-')
        .map(|c| c.to_ascii_lowercase())
        .collect();
    match normalized.as_str() {
        "scalenormalizedstress" | "sns" => Ok(MetricTarget::ScaleNormalizedStress),
        "sampledscalenormalizedstress" | "sampledsns" => {
            Ok(MetricTarget::SampledScaleNormalizedStress)
        }
        "crossingnumber" | "sweepcrossingnumber" => Ok(MetricTarget::CrossingNumber),
        "crossingangle" | "sweepcrossingangle" => Ok(MetricTarget::CrossingAngle),
        "sampledcrossingnumber" | "sampledcrossing" => Ok(MetricTarget::SampledCrossingNumber),
        "neighborhoodpreservation1hop" | "neighborhoodpreservation" | "np1" => {
            Ok(MetricTarget::NeighborhoodPreservation1Hop)
        }
        "neighborhoodpreservation2hop" | "neighborhoodpreservation2hopv6" | "np2" => {
            Ok(MetricTarget::NeighborhoodPreservation2Hop)
        }
        "gabrielgraphproperty" | "gabrielgraphpropertyv6" | "gabrielproperty" => {
            Ok(MetricTarget::GabrielGraphProperty)
        }
        "noderesolution" | "noderesolutionv6" => Ok(MetricTarget::NodeResolution),
        "relativeneighborhoodpreservation" | "drng" | "drngpreservation" => {
            Ok(MetricTarget::RelativeNeighborhoodPreservation)
        }
        "gabrielgraphpreservation" | "dgg" | "dggpreservation" => {
            Ok(MetricTarget::GabrielGraphPreservation)
        }
        "aspectratio" | "aspectratiov6" => Ok(MetricTarget::AspectRatio),
        "angularresolution" | "angularresolutionv6" => Ok(MetricTarget::AngularResolution),
        "idealedgelengths" | "idealedgelengthsv6" => Ok(MetricTarget::IdealEdgeLengths),
        _ => Err(format!("unknown metric target {raw:?}")),
    }
}

fn extract_metric_targets(targets: &Bound<'_, PyAny>) -> PyResult<Vec<MetricTarget>> {
    if is_non_sequence_container(targets) {
        return Err(PyTypeError::new_err(
            "targets must be a sequence or iterable of metric target strings",
        ));
    }
    let iter = targets.try_iter().map_err(|_| {
        PyTypeError::new_err("targets must be a sequence or iterable of metric target strings")
    })?;
    let mut parsed = Vec::new();
    for item_res in iter {
        let item = item_res?;
        let raw = item
            .extract::<String>()
            .map_err(|_| PyTypeError::new_err("metric target entries must be strings"))?;
        parsed.push(parse_metric_target(&raw).map_err(PyValueError::new_err)?);
    }
    Ok(parsed)
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

pub(super) fn extract_layout_inner(coords: &Bound<'_, PyAny>) -> PyResult<Layout2dInner> {
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
    pub(super) fn from_f64_nd(nd: LayoutNd<f64>) -> Self {
        Self {
            inner: LayoutNdInner::F64(nd),
        }
    }

    pub(super) fn from_f32_nd(nd: LayoutNd<f32>) -> Self {
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

    let (flat, nrows, ncols) = v6_m2::extract_f64_2d(coords, "coords")?;
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

pub(crate) fn v6_crossing_number_impl(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
) -> PyResult<f64> {
    let g = graph.inner();
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::crossing_number(g, &soa).map(|c| c as f64)
    })
}

pub(crate) fn v6_crossing_angle_impl(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
) -> PyResult<f64> {
    let g = graph.inner();
    dispatch_layout2d!(py, layout, |soa| egraph_metrics::crossing_angle(g, &soa))
}

pub(crate) fn v6_neighborhood_preservation_1hop_impl(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
) -> PyResult<f64> {
    let g = graph.inner();
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::neighborhood_preservation_1hop(g, &soa)
    })
}

pub(crate) fn v6_neighborhood_preservation_2hop_impl(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
) -> PyResult<f64> {
    let g = graph.inner();
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::neighborhood_preservation_2hop(g, &soa)
    })
}

pub(crate) fn v6_gabriel_graph_property_impl(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
) -> PyResult<f64> {
    let g = graph.inner();
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::gabriel_graph_property(g, &soa)
    })
}

pub(crate) fn v6_node_resolution_impl(py: Python<'_>, layout: &PyLayout2d) -> PyResult<f64> {
    dispatch_layout2d!(py, layout, |soa| egraph_metrics::node_resolution(&soa))
}

pub(crate) fn v6_aspect_ratio_impl(py: Python<'_>, layout: &PyLayout2d) -> PyResult<f64> {
    dispatch_layout2d!(py, layout, |soa| egraph_metrics::aspect_ratio(&soa))
}

pub(crate) fn v6_angular_resolution_impl(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
) -> PyResult<f64> {
    let g = graph.inner();
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::angular_resolution(g, &soa)
    })
}

pub(crate) fn v6_ideal_edge_lengths_impl(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
) -> PyResult<f64> {
    let g = graph.inner();
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::ideal_edge_lengths(g, &soa)
    })
}

/// Compute closed-form Scale-Normalized Stress (`SNS`) in `[0.0, 1.0]` over `graph` and `layout`.
///
/// Uses unweighted BFS hop distances with inverse-squared weights `w_ij = 1 / d_ij^2` and
/// analytical optimal coordinate rescaling so uniform scaling leaves the metric invariant.
/// Releases the Python GIL during computation.
#[pyfunction]
fn scale_normalized_stress(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
) -> PyResult<f64> {
    let g = graph.inner();
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::scale_normalized_stress(g, &soa)
    })
}

/// Compute O(kN) pivot-sampled Scale-Normalized Stress (`SNS`) from `num_pivots` deterministic BFS sources.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (graph, layout, *, num_pivots = StrictUsize(32), seed = StrictU64(42)),
    text_signature = "(graph, layout, *, num_pivots=32, seed=42)"
)]
fn sampled_scale_normalized_stress(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
    num_pivots: StrictUsize,
    seed: StrictU64,
) -> PyResult<f64> {
    let g = graph.inner();
    let pivots = num_pivots.0;
    let seed_val = seed.0;
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::sampled_scale_normalized_stress(g, &soa, pivots, seed_val)
    })
}

/// Count exact non-adjacent edge crossings using the ``O((|E| + K) log |E|)`` sweep-line detector.
///
/// Releases the Python GIL during computation.
#[pyfunction]
fn sweep_crossing_number(py: Python<'_>, graph: &PyCsrGraph, layout: &PyLayout2d) -> PyResult<f64> {
    v6_crossing_number_impl(py, graph, layout)
}

/// Compute the sweep-line crossing-angle penalty `sum cos^2(theta)` over all crossing edge pairs.
///
/// Releases the Python GIL during computation.
#[pyfunction]
fn sweep_crossing_angle(py: Python<'_>, graph: &PyCsrGraph, layout: &PyLayout2d) -> PyResult<f64> {
    v6_crossing_angle_impl(py, graph, layout)
}

/// Estimate total edge crossings in O(num_samples) time via deterministic edge-pair sampling.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (graph, layout, *, num_samples = StrictUsize(4096), seed = StrictU64(42)),
    text_signature = "(graph, layout, *, num_samples=4096, seed=42)"
)]
fn sampled_crossing_number(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
    num_samples: StrictUsize,
    seed: StrictU64,
) -> PyResult<f64> {
    let g = graph.inner();
    let samples = num_samples.0;
    let seed_val = seed.0;
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::sampled_crossing_number(g, &soa, samples, seed_val)
    })
}

/// Count crossing edge sub-segments on the periodic unit torus `[0, 1)^2`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
fn torus_crossing_number(py: Python<'_>, graph: &PyCsrGraph, layout: &PyLayout2d) -> PyResult<f64> {
    let g = graph.inner();
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::crossing_edges_torus(g, &soa).map(|pairs| pairs.len() as f64)
    })
}

/// Compute 1-hop neighborhood preservation (`NP1`) in `[0.0, 1.0]` backed by `StaticKdTree`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
fn neighborhood_preservation_1hop(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
) -> PyResult<f64> {
    v6_neighborhood_preservation_1hop_impl(py, graph, layout)
}

/// Compute 2-hop neighborhood preservation (`NP2`) in `[0.0, 1.0]` backed by `StaticKdTree`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
fn neighborhood_preservation_2hop_v6(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
) -> PyResult<f64> {
    v6_neighborhood_preservation_2hop_impl(py, graph, layout)
}

/// Compute the undirected angle in `[0.0, pi]` radians between 2D vectors `(x1, y1)` and `(x2, y2)`.
///
/// Returns `None` if either vector is non-finite or has zero length.
#[pyfunction]
fn edge_angle(x1: StrictF64, y1: StrictF64, x2: StrictF64, y2: StrictF64) -> Option<f64> {
    egraph_metrics::edge_angle(x1.0, y1.0, x2.0, y2.0)
}

/// Compute PRISM layout-preservation distortion `(sigma_dist, sigma_disp)` between
/// `original_layout` and `adjusted_layout`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
fn prism_distortion(
    py: Python<'_>,
    original_layout: &PyLayout2d,
    adjusted_layout: &PyLayout2d,
) -> PyResult<(f64, f64)> {
    let res = py
        .detach(
            || match (original_layout.inner(), adjusted_layout.inner()) {
                (Layout2dInner::F64(orig), Layout2dInner::F64(adj)) => {
                    let o = orig.to_layout2d();
                    let a = adj.to_layout2d();
                    egraph_metrics::prism_distortion(&o, &a)
                }
                (Layout2dInner::F32(orig), Layout2dInner::F32(adj)) => {
                    let o = orig.to_layout2d();
                    let a = adj.to_layout2d();
                    egraph_metrics::prism_distortion(&o, &a)
                }
                (Layout2dInner::F64(orig), Layout2dInner::F32(adj)) => {
                    let o = orig.to_layout2d();
                    let a = layout_f32_to_f64(&adj.to_layout2d());
                    egraph_metrics::prism_distortion(&o, &a)
                }
                (Layout2dInner::F32(orig), Layout2dInner::F64(adj)) => {
                    let o = layout_f32_to_f64(&orig.to_layout2d());
                    let a = adj.to_layout2d();
                    egraph_metrics::prism_distortion(&o, &a)
                }
            },
        )
        .map_err(core_error_to_py)?;
    Ok((res.sigma_dist, res.sigma_disp))
}

/// Compute shape-based Relative Neighborhood Graph preservation (`dRNG`) in `[0.0, 1.0]`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
fn drng_preservation(py: Python<'_>, graph: &PyCsrGraph, layout: &PyLayout2d) -> PyResult<f64> {
    let g = graph.inner();
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::relative_neighborhood_preservation(g, &soa)
    })
}

/// Compute shape-based Gabriel Graph preservation (`dGG`) in `[0.0, 1.0]`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
fn dgg_preservation(py: Python<'_>, graph: &PyCsrGraph, layout: &PyLayout2d) -> PyResult<f64> {
    let g = graph.inner();
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::gabriel_graph_preservation(g, &soa)
    })
}

/// Compute Gabriel graph property violation sum over `graph` and `layout`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
fn gabriel_graph_property_v6(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
) -> PyResult<f64> {
    v6_gabriel_graph_property_impl(py, graph, layout)
}

/// Compute hinge node-resolution violation penalty for `layout`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
fn node_resolution_v6(py: Python<'_>, layout: &PyLayout2d) -> PyResult<f64> {
    v6_node_resolution_impl(py, layout)
}

/// Compute PCA singular-value aspect ratio `sigma_2 / sigma_1` in `[0.0, 1.0]` for `layout`.
///
/// Releases the Python GIL during computation.
#[pyfunction]
fn aspect_ratio_v6(py: Python<'_>, layout: &PyLayout2d) -> PyResult<f64> {
    v6_aspect_ratio_impl(py, layout)
}

/// Compute exponential angular-resolution penalty across incident edge pairs at each vertex.
///
/// Releases the Python GIL during computation.
#[pyfunction]
fn angular_resolution_v6(py: Python<'_>, graph: &PyCsrGraph, layout: &PyLayout2d) -> PyResult<f64> {
    v6_angular_resolution_impl(py, graph, layout)
}

/// Compute relative squared ideal edge-length error using edge weights (or `1.0` if unweighted).
///
/// Releases the Python GIL during computation.
#[pyfunction]
fn ideal_edge_lengths_v6(py: Python<'_>, graph: &PyCsrGraph, layout: &PyLayout2d) -> PyResult<f64> {
    v6_ideal_edge_lengths_impl(py, graph, layout)
}

/// Selectively evaluate requested quality metrics on `(graph, layout)` without running unrequested passes.
///
/// Supported target names and aliases include `"scale_normalized_stress"` (`"sns"`),
/// `"sampled_scale_normalized_stress"` (`"sampled_sns"`), `"crossing_number"`,
/// `"crossing_angle"`, `"sampled_crossing_number"`, `"neighborhood_preservation_1hop"`
/// (`"np1"`), `"neighborhood_preservation_2hop"` (`"np2"`), `"gabriel_graph_property"`,
/// `"node_resolution"`, `"relative_neighborhood_preservation"` (`"drng"`),
/// `"gabriel_graph_preservation"` (`"dgg"`), `"aspect_ratio"`, `"angular_resolution"`,
/// and `"ideal_edge_lengths"`.
///
/// Releases the Python GIL during computation and returns a dict keyed by canonical
/// `MetricReport` snake_case field names.
#[pyfunction]
#[pyo3(
    signature = (graph, layout, targets, *, num_pivots = StrictUsize(32), num_samples = StrictUsize(4096), seed = StrictU64(42), ideal_edge_length = StrictF64(1.0)),
    text_signature = "(graph, layout, targets, *, num_pivots=32, num_samples=4096, seed=42, ideal_edge_length=1.0)"
)]
#[allow(clippy::too_many_arguments)]
fn evaluate_metrics(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
    targets: &Bound<'_, PyAny>,
    num_pivots: StrictUsize,
    num_samples: StrictUsize,
    seed: StrictU64,
    ideal_edge_length: StrictF64,
) -> PyResult<BTreeMap<String, f64>> {
    let target_list = extract_metric_targets(targets)?;
    let config = MetricConfig {
        sns_num_pivots: num_pivots.0,
        sns_seed: seed.0,
        crossing_num_samples: num_samples.0,
        crossing_seed: seed.0,
        ideal_edge_length: ideal_edge_length.0,
    };
    let g = graph.inner();
    let report = dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::evaluate_metrics(g, &soa, &target_list, &config)
    })?;

    let mut out = BTreeMap::new();
    if let Some(v) = report.scale_normalized_stress {
        out.insert("scale_normalized_stress".to_owned(), v);
    }
    if let Some(v) = report.sampled_scale_normalized_stress {
        out.insert("sampled_scale_normalized_stress".to_owned(), v);
    }
    if let Some(v) = report.crossing_number {
        out.insert("crossing_number".to_owned(), v as f64);
    }
    if let Some(v) = report.crossing_angle {
        out.insert("crossing_angle".to_owned(), v);
    }
    if let Some(v) = report.sampled_crossing_number {
        out.insert("sampled_crossing_number".to_owned(), v);
    }
    if let Some(v) = report.neighborhood_preservation_1hop {
        out.insert("neighborhood_preservation_1hop".to_owned(), v);
    }
    if let Some(v) = report.neighborhood_preservation_2hop {
        out.insert("neighborhood_preservation_2hop".to_owned(), v);
    }
    if let Some(v) = report.gabriel_graph_property {
        out.insert("gabriel_graph_property".to_owned(), v);
    }
    if let Some(v) = report.node_resolution {
        out.insert("node_resolution".to_owned(), v);
    }
    if let Some(v) = report.relative_neighborhood_preservation {
        out.insert("relative_neighborhood_preservation".to_owned(), v);
    }
    if let Some(v) = report.gabriel_graph_preservation {
        out.insert("gabriel_graph_preservation".to_owned(), v);
    }
    if let Some(v) = report.aspect_ratio {
        out.insert("aspect_ratio".to_owned(), v);
    }
    if let Some(v) = report.angular_resolution {
        out.insert("angular_resolution".to_owned(), v);
    }
    if let Some(v) = report.ideal_edge_lengths {
        out.insert("ideal_edge_lengths".to_owned(), v);
    }
    Ok(out)
}

/// Evaluate relative squared ideal edge-length error against uniform `unit_edge_length`.
#[pyfunction]
#[pyo3(
    signature = (graph, layout, unit_edge_length = StrictF64(1.0)),
    text_signature = "(graph, layout, unit_edge_length=1.0)"
)]
fn ideal_edge_lengths_with_unit(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
    unit_edge_length: StrictF64,
) -> PyResult<f64> {
    let g = graph.inner();
    let unit = unit_edge_length.0;
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::ideal_edge_lengths_with_unit(g, &soa, unit)
    })
}

pub(crate) fn v6_crossing_edges_impl(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
) -> PyResult<Vec<Py<PyDict>>> {
    let g = graph.inner();
    let pairs = dispatch_layout2d!(py, layout, |soa| egraph_metrics::crossing_edges(g, &soa))?;
    let mut out = Vec::with_capacity(pairs.len());
    for pair in pairs {
        let dict = PyDict::new(py);
        dict.set_item("edge_a", pair.edge_a)?;
        dict.set_item("edge_b", pair.edge_b)?;
        dict.set_item("u1", pair.u1)?;
        dict.set_item("v1", pair.v1)?;
        dict.set_item("u2", pair.u2)?;
        dict.set_item("v2", pair.v2)?;
        dict.set_item("angle_rad", pair.angle_rad)?;
        dict.set_item("cos_sq", pair.cos_sq)?;
        out.push(dict.unbind());
    }
    Ok(out)
}

/// Find all crossing edge sub-segment pairs for `graph` laid out on the unit torus `[0, 1)^2`.
#[pyfunction]
#[pyo3(signature = (graph, layout), text_signature = "(graph, layout)")]
fn crossing_edges_torus(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
) -> PyResult<Vec<Py<PyDict>>> {
    let g = graph.inner();
    let pairs = dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::crossing_edges_torus(g, &soa)
    })?;
    let mut out = Vec::with_capacity(pairs.len());
    for pair in pairs {
        let dict = PyDict::new(py);
        dict.set_item("edge_a", pair.edge_a)?;
        dict.set_item("edge_b", pair.edge_b)?;
        dict.set_item("seg_a", pair.seg_a.to_vec())?;
        dict.set_item("seg_b", pair.seg_b.to_vec())?;
        out.push(dict.unbind());
    }
    Ok(out)
}

/// Compute crossing count and crossing-angle summary statistics in a single sweep pass.
#[pyfunction]
#[pyo3(signature = (graph, layout), text_signature = "(graph, layout)")]
fn crossing_summary(
    py: Python<'_>,
    graph: &PyCsrGraph,
    layout: &PyLayout2d,
) -> PyResult<Py<PyDict>> {
    let g = graph.inner();
    let summary = dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::crossing_summary(g, &soa)
    })?;
    let dict = PyDict::new(py);
    dict.set_item("count", summary.count)?;
    dict.set_item("angle_cos_sq_sum", summary.angle_cos_sq_sum)?;
    dict.set_item("min_angle_rad", summary.min_angle_rad)?;
    Ok(dict.unbind())
}

fn pairs_to_u32_array2<'py>(
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

/// Extract exact undirected Gabriel Graph edges `(u, v)` (`u < v`) as a `(E, 2)` `uint32` NumPy array.
#[pyfunction]
#[pyo3(signature = (layout), text_signature = "(layout)")]
fn gabriel_graph_edges<'py>(
    py: Python<'py>,
    layout: &PyLayout2d,
) -> PyResult<Bound<'py, PyArray2<u32>>> {
    let edges = dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::gabriel_graph_edges(&soa)
    })?;
    pairs_to_u32_array2(py, edges)
}

/// Extract exact undirected Relative Neighborhood Graph edges `(u, v)` (`u < v`) as a `(E, 2)` `uint32` NumPy array.
#[pyfunction]
#[pyo3(signature = (layout), text_signature = "(layout)")]
fn relative_neighborhood_edges<'py>(
    py: Python<'py>,
    layout: &PyLayout2d,
) -> PyResult<Bound<'py, PyArray2<u32>>> {
    let edges = dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::relative_neighborhood_edges(&soa)
    })?;
    pairs_to_u32_array2(py, edges)
}

/// Compute an all-pairs shortest-path `(N, N)` `float64` distance matrix via BFS.
#[pyfunction]
#[pyo3(
    signature = (graph, unit_edge_length = StrictF64(1.0)),
    text_signature = "(graph, unit_edge_length=1.0)"
)]
fn all_pairs_bfs<'py>(
    py: Python<'py>,
    graph: &PyCsrGraph,
    unit_edge_length: StrictF64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let g = graph.inner();
    let n = g.num_nodes();
    let unit = unit_edge_length.0;
    let dists = py
        .detach(|| egraph_metrics::all_pairs_bfs(g, unit))
        .map_err(core_error_to_py)?;
    let arr =
        Array2::from_shape_vec((n, n), dists).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

/// Compute single-source BFS shortest-path distances from `source` as a `(N,)` `float64` NumPy array.
#[pyfunction]
#[pyo3(
    signature = (graph, source, unit_edge_length = StrictF64(1.0)),
    text_signature = "(graph, source, unit_edge_length=1.0)"
)]
fn bfs_single_source<'py>(
    py: Python<'py>,
    graph: &PyCsrGraph,
    source: StrictUsize,
    unit_edge_length: StrictF64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let g = graph.inner();
    let src = source.0;
    let unit = unit_edge_length.0;
    let dists = py
        .detach(|| -> CoreResult<Vec<f64>> {
            let mut scratch = egraph_metrics::BfsScratch::new(g.num_nodes());
            let slice = egraph_metrics::bfs_single_source(g, src, unit, &mut scratch)?;
            Ok(slice.to_vec())
        })
        .map_err(core_error_to_py)?;
    Ok(PyArray1::from_vec(py, dists))
}

/// Compute closed-form Scale-Normalized Stress (`SNS`) from an `(N, N)` distance matrix and optional weights.
#[pyfunction]
#[pyo3(
    signature = (layout, distances, weights = None),
    text_signature = "(layout, distances, weights=None)"
)]
fn scale_normalized_stress_from_distances(
    py: Python<'_>,
    layout: &PyLayout2d,
    distances: &Bound<'_, PyAny>,
    weights: Option<&Bound<'_, PyAny>>,
) -> PyResult<f64> {
    let (dists_flat, d_rows, d_cols) = v6_m2::extract_f64_2d_raw(distances, "distances")?;
    if d_rows != d_cols {
        return Err(PyValueError::new_err(format!(
            "distances must be square (N, N), got ({d_rows}, {d_cols})"
        )));
    }
    let weights_flat = match weights {
        Some(w_obj) if !w_obj.is_none() => {
            let (wf, w_rows, w_cols) = v6_m2::extract_f64_2d_raw(w_obj, "weights")?;
            if w_rows != w_cols || w_rows != d_rows {
                return Err(PyValueError::new_err(format!(
                    "weights shape ({w_rows}, {w_cols}) does not match distances shape ({d_rows}, {d_cols})"
                )));
            }
            Some(wf)
        }
        _ => None,
    };
    dispatch_layout2d!(py, layout, |soa| {
        egraph_metrics::scale_normalized_stress_from_distances(
            &soa,
            &dists_flat,
            weights_flat.as_deref(),
        )
    })
}

/// Compute PRISM layout-preservation distortion `(sigma_dist, sigma_disp)` using explicit proximity edges.
#[pyfunction]
#[pyo3(
    signature = (original_layout, adjusted_layout, edges),
    text_signature = "(original_layout, adjusted_layout, edges)"
)]
fn prism_distortion_with_edges(
    py: Python<'_>,
    original_layout: &PyLayout2d,
    adjusted_layout: &PyLayout2d,
    edges: &Bound<'_, PyAny>,
) -> PyResult<(f64, f64)> {
    let flat = extract_flat_edges(edges)?;
    let (pair_chunks, _) = flat.as_chunks::<2>();
    let pairs: Vec<(u32, u32)> = pair_chunks.iter().map(|&[u, v]| (u, v)).collect();
    let res = py
        .detach(
            || match (original_layout.inner(), adjusted_layout.inner()) {
                (Layout2dInner::F64(orig), Layout2dInner::F64(adj)) => {
                    let o = orig.to_layout2d();
                    let a = adj.to_layout2d();
                    egraph_metrics::prism_distortion_with_edges(&o, &a, &pairs)
                }
                (Layout2dInner::F32(orig), Layout2dInner::F32(adj)) => {
                    let o = orig.to_layout2d();
                    let a = adj.to_layout2d();
                    egraph_metrics::prism_distortion_with_edges(&o, &a, &pairs)
                }
                (Layout2dInner::F64(orig), Layout2dInner::F32(adj)) => {
                    let o = orig.to_layout2d();
                    let a = layout_f32_to_f64(&adj.to_layout2d());
                    egraph_metrics::prism_distortion_with_edges(&o, &a, &pairs)
                }
                (Layout2dInner::F32(orig), Layout2dInner::F64(adj)) => {
                    let o = layout_f32_to_f64(&orig.to_layout2d());
                    let a = adj.to_layout2d();
                    egraph_metrics::prism_distortion_with_edges(&o, &a, &pairs)
                }
            },
        )
        .map_err(core_error_to_py)?;
    Ok((res.sigma_dist, res.sigma_disp))
}

type DelaunayArrays<'py> = (Bound<'py, PyArray2<u32>>, Bound<'py, PyArray2<u32>>);

/// Compute the deterministic 2D Delaunay triangulation of `layout`.
///
/// Returns `(edges, triangles)` as `uint32` NumPy arrays with shapes `(E, 2)` and `(T, 3)`.
/// Deduplicates coincident vertices deterministically and handles collinear layouts without
/// panicking. Releases the Python GIL during computation.
#[pyfunction]
fn delaunay_triangulation<'py>(
    py: Python<'py>,
    layout: &PyLayout2d,
) -> PyResult<DelaunayArrays<'py>> {
    let (flat_edges, e_len, flat_tris, t_len) = py
        .detach(|| -> CoreResult<(Vec<u32>, usize, Vec<u32>, usize)> {
            let dt = match layout.inner() {
                Layout2dInner::F64(buf) => Delaunay2d::triangulate(&buf.to_layout2d())?,
                Layout2dInner::F32(buf) => Delaunay2d::triangulate(&buf.to_layout2d())?,
            };
            let edges = dt.edges();
            let mut flat_edges = Vec::with_capacity(edges.len() * 2);
            for &(u, v) in edges {
                flat_edges.push(u);
                flat_edges.push(v);
            }
            let tris = dt.triangles();
            let mut flat_tris = Vec::with_capacity(tris.len() * 3);
            for &[a, b, c] in tris {
                flat_tris.push(a);
                flat_tris.push(b);
                flat_tris.push(c);
            }
            Ok((flat_edges, edges.len(), flat_tris, tris.len()))
        })
        .map_err(core_error_to_py)?;

    let edges_arr = Array2::from_shape_vec((e_len, 2), flat_edges)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let tris_arr = Array2::from_shape_vec((t_len, 3), flat_tris)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        PyArray2::from_owned_array(py, edges_arr),
        PyArray2::from_owned_array(py, tris_arr),
    ))
}

/// Compute the deterministic 2D Delaunay triangulation of `layout` and return an undirected `CsrGraph`.
#[pyfunction]
#[pyo3(signature = (layout), text_signature = "(layout)")]
fn delaunay_to_csr(py: Python<'_>, layout: &PyLayout2d) -> PyResult<PyCsrGraph> {
    let csr = dispatch_layout2d!(py, layout, |soa| Delaunay2d::triangulate(&soa)?.to_csr())?;
    Ok(PyCsrGraph::from_csr_graph(csr))
}

/// Find all strictly overlapping axis-aligned rectangle pairs `(i, j)` (`i < j`) as a `(K, 2)` `uint32` array.
#[pyfunction]
#[pyo3(
    signature = (layout, half_widths, half_heights),
    text_signature = "(layout, half_widths, half_heights)"
)]
fn rectangle_overlaps<'py>(
    py: Python<'py>,
    layout: &PyLayout2d,
    half_widths: &Bound<'_, PyAny>,
    half_heights: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyArray2<u32>>> {
    let hw_f64 = v6_m2::extract_f64_1d(half_widths, "half_widths")?;
    let hh_f64 = v6_m2::extract_f64_1d(half_heights, "half_heights")?;
    let pairs = py
        .detach(|| -> CoreResult<Vec<(usize, usize)>> {
            match layout.inner() {
                Layout2dInner::F64(buf) => RectangleOverlapSweep::find_layout_overlaps(
                    &buf.to_layout2d(),
                    &hw_f64,
                    &hh_f64,
                ),
                Layout2dInner::F32(buf) => {
                    let hw32: Vec<f32> = hw_f64.iter().map(|&v| v as f32).collect();
                    let hh32: Vec<f32> = hh_f64.iter().map(|&v| v as f32).collect();
                    if hw32.iter().any(|v| !v.is_finite()) {
                        return Err(CoreError::non_finite("rectangle_overlaps::half_widths"));
                    }
                    if hh32.iter().any(|v| !v.is_finite()) {
                        return Err(CoreError::non_finite("rectangle_overlaps::half_heights"));
                    }
                    RectangleOverlapSweep::find_layout_overlaps(&buf.to_layout2d(), &hw32, &hh32)
                }
            }
        })
        .map_err(core_error_to_py)?;
    let mut flat = Vec::with_capacity(pairs.len() * 2);
    for (i, j) in &pairs {
        let iu = u32::try_from(*i)
            .map_err(|_| PyValueError::new_err(format!("rectangle index {i} exceeds u32")))?;
        let ju = u32::try_from(*j)
            .map_err(|_| PyValueError::new_err(format!("rectangle index {j} exceeds u32")))?;
        flat.push(iu);
        flat.push(ju);
    }
    let arr = Array2::from_shape_vec((pairs.len(), 2), flat)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

/// Load a built-in SuiteSparse / classic benchmark dataset as an undirected `CsrGraph`.
#[pyfunction]
#[pyo3(signature = (name), text_signature = "(name)")]
fn load_dataset(py: Python<'_>, name: &str) -> PyResult<PyCsrGraph> {
    let key: String = name
        .trim()
        .chars()
        .filter(|c| *c != '_' && *c != '-' && *c != ' ')
        .map(|c| c.to_ascii_lowercase())
        .collect();
    let csr = py.detach(|| -> CoreResult<CsrGraph<f64>> {
        match key.as_str() {
            "1138bus" => Ok(egraph_dataset::dataset_1138_bus_csr()),
            "3elt" => Ok(egraph_dataset::dataset_3_elt_csr()),
            "dwt1005" => Ok(egraph_dataset::dataset_dwt_1005_csr()),
            "dwt2680" => Ok(egraph_dataset::dataset_dwt_2680_csr()),
            "poli" => Ok(egraph_dataset::dataset_poli_csr()),
            "qh882" => Ok(egraph_dataset::dataset_qh882_csr()),
            "uspowergrid" => Ok(egraph_dataset::dataset_uspower_grid_csr()),
            "karate" => Ok(egraph_dataset::dataset_karate_csr()),
            "lesmis" => Ok(egraph_dataset::dataset_lesmis_csr()),
            _ => Err(CoreError::invalid_input(format!(
                "unknown dataset {name:?}; expected one of '1138_bus', '3_elt', 'dwt_1005', 'dwt_2680', 'poli', 'qh882', 'uspower_grid', 'karate', 'lesmis'"
            ))),
        }
    })
    .map_err(core_error_to_py)?;
    Ok(PyCsrGraph::from_csr_graph(csr))
}

/// Parse an edge-list CSV string (`num_nodes,num_edges` header followed by `u,v` lines) into a `CsrGraph`.
#[pyfunction]
#[pyo3(
    signature = (csv, *, directed = false, sanitize = true),
    text_signature = "(csv, *, directed=False, sanitize=True)"
)]
fn parse_csr_csv(
    py: Python<'_>,
    csv: &str,
    directed: bool,
    sanitize: bool,
) -> PyResult<PyCsrGraph> {
    let csr = py
        .detach(|| egraph_dataset::parse_csr_csv(csv, directed, sanitize))
        .map_err(core_error_to_py)?;
    Ok(PyCsrGraph::from_csr_graph(csr))
}

/// Register the v6 zero-copy `CsrGraph`, `Layout2d`, `LayoutNd`, metrics, dataset, and spatial functions on the Python extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCsrGraph>()?;
    m.add_class::<PyLayout2d>()?;
    m.add_class::<PyLayoutNd>()?;
    m.add_function(wrap_pyfunction!(scale_normalized_stress, m)?)?;
    m.add_function(wrap_pyfunction!(sampled_scale_normalized_stress, m)?)?;
    m.add_function(wrap_pyfunction!(sweep_crossing_number, m)?)?;
    m.add_function(wrap_pyfunction!(sweep_crossing_angle, m)?)?;
    m.add_function(wrap_pyfunction!(sampled_crossing_number, m)?)?;
    m.add_function(wrap_pyfunction!(torus_crossing_number, m)?)?;
    m.add_function(wrap_pyfunction!(neighborhood_preservation_1hop, m)?)?;
    m.add_function(wrap_pyfunction!(neighborhood_preservation_2hop_v6, m)?)?;
    m.add_function(wrap_pyfunction!(edge_angle, m)?)?;
    m.add_function(wrap_pyfunction!(prism_distortion, m)?)?;
    m.add_function(wrap_pyfunction!(prism_distortion_with_edges, m)?)?;
    m.add_function(wrap_pyfunction!(drng_preservation, m)?)?;
    m.add_function(wrap_pyfunction!(dgg_preservation, m)?)?;
    m.add_function(wrap_pyfunction!(gabriel_graph_property_v6, m)?)?;
    m.add_function(wrap_pyfunction!(node_resolution_v6, m)?)?;
    m.add_function(wrap_pyfunction!(aspect_ratio_v6, m)?)?;
    m.add_function(wrap_pyfunction!(angular_resolution_v6, m)?)?;
    m.add_function(wrap_pyfunction!(ideal_edge_lengths_v6, m)?)?;
    m.add_function(wrap_pyfunction!(ideal_edge_lengths_with_unit, m)?)?;
    m.add_function(wrap_pyfunction!(crossing_edges_torus, m)?)?;
    m.add_function(wrap_pyfunction!(crossing_summary, m)?)?;
    m.add_function(wrap_pyfunction!(gabriel_graph_edges, m)?)?;
    m.add_function(wrap_pyfunction!(relative_neighborhood_edges, m)?)?;
    m.add_function(wrap_pyfunction!(all_pairs_bfs, m)?)?;
    m.add_function(wrap_pyfunction!(bfs_single_source, m)?)?;
    m.add_function(wrap_pyfunction!(scale_normalized_stress_from_distances, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(delaunay_triangulation, m)?)?;
    m.add_function(wrap_pyfunction!(delaunay_to_csr, m)?)?;
    m.add_function(wrap_pyfunction!(rectangle_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(load_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(parse_csr_csv, m)?)?;
    v6_m2::register_m2(m)?;
    v6_m3::register_m3(m)?;
    v6_m4::register_m4(m)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{MetricTarget, ScalarDtype, parse_metric_target};

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

    #[test]
    fn test_metric_target_aliases() {
        assert_eq!(
            parse_metric_target("sns").unwrap(),
            MetricTarget::ScaleNormalizedStress
        );
        assert_eq!(
            parse_metric_target("ScaleNormalizedStress").unwrap(),
            MetricTarget::ScaleNormalizedStress
        );
        assert_eq!(
            parse_metric_target("sampled_sns").unwrap(),
            MetricTarget::SampledScaleNormalizedStress
        );
        assert_eq!(
            parse_metric_target("np1").unwrap(),
            MetricTarget::NeighborhoodPreservation1Hop
        );
        assert_eq!(
            parse_metric_target("np2").unwrap(),
            MetricTarget::NeighborhoodPreservation2Hop
        );
        assert_eq!(
            parse_metric_target("drng").unwrap(),
            MetricTarget::RelativeNeighborhoodPreservation
        );
        assert_eq!(
            parse_metric_target("dgg").unwrap(),
            MetricTarget::GabrielGraphPreservation
        );
        assert!(parse_metric_target("not_a_metric").is_err());
    }
}
