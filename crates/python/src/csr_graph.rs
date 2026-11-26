//! Compressed Sparse Row (`CsrGraph`) Python bindings and edge-list extractors.

use egraph::core::CsrGraph;
use egraph_ffi::{build_csr_f64, build_edges_f64};
use numpy::{PyArray1, PyReadonlyArray2, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::ffi_convert::{
    StrictUsize, core_error_to_py, extract_non_negative_usize, extract_optional_weights_1d,
    extract_u32_1d, extract_u32_item, ffi_error_to_py, is_non_sequence_container,
};

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

pub(crate) fn extract_flat_edges(edges: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
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

    pub(crate) fn from_csr_graph(inner: CsrGraph<f64>) -> Self {
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

/// Register `CsrGraph` on the Python extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCsrGraph>()?;
    Ok(())
}
