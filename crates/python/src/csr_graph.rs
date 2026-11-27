//! Compressed Sparse Row (`CsrGraph`) Python bindings.

use std::borrow::Cow;

use egraph::core::CsrGraph;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;

pub(crate) use crate::csr_helpers::extract_flat_edges;
use crate::csr_helpers::{CsrGraphInner, build_csr_inner, build_edges_inner};
use crate::ffi_convert::{
    ScalarDtype, StrictUsize, core_error_to_py, extract_non_negative_usize,
    extract_optional_weights_1d, extract_u32_1d, pairs_to_u32_array2,
};

/// Compact `u32`-indexed Compressed Sparse Row (`CsrGraph`) graph representation.
///
/// Stores vertex neighborhoods in flat CSR arrays with optional `float64` or `float32` edge
/// weights, supporting undirected/directed graphs, connectivity queries, and LCC extraction.
#[pyclass(name = "CsrGraph", module = "egraph._core", from_py_object)]
#[derive(Clone)]
pub struct PyCsrGraph {
    inner: CsrGraphInner,
}

impl PyCsrGraph {
    pub(crate) fn inner(&self) -> Cow<'_, CsrGraph<f64>> {
        self.inner.as_f64_cow()
    }

    pub(crate) fn from_csr_graph(inner: CsrGraph<f64>) -> Self {
        Self {
            inner: CsrGraphInner::F64(inner),
        }
    }
}

#[pymethods]
impl PyCsrGraph {
    /// Construct a `CsrGraph` from CSR `indptr`, `indices`, and optional `weights`.
    #[staticmethod]
    #[pyo3(
        signature = (indptr, indices, weights = None, *, directed = false, sanitize = true, weight_dtype = ScalarDtype::F64),
        text_signature = "(indptr, indices, weights=None, *, directed=False, sanitize=True, weight_dtype='float64')"
    )]
    fn from_csr(
        py: Python<'_>,
        indptr: &Bound<'_, PyAny>,
        indices: &Bound<'_, PyAny>,
        weights: Option<&Bound<'_, PyAny>>,
        directed: bool,
        sanitize: bool,
        weight_dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let indptr_vec = extract_u32_1d(indptr, "indptr")?;
        let indices_vec = extract_u32_1d(indices, "indices")?;
        let weights_vec = extract_optional_weights_1d(weights, "weights")?;
        let inner = py.detach(|| {
            build_csr_inner(
                &indptr_vec,
                &indices_vec,
                weights_vec.as_deref(),
                directed,
                sanitize,
                weight_dtype,
            )
        })?;
        Ok(Self { inner })
    }

    /// Construct a `CsrGraph` from `num_nodes` and an `(M, 2)` edge list with optional `weights`.
    #[staticmethod]
    #[pyo3(
        signature = (num_nodes, edges, weights = None, *, directed = false, sanitize = true, weight_dtype = ScalarDtype::F64),
        text_signature = "(num_nodes, edges, weights=None, *, directed=False, sanitize=True, weight_dtype='float64')"
    )]
    fn from_edges(
        py: Python<'_>,
        num_nodes: &Bound<'_, PyAny>,
        edges: &Bound<'_, PyAny>,
        weights: Option<&Bound<'_, PyAny>>,
        directed: bool,
        sanitize: bool,
        weight_dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let num_nodes_usize = extract_non_negative_usize(num_nodes, "num_nodes")?;
        let flat_edges = extract_flat_edges(edges)?;
        let weights_vec = extract_optional_weights_1d(weights, "weights")?;
        let inner = py.detach(|| {
            build_edges_inner(
                num_nodes_usize,
                &flat_edges,
                weights_vec.as_deref(),
                directed,
                sanitize,
                weight_dtype,
            )
        })?;
        Ok(Self { inner })
    }

    /// Alias for :meth:`CsrGraph.from_edges` matching WASM `CsrGraph.fromEdgeList`.
    #[staticmethod]
    #[pyo3(
        signature = (num_nodes, edges, weights = None, *, directed = false, sanitize = true, weight_dtype = ScalarDtype::F64),
        text_signature = "(num_nodes, edges, weights=None, *, directed=False, sanitize=True, weight_dtype='float64')"
    )]
    fn from_edge_list(
        py: Python<'_>,
        num_nodes: &Bound<'_, PyAny>,
        edges: &Bound<'_, PyAny>,
        weights: Option<&Bound<'_, PyAny>>,
        directed: bool,
        sanitize: bool,
        weight_dtype: ScalarDtype,
    ) -> PyResult<Self> {
        Self::from_edges(
            py,
            num_nodes,
            edges,
            weights,
            directed,
            sanitize,
            weight_dtype,
        )
    }

    /// Number of vertices in the graph.
    #[getter]
    fn num_nodes(&self) -> usize {
        match &self.inner {
            CsrGraphInner::F64(g) => g.num_nodes(),
            CsrGraphInner::F32(g) => g.num_nodes(),
        }
    }

    /// Number of vertices in the graph.
    fn __len__(&self) -> usize {
        self.num_nodes()
    }

    /// Number of logical edges in the graph (`nnz` if directed, `nnz // 2` if undirected).
    #[getter]
    fn num_edges(&self) -> usize {
        match &self.inner {
            CsrGraphInner::F64(g) => g.num_edges(),
            CsrGraphInner::F32(g) => g.num_edges(),
        }
    }

    /// Number of stored CSR half-edge entries (`indices().len()`).
    #[getter]
    fn nnz(&self) -> usize {
        match &self.inner {
            CsrGraphInner::F64(g) => g.nnz(),
            CsrGraphInner::F32(g) => g.nnz(),
        }
    }

    /// Whether the graph is directed.
    #[getter]
    fn is_directed(&self) -> bool {
        match &self.inner {
            CsrGraphInner::F64(g) => g.is_directed(),
            CsrGraphInner::F32(g) => g.is_directed(),
        }
    }

    /// Whether explicit edge weights are stored.
    #[getter]
    fn is_weighted(&self) -> bool {
        match &self.inner {
            CsrGraphInner::F64(g) => g.is_weighted(),
            CsrGraphInner::F32(g) => g.is_weighted(),
        }
    }

    /// Stored edge weight scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn weight_dtype(&self) -> &'static str {
        self.inner.weight_dtype().as_str()
    }

    /// Return the degree (outgoing degree for directed graphs) of vertex `u`.
    #[pyo3(signature = (u), text_signature = "($self, u)")]
    fn degree(&self, u: StrictUsize) -> PyResult<usize> {
        match &self.inner {
            CsrGraphInner::F64(g) => g.try_degree(u.0),
            CsrGraphInner::F32(g) => g.try_degree(u.0),
        }
        .map_err(core_error_to_py)
    }

    /// Return a `uint32` NumPy copy of neighbor vertex IDs adjacent to `u`.
    #[pyo3(signature = (u), text_signature = "($self, u)")]
    fn neighbors<'py>(
        &self,
        py: Python<'py>,
        u: StrictUsize,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let slice = match &self.inner {
            CsrGraphInner::F64(g) => g.try_neighbors(u.0),
            CsrGraphInner::F32(g) => g.try_neighbors(u.0),
        }
        .map_err(core_error_to_py)?;
        Ok(PyArray1::from_slice(py, slice))
    }

    /// Return all logical edge pairs `(u, v)` as a `uint32` NumPy array of shape `(M, 2)`.
    fn edges<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u32>>> {
        let pairs = py.detach(|| match &self.inner {
            CsrGraphInner::F64(g) => g.edges(),
            CsrGraphInner::F32(g) => g.edges(),
        });
        pairs_to_u32_array2(py, pairs)
    }

    /// Return `True` if edge `(u, v)` exists in the graph (raises `IndexError` if `u`/`v` is out of bounds).
    #[pyo3(signature = (u, v), text_signature = "($self, u, v)")]
    fn has_edge(&self, u: StrictUsize, v: StrictUsize) -> PyResult<bool> {
        match &self.inner {
            CsrGraphInner::F64(g) => g.try_has_edge(u.0, v.0),
            CsrGraphInner::F32(g) => g.try_has_edge(u.0, v.0),
        }
        .map_err(core_error_to_py)
    }

    /// Return the weight of edge `(u, v)`, or `None` if absent or unweighted (`IndexError` if out of bounds).
    #[pyo3(signature = (u, v), text_signature = "($self, u, v)")]
    fn edge_weight(&self, u: StrictUsize, v: StrictUsize) -> PyResult<Option<f64>> {
        match &self.inner {
            CsrGraphInner::F64(g) => g.try_edge_weight(u.0, v.0),
            CsrGraphInner::F32(g) => g.try_edge_weight(u.0, v.0).map(|opt| opt.map(f64::from)),
        }
        .map_err(core_error_to_py)
    }

    /// Return `True` if the graph is (weakly) connected.
    fn is_connected(&self, py: Python<'_>) -> bool {
        py.detach(|| match &self.inner {
            CsrGraphInner::F64(g) => g.is_connected(),
            CsrGraphInner::F32(g) => g.is_connected(),
        })
    }

    /// Extract the induced subgraph on `nodes` (with compact `0..len(nodes)` vertex IDs).
    #[pyo3(signature = (nodes), text_signature = "($self, nodes)")]
    fn induced_subgraph(&self, py: Python<'_>, nodes: &Bound<'_, PyAny>) -> PyResult<Self> {
        let nodes_vec = extract_u32_1d(nodes, "nodes")?;
        let inner = py
            .detach(|| match &self.inner {
                CsrGraphInner::F64(g) => g.induced_subgraph(&nodes_vec).map(CsrGraphInner::F64),
                CsrGraphInner::F32(g) => g.induced_subgraph(&nodes_vec).map(CsrGraphInner::F32),
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Return a `uint32` NumPy copy of the CSR row pointer array of length `num_nodes + 1`.
    fn indptr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        match &self.inner {
            CsrGraphInner::F64(g) => PyArray1::from_slice(py, g.indptr()),
            CsrGraphInner::F32(g) => PyArray1::from_slice(py, g.indptr()),
        }
    }

    /// Return a `uint32` NumPy copy of the CSR neighbor index array.
    fn indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        match &self.inner {
            CsrGraphInner::F64(g) => PyArray1::from_slice(py, g.indices()),
            CsrGraphInner::F32(g) => PyArray1::from_slice(py, g.indices()),
        }
    }

    /// Return a NumPy copy of edge weights (`float64` or `float32`) aligned with `indices()`, or `None`.
    fn weights<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyAny>> {
        match &self.inner {
            CsrGraphInner::F64(g) => g.weights().map(|w| PyArray1::from_slice(py, w).into_any()),
            CsrGraphInner::F32(g) => g.weights().map(|w| PyArray1::from_slice(py, w).into_any()),
        }
    }

    /// Extract the largest (weakly) connected component as `(subgraph, original_node_ids)`.
    fn largest_connected_component<'py>(
        &self,
        py: Python<'py>,
    ) -> (Self, Bound<'py, PyArray1<u32>>) {
        let (inner, original_node_ids) = py.detach(|| match &self.inner {
            CsrGraphInner::F64(g) => {
                let (sub, ids) = g.largest_connected_component();
                (CsrGraphInner::F64(sub), ids)
            }
            CsrGraphInner::F32(g) => {
                let (sub, ids) = g.largest_connected_component();
                (CsrGraphInner::F32(sub), ids)
            }
        });
        let ids_array = PyArray1::from_slice(py, &original_node_ids);
        (Self { inner }, ids_array)
    }

    /// Return a concise string representation of the `CsrGraph`.
    fn __repr__(&self) -> String {
        format!(
            "CsrGraph(num_nodes={}, num_edges={}, directed={})",
            self.num_nodes(),
            self.num_edges(),
            if self.is_directed() { "True" } else { "False" }
        )
    }
}

/// Register `CsrGraph` on the Python extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCsrGraph>()?;
    Ok(())
}
