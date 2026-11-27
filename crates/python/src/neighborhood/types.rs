//! Core neighborhood wrapper types (`KnnGraph`, `SparseProbabilities`, `Sparse12HopPairs`).

use egraph::neighborhood::{KnnGraph, KnnStats, Sparse12HopPairs, SparseProbabilities};
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::array_extract::extract_f64_1d;
use crate::ffi_convert::{StrictUsize, core_error_to_py, extract_u32_1d};

pub(super) fn knn_stats_float_fields(stats: &KnnStats) -> [(&'static str, f64); 3] {
    [
        ("avg_deg1", stats.avg_deg1),
        ("avg_deg2", stats.avg_deg2),
        ("avg_row_k", stats.avg_row_k),
    ]
}

pub(super) fn knn_stats_int_fields(stats: &KnnStats) -> [(&'static str, usize); 9] {
    [
        ("min_row_k", stats.min_row_k),
        ("max_row_k", stats.max_row_k),
        ("nnz", stats.nnz),
        ("n_1hop", stats.n_1hop),
        ("n_2hop", stats.n_2hop),
        ("n_truncated_1hop", stats.n_truncated_1hop),
        ("n_truncated_2hop", stats.n_truncated_2hop),
        ("n_filled_residual", stats.n_filled_residual),
        ("n_pure_2hop_rows", stats.n_pure_2hop_rows),
    ]
}

pub(super) fn knn_stats_to_py_dict(py: Python<'_>, stats: &KnnStats) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    for (k, v) in knn_stats_float_fields(stats) {
        dict.set_item(k, v)?;
    }
    for (k, v) in knn_stats_int_fields(stats) {
        dict.set_item(k, v)?;
    }
    Ok(dict.unbind())
}

/// Validated Compressed Sparse Row (CSR) `k`-nearest-neighbor graph in `float64`.
#[pyclass(name = "KnnGraph", module = "egraph._core", from_py_object)]
#[derive(Clone)]
pub struct PyKnnGraph {
    inner: KnnGraph,
}

impl PyKnnGraph {
    pub(crate) fn inner(&self) -> &KnnGraph {
        &self.inner
    }

    pub(super) fn from_inner(inner: KnnGraph) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyKnnGraph {
    /// Construct a validated `KnnGraph` from raw CSR arrays `(num_nodes, indptr, indices, distances)`.
    #[staticmethod]
    #[pyo3(
        signature = (num_nodes, indptr, indices, distances),
        text_signature = "(num_nodes, indptr, indices, distances)"
    )]
    fn from_csr(
        py: Python<'_>,
        num_nodes: StrictUsize,
        indptr: &Bound<'_, PyAny>,
        indices: &Bound<'_, PyAny>,
        distances: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let indptr_vec = extract_u32_1d(indptr, "indptr")?;
        let indices_vec = extract_u32_1d(indices, "indices")?;
        let distances_vec = extract_f64_1d(distances, "distances")?;

        let inner = py
            .detach(|| KnnGraph::new_validated(num_nodes.0, indptr_vec, indices_vec, distances_vec))
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Number of vertices in the `k`-NN graph.
    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Total number of stored directed neighbor entries.
    #[getter]
    fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    /// Return a `uint32` NumPy copy of the CSR row pointer array of length `num_nodes + 1`.
    fn indptr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_slice(py, self.inner.indptr())
    }

    /// Return a `uint32` NumPy copy of the CSR neighbor index array.
    fn indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_slice(py, self.inner.indices())
    }

    /// Return a `float64` NumPy copy of neighbor distances aligned with `indices()`.
    fn distances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, self.inner.distances())
    }

    /// Return the number of vertices in the `k`-NN graph.
    fn __len__(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Return a concise string representation of the `KnnGraph`.
    fn __repr__(&self) -> String {
        format!(
            "KnnGraph(num_nodes={}, nnz={})",
            self.inner.num_nodes(),
            self.inner.nnz()
        )
    }
}

/// Canonical sorted symmetric CSR joint probability matrix `P`.
#[pyclass(name = "SparseProbabilities", module = "egraph._core", from_py_object)]
#[derive(Clone)]
pub struct PySparseProbabilities {
    inner: SparseProbabilities,
}

impl PySparseProbabilities {
    pub(crate) fn inner(&self) -> &SparseProbabilities {
        &self.inner
    }

    pub(super) fn from_inner(inner: SparseProbabilities) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySparseProbabilities {
    /// Construct a validated `SparseProbabilities` from CSR arrays `(num_nodes, indptr, indices, probabilities)`.
    #[staticmethod]
    #[pyo3(
        signature = (num_nodes, indptr, indices, probabilities),
        text_signature = "(num_nodes, indptr, indices, probabilities)"
    )]
    fn from_csr(
        py: Python<'_>,
        num_nodes: StrictUsize,
        indptr: &Bound<'_, PyAny>,
        indices: &Bound<'_, PyAny>,
        probabilities: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let indptr_vec = extract_u32_1d(indptr, "indptr")?;
        let indices_vec = extract_u32_1d(indices, "indices")?;
        let probs_vec = extract_f64_1d(probabilities, "probabilities")?;
        let inner = py
            .detach(|| {
                SparseProbabilities::from_csr(num_nodes.0, indptr_vec, indices_vec, probs_vec)
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Number of vertices in the probability matrix.
    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Total number of stored symmetric probability entries.
    #[getter]
    fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    /// Return a `uint32` NumPy copy of the CSR row pointer array of length `num_nodes + 1`.
    fn indptr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_slice(py, &self.inner.indptr)
    }

    /// Return a `uint32` NumPy copy of the sorted CSR neighbor column index array.
    fn indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_slice(py, &self.inner.indices)
    }

    /// Return a `float64` NumPy copy of joint probabilities aligned with `indices()`.
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.values)
    }

    /// Return the number of vertices in the probability matrix.
    fn __len__(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Return a concise string representation of the `SparseProbabilities`.
    fn __repr__(&self) -> String {
        format!(
            "SparseProbabilities(num_nodes={}, nnz={})",
            self.inner.num_nodes(),
            self.inner.nnz()
        )
    }
}

/// Extracted sparse undirected 1-hop and capped 2-hop vertex pair arrays `(u, v)` with `u < v`.
#[pyclass(name = "Sparse12HopPairs", module = "egraph._core", from_py_object)]
#[derive(Clone)]
pub struct PySparse12HopPairs {
    inner: Sparse12HopPairs,
}

impl PySparse12HopPairs {
    pub(super) fn from_inner(inner: Sparse12HopPairs) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySparse12HopPairs {
    /// Number of unique undirected 1-hop pairs (`u1 < v1`).
    #[getter]
    fn num_1hop(&self) -> usize {
        self.inner.u1.len()
    }

    /// Number of unique undirected 2-hop pairs (`u2 < v2`).
    #[getter]
    fn num_2hop(&self) -> usize {
        self.inner.u2.len()
    }

    /// Return a `uint32` NumPy copy of 1-hop source vertex indices.
    fn u1<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_slice(py, &self.inner.u1)
    }

    /// Return a `uint32` NumPy copy of 1-hop target vertex indices.
    fn v1<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_slice(py, &self.inner.v1)
    }

    /// Return a `uint32` NumPy copy of 2-hop source vertex indices.
    fn u2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_slice(py, &self.inner.u2)
    }

    /// Return a `uint32` NumPy copy of 2-hop target vertex indices.
    fn v2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_slice(py, &self.inner.v2)
    }

    /// Return a concise string representation of the `Sparse12HopPairs`.
    fn __repr__(&self) -> String {
        format!(
            "Sparse12HopPairs(num_1hop={}, num_2hop={})",
            self.inner.u1.len(),
            self.inner.u2.len()
        )
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKnnGraph>()?;
    m.add_class::<PySparseProbabilities>()?;
    m.add_class::<PySparse12HopPairs>()?;
    Ok(())
}
