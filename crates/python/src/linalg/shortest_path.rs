//! Weighted shortest-path (`DijkstraWorkspace`, `dijkstra_single_source`, `all_pairs_dijkstra`) bindings.

use egraph::linalg::{
    DijkstraWorkspace, all_pairs_dijkstra as linalg_all_pairs_dijkstra,
    dijkstra_single_source as linalg_dijkstra_single_source,
};
use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{StrictUsize, core_error_to_py};

/// Reusable single-source Dijkstra workspace avoiding repeated heap allocations across queries.
#[pyclass(name = "DijkstraWorkspace", module = "egraph._core")]
pub struct PyDijkstraWorkspace {
    inner: DijkstraWorkspace,
}

#[pymethods]
impl PyDijkstraWorkspace {
    /// Construct a Dijkstra workspace pre-sized for `num_nodes` vertices.
    #[new]
    #[pyo3(signature = (num_nodes = StrictUsize(0)), text_signature = "(num_nodes=0)")]
    fn new(num_nodes: StrictUsize) -> Self {
        Self {
            inner: DijkstraWorkspace::new(num_nodes.0),
        }
    }

    /// Ensure internal scratch buffers cover at least `num_nodes` vertices and reset marks.
    #[pyo3(signature = (num_nodes), text_signature = "($self, num_nodes)")]
    fn ensure_capacity(&mut self, num_nodes: StrictUsize) {
        self.inner.ensure_capacity(num_nodes.0);
    }

    /// Compute weighted single-source shortest-path distances from `source` on `graph`.
    #[pyo3(signature = (graph, source), text_signature = "($self, graph, source)")]
    fn dijkstra_from<'py>(
        &mut self,
        py: Python<'py>,
        graph: &PyCsrGraph,
        source: StrictUsize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let g = graph.inner();
        let n = g.num_nodes();
        let src = source.0;
        let ws = &mut self.inner;
        let dists = py
            .detach(|| -> egraph::core::CoreResult<Vec<f64>> {
                let mut out = vec![f64::INFINITY; n];
                ws.dijkstra_from(&*g, src, &mut out)?;
                Ok(out)
            })
            .map_err(core_error_to_py)?;
        Ok(PyArray1::from_vec(py, dists))
    }

    /// Return a concise string representation of the `DijkstraWorkspace`.
    fn __repr__(&self) -> &'static str {
        "DijkstraWorkspace()"
    }
}

/// Compute single-source weighted shortest-path distances from `source` on `graph`.
#[pyfunction]
#[pyo3(signature = (graph, source), text_signature = "(graph, source)")]
fn dijkstra_single_source<'py>(
    py: Python<'py>,
    graph: &PyCsrGraph,
    source: StrictUsize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let g = graph.inner();
    let dists = py
        .detach(|| linalg_dijkstra_single_source(&*g, source.0))
        .map_err(core_error_to_py)?;
    Ok(PyArray1::from_vec(py, dists))
}

/// Compute an `(N, N)` all-pairs weighted shortest-path distance matrix via repeated Dijkstra.
#[pyfunction]
#[pyo3(signature = (graph), text_signature = "(graph)")]
fn all_pairs_dijkstra<'py>(
    py: Python<'py>,
    graph: &PyCsrGraph,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let g = graph.inner();
    let n = g.num_nodes();
    let flat = py
        .detach(|| linalg_all_pairs_dijkstra(&*g))
        .map_err(core_error_to_py)?;
    let arr =
        Array2::from_shape_vec((n, n), flat).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDijkstraWorkspace>()?;
    m.add_function(wrap_pyfunction!(dijkstra_single_source, m)?)?;
    m.add_function(wrap_pyfunction!(all_pairs_dijkstra, m)?)?;
    Ok(())
}
