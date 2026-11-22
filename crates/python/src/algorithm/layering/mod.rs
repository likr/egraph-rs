/// Python bindings for petgraph-algorithm-layering.
///
/// This module provides Python bindings for the petgraph-algorithm-layering Rust crate,
/// which implements algorithms for assigning layers to nodes in directed graphs.
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

mod longest_path;

use crate::graph::{GraphType, PyGraphAdapter};
use crate::v6::{PyCsrGraph, v6_cycle_edges_impl};
use longest_path::PyLongestPath;

/// Register the layering module and its classes with Python.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLongestPath>()?;

    // Register cycle detection and removal functions
    m.add_function(wrap_pyfunction!(cycle_edges, m)?)?;
    m.add_function(wrap_pyfunction!(remove_cycle, m)?)?;

    Ok(())
}

/// Get edges that form cycles in a directed graph (`CsrGraph` or `DiGraph`).
///
/// For `CsrGraph`, returns a `uint32` NumPy array of shape `(M, 2)` and releases the GIL.
/// For legacy `DiGraph`, returns a list of `(u, v)` index tuples.
#[pyfunction]
fn cycle_edges<'py>(py: Python<'py>, graph: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(csr) = graph.cast::<PyCsrGraph>() {
        return v6_cycle_edges_impl(py, &csr.borrow()).map(Bound::into_any);
    }
    if let Ok(adapter) = graph.cast::<PyGraphAdapter>() {
        let borrowed = adapter.borrow();
        return match borrowed.graph() {
            GraphType::Graph(_) => Err(PyValueError::new_err(
                "cycle_edges only works with directed graphs",
            )),
            GraphType::DiGraph(graph) => {
                let edges: Vec<(usize, usize)> =
                    petgraph_algorithm_layering::cycle::cycle_edges(graph)
                        .into_iter()
                        .map(|(u, v)| (u.index(), v.index()))
                        .collect();
                Ok(edges.into_pyobject(py)?.into_any())
            }
        };
    }
    Err(PyTypeError::new_err(
        "cycle_edges: graph must be a directed CsrGraph or DiGraph",
    ))
}

/// Remove cycles from a directed graph.
///
/// This function removes the minimum number of edges to make the graph acyclic.
///
/// Args:
///     graph: A directed graph.
///
/// Returns:
///     None: The graph is modified in-place.
#[pyfunction]
fn remove_cycle(graph: &mut PyGraphAdapter) -> PyResult<()> {
    match graph.graph_mut() {
        GraphType::Graph(_) => Err(pyo3::exceptions::PyValueError::new_err(
            "remove_cycle only works with directed graphs",
        )),
        GraphType::DiGraph(graph) => {
            petgraph_algorithm_layering::cycle::remove_cycle(graph);
            Ok(())
        }
    }
}
