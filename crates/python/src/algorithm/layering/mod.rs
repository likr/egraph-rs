/// Python bindings for petgraph-algorithm-layering.
///
/// This module provides Python bindings for the petgraph-algorithm-layering Rust crate,
/// which implements algorithms for assigning layers to nodes in directed graphs.
use pyo3::prelude::*;

mod longest_path;

use crate::graph::{GraphType, PyGraphAdapter};
use longest_path::PyLongestPath;

/// Register the layering module and its classes with Python.
pub fn register(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyLongestPath>()?;

    // Register cycle detection and removal functions
    m.add_function(wrap_pyfunction!(cycle_edges, m)?)?;
    m.add_function(wrap_pyfunction!(remove_cycle, m)?)?;

    Ok(())
}

/// Get edges that form cycles in a directed graph.
///
/// Args:
///     graph: A directed graph.
///
/// Returns:
///     list: A list of edge indices that form cycles.
#[pyfunction]
fn cycle_edges(graph: &PyGraphAdapter) -> PyResult<Vec<(usize, usize)>> {
    match graph.graph() {
        GraphType::Graph(_) => Err(pyo3::exceptions::PyValueError::new_err(
            "cycle_edges only works with directed graphs",
        )),
        GraphType::DiGraph(graph) => {
            let edges = petgraph_algorithm_layering::cycle::cycle_edges(graph);
            Ok(edges
                .into_iter()
                .map(|(u, v)| (u.index(), v.index()))
                .collect())
        }
    }
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
