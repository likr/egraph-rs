use crate::{
    distance_matrix::PyDistanceMatrix,
    graph::{GraphType, PyGraphAdapter},
};
use petgraph::visit::EdgeRef;
use petgraph_algorithm_shortest_path::{all_sources_bfs, all_sources_dijkstra, warshall_floyd};
use pyo3::prelude::*;

/// Computes shortest paths from all nodes to all other nodes using BFS
///
/// This function computes shortest paths in terms of the number of edges
/// (not their weights), and then multiplies each path length by the unit edge length.
/// It's suitable for unweighted graphs or when all edges have the same weight.
///
/// # Parameters
/// * `graph` - The graph to compute paths for
/// * `unit_edge_length` - The length to assign to each edge
///
/// # Returns
/// A distance matrix containing the shortest path distances between all pairs of nodes
#[pyfunction]
#[pyo3(name = "all_sources_bfs")]
fn py_all_sources_bfs(graph: &PyGraphAdapter, unit_edge_length: f32) -> PyDistanceMatrix {
    let distance_matrix = match graph.graph() {
        GraphType::Graph(g) => all_sources_bfs(g, unit_edge_length),
        GraphType::DiGraph(g) => all_sources_bfs(g, unit_edge_length),
    };
    PyDistanceMatrix::new_with_full_distance_matrix(distance_matrix)
}

/// Computes shortest paths from all nodes to all other nodes using Dijkstra's algorithm
///
/// This function uses Dijkstra's algorithm to compute shortest paths based on edge
/// weights provided by a Python function. It's suitable for weighted graphs where
/// all edge weights are positive.
///
/// # Parameters
/// * `graph` - The graph to compute paths for
/// * `f` - A Python function that takes an edge index and returns its weight
///
/// # Returns
/// A distance matrix containing the shortest path distances between all pairs of nodes
#[pyfunction]
#[pyo3(name = "all_sources_dijkstra")]
fn py_all_sources_dijkstra(graph: &PyGraphAdapter, f: &Bound<PyAny>) -> PyDistanceMatrix {
    let distance_matrix = match graph.graph() {
        GraphType::Graph(g) => all_sources_dijkstra(g, |e| {
            f.call1((e.id().index(),)).unwrap().extract().unwrap()
        }),
        GraphType::DiGraph(g) => all_sources_dijkstra(g, |e| {
            f.call1((e.id().index(),)).unwrap().extract().unwrap()
        }),
    };
    PyDistanceMatrix::new_with_full_distance_matrix(distance_matrix)
}

/// Computes shortest paths from all nodes to all other nodes using the Warshall-Floyd algorithm
///
/// This function uses the Warshall-Floyd (aka Floyd-Warshall) algorithm to compute
/// shortest paths based on edge weights provided by a Python function. It can handle
/// negative edge weights, unlike Dijkstra's algorithm.
///
/// # Parameters
/// * `graph` - The graph to compute paths for
/// * `f` - A Python function that takes an edge index and returns its weight
///
/// # Returns
/// A distance matrix containing the shortest path distances between all pairs of nodes
#[pyfunction]
#[pyo3(name = "warshall_floyd")]
fn py_warshall_floyd(graph: &PyGraphAdapter, f: &Bound<PyAny>) -> PyDistanceMatrix {
    let distance_matrix = match graph.graph() {
        GraphType::Graph(g) => warshall_floyd(g, |e| {
            f.call1((e.id().index(),)).unwrap().extract().unwrap()
        }),
        GraphType::DiGraph(g) => warshall_floyd(g, |e| {
            f.call1((e.id().index(),)).unwrap().extract().unwrap()
        }),
    };
    PyDistanceMatrix::new_with_full_distance_matrix(distance_matrix)
}

/// Registers shortest path functions with the Python module
pub fn register(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_all_sources_bfs, m)?)?;
    m.add_function(wrap_pyfunction!(py_all_sources_dijkstra, m)?)?;
    m.add_function(wrap_pyfunction!(py_warshall_floyd, m)?)?;
    Ok(())
}
