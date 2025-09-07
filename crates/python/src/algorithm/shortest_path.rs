use crate::{
    distance_matrix::PyDistanceMatrix,
    graph::{GraphType, PyGraphAdapter},
    FloatType,
};
use petgraph::visit::EdgeRef;
use petgraph_algorithm_shortest_path::{
    all_sources_bfs, all_sources_dijkstra, warshall_floyd, WeightedEdgeLength,
};
use pyo3::prelude::*;

/// Computes shortest paths from all nodes to all other nodes using BFS
///
/// This function computes shortest paths in terms of the number of edges
/// (not their weights), and then multiplies each path length by the unit edge length.
/// It's suitable for unweighted graphs or when all edges have the same weight.
///
/// :param graph: The graph to compute paths for
/// :type graph: Graph or DiGraph
/// :param unit_edge_length: The length to assign to each edge
/// :type unit_edge_length: float
/// :return: A distance matrix containing the shortest path distances between all pairs of nodes
/// :rtype: DistanceMatrix
#[pyfunction]
#[pyo3(name = "all_sources_bfs")]
fn py_all_sources_bfs(graph: &PyGraphAdapter, unit_edge_length: FloatType) -> PyDistanceMatrix {
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
/// :param graph: The graph to compute paths for
/// :type graph: Graph or DiGraph
/// :param f: A Python function that takes an edge index and returns its weight
/// :type f: callable
/// :return: A distance matrix containing the shortest path distances between all pairs of nodes
/// :rtype: DistanceMatrix
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
/// :param graph: The graph to compute paths for
/// :type graph: Graph or DiGraph
/// :param f: A Python function that takes an edge index and returns its weight
/// :type f: callable
/// :return: A distance matrix containing the shortest path distances between all pairs of nodes
/// :rtype: DistanceMatrix
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

/// Python wrapper for the WeightedEdgeLength calculator
///
/// This class calculates edge weights based on node degrees and common neighbors.
/// It implements the same interface as the original Python WeightedEdgeLength class.
#[pyclass]
#[pyo3(name = "WeightedEdgeLength")]
pub struct PyWeightedEdgeLength {
    calculator: WeightedEdgeLength,
}

#[pymethods]
impl PyWeightedEdgeLength {
    /// Creates a new WeightedEdgeLength calculator for the given graph.
    ///
    /// :param graph: The graph to analyze
    /// :type graph: Graph or DiGraph
    #[new]
    fn new(graph: &PyGraphAdapter) -> PyResult<Self> {
        match graph.graph() {
            GraphType::Graph(g) => Ok(Self {
                calculator: WeightedEdgeLength::new(g),
            }),
            GraphType::DiGraph(_) => Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "WeightedEdgeLength currently only supports undirected graphs",
            )),
        }
    }

    /// Calculates the weighted length for an edge.
    ///
    /// This method is called when the instance is used as a function.
    /// It matches the interface of the original Python WeightedEdgeLength class.
    ///
    /// :param edge_index: The index of the edge
    /// :type edge_index: int
    /// :return: The calculated edge weight
    /// :rtype: int
    fn __call__(&self, edge_index: usize) -> PyResult<usize> {
        Ok(self.calculator.edge_weight(edge_index))
    }
}

/// Registers shortest path functions with the Python module
pub fn register(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_all_sources_bfs, m)?)?;
    m.add_function(wrap_pyfunction!(py_all_sources_dijkstra, m)?)?;
    m.add_function(wrap_pyfunction!(py_warshall_floyd, m)?)?;
    m.add_class::<PyWeightedEdgeLength>()?;
    Ok(())
}
