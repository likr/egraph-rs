/// Distance matrix implementation for the Python bindings
///
/// This module provides classes and functions for working with distance matrices,
/// which store the distance between pairs of nodes in a graph. These distances
/// are typically computed using shortest path algorithms and can be used by
/// various graph layout algorithms.
///
/// The implementation supports both full distance matrices (containing distances
/// between all pairs of nodes) and sub-distance matrices (containing distances
/// between a subset of node pairs), allowing for efficient memory usage for
/// different use cases.
use crate::graph::{GraphType, IndexType, PyGraphAdapter};
use petgraph::{graph::NodeIndex, stable_graph::node_index};
use petgraph_algorithm_shortest_path::{DistanceMatrix, FullDistanceMatrix, SubDistanceMatrix};
use pyo3::prelude::*;

/// Enum representing different types of distance matrices
///
/// This enum allows the code to work with either a full distance matrix
/// (containing distances between all pairs of nodes) or a sub-distance matrix
/// (containing distances between a subset of node pairs).
///
/// # Variants
///
/// * `Full` - A complete distance matrix containing distances between all pairs of nodes
/// * `Sub` - A partial distance matrix containing distances between a subset of node pairs,
///           typically used in sparse algorithms for improved memory efficiency
pub enum DistanceMatrixType {
    /// Full distance matrix containing distances between all pairs of nodes
    Full(FullDistanceMatrix<NodeIndex<IndexType>, f32>),
    /// Sub-distance matrix containing distances between a subset of node pairs
    Sub(SubDistanceMatrix<NodeIndex<IndexType>, f32>),
}

/// Python class for working with distance matrices
///
/// A distance matrix stores the distance between pairs of nodes in a graph.
/// These distances are typically computed using shortest path algorithms and
/// can be used by various graph layout algorithms.
#[pyclass]
#[pyo3(name = "DistanceMatrix")]
pub struct PyDistanceMatrix {
    distance_matrix: DistanceMatrixType,
}

impl PyDistanceMatrix {
    /// Creates a new distance matrix from a full distance matrix
    ///
    /// # Parameters
    /// * `distance_matrix` - The full distance matrix to wrap
    pub fn new_with_full_distance_matrix(
        distance_matrix: FullDistanceMatrix<NodeIndex<IndexType>, f32>,
    ) -> Self {
        PyDistanceMatrix {
            distance_matrix: DistanceMatrixType::Full(distance_matrix),
        }
    }

    /// Creates a new distance matrix from a sub-distance matrix
    ///
    /// # Parameters
    /// * `distance_matrix` - The sub-distance matrix to wrap
    pub fn new_with_sub_distance_matrix(
        distance_matrix: SubDistanceMatrix<NodeIndex<IndexType>, f32>,
    ) -> Self {
        PyDistanceMatrix {
            distance_matrix: DistanceMatrixType::Sub(distance_matrix),
        }
    }

    /// Returns a reference to the underlying distance matrix
    pub fn distance_matrix(&self) -> &DistanceMatrixType {
        &self.distance_matrix
    }

    /// Returns a mutable reference to the underlying distance matrix
    pub fn distance_matrix_mut(&mut self) -> &mut DistanceMatrixType {
        &mut self.distance_matrix
    }
}

#[pymethods]
impl PyDistanceMatrix {
    /// Creates a new distance matrix from a graph
    ///
    /// This constructor computes shortest path distances between all pairs of nodes
    /// in the given graph and stores them in a full distance matrix.
    ///
    /// :param graph: The graph to compute distances for
    /// :type graph: Graph or DiGraph
    /// :return: A new distance matrix
    /// :rtype: DistanceMatrix
    #[new]
    pub fn new(graph: &PyGraphAdapter) -> PyDistanceMatrix {
        match graph.graph() {
            GraphType::Graph(g) => Self::new_with_full_distance_matrix(FullDistanceMatrix::new(g)),
            GraphType::DiGraph(g) => {
                Self::new_with_full_distance_matrix(FullDistanceMatrix::new(g))
            }
        }
    }

    /// Gets the distance between two nodes
    ///
    /// :param u: The source node index
    /// :type u: int
    /// :param v: The target node index
    /// :type v: int
    /// :return: The distance between the nodes if it exists, None otherwise
    /// :rtype: float or None
    pub fn get(&self, u: usize, v: usize) -> Option<f32> {
        match self.distance_matrix() {
            DistanceMatrixType::Full(distance_matrix) => {
                distance_matrix.get(node_index(u), node_index(v))
            }
            DistanceMatrixType::Sub(distance_matrix) => {
                distance_matrix.get(node_index(u), node_index(v))
            }
        }
    }

    /// Sets the distance between two nodes
    ///
    /// :param u: The source node index
    /// :type u: int
    /// :param v: The target node index
    /// :type v: int
    /// :param d: The new distance value
    /// :type d: float
    /// :return: Some(()) if the distance was set successfully, None otherwise
    /// :rtype: Some(()) or None
    pub fn set(&mut self, u: usize, v: usize, d: f32) -> Option<()> {
        match self.distance_matrix_mut() {
            DistanceMatrixType::Full(distance_matrix) => {
                distance_matrix.set(node_index(u), node_index(v), d)
            }
            DistanceMatrixType::Sub(distance_matrix) => {
                distance_matrix.set(node_index(u), node_index(v), d)
            }
        }
    }
}

/// Registers distance matrix classes with the Python module
pub fn register(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyDistanceMatrix>()?;
    Ok(())
}
