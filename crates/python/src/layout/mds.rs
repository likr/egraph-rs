/// Multidimensional Scaling (MDS) layout algorithms for Python
///
/// This module provides Python bindings for Multidimensional Scaling algorithms,
/// which visualize graph structures in lower dimensional spaces based on pairwise
/// distances between nodes. MDS aims to place nodes such that the distances in the
/// layout closely match the graph-theoretical distances (e.g., shortest paths).
///
/// Two MDS variants are implemented:
/// - ClassicalMds: The standard MDS algorithm that computes a full distance matrix
/// - PivotMds: An efficient approximation that uses a subset of nodes as pivots
use crate::{
    distance_matrix::{DistanceMatrixType, PyDistanceMatrix},
    drawing::PyDrawing,
    graph::{GraphType, PyGraphAdapter},
    FloatType,
};
use petgraph::{graph::node_index, stable_graph::NodeIndex, visit::EdgeRef};
use petgraph_layout_mds::{ClassicalMds, PivotMds};
use pyo3::prelude::*;

/// Python class for Classical Multidimensional Scaling
///
/// Classical MDS (also known as Principal Coordinates Analysis) is a dimension
/// reduction technique that projects high-dimensional data into a lower-dimensional
/// space while preserving pairwise distances as much as possible.
///
/// The algorithm works by:
/// 1. Computing a distance matrix between all pairs of nodes (if not provided)
/// 2. Double-centering this matrix to create a dot-product matrix
/// 3. Computing the eigendecomposition of this matrix
/// 4. Using the top eigenvectors to project the data into the desired dimension
///
/// This implementation is suitable for smaller graphs as it requires O(n²) memory
/// for the distance matrix.
///
/// Reference: Cox, T. F., & Cox, M. A. (2000). Multidimensional scaling. Chapman & Hall/CRC.
#[pyclass]
#[pyo3(name = "ClassicalMds")]
struct PyClassicalMds {
    mds: ClassicalMds<NodeIndex, FloatType>,
}

#[pymethods]
impl PyClassicalMds {
    /// Creates a new ClassicalMds instance from a graph
    ///
    /// :param graph: The graph to layout
    /// :type graph: Graph or DiGraph
    /// :param f: A Python function that takes an edge index and returns its weight
    /// :type f: callable
    /// :return: A new ClassicalMds instance
    /// :rtype: ClassicalMds
    #[new]
    fn new(graph: &PyGraphAdapter, f: &Bound<PyAny>) -> PyClassicalMds {
        match graph.graph() {
            GraphType::Graph(native_graph) => PyClassicalMds {
                mds: ClassicalMds::new(native_graph, |e| {
                    f.call1((e.id().index(),)).unwrap().extract().unwrap()
                }),
            },
            _ => panic!("unsupported graph type"),
        }
    }

    /// Creates a new ClassicalMds instance from a distance matrix
    ///
    /// :param d: A pre-computed matrix of distances between nodes
    /// :type d: DistanceMatrix
    /// :return: A new ClassicalMds instance
    /// :rtype: ClassicalMds
    #[staticmethod]
    fn new_with_distance_matrix(d: &PyDistanceMatrix) -> Self {
        match d.distance_matrix() {
            DistanceMatrixType::Full(d) => Self {
                mds: ClassicalMds::new_with_distance_matrix(d),
            },
            _ => panic!("unsupported distance matrix type"),
        }
    }

    /// Runs the Classical MDS algorithm to generate a layout in the specified dimension
    ///
    /// :param d: The number of dimensions for the layout
    /// :type d: int
    /// :return: A drawing with node positions in d-dimensional Euclidean space
    /// :rtype: DrawingEuclidean
    fn run(&self, d: usize) -> PyObject {
        PyDrawing::new_drawing_euclidean(self.mds.run(d))
    }

    /// Runs the Classical MDS algorithm to generate a 2D layout
    ///
    /// This is a convenience method equivalent to run(2) but returns a 2D drawing.
    ///
    /// :return: A drawing with node positions in 2D Euclidean space
    /// :rtype: DrawingEuclidean2d
    fn run_2d(&self) -> PyObject {
        PyDrawing::new_drawing_euclidean_2d(self.mds.run_2d())
    }

    /// Gets the eigenvalue threshold used for numerical stability
    ///
    /// :return: The current epsilon value
    /// :rtype: float
    #[getter]
    fn eps(&self) -> FloatType {
        self.mds.eps
    }

    /// Sets the eigenvalue threshold used for numerical stability
    ///
    /// :param value: The new epsilon value
    /// :type value: float
    /// :return: None
    /// :rtype: None
    #[setter]
    fn set_eps(&mut self, value: FloatType) {
        self.mds.eps = value;
    }
}

/// Python class for Pivot-based Multidimensional Scaling
///
/// Pivot MDS is an efficient approximation of Classical MDS that uses a subset of
/// nodes (called pivots) to reduce the computational complexity. Instead of computing
/// distances between all pairs of nodes, it only computes distances between each node
/// and the pivot nodes.
///
/// The algorithm works by:
/// 1. Selecting a subset of nodes as pivots
/// 2. Computing distances between all nodes and these pivots
/// 3. Double-centering this partial distance matrix
/// 4. Computing eigendecomposition of a smaller matrix
/// 5. Projecting into the desired dimension using these eigenvectors
///
/// This implementation is more suitable for larger graphs as it requires O(n·h) memory
/// where h is the number of pivot nodes, which is typically much smaller than n.
///
/// Reference: Brandes, U., & Pich, C. (2007). Eigensolver methods for progressive
/// multidimensional scaling of large data. Graph Drawing, 42-53.
#[pyclass]
#[pyo3(name = "PivotMds")]
struct PyPivotMds {
    mds: PivotMds<NodeIndex, FloatType>,
}

#[pymethods]
impl PyPivotMds {
    /// Creates a new PivotMds instance from a graph
    ///
    /// :param graph: The graph to layout
    /// :type graph: Graph or DiGraph
    /// :param f: A Python function that takes an edge index and returns its weight
    /// :type f: callable
    /// :param pivot: A list of node indices to use as pivots
    /// :type pivot: list[int]
    /// :return: A new PivotMds instance
    /// :rtype: PivotMds
    #[new]
    fn new(graph: &PyGraphAdapter, f: &Bound<PyAny>, pivot: Vec<usize>) -> PyPivotMds {
        match graph.graph() {
            GraphType::Graph(native_graph) => {
                let pivot = pivot.into_iter().map(node_index).collect::<Vec<_>>();
                PyPivotMds {
                    mds: PivotMds::new(
                        native_graph,
                        |e| f.call1((e.id().index(),)).unwrap().extract().unwrap(),
                        &pivot,
                    ),
                }
            }
            _ => panic!("unsupported graph type"),
        }
    }

    /// Creates a new PivotMds instance from a distance matrix
    ///
    /// :param d: A pre-computed matrix of distances between nodes
    /// :type d: DistanceMatrix
    /// :return: A new PivotMds instance
    /// :rtype: PivotMds
    #[staticmethod]
    fn new_with_distance_matrix(d: &PyDistanceMatrix) -> Self {
        match d.distance_matrix() {
            DistanceMatrixType::Full(d) => Self {
                mds: PivotMds::new_with_distance_matrix(d),
            },
            DistanceMatrixType::Sub(d) => Self {
                mds: PivotMds::new_with_distance_matrix(d),
            },
        }
    }

    /// Runs the Pivot MDS algorithm to generate a layout in the specified dimension
    ///
    /// :param d: The number of dimensions for the layout
    /// :type d: int
    /// :return: A drawing with node positions in d-dimensional Euclidean space
    /// :rtype: DrawingEuclidean
    fn run(&self, d: usize) -> PyObject {
        PyDrawing::new_drawing_euclidean(self.mds.run(d))
    }

    /// Runs the Pivot MDS algorithm to generate a 2D layout
    ///
    /// This is a convenience method equivalent to run(2) but returns a 2D drawing.
    ///
    /// :return: A drawing with node positions in 2D Euclidean space
    /// :rtype: DrawingEuclidean2d
    fn run_2d(&self) -> PyObject {
        PyDrawing::new_drawing_euclidean_2d(self.mds.run_2d())
    }

    /// Gets the eigenvalue threshold used for numerical stability
    ///
    /// :return: The current epsilon value
    /// :rtype: float
    #[getter]
    fn eps(&self) -> FloatType {
        self.mds.eps
    }

    /// Sets the eigenvalue threshold used for numerical stability
    ///
    /// :param value: The new epsilon value
    /// :type value: float
    /// :return: None
    /// :rtype: None
    #[setter]
    fn set_eps(&mut self, value: FloatType) {
        self.mds.eps = value;
    }
}

pub fn register(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyClassicalMds>()?;
    m.add_class::<PyPivotMds>()?;
    Ok(())
}
