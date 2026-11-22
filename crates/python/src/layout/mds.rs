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
    FloatType,
    distance_matrix::{with_full_distance, with_pivot_mds_distance},
    drawing::PyDrawing,
    graph::{GraphType, PyGraphAdapter},
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
    fn new(graph: &PyGraphAdapter, f: &Bound<PyAny>) -> PyResult<PyClassicalMds> {
        use petgraph::visit::EdgeIndexable;
        if !f.is_callable() {
            return Err(pyo3::exceptions::PyTypeError::new_err("f must be callable"));
        }
        match graph.graph() {
            GraphType::Graph(native_graph) => {
                let mut edge_lengths = vec![0.0 as FloatType; native_graph.edge_bound()];
                for e in native_graph.edge_references() {
                    let val: FloatType = f.call1((e.id().index(),))?.extract()?;
                    if !val.is_finite() || val < 0.0 {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "edge length must be finite and non-negative",
                        ));
                    }
                    edge_lengths[e.id().index()] = val;
                }
                Ok(PyClassicalMds {
                    mds: ClassicalMds::new(native_graph, |e| edge_lengths[e.id().index()]),
                })
            }
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
                "unsupported graph type for ClassicalMds",
            )),
        }
    }

    /// Creates a new ClassicalMds instance from a distance matrix
    ///
    /// :param d: A pre-computed matrix of distances between nodes
    /// :type d: DistanceMatrix or KernelDistance or NegLogDistance or NegLogSimDistance
    /// :return: A new ClassicalMds instance
    /// :rtype: ClassicalMds
    #[staticmethod]
    fn new_with_distance_matrix(d: &Bound<PyAny>) -> PyResult<Self> {
        let mds = with_full_distance(d, |distance_matrix| {
            ClassicalMds::try_new_with_distance_matrix(distance_matrix)
                .map_err(pyo3::exceptions::PyValueError::new_err)
        })??;
        Ok(Self { mds })
    }

    /// Runs the Classical MDS algorithm to generate a layout in the specified dimension
    ///
    /// :param d: The number of dimensions for the layout
    /// :type d: int
    /// :return: A drawing with node positions in d-dimensional Euclidean space
    /// :rtype: DrawingEuclidean
    fn run(&self, d: usize) -> Py<PyAny> {
        PyDrawing::new_drawing_euclidean(self.mds.run(d))
    }

    /// Runs the Classical MDS algorithm to generate a 2D layout
    ///
    /// This is a convenience method equivalent to run(2) but returns a 2D drawing.
    ///
    /// :return: A drawing with node positions in 2D Euclidean space
    /// :rtype: DrawingEuclidean2d
    fn run_2d(&self) -> Py<PyAny> {
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
    fn set_eps(&mut self, value: FloatType) -> PyResult<()> {
        if !value.is_finite() || value <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "eps must be finite and positive",
            ));
        }
        self.mds.eps = value;
        Ok(())
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
    fn new(graph: &PyGraphAdapter, f: &Bound<PyAny>, pivot: Vec<usize>) -> PyResult<PyPivotMds> {
        use petgraph::visit::EdgeIndexable;
        if !f.is_callable() {
            return Err(pyo3::exceptions::PyTypeError::new_err("f must be callable"));
        }
        match graph.graph() {
            GraphType::Graph(native_graph) => {
                let mut pivot_indices = Vec::with_capacity(pivot.len());
                for p in pivot {
                    if p > crate::graph::IndexType::MAX as usize
                        || native_graph.node_weight(node_index(p)).is_none()
                    {
                        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                            "pivot index {p} out of bounds for graph"
                        )));
                    }
                    pivot_indices.push(node_index(p));
                }
                let mut edge_lengths = vec![0.0 as FloatType; native_graph.edge_bound()];
                for e in native_graph.edge_references() {
                    let val: FloatType = f.call1((e.id().index(),))?.extract()?;
                    if !val.is_finite() || val < 0.0 {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "edge length must be finite and non-negative",
                        ));
                    }
                    edge_lengths[e.id().index()] = val;
                }
                Ok(PyPivotMds {
                    mds: PivotMds::new(
                        native_graph,
                        |e| edge_lengths[e.id().index()],
                        &pivot_indices,
                    ),
                })
            }
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
                "unsupported graph type for PivotMds",
            )),
        }
    }

    /// Creates a new PivotMds instance from a distance matrix
    ///
    /// :param d: A pre-computed matrix of distances between nodes
    /// :type d: DistanceMatrix or KernelDistance or NegLogDistance or NegLogSimDistance or PivotedNegLogDistance
    /// :return: A new PivotMds instance
    /// :rtype: PivotMds
    #[staticmethod]
    fn new_with_distance_matrix(d: &Bound<PyAny>) -> PyResult<Self> {
        with_pivot_mds_distance(d, |distance_matrix| Self {
            mds: PivotMds::new_with_distance_matrix(distance_matrix),
        })
    }

    /// Runs the Pivot MDS algorithm to generate a layout in the specified dimension
    ///
    /// :param d: The number of dimensions for the layout
    /// :type d: int
    /// :return: A drawing with node positions in d-dimensional Euclidean space
    /// :rtype: DrawingEuclidean
    fn run(&self, d: usize) -> Py<PyAny> {
        PyDrawing::new_drawing_euclidean(self.mds.run(d))
    }

    /// Runs the Pivot MDS algorithm to generate a 2D layout
    ///
    /// This is a convenience method equivalent to run(2) but returns a 2D drawing.
    ///
    /// :return: A drawing with node positions in 2D Euclidean space
    /// :rtype: DrawingEuclidean2d
    fn run_2d(&self) -> Py<PyAny> {
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
    fn set_eps(&mut self, value: FloatType) -> PyResult<()> {
        if !value.is_finite() || value <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "eps must be finite and positive",
            ));
        }
        self.mds.eps = value;
        Ok(())
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyClassicalMds>()?;
    m.add_class::<PyPivotMds>()?;
    Ok(())
}
