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
use crate::{
    graph::{GraphType, IndexType, PyGraphAdapter},
    FloatType,
};
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
///   typically used in sparse algorithms for improved memory efficiency
pub enum DistanceMatrixType {
    /// Full distance matrix containing distances between all pairs of nodes
    Full(FullDistanceMatrix<NodeIndex<IndexType>, FloatType>),
    /// Sub-distance matrix containing distances between a subset of node pairs
    Sub(SubDistanceMatrix<NodeIndex<IndexType>, FloatType>),
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
        distance_matrix: FullDistanceMatrix<NodeIndex<IndexType>, FloatType>,
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
        distance_matrix: SubDistanceMatrix<NodeIndex<IndexType>, FloatType>,
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
    pub fn get(&self, u: usize, v: usize) -> Option<FloatType> {
        match self.distance_matrix() {
            DistanceMatrixType::Full(distance_matrix) => {
                DistanceMatrix::get(distance_matrix, node_index(u), node_index(v))
            }
            DistanceMatrixType::Sub(distance_matrix) => {
                DistanceMatrix::get(distance_matrix, node_index(u), node_index(v))
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
    pub fn set(&mut self, u: usize, v: usize, d: FloatType) -> Option<()> {
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

use petgraph_distance::{Distance, GaussianKernel, KernelDistance};
use petgraph_linalg_diffusion_kernel::DiffusionDistanceMatrix;
use petgraph_linalg_embedding_distance::EmbeddingDistanceMatrix;

/// Helper enum to store any underlying Rust distance matrix in python bindings
#[derive(Clone)]
pub enum InnerDistanceMatrix {
    Full(FullDistanceMatrix<NodeIndex<IndexType>, FloatType>),
    Sub(SubDistanceMatrix<NodeIndex<IndexType>, FloatType>),
    Diffusion(DiffusionDistanceMatrix<NodeIndex<IndexType>, FloatType>),
    Embedding(EmbeddingDistanceMatrix<NodeIndex<IndexType>, FloatType>),
}

impl Distance<NodeIndex<IndexType>, FloatType> for InnerDistanceMatrix {
    fn get(&self, u: NodeIndex<IndexType>, v: NodeIndex<IndexType>) -> Option<FloatType> {
        match self {
            Self::Full(d) => Distance::get(d, u, v),
            Self::Sub(d) => Distance::get(d, u, v),
            Self::Diffusion(d) => Distance::get(d, u, v),
            Self::Embedding(d) => Distance::get(d, u, v),
        }
    }

    fn get_by_index(&self, i: usize, j: usize) -> FloatType {
        match self {
            Self::Full(d) => Distance::get_by_index(d, i, j),
            Self::Sub(d) => Distance::get_by_index(d, i, j),
            Self::Diffusion(d) => Distance::get_by_index(d, i, j),
            Self::Embedding(d) => Distance::get_by_index(d, i, j),
        }
    }

    fn shape(&self) -> (usize, usize) {
        match self {
            Self::Full(d) => Distance::shape(d),
            Self::Sub(d) => Distance::shape(d),
            Self::Diffusion(d) => Distance::shape(d),
            Self::Embedding(d) => Distance::shape(d),
        }
    }

    fn row_index(&self, u: NodeIndex<IndexType>) -> Option<usize> {
        match self {
            Self::Full(d) => Distance::row_index(d, u),
            Self::Sub(d) => Distance::row_index(d, u),
            Self::Diffusion(d) => Distance::row_index(d, u),
            Self::Embedding(d) => Distance::row_index(d, u),
        }
    }

    fn col_index(&self, u: NodeIndex<IndexType>) -> Option<usize> {
        match self {
            Self::Full(d) => Distance::col_index(d, u),
            Self::Sub(d) => Distance::col_index(d, u),
            Self::Diffusion(d) => Distance::col_index(d, u),
            Self::Embedding(d) => Distance::col_index(d, u),
        }
    }
}

pub fn extract_inner_distance(distance_matrix: &Bound<PyAny>) -> PyResult<InnerDistanceMatrix> {
    if let Ok(dm) = distance_matrix.extract::<PyRef<PyDistanceMatrix>>() {
        match dm.distance_matrix() {
            DistanceMatrixType::Full(d) => Ok(InnerDistanceMatrix::Full(d.clone())),
            DistanceMatrixType::Sub(d) => Ok(InnerDistanceMatrix::Sub(d.clone())),
        }
    } else if let Ok(dm) = distance_matrix.extract::<PyRef<PyDiffusionDistanceMatrix>>() {
        Ok(InnerDistanceMatrix::Diffusion(dm.matrix.clone()))
    } else if let Ok(dm) = distance_matrix.extract::<PyRef<PyEmbeddingDistanceMatrix>>() {
        Ok(InnerDistanceMatrix::Embedding(dm.matrix.clone()))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Unsupported distance matrix type for wrapping",
        ))
    }
}

/// Helper function to perform dynamic dispatch on any distance matrix in python bindings
pub fn with_distance<R>(
    distance_matrix: &Bound<PyAny>,
    f: impl FnOnce(&dyn Distance<NodeIndex<IndexType>, FloatType>) -> R,
) -> PyResult<R> {
    if let Ok(dm) = distance_matrix.extract::<PyRef<PyDistanceMatrix>>() {
        match dm.distance_matrix() {
            DistanceMatrixType::Full(d) => Ok(f(d)),
            DistanceMatrixType::Sub(d) => Ok(f(d)),
        }
    } else if let Ok(dm) = distance_matrix.extract::<PyRef<PyDiffusionDistanceMatrix>>() {
        Ok(f(&dm.matrix))
    } else if let Ok(dm) = distance_matrix.extract::<PyRef<PyEmbeddingDistanceMatrix>>() {
        Ok(f(&dm.matrix))
    } else if let Ok(dm) = distance_matrix.extract::<PyRef<PyKernelDistance>>() {
        Ok(f(&dm.matrix))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Unsupported distance matrix type",
        ))
    }
}

#[pyclass]
#[pyo3(name = "DiffusionDistanceMatrix")]
pub struct PyDiffusionDistanceMatrix {
    pub(crate) matrix: DiffusionDistanceMatrix<NodeIndex<IndexType>, FloatType>,
}

#[pymethods]
impl PyDiffusionDistanceMatrix {
    #[new]
    pub fn new(
        graph: &PyGraphAdapter,
        kernel: &crate::layout::sgd::PyDiffusionKernel,
        min_dist: FloatType,
    ) -> PyResult<Self> {
        let matrix = match graph.graph() {
            GraphType::Graph(native_graph) => {
                DiffusionDistanceMatrix::new(native_graph, kernel.kernel.clone(), min_dist)
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Unsupported graph type",
                ))
            }
        };
        Ok(Self { matrix })
    }

    pub fn get(&self, u: usize, v: usize) -> Option<FloatType> {
        self.matrix.get(node_index(u), node_index(v))
    }
}

#[pyclass]
#[pyo3(name = "EmbeddingDistanceMatrix")]
pub struct PyEmbeddingDistanceMatrix {
    pub(crate) matrix: EmbeddingDistanceMatrix<NodeIndex<IndexType>, FloatType>,
}

#[pymethods]
impl PyEmbeddingDistanceMatrix {
    #[new]
    pub fn new(
        graph: &PyGraphAdapter,
        embedding: &crate::array::PyArray2,
        min_dist: FloatType,
    ) -> PyResult<Self> {
        let matrix = match graph.graph() {
            GraphType::Graph(native_graph) => {
                EmbeddingDistanceMatrix::new(native_graph, embedding.as_array().clone(), min_dist)
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Unsupported graph type",
                ))
            }
        };
        Ok(Self { matrix })
    }

    pub fn get(&self, u: usize, v: usize) -> Option<FloatType> {
        self.matrix.get(node_index(u), node_index(v))
    }
}

#[pyclass]
#[pyo3(name = "KernelDistance")]
pub struct PyKernelDistance {
    pub(crate) matrix: KernelDistance<InnerDistanceMatrix, GaussianKernel<FloatType>>,
}

#[pymethods]
impl PyKernelDistance {
    #[new]
    pub fn new(distance_matrix: &Bound<PyAny>, gamma: FloatType) -> PyResult<Self> {
        let inner = extract_inner_distance(distance_matrix)?;
        let kernel = GaussianKernel::new(gamma);
        let matrix = KernelDistance::new(inner, kernel);
        Ok(Self { matrix })
    }

    pub fn get(&self, u: usize, v: usize) -> Option<FloatType> {
        self.matrix.get(node_index(u), node_index(v))
    }
}

/// Registers distance matrix classes with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDistanceMatrix>()?;
    m.add_class::<PyDiffusionDistanceMatrix>()?;
    m.add_class::<PyEmbeddingDistanceMatrix>()?;
    m.add_class::<PyKernelDistance>()?;
    Ok(())
}
