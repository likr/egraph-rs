//! Python bindings for DiffusionKernel
//!
//! This module provides Python access to the diffusion kernel matrix exp(-tL),
//! allowing random access to individual matrix elements.

use crate::{
    graph::{GraphType, PyGraphAdapter},
    FloatType,
};
use petgraph::visit::EdgeRef;
use petgraph_linalg_diffusion_kernel::{DiffusionKernel, MultiscaleDiffusionKernel};
use pyo3::prelude::*;

/// Python class for querying diffusion kernel matrix elements
///
/// DiffusionKernel provides random access to elements of the matrix exp(-tL),
/// where L is the graph Laplacian. The kernel is approximated using Chebyshev
/// polynomials and element queries use the Hutchinson trace estimator with
/// symmetry optimization.
///
/// :param graph: The input graph
/// :type graph: Graph or DiGraph
/// :param length: A function that maps edge indices to their lengths/weights
/// :type length: callable
/// :param t: Diffusion time parameter
/// :type t: float
/// :param degree: Degree of Chebyshev polynomial approximation
/// :type degree: int
/// :param num_vectors: Number of Hutchinson random vectors (effective 2x due to symmetry)
/// :type num_vectors: int
/// :param rng: Random number generator
/// :type rng: Rng
///
/// Example:
///     >>> import egraph as eg
///     >>> graph = eg.Graph()
///     >>> # ... add nodes and edges
///     >>> rng = eg.Rng.seed_from(42)
///     >>> dk = eg.DiffusionKernel(graph, lambda i: 1.0, 1000.0, 10, 50, rng)
///     >>> k_ij = dk.get(0, 1)
#[pyclass]
#[pyo3(name = "DiffusionKernel")]
pub struct PyDiffusionKernel {
    pub(crate) kernel: DiffusionKernel<FloatType>,
}

#[pymethods]
impl PyDiffusionKernel {
    /// Creates a new DiffusionKernel with automatic lambda_max estimation
    ///
    /// :param graph: The input graph
    /// :type graph: Graph or DiGraph
    /// :param length: A function that maps edge indices to their lengths/weights
    /// :type length: callable
    /// :param t: Diffusion time parameter
    /// :type t: float
    /// :param degree: Degree of Chebyshev polynomial approximation
    /// :type degree: int
    /// :param num_vectors: Number of Hutchinson random vectors
    /// :type num_vectors: int
    /// :param rng: Random number generator
    /// :type rng: Rng
    /// :return: A new DiffusionKernel instance
    /// :rtype: DiffusionKernel
    #[new]
    fn new(
        graph: &PyGraphAdapter,
        length: Py<PyAny>,
        t: FloatType,
        degree: usize,
        num_vectors: usize,
        rng: &mut crate::rng::PyRng,
    ) -> PyResult<Self> {
        let kernel = match graph.graph() {
            GraphType::Graph(native_graph) => {
                let length_fn = |edge: petgraph::graph::EdgeReference<Py<PyAny>>| -> FloatType {
                    Python::attach(|py| {
                        let result = length.call1(py, (edge.id().index(),));
                        match result {
                            Ok(value) => value.extract::<FloatType>(py).unwrap_or(1.0),
                            Err(_) => 1.0,
                        }
                    })
                };
                DiffusionKernel::new(
                    native_graph,
                    length_fn,
                    t,
                    degree,
                    num_vectors,
                    petgraph_distance::StandardLaplacian,
                    rng.get_mut(),
                )
            }
            _ => panic!("unsupported graph type"),
        };

        Ok(PyDiffusionKernel { kernel })
    }

    /// Creates a new DiffusionKernel with externally provided lambda_max
    ///
    /// :param graph: The input graph
    /// :type graph: Graph or DiGraph
    /// :param length: A function that maps edge indices to their lengths/weights
    /// :type length: callable
    /// :param t: Diffusion time parameter
    /// :type t: float
    /// :param degree: Degree of Chebyshev polynomial approximation
    /// :type degree: int
    /// :param lambda_max: Maximum eigenvalue of the Laplacian
    /// :type lambda_max: float
    /// :param num_vectors: Number of Hutchinson random vectors
    /// :type num_vectors: int
    /// :param rng: Random number generator
    /// :type rng: Rng
    /// :return: A new DiffusionKernel instance
    /// :rtype: DiffusionKernel
    #[staticmethod]
    fn new_with_lambda_max(
        graph: &PyGraphAdapter,
        length: Py<PyAny>,
        t: FloatType,
        degree: usize,
        lambda_max: FloatType,
        num_vectors: usize,
        rng: &mut crate::rng::PyRng,
    ) -> PyResult<Self> {
        let kernel = match graph.graph() {
            GraphType::Graph(native_graph) => {
                let length_fn = |edge: petgraph::graph::EdgeReference<Py<PyAny>>| -> FloatType {
                    Python::attach(|py| {
                        let result = length.call1(py, (edge.id().index(),));
                        match result {
                            Ok(value) => value.extract::<FloatType>(py).unwrap_or(1.0),
                            Err(_) => 1.0,
                        }
                    })
                };
                DiffusionKernel::new_with_lambda_max(
                    native_graph,
                    length_fn,
                    t,
                    degree,
                    lambda_max,
                    num_vectors,
                    petgraph_distance::StandardLaplacian,
                    rng.get_mut(),
                )
            }
            _ => panic!("unsupported graph type"),
        };

        Ok(PyDiffusionKernel { kernel })
    }

    /// Queries the (i, j) element of the diffusion kernel matrix
    ///
    /// Returns an approximation of exp(-tL)[i,j] using the Hutchinson estimator
    /// with symmetry optimization.
    ///
    /// :param i: Row index
    /// :type i: int
    /// :param j: Column index
    /// :type j: int
    /// :return: Approximated kernel value at (i, j)
    /// :rtype: float
    ///
    /// Example:
    ///     >>> k_00 = dk.get(0, 0)  # Diagonal element
    ///     >>> k_01 = dk.get(0, 1)  # Off-diagonal element
    ///     >>> k_10 = dk.get(1, 0)  # Symmetric: k_01 == k_10
    fn get(&self, i: usize, j: usize) -> FloatType {
        self.kernel.get(i, j)
    }

    /// Returns the number of nodes in the graph
    ///
    /// :return: The dimension of the kernel matrix (number of nodes)
    /// :rtype: int
    fn n(&self) -> usize {
        self.kernel.n()
    }
}

/// Python class for computing multiscale diffusion distance between nodes
///
/// MultiscaleDiffusionKernel provides fast distance queries between graph nodes
/// using Hutchinson sampling and batched BiCGSTAB linear solves.
///
/// :param graph: The input graph
/// :type graph: Graph or DiGraph
/// :param length: A function that maps edge indices to their lengths/weights
/// :type length: callable
/// :param alpha: Restart/scaling parameter alpha (e.g. 0.85)
/// :type alpha: float
/// :param num_samples: Number of Hutchinson random samples (e.g. 32)
/// :type num_samples: int
/// :param tol: BiCGSTAB solver convergence tolerance (e.g. 1e-7)
/// :type tol: float
/// :param max_iter: Maximum BiCGSTAB solver iterations (e.g. 100)
/// :type max_iter: int
/// :param rng: Random number generator
/// :type rng: Rng
///
/// Example:
///     >>> import egraph as eg
///     >>> graph = eg.Graph()
///     >>> # ... add nodes and edges
///     >>> rng = eg.Rng.seed_from(42)
///     >>> mdk = eg.MultiscaleDiffusionKernel(graph, lambda i: 1.0, 0.85, 32, 1e-7, 100, rng)
///     >>> dist_01 = mdk.get(0, 1)
#[pyclass]
#[pyo3(name = "MultiscaleDiffusionKernel")]
pub struct PyMultiscaleDiffusionKernel {
    pub(crate) kernel: MultiscaleDiffusionKernel,
}

#[pymethods]
impl PyMultiscaleDiffusionKernel {
    /// Creates a new MultiscaleDiffusionKernel and builds its Hutchinson index
    ///
    /// :param graph: The input graph
    /// :type graph: Graph or DiGraph
    /// :param length: A function that maps edge indices to their lengths/weights
    /// :type length: callable
    /// :param alpha: Restart/scaling parameter alpha
    /// :type alpha: float
    /// :param num_samples: Number of Hutchinson random samples
    /// :type num_samples: int
    /// :param tol: BiCGSTAB solver tolerance
    /// :type tol: float
    /// :param max_iter: Maximum BiCGSTAB solver iterations
    /// :type max_iter: int
    /// :param rng: Random number generator
    /// :type rng: Rng
    /// :return: A new MultiscaleDiffusionKernel instance
    /// :rtype: MultiscaleDiffusionKernel
    #[new]
    fn new(
        graph: &PyGraphAdapter,
        length: Py<PyAny>,
        alpha: FloatType,
        num_samples: usize,
        tol: FloatType,
        max_iter: usize,
        rng: &mut crate::rng::PyRng,
    ) -> PyResult<Self> {
        let mut kernel = match graph.graph() {
            GraphType::Graph(native_graph) => {
                let length_fn = |edge: petgraph::graph::EdgeReference<Py<PyAny>>| -> FloatType {
                    Python::attach(|py| {
                        let result = length.call1(py, (edge.id().index(),));
                        match result {
                            Ok(value) => value.extract::<FloatType>(py).unwrap_or(1.0),
                            Err(_) => 1.0,
                        }
                    })
                };
                MultiscaleDiffusionKernel::from_petgraph(
                    native_graph,
                    length_fn,
                    alpha,
                    num_samples,
                    tol,
                    max_iter,
                )
            }
            _ => panic!("unsupported graph type"),
        };

        kernel.build_index_with_rng(rng.get_mut());

        Ok(PyMultiscaleDiffusionKernel { kernel })
    }

    /// Queries the multiscale diffusion distance between nodes i and j
    ///
    /// :param i: Row index
    /// :type i: int
    /// :param j: Column index
    /// :type j: int
    /// :return: Approximated multiscale diffusion distance between (i, j)
    /// :rtype: float
    fn get(&self, i: usize, j: usize) -> FloatType {
        self.kernel.sample_distance(i, j).unwrap_or(0.0)
    }

    /// Computes multiscale diffusion distance between nodes i and j
    ///
    /// :param i: Row index
    /// :type i: int
    /// :param j: Column index
    /// :type j: int
    /// :return: Approximated multiscale diffusion distance between (i, j)
    /// :rtype: float
    fn sample_distance(&self, i: usize, j: usize) -> PyResult<FloatType> {
        self.kernel
            .sample_distance(i, j)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Re-builds the Hutchinson index using a custom random number generator
    ///
    /// :param rng: Random number generator
    /// :type rng: Rng
    fn build_index(&mut self, rng: &mut crate::rng::PyRng) {
        self.kernel.build_index_with_rng(rng.get_mut());
    }

    /// Returns the number of nodes in the graph
    ///
    /// :return: The dimension of the kernel matrix (number of nodes)
    /// :rtype: int
    fn n(&self) -> usize {
        self.kernel.n()
    }
}
