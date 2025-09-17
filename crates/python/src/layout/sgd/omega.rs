//! Omega SGD layout algorithm
//!
//! This module provides Python bindings for the Omega SGD layout algorithm,
//! which uses spectral coordinates derived from graph Laplacian eigenvalues.

use crate::{
    array::{PyArray1, PyArray2},
    graph::{GraphType, PyGraphAdapter},
    layout::sgd::PySgd,
    FloatType,
};
use petgraph::visit::EdgeRef;
use petgraph_layout_omega::Omega;
use pyo3::prelude::*;

/// Python class for configuring the Omega SGD algorithm
///
/// This builder provides configuration options for the Omega algorithm, including
/// spectral dimensions, random pairs, distance constraints, and eigenvalue solver parameters.
///
/// The Omega algorithm uses spectral analysis of the graph Laplacian to create initial
/// coordinates, then applies SGD optimization using both edge-based and random node pairs.
///
/// :param d: Number of spectral dimensions (default: 2)
/// :type d: int
/// :param k: Number of random pairs per node (default: 30)
/// :type k: int
/// :param min_dist: Minimum distance between node pairs (default: 1e-3)
/// :type min_dist: float
/// :param eigenvalue_max_iterations: Maximum iterations for eigenvalue computation (default: 1000)
/// :type eigenvalue_max_iterations: int
/// :param cg_max_iterations: Maximum iterations for conjugate gradient method (default: 100)
/// :type cg_max_iterations: int
/// :param eigenvalue_tolerance: Convergence tolerance for eigenvalue computation (default: 1e-4)
/// :type eigenvalue_tolerance: float
/// :param cg_tolerance: Convergence tolerance for conjugate gradient method (default: 1e-4)
/// :type cg_tolerance: float
#[pyclass]
#[pyo3(name = "Omega")]
pub struct PyOmega {
    builder: Omega<FloatType>,
}

#[pymethods]
impl PyOmega {
    /// Creates a new Omega with default parameters
    ///
    /// Default values:
    /// - d: 2 (spectral dimensions)
    /// - k: 30 (random pairs per node)
    /// - min_dist: 1e-3 (minimum distance)
    /// - eigenvalue_max_iterations: 1000
    /// - cg_max_iterations: 100
    /// - eigenvalue_tolerance: 1e-4
    /// - cg_tolerance: 1e-4
    ///
    /// :return: A new Omega instance
    /// :rtype: Omega
    #[new]
    fn new() -> Self {
        PyOmega {
            builder: Omega::new(),
        }
    }

    /// Sets the number of spectral dimensions
    ///
    /// :param d: Number of spectral dimensions to use
    /// :type d: int
    /// :return: Self for method chaining
    /// :rtype: Omega
    fn d(mut slf: PyRefMut<Self>, d: usize) -> Py<Self> {
        slf.builder.d(d);
        slf.into()
    }

    /// Sets the number of random pairs per node
    ///
    /// :param k: Number of random pairs per node
    /// :type k: int
    /// :return: Self for method chaining
    /// :rtype: Omega
    fn k(mut slf: PyRefMut<Self>, k: usize) -> Py<Self> {
        slf.builder.k(k);
        slf.into()
    }

    fn shift(mut slf: PyRefMut<Self>, shift: FloatType) -> Py<Self> {
        slf.builder.shift(shift);
        slf.into()
    }

    /// Sets the minimum distance between node pairs
    ///
    /// :param min_dist: Minimum distance between node pairs
    /// :type min_dist: float
    /// :return: Self for method chaining
    /// :rtype: Omega
    fn min_dist(mut slf: PyRefMut<Self>, min_dist: FloatType) -> Py<Self> {
        slf.builder.min_dist(min_dist);
        slf.into()
    }

    /// Sets maximum iterations for eigenvalue computation
    ///
    /// :param eigenvalue_max_iterations: Maximum iterations for eigenvalue computation
    /// :type eigenvalue_max_iterations: int
    /// :return: Self for method chaining
    /// :rtype: Omega
    fn eigenvalue_max_iterations(
        mut slf: PyRefMut<Self>,
        eigenvalue_max_iterations: usize,
    ) -> Py<Self> {
        slf.builder
            .eigenvalue_max_iterations(eigenvalue_max_iterations);
        slf.into()
    }

    /// Sets maximum iterations for conjugate gradient method
    ///
    /// :param cg_max_iterations: Maximum iterations for conjugate gradient method
    /// :type cg_max_iterations: int
    /// :return: Self for method chaining
    /// :rtype: Omega
    fn cg_max_iterations(mut slf: PyRefMut<Self>, cg_max_iterations: usize) -> Py<Self> {
        slf.builder.cg_max_iterations(cg_max_iterations);
        slf.into()
    }

    /// Sets convergence tolerance for eigenvalue computation
    ///
    /// :param eigenvalue_tolerance: Convergence tolerance for eigenvalue computation
    /// :type eigenvalue_tolerance: float
    /// :return: Self for method chaining
    /// :rtype: Omega
    fn eigenvalue_tolerance(mut slf: PyRefMut<Self>, eigenvalue_tolerance: FloatType) -> Py<Self> {
        slf.builder.eigenvalue_tolerance(eigenvalue_tolerance);
        slf.into()
    }

    /// Sets convergence tolerance for conjugate gradient method
    ///
    /// :param cg_tolerance: Convergence tolerance for conjugate gradient method
    /// :type cg_tolerance: float
    /// :return: Self for method chaining
    /// :rtype: Omega
    fn cg_tolerance(mut slf: PyRefMut<Self>, cg_tolerance: FloatType) -> Py<Self> {
        slf.builder.cg_tolerance(cg_tolerance);
        slf.into()
    }

    /// Computes spectral coordinates using the configured parameters
    ///
    /// :param graph: The graph to layout
    /// :type graph: Graph or DiGraph
    /// :param f: A Python function that takes an edge index and returns its weight
    /// :type f: callable
    /// :param rng: Random number generator for spectral coordinate computation
    /// :type rng: Rng
    /// :return: Spectral coordinates as a 2D Array2 where each row is a node's coordinate
    /// :rtype: Array2
    /// :raises: ValueError if the graph type is not supported
    fn embedding(
        &self,
        graph: &PyGraphAdapter,
        f: &Bound<PyAny>,
        rng: &mut crate::rng::PyRng,
    ) -> PyResult<PyArray2> {
        let coordinates = match graph.graph() {
            GraphType::Graph(native_graph) => self.builder.embedding(
                native_graph,
                |e| f.call1((e.id().index(),)).unwrap().extract().unwrap(),
                rng.get_mut(),
            ),
            _ => panic!("unsupported graph type"),
        };

        // Convert to f64 for compatibility with FloatType
        let converted_coords = coordinates.mapv(|v| v as FloatType);
        Ok(PyArray2::new(converted_coords))
    }

    /// Computes spectral coordinates and eigenvalues using the configured parameters
    ///
    /// :param graph: The graph to layout
    /// :type graph: Graph or DiGraph
    /// :param f: A Python function that takes an edge index and returns its weight
    /// :type f: callable
    /// :param rng: Random number generator for spectral coordinate computation
    /// :type rng: Rng
    /// :return: A tuple containing (coordinates, eigenvalues) as Array2 and Array1
    /// :rtype: tuple[Array2, Array1]
    /// :raises: ValueError if the graph type is not supported
    fn eigendecomposition(
        &self,
        graph: &PyGraphAdapter,
        f: &Bound<PyAny>,
        rng: &mut crate::rng::PyRng,
    ) -> PyResult<(PyArray2, PyArray1)> {
        let (coordinates, eigenvalues) = match graph.graph() {
            GraphType::Graph(native_graph) => self.builder.eigendecomposition(
                native_graph,
                |e| f.call1((e.id().index(),)).unwrap().extract().unwrap(),
                rng.get_mut(),
            ),
            _ => panic!("unsupported graph type"),
        };

        // Convert to f64 for compatibility with FloatType
        let converted_coords = coordinates.mapv(|v| v as FloatType);
        let converted_eigenvals = eigenvalues.mapv(|v| v as FloatType);

        Ok((
            PyArray2::new(converted_coords),
            PyArray1::new(converted_eigenvals),
        ))
    }

    /// Builds an Sgd instance using precomputed embedding
    ///
    /// :param graph: The graph to layout
    /// :type graph: Graph or DiGraph
    /// :param embedding: Precomputed spectral coordinates as a 2D Array2
    /// :type embedding: Array2
    /// :param rng: Random number generator for selecting random node pairs
    /// :type rng: Rng
    /// :return: A new Sgd instance
    /// :rtype: Sgd
    /// :raises: ValueError if the graph type is not supported
    fn build_with_embedding(
        &self,
        graph: &PyGraphAdapter,
        embedding: &PyArray2,
        rng: &mut crate::rng::PyRng,
    ) -> PyResult<PySgd> {
        // Get the ndarray directly from PyArray2
        let coordinates = embedding.as_array();

        let sgd = match graph.graph() {
            GraphType::Graph(native_graph) => {
                self.builder
                    .build_with_embedding(native_graph, coordinates, rng.get_mut())
            }
            _ => panic!("unsupported graph type"),
        };

        Ok(PySgd::new_with_sgd(sgd))
    }

    /// Builds an Sgd instance using the configured parameters
    ///
    /// :param graph: The graph to layout
    /// :type graph: Graph or DiGraph
    /// :param f: A Python function that takes an edge index and returns its weight
    /// :type f: callable
    /// :param rng: Random number generator for selecting random node pairs
    /// :type rng: Rng
    /// :return: A new Sgd instance
    /// :rtype: Sgd
    /// :raises: ValueError if the graph type is not supported
    fn build(
        &self,
        graph: &PyGraphAdapter,
        f: &Bound<PyAny>,
        rng: &mut crate::rng::PyRng,
    ) -> PySgd {
        PySgd::new_with_sgd(match graph.graph() {
            GraphType::Graph(native_graph) => self.builder.build(
                native_graph,
                |e| f.call1((e.id().index(),)).unwrap().extract().unwrap(),
                rng.get_mut(),
            ),
            _ => panic!("unsupported graph type"),
        })
    }
}
