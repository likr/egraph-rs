//! RdMds (Resistance-distance MDS) layout algorithm
//!
//! This module provides Python bindings for the RdMds algorithm,
//! which computes spectral embeddings from graph Laplacian eigenvalues.

use crate::layout::rdmds_solvers::PySolverEnum;
use crate::{
    array::{PyArray1, PyArray2},
    graph::{GraphType, PyGraphAdapter},
    FloatType,
};
use petgraph::visit::EdgeRef;
use petgraph_linalg_rdmds::RdMds;
use pyo3::prelude::*;

/// Result of an eigendecomposition computation
#[pyclass(get_all)]
#[pyo3(name = "EigendecompositionResult")]
pub struct PyEigendecompositionResult {
    pub eigenvectors: PyArray2,
    pub eigenvalues: PyArray1,
    pub cg_iterations: Vec<usize>,
    pub power_iterations: Vec<usize>,
}

/// Python class for computing spectral embeddings using RdMds
///
/// RdMds (Resistance-distance Multidimensional Scaling) computes d-dimensional
/// spectral coordinates by finding the smallest non-zero eigenvalues and eigenvectors
/// of the graph Laplacian matrix. These spectral coordinates can be used as initial
/// embeddings for graph layout algorithms.
///
/// :param d: Number of spectral dimensions (default: 2)
/// :type d: int
/// :param shift: Shift parameter for positive definite matrix L + cI (default: 1e-3)
/// :type shift: float
/// :param eigenvalue_max_iterations: Maximum iterations for eigenvalue computation (default: 1000)
/// :type eigenvalue_max_iterations: int
/// :param cg_max_iterations: Maximum iterations for conjugate gradient method (default: 100)
/// :type cg_max_iterations: int
/// :param eigenvalue_tolerance: Convergence tolerance for eigenvalue computation (default: 1e-1)
/// :type eigenvalue_tolerance: float
/// :param cg_tolerance: Convergence tolerance for conjugate gradient method (default: 1e-4)
/// :type cg_tolerance: float
#[pyclass]
#[pyo3(name = "RdMds")]
pub struct PyRdMds {
    rdmds: RdMds<FloatType>,
}

#[pymethods]
impl PyRdMds {
    /// Creates a new RdMds with default parameters
    ///
    /// Default values:
    /// - d: 2 (spectral dimensions)
    /// - shift: 1e-3 (shift parameter)
    /// - eigenvalue_max_iterations: 1000
    /// - cg_max_iterations: 100
    /// - eigenvalue_tolerance: 1e-1
    /// - cg_tolerance: 1e-4
    ///
    /// :return: A new RdMds instance
    /// :rtype: RdMds
    #[new]
    fn new() -> Self {
        PyRdMds {
            rdmds: RdMds::new(),
        }
    }

    /// Sets the number of spectral dimensions
    ///
    /// :param d: Number of spectral dimensions to use
    /// :type d: int
    /// :return: Self for method chaining
    /// :rtype: RdMds
    fn d(mut slf: PyRefMut<Self>, d: usize) -> Py<Self> {
        slf.rdmds.d(d);
        slf.into()
    }

    /// Sets the shift parameter for creating positive definite matrix L + cI
    ///
    /// :param shift: Shift parameter value
    /// :type shift: float
    /// :return: Self for method chaining
    /// :rtype: RdMds
    fn shift(mut slf: PyRefMut<Self>, shift: FloatType) -> Py<Self> {
        slf.rdmds.shift(shift);
        slf.into()
    }

    /// Sets maximum iterations for eigenvalue computation
    ///
    /// :param eigenvalue_max_iterations: Maximum iterations for eigenvalue computation
    /// :type eigenvalue_max_iterations: int
    /// :return: Self for method chaining
    /// :rtype: RdMds
    fn eigenvalue_max_iterations(
        mut slf: PyRefMut<Self>,
        eigenvalue_max_iterations: usize,
    ) -> Py<Self> {
        slf.rdmds
            .eigenvalue_max_iterations(eigenvalue_max_iterations);
        slf.into()
    }

    /// Sets convergence tolerance for eigenvalue computation
    ///
    /// :param eigenvalue_tolerance: Convergence tolerance for eigenvalue computation
    /// :type eigenvalue_tolerance: float
    /// :return: Self for method chaining
    /// :rtype: RdMds
    fn eigenvalue_tolerance(mut slf: PyRefMut<Self>, eigenvalue_tolerance: FloatType) -> Py<Self> {
        slf.rdmds.eigenvalue_tolerance(eigenvalue_tolerance);
        slf.into()
    }

    /// Computes spectral coordinates (embedding) using the configured parameters
    ///
    /// :param graph: The graph to compute embedding for
    /// :type graph: Graph or DiGraph
    /// :param length: A Python function that takes an edge index and returns its weight
    /// :type length: callable
    /// :param rng: Random number generator for spectral coordinate computation
    /// :type rng: Rng
    /// :return: Spectral coordinates as a 2D Array2 where each row is a node's coordinate
    /// :rtype: Array2
    /// :raises: ValueError if the graph type is not supported
    fn embedding(
        &self,
        graph: &PyGraphAdapter,
        length: &Bound<PyAny>,
        solver: PySolverEnum,
        rng: &mut crate::rng::PyRng,
    ) -> PyResult<PyArray2> {
        let coordinates = match graph.graph() {
            GraphType::Graph(native_graph) => self.rdmds.embedding(
                native_graph,
                |e| length.call1((e.id().index(),)).unwrap().extract().unwrap(),
                &solver,
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
    /// :param graph: The graph to compute embedding for
    /// :type graph: Graph or DiGraph
    /// :param length: A Python function that takes an edge index and returns its weight
    /// :type length: callable
    /// :param rng: Random number generator for spectral coordinate computation
    /// :type rng: Rng
    /// :return: The result of eigendecomposition
    /// :rtype: EigendecompositionResult
    /// :raises: ValueError if the graph type is not supported
    fn embedding_symmetric_normalized(
        &self,
        graph: &PyGraphAdapter,
        length: &Bound<PyAny>,
        solver: PySolverEnum,
        rng: &mut crate::rng::PyRng,
    ) -> PyResult<PyArray2> {
        let coordinates = match graph.graph() {
            GraphType::Graph(native_graph) => self.rdmds.embedding_symmetric_normalized(
                native_graph,
                |e| length.call1((e.id().index(),)).unwrap().extract().unwrap(),
                &solver,
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
    /// :param graph: The graph to compute embedding for
    /// :type graph: Graph or DiGraph
    /// :param length: A Python function that takes an edge index and returns its weight
    /// :type length: callable
    /// :param rng: Random number generator for spectral coordinate computation
    /// :type rng: Rng
    /// :return: The result of eigendecomposition
    /// :rtype: EigendecompositionResult
    /// :raises: ValueError if the graph type is not supported
    fn embedding_random_walk_normalized(
        &self,
        graph: &PyGraphAdapter,
        length: &Bound<PyAny>,
        solver: PySolverEnum,
        rng: &mut crate::rng::PyRng,
    ) -> PyResult<PyArray2> {
        let coordinates = match graph.graph() {
            GraphType::Graph(native_graph) => self.rdmds.embedding_random_walk_normalized(
                native_graph,
                |e| length.call1((e.id().index(),)).unwrap().extract().unwrap(),
                &solver,
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
    /// :param graph: The graph to compute embedding for
    /// :type graph: Graph or DiGraph
    /// :param length: A Python function that takes an edge index and returns its weight
    /// :type length: callable
    /// :param rng: Random number generator for spectral coordinate computation
    /// :type rng: Rng
    /// :return: The result of eigendecomposition
    /// :rtype: EigendecompositionResult
    /// :raises: ValueError if the graph type is not supported
    fn eigendecomposition(
        &self,
        graph: &PyGraphAdapter,
        length: &Bound<PyAny>,
        solver: PySolverEnum,
        rng: &mut crate::rng::PyRng,
    ) -> PyResult<PyEigendecompositionResult> {
        let result = match graph.graph() {
            GraphType::Graph(native_graph) => self.rdmds.eigendecomposition(
                native_graph,
                |e| length.call1((e.id().index(),)).unwrap().extract().unwrap(),
                &solver,
                rng.get_mut(),
            ),
            _ => panic!("unsupported graph type"),
        };
        let coordinates = result.eigenvectors;
        let eigenvalues = result.eigenvalues;

        // Convert to f64 for compatibility with FloatType
        let converted_coords = coordinates.mapv(|v| v as FloatType);
        let converted_eigenvals = eigenvalues.mapv(|v| v as FloatType);

        Ok(PyEigendecompositionResult {
            eigenvectors: PyArray2::new(converted_coords),
            eigenvalues: PyArray1::new(converted_eigenvals),
            cg_iterations: result.cg_iterations,
            power_iterations: result.power_iterations,
        })
    }
    fn eigendecomposition_symmetric_normalized(
        &self,
        graph: &PyGraphAdapter,
        length: &Bound<PyAny>,
        solver: PySolverEnum,
        rng: &mut crate::rng::PyRng,
    ) -> PyResult<PyEigendecompositionResult> {
        let result = match graph.graph() {
            GraphType::Graph(native_graph) => self.rdmds.eigendecomposition_symmetric_normalized(
                native_graph,
                |e| length.call1((e.id().index(),)).unwrap().extract().unwrap(),
                &solver,
                rng.get_mut(),
            ),
            _ => panic!("unsupported graph type"),
        };
        let coordinates = result.eigenvectors;
        let eigenvalues = result.eigenvalues;

        // Convert to f64 for compatibility with FloatType
        let converted_coords = coordinates.mapv(|v| v as FloatType);
        let converted_eigenvals = eigenvalues.mapv(|v| v as FloatType);

        Ok(PyEigendecompositionResult {
            eigenvectors: PyArray2::new(converted_coords),
            eigenvalues: PyArray1::new(converted_eigenvals),
            cg_iterations: result.cg_iterations,
            power_iterations: result.power_iterations,
        })
    }
    fn eigendecomposition_random_walk_normalized(
        &self,
        graph: &PyGraphAdapter,
        length: &Bound<PyAny>,
        solver: PySolverEnum,
        rng: &mut crate::rng::PyRng,
    ) -> PyResult<PyEigendecompositionResult> {
        let result = match graph.graph() {
            GraphType::Graph(native_graph) => self.rdmds.eigendecomposition_random_walk_normalized(
                native_graph,
                |e| length.call1((e.id().index(),)).unwrap().extract().unwrap(),
                &solver,
                rng.get_mut(),
            ),
            _ => panic!("unsupported graph type"),
        };
        let coordinates = result.eigenvectors;
        let eigenvalues = result.eigenvalues;

        // Convert to f64 for compatibility with FloatType
        let converted_coords = coordinates.mapv(|v| v as FloatType);
        let converted_eigenvals = eigenvalues.mapv(|v| v as FloatType);

        Ok(PyEigendecompositionResult {
            eigenvectors: PyArray2::new(converted_coords),
            eigenvalues: PyArray1::new(converted_eigenvals),
            cg_iterations: result.cg_iterations,
            power_iterations: result.power_iterations,
        })
    }
}

/// Register RdMds class with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEigendecompositionResult>()?;
    m.add_class::<PyRdMds>()?;
    Ok(())
}
