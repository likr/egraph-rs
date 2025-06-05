//! Omega SGD layout algorithm
//!
//! This module provides Python bindings for the Omega SGD layout algorithm,
//! which uses spectral coordinates derived from graph Laplacian eigenvalues.

use crate::{
    drawing::{
        DrawingType, PyDrawing, PyDrawingEuclidean, PyDrawingEuclidean2d, PyDrawingHyperbolic2d,
        PyDrawingSpherical2d, PyDrawingTorus2d,
    },
    graph::{GraphType, PyGraphAdapter},
    layout::sgd::schedulers::{
        PySchedulerConstant, PySchedulerExponential, PySchedulerLinear, PySchedulerQuadratic,
        PySchedulerReciprocal,
    },
};
use petgraph::visit::EdgeRef;
use petgraph_layout_omega::{Omega, OmegaBuilder};
use petgraph_layout_sgd::Sgd;
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
#[pyo3(name = "OmegaBuilder")]
pub struct PyOmegaBuilder {
    builder: OmegaBuilder<f32>,
}

#[pymethods]
impl PyOmegaBuilder {
    /// Creates a new OmegaBuilder with default parameters
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
    /// :return: A new OmegaBuilder instance
    /// :rtype: OmegaBuilder
    #[new]
    fn new() -> Self {
        PyOmegaBuilder {
            builder: OmegaBuilder::new(),
        }
    }

    /// Sets the number of spectral dimensions
    ///
    /// :param d: Number of spectral dimensions to use
    /// :type d: int
    /// :return: Self for method chaining
    /// :rtype: OmegaBuilder
    fn d(mut slf: PyRefMut<Self>, d: usize) -> Py<Self> {
        slf.builder.d(d);
        slf.into()
    }

    /// Sets the number of random pairs per node
    ///
    /// :param k: Number of random pairs per node
    /// :type k: int
    /// :return: Self for method chaining
    /// :rtype: OmegaBuilder
    fn k(mut slf: PyRefMut<Self>, k: usize) -> Py<Self> {
        slf.builder.k(k);
        slf.into()
    }

    /// Sets the minimum distance between node pairs
    ///
    /// :param min_dist: Minimum distance between node pairs
    /// :type min_dist: float
    /// :return: Self for method chaining
    /// :rtype: OmegaBuilder
    fn min_dist(mut slf: PyRefMut<Self>, min_dist: f32) -> Py<Self> {
        slf.builder.min_dist(min_dist);
        slf.into()
    }

    /// Sets maximum iterations for eigenvalue computation
    ///
    /// :param eigenvalue_max_iterations: Maximum iterations for eigenvalue computation
    /// :type eigenvalue_max_iterations: int
    /// :return: Self for method chaining
    /// :rtype: OmegaBuilder
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
    /// :rtype: OmegaBuilder
    fn cg_max_iterations(mut slf: PyRefMut<Self>, cg_max_iterations: usize) -> Py<Self> {
        slf.builder.cg_max_iterations(cg_max_iterations);
        slf.into()
    }

    /// Sets convergence tolerance for eigenvalue computation
    ///
    /// :param eigenvalue_tolerance: Convergence tolerance for eigenvalue computation
    /// :type eigenvalue_tolerance: float
    /// :return: Self for method chaining
    /// :rtype: OmegaBuilder
    fn eigenvalue_tolerance(mut slf: PyRefMut<Self>, eigenvalue_tolerance: f32) -> Py<Self> {
        slf.builder.eigenvalue_tolerance(eigenvalue_tolerance);
        slf.into()
    }

    /// Sets convergence tolerance for conjugate gradient method
    ///
    /// :param cg_tolerance: Convergence tolerance for conjugate gradient method
    /// :type cg_tolerance: float
    /// :return: Self for method chaining
    /// :rtype: OmegaBuilder
    fn cg_tolerance(mut slf: PyRefMut<Self>, cg_tolerance: f32) -> Py<Self> {
        slf.builder.cg_tolerance(cg_tolerance);
        slf.into()
    }

    /// Builds an Omega instance using the configured parameters
    ///
    /// :param graph: The graph to layout
    /// :type graph: Graph or DiGraph
    /// :param f: A Python function that takes an edge index and returns its weight
    /// :type f: callable
    /// :param rng: Random number generator for selecting random node pairs
    /// :type rng: Rng
    /// :return: A new Omega instance
    /// :rtype: Omega
    /// :raises: ValueError if the graph type is not supported
    fn build(
        &self,
        graph: &PyGraphAdapter,
        f: &Bound<PyAny>,
        rng: &mut crate::rng::PyRng,
    ) -> PyOmega {
        PyOmega {
            sgd: match graph.graph() {
                GraphType::Graph(native_graph) => self.builder.build(
                    native_graph,
                    |e| f.call1((e.id().index(),)).unwrap().extract().unwrap(),
                    rng.get_mut(),
                ),
                _ => panic!("unsupported graph type"),
            },
        }
    }
}

/// Python class for Omega stochastic gradient descent (SGD) layout algorithm
///
/// The Omega algorithm uses spectral analysis of the graph Laplacian to create initial
/// coordinates for nodes, then applies SGD optimization using both edge-based and
/// random node pairs. This approach often produces higher-quality layouts than
/// traditional SGD methods.
///
/// The algorithm follows these steps:
/// 1. Compute the smallest d non-zero eigenvalues and eigenvectors of the graph Laplacian
/// 2. Create d-dimensional coordinates by dividing eigenvectors by sqrt of eigenvalues
/// 3. Add edge-based node pairs using Euclidean distances from coordinates
/// 4. Add k random node pairs per node using Euclidean distances (avoiding duplicates)
///
/// :param graph: The graph to layout
/// :type graph: Graph or DiGraph
/// :param f: A Python function that takes an edge index and returns its weight
/// :type f: callable
/// :param rng: Random number generator for selecting random node pairs
/// :type rng: Rng
/// :raises: ValueError if the graph type is not supported
#[pyclass]
#[pyo3(name = "Omega")]
pub struct PyOmega {
    sgd: Omega<f32>,
}

#[pymethods]
impl PyOmega {
    /// Creates a new Omega instance with default parameters
    ///
    /// This uses the default OmegaBuilder configuration:
    /// - d=2, k=30, min_dist=1e-3
    /// - eigenvalue_max_iterations=1000, cg_max_iterations=100
    /// - eigenvalue_tolerance=1e-4, cg_tolerance=1e-4
    ///
    /// :param graph: The graph to layout
    /// :type graph: Graph or DiGraph
    /// :param f: A Python function that takes an edge index and returns its weight
    /// :type f: callable
    /// :param rng: Random number generator for selecting random node pairs
    /// :type rng: Rng
    /// :return: A new Omega instance
    /// :rtype: Omega
    #[new]
    fn new(graph: &PyGraphAdapter, f: &Bound<PyAny>, rng: &mut crate::rng::PyRng) -> PyOmega {
        PyOmegaBuilder::new().build(graph, f, rng)
    }

    /// Shuffles the order of node pairs used in the SGD algorithm
    ///
    /// Randomizing the order of node pairs can help avoid local minima
    /// and improve convergence.
    ///
    /// :param rng: Random number generator for shuffling
    /// :type rng: Rng
    /// :return: None
    /// :rtype: None
    fn shuffle(&mut self, rng: &mut crate::rng::PyRng) {
        self.sgd.shuffle(rng.get_mut())
    }

    /// Applies one iteration of the SGD algorithm to the drawing
    ///
    /// :param drawing: The drawing to modify
    /// :type drawing: Drawing
    /// :param eta: The learning rate for this iteration
    /// :type eta: float
    /// :return: None
    /// :rtype: None
    fn apply(&self, drawing: &Bound<PyDrawing>, eta: f32) {
        let drawing_type = drawing.borrow().drawing_type();
        Python::with_gil(|py| match drawing_type {
            DrawingType::Euclidean2d => {
                let mut drawing = drawing
                    .into_py(py)
                    .downcast_bound::<PyDrawingEuclidean2d>(py)
                    .unwrap()
                    .borrow_mut();
                self.sgd.apply(drawing.drawing_mut(), eta)
            }
            DrawingType::Euclidean => {
                let mut drawing = drawing
                    .into_py(py)
                    .downcast_bound::<PyDrawingEuclidean>(py)
                    .unwrap()
                    .borrow_mut();
                self.sgd.apply(drawing.drawing_mut(), eta)
            }
            DrawingType::Hyperbolic2d => {
                let mut drawing = drawing
                    .into_py(py)
                    .downcast_bound::<PyDrawingHyperbolic2d>(py)
                    .unwrap()
                    .borrow_mut();
                self.sgd.apply(drawing.drawing_mut(), eta)
            }
            DrawingType::Spherical2d => {
                let mut drawing = drawing
                    .into_py(py)
                    .downcast_bound::<PyDrawingSpherical2d>(py)
                    .unwrap()
                    .borrow_mut();
                self.sgd.apply(drawing.drawing_mut(), eta)
            }
            DrawingType::Torus2d => {
                let mut drawing = drawing
                    .into_py(py)
                    .downcast_bound::<PyDrawingTorus2d>(py)
                    .unwrap()
                    .borrow_mut();
                self.sgd.apply(drawing.drawing_mut(), eta)
            }
        })
    }

    /// Creates a default scheduler (exponential) for this SGD algorithm
    ///
    /// :param t_max: The maximum number of iterations
    /// :type t_max: int
    /// :param epsilon: The final learning rate (initial rate is 1.0)
    /// :type epsilon: float
    /// :return: An exponential scheduler
    /// :rtype: SchedulerExponential
    pub fn scheduler(&self, t_max: usize, epsilon: f32) -> PySchedulerExponential {
        self.scheduler_exponential(t_max, epsilon)
    }

    /// Creates a constant scheduler for this SGD algorithm
    ///
    /// :param t_max: The maximum number of iterations
    /// :type t_max: int
    /// :param epsilon: The constant learning rate
    /// :type epsilon: float
    /// :return: A constant scheduler
    /// :rtype: SchedulerConstant
    pub fn scheduler_constant(&self, t_max: usize, epsilon: f32) -> PySchedulerConstant {
        PySchedulerConstant {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    /// Creates a linear scheduler for this SGD algorithm
    ///
    /// :param t_max: The maximum number of iterations
    /// :type t_max: int
    /// :param epsilon: The final learning rate (initial rate is 1.0)
    /// :type epsilon: float
    /// :return: A linear scheduler
    /// :rtype: SchedulerLinear
    pub fn scheduler_linear(&self, t_max: usize, epsilon: f32) -> PySchedulerLinear {
        PySchedulerLinear {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    /// Creates a quadratic scheduler for this SGD algorithm
    ///
    /// :param t_max: The maximum number of iterations
    /// :type t_max: int
    /// :param epsilon: The final learning rate (initial rate is 1.0)
    /// :type epsilon: float
    /// :return: A quadratic scheduler
    /// :rtype: SchedulerQuadratic
    pub fn scheduler_quadratic(&self, t_max: usize, epsilon: f32) -> PySchedulerQuadratic {
        PySchedulerQuadratic {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    /// Creates an exponential scheduler for this SGD algorithm
    ///
    /// :param t_max: The maximum number of iterations
    /// :type t_max: int
    /// :param epsilon: The final learning rate (initial rate is 1.0)
    /// :type epsilon: float
    /// :return: An exponential scheduler
    /// :rtype: SchedulerExponential
    pub fn scheduler_exponential(&self, t_max: usize, epsilon: f32) -> PySchedulerExponential {
        PySchedulerExponential {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    /// Creates a reciprocal scheduler for this SGD algorithm
    ///
    /// :param t_max: The maximum number of iterations
    /// :type t_max: int
    /// :param epsilon: The final learning rate (initial rate is 1.0)
    /// :type epsilon: float
    /// :return: A reciprocal scheduler
    /// :rtype: SchedulerReciprocal
    pub fn scheduler_reciprocal(&self, t_max: usize, epsilon: f32) -> PySchedulerReciprocal {
        PySchedulerReciprocal {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    /// Updates the distance matrix using a Python function
    ///
    /// :param f: A Python function that takes (i, j, distance, weight) and returns a new distance value
    /// :type f: callable
    /// :return: None
    /// :rtype: None
    pub fn update_distance(&mut self, f: &Bound<PyAny>) {
        self.sgd
            .update_distance(|i, j, dij, wij| f.call1((i, j, dij, wij)).unwrap().extract().unwrap())
    }

    /// Updates the weight matrix using a Python function
    ///
    /// :param f: A Python function that takes (i, j, distance, weight) and returns a new weight value
    /// :type f: callable
    /// :return: None
    /// :rtype: None
    pub fn update_weight(&mut self, f: &Bound<PyAny>) {
        self.sgd
            .update_weight(|i, j, dij, wij| f.call1((i, j, dij, wij)).unwrap().extract().unwrap())
    }
}
