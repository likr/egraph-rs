//! Sparse SGD layout algorithm
//!
//! This module provides Python bindings for the sparse SGD layout algorithm.

use crate::{
    distance_matrix::{DistanceMatrixType, PyDistanceMatrix},
    drawing::{
        DrawingType, PyDrawing, PyDrawingEuclidean, PyDrawingEuclidean2d, PyDrawingHyperbolic2d,
        PyDrawingSpherical2d, PyDrawingTorus2d,
    },
    graph::{GraphType, PyGraphAdapter},
    layout::sgd::schedulers::{
        PySchedulerConstant, PySchedulerExponential, PySchedulerLinear, PySchedulerQuadratic,
        PySchedulerReciprocal,
    },
    rng::PyRng,
};
use petgraph::visit::{EdgeRef, IntoNodeIdentifiers};
use petgraph_layout_sgd::{Sgd, SparseSgd};
use pyo3::prelude::*;

/// Python class for sparse stochastic gradient descent (SGD) layout algorithm
///
/// This class implements SGD with sparse distance approximation, which is more
/// efficient for large graphs. It uses a subset of "pivot" nodes to approximate
/// distances, reducing computational complexity from O(n²) to O(nh) where h is
/// the number of pivot nodes.
///
/// :param graph: The graph to layout
/// :type graph: Graph or DiGraph
/// :param f: A Python function that takes an edge index and returns its weight
/// :type f: callable
/// :param h: The number of pivot nodes to use
/// :type h: int
/// :param rng: Random number generator for selecting pivot nodes
/// :type rng: Rng
#[pyclass]
#[pyo3(name = "SparseSgd")]
pub struct PySparseSgd {
    sgd: SparseSgd<f32>,
}

#[pymethods]
impl PySparseSgd {
    /// Creates a new sparse SGD instance with randomly selected pivot nodes
    ///
    /// :param graph: The graph to layout
    /// :type graph: Graph or DiGraph
    /// :param f: A Python function that takes an edge index and returns its weight
    /// :type f: callable
    /// :param h: The number of pivot nodes to use
    /// :type h: int
    /// :param rng: Random number generator for selecting pivot nodes
    /// :type rng: Rng
    /// :return: A new SparseSgd instance
    /// :rtype: SparseSgd
    #[new]
    fn new(graph: &PyGraphAdapter, f: &Bound<PyAny>, h: usize, rng: &mut PyRng) -> PySparseSgd {
        PySparseSgd {
            sgd: match graph.graph() {
                GraphType::Graph(native_graph) => SparseSgd::new_with_rng(
                    native_graph,
                    |e| f.call1((e.id().index(),)).unwrap().extract().unwrap(),
                    h,
                    rng.get_mut(),
                ),
                _ => panic!("unsupported graph type"),
            },
        }
    }

    /// Creates a new sparse SGD instance with specified pivot nodes
    ///
    /// :param graph: The graph to layout
    /// :type graph: Graph or DiGraph
    /// :param f: A Python function that takes an edge index and returns its weight
    /// :type f: callable
    /// :param pivot: A list of node indices to use as pivot nodes
    /// :type pivot: list[int]
    /// :return: A new SparseSgd instance
    /// :rtype: SparseSgd
    #[staticmethod]
    pub fn new_with_pivot(graph: &PyGraphAdapter, f: &Bound<PyAny>, pivot: Vec<usize>) -> Self {
        PySparseSgd {
            sgd: match graph.graph() {
                GraphType::Graph(native_graph) => {
                    let nodes = native_graph.node_identifiers().collect::<Vec<_>>();
                    SparseSgd::new_with_pivot(
                        native_graph,
                        |e| f.call1((e.id().index(),)).unwrap().extract().unwrap(),
                        &pivot.iter().map(|&i| nodes[i]).collect::<Vec<_>>(),
                    )
                }
                _ => panic!("unsupported graph type"),
            },
        }
    }

    /// Creates a new sparse SGD instance with specified pivot nodes and distance matrix
    ///
    /// :param graph: The graph to layout
    /// :type graph: Graph or DiGraph
    /// :param f: A Python function that takes an edge index and returns its weight
    /// :type f: callable
    /// :param pivot: A list of node indices to use as pivot nodes
    /// :type pivot: list[int]
    /// :param d: A pre-computed distance matrix
    /// :type d: DistanceMatrix
    /// :return: A new SparseSgd instance
    /// :rtype: SparseSgd
    #[staticmethod]
    pub fn new_with_pivot_and_distance_matrix(
        graph: &PyGraphAdapter,
        f: &Bound<PyAny>,
        pivot: Vec<usize>,
        d: &PyDistanceMatrix,
    ) -> Self {
        PySparseSgd {
            sgd: match graph.graph() {
                GraphType::Graph(native_graph) => {
                    let nodes = native_graph.node_identifiers().collect::<Vec<_>>();
                    match d.distance_matrix() {
                        DistanceMatrixType::Full(d) => {
                            SparseSgd::new_with_pivot_and_distance_matrix(
                                native_graph,
                                |e| f.call1((e.id().index(),)).unwrap().extract().unwrap(),
                                &pivot.iter().map(|&i| nodes[i]).collect::<Vec<_>>(),
                                d,
                            )
                        }
                        DistanceMatrixType::Sub(d) => {
                            SparseSgd::new_with_pivot_and_distance_matrix(
                                native_graph,
                                |e| f.call1((e.id().index(),)).unwrap().extract().unwrap(),
                                &pivot.iter().map(|&i| nodes[i]).collect::<Vec<_>>(),
                                d,
                            )
                        }
                    }
                }
                _ => panic!("unsupported graph type"),
            },
        }
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
    fn shuffle(&mut self, rng: &mut PyRng) {
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

    /// Selects pivot nodes and computes a distance matrix using them
    ///
    /// This static method selects a set of pivot nodes and then computes
    /// a sub-distance matrix containing distances between these pivots
    /// and all other nodes in the graph.
    ///
    /// :param graph: The graph to select pivots from
    /// :type graph: Graph or DiGraph
    /// :param f: A Python function that takes an edge index and returns its weight
    /// :type f: callable
    /// :param h: The number of pivot nodes to select
    /// :type h: int
    /// :param rng: Random number generator for selecting pivots
    /// :type rng: Rng
    /// :return: A tuple containing the pivot node indices and a distance matrix
    /// :rtype: tuple[list[int], DistanceMatrix]
    /// :raises: ValueError if the graph type is not supported
    #[staticmethod]
    pub fn choose_pivot(
        graph: &PyGraphAdapter,
        f: &Bound<PyAny>,
        h: usize,
        rng: &mut PyRng,
    ) -> (Vec<usize>, PyDistanceMatrix) {
        match graph.graph() {
            GraphType::Graph(native_graph) => {
                let (pivot, d) = SparseSgd::choose_pivot(
                    native_graph,
                    |e| f.call1((e.id().index(),)).unwrap().extract().unwrap(),
                    h,
                    rng.get_mut(),
                );
                (
                    pivot.into_iter().map(|u| u.index()).collect::<Vec<_>>(),
                    PyDistanceMatrix::new_with_sub_distance_matrix(d),
                )
            }
            _ => panic!("unsupported graph type"),
        }
    }
}
