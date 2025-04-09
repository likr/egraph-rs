//! Distance-adjusted full SGD layout algorithm
//!
//! This module provides Python bindings for the distance-adjusted full SGD layout algorithm.

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
use petgraph::visit::EdgeRef;
use petgraph_layout_sgd::{DistanceAdjustedSgd, FullSgd, Sgd};
use pyo3::prelude::*;

/// Python class for distance-adjusted full SGD layout algorithm
///
/// This class combines the full SGD algorithm with distance adjustments to
/// create more aesthetically pleasing layouts. It uses all pairs of nodes
/// for layout computation but applies distance-dependent forces adjustments.
///
/// Similar to DistanceAdjustedSparseSgd, this algorithm applies stronger forces
/// between nodes that are already close to each other and weaker forces between
/// distant nodes, but operates on a full distance matrix rather than using
/// sparse approximation.
///
/// :param graph: The graph to layout
/// :type graph: Graph or DiGraph
/// :param f: A Python function that takes an edge index and returns its weight
/// :type f: callable
/// :raises: ValueError if the graph type is not supported
#[pyclass]
#[pyo3(name = "DistanceAdjustedFullSgd")]
pub struct PyDistanceAdjustedFullSgd {
    sgd: DistanceAdjustedSgd<FullSgd<f32>, f32>,
}

#[pymethods]
impl PyDistanceAdjustedFullSgd {
    /// Creates a new distance-adjusted full SGD instance
    ///
    /// :param graph: The graph to layout
    /// :type graph: Graph or DiGraph
    /// :param f: A Python function that takes an edge index and returns its weight
    /// :type f: callable
    /// :return: A new DistanceAdjustedFullSgd instance
    /// :rtype: DistanceAdjustedFullSgd
    #[new]
    fn new(graph: &PyGraphAdapter, f: &Bound<PyAny>) -> Self {
        Self {
            sgd: DistanceAdjustedSgd::new(match graph.graph() {
                GraphType::Graph(native_graph) => FullSgd::new(native_graph, |e| {
                    f.call1((e.id().index(),)).unwrap().extract().unwrap()
                }),
                _ => panic!("unsupported graph type"),
            }),
        }
    }

    /// Creates a new distance-adjusted full SGD instance from a distance matrix
    ///
    /// :param d: A pre-computed matrix of distances between nodes
    /// :type d: DistanceMatrix
    /// :return: A new DistanceAdjustedFullSgd instance
    /// :rtype: DistanceAdjustedFullSgd
    /// :raises: ValueError if the distance matrix type is not supported
    #[staticmethod]
    fn new_with_distance_matrix(d: &PyDistanceMatrix) -> Self {
        match d.distance_matrix() {
            DistanceMatrixType::Full(d) => Self {
                sgd: DistanceAdjustedSgd::new(FullSgd::new_with_distance_matrix(d)),
            },
            _ => panic!("unsupported distance matrix type"),
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

    /// Applies one iteration of the SGD algorithm with distance adjustment
    ///
    /// This method applies distance-dependent forces between nodes, with
    /// stronger forces between nodes that are already close in the layout.
    /// The force between nodes i and j is scaled by (d_ij)^(-alpha) where
    /// d_ij is the current distance between the nodes in the drawing and
    /// alpha is the distance adjustment exponent (default: 1.0).
    ///
    /// :param drawing: The drawing to modify
    /// :type drawing: Drawing
    /// :param eta: The learning rate for this iteration
    /// :type eta: float
    /// :return: None
    /// :rtype: None
    pub fn apply_with_distance_adjustment(&mut self, drawing: &Bound<PyDrawing>, eta: f32) {
        let drawing_type = drawing.borrow().drawing_type();
        Python::with_gil(|py| match drawing_type {
            DrawingType::Euclidean2d => {
                let mut drawing = drawing
                    .into_py(py)
                    .downcast_bound::<PyDrawingEuclidean2d>(py)
                    .unwrap()
                    .borrow_mut();
                self.sgd
                    .apply_with_distance_adjustment(drawing.drawing_mut(), eta)
            }
            DrawingType::Euclidean => {
                let mut drawing = drawing
                    .into_py(py)
                    .downcast_bound::<PyDrawingEuclidean>(py)
                    .unwrap()
                    .borrow_mut();
                self.sgd
                    .apply_with_distance_adjustment(drawing.drawing_mut(), eta)
            }
            DrawingType::Hyperbolic2d => {
                let mut drawing = drawing
                    .into_py(py)
                    .downcast_bound::<PyDrawingHyperbolic2d>(py)
                    .unwrap()
                    .borrow_mut();
                self.sgd
                    .apply_with_distance_adjustment(drawing.drawing_mut(), eta)
            }
            DrawingType::Spherical2d => {
                let mut drawing = drawing
                    .into_py(py)
                    .downcast_bound::<PyDrawingSpherical2d>(py)
                    .unwrap()
                    .borrow_mut();
                self.sgd
                    .apply_with_distance_adjustment(drawing.drawing_mut(), eta)
            }
            DrawingType::Torus2d => {
                let mut drawing = drawing
                    .into_py(py)
                    .downcast_bound::<PyDrawingTorus2d>(py)
                    .unwrap()
                    .borrow_mut();
                self.sgd
                    .apply_with_distance_adjustment(drawing.drawing_mut(), eta)
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

    /// Gets the distance adjustment exponent
    ///
    /// The alpha parameter controls how strongly distance affects the force.
    /// Higher values make the distance adjustment more pronounced.
    /// The force between nodes i and j is scaled by (d_ij)^(-alpha).
    ///
    /// :return: The current alpha value
    /// :rtype: float
    #[getter]
    fn alpha(&self) -> f32 {
        self.sgd.alpha
    }

    /// Sets the distance adjustment exponent
    ///
    /// :param value: The new alpha value
    /// :type value: float
    /// :return: None
    /// :rtype: None
    #[setter]
    fn set_alpha(&mut self, value: f32) {
        self.sgd.alpha = value;
    }

    /// Gets the minimum distance threshold
    ///
    /// This parameter prevents division by zero when nodes are very close.
    ///
    /// :return: The current minimum distance value
    /// :rtype: float
    #[getter]
    fn minimum_distance(&self) -> f32 {
        self.sgd.minimum_distance
    }

    /// Sets the minimum distance threshold
    ///
    /// :param value: The new minimum distance value
    /// :type value: float
    /// :return: None
    /// :rtype: None
    #[setter]
    fn set_minimum_distance(&mut self, value: f32) {
        self.sgd.minimum_distance = value;
    }
}
