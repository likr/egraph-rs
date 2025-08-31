use crate::drawing::{
    DrawingType, PyDrawing, PyDrawingEuclidean, PyDrawingEuclidean2d, PyDrawingHyperbolic2d,
    PyDrawingSpherical2d, PyDrawingTorus2d,
};
use crate::layout::sgd::schedulers::{
    PySchedulerConstant, PySchedulerExponential, PySchedulerLinear, PySchedulerQuadratic,
    PySchedulerReciprocal,
};
use crate::FloatType;
use petgraph_layout_sgd::{
    SchedulerConstant, SchedulerExponential, SchedulerLinear, SchedulerQuadratic,
    SchedulerReciprocal, Sgd,
};
use pyo3::prelude::*;

#[pyclass]
#[pyo3(name = "Sgd")]
pub struct PySgd {
    sgd: Sgd<FloatType>,
}

impl PySgd {
    pub fn new_with_sgd(sgd: Sgd<FloatType>) -> Self {
        Self { sgd }
    }
}

#[pymethods]
impl PySgd {
    /// Creates a new SGD instance with the given node pairs and epsilon
    ///
    /// :param node_pairs: List of tuples (i, j, dij, dji, wij, wji) representing node pairs
    /// :type node_pairs: list[tuple[int, int, float, float, float, float]]
    /// :return: A new SGD instance
    /// :rtype: Sgd
    /// :raises ValueError: If node_pairs is malformed or contains invalid values
    #[new]
    fn new(
        node_pairs: Vec<(usize, usize, FloatType, FloatType, FloatType, FloatType)>,
    ) -> PyResult<Self> {
        let sgd = Sgd::new(node_pairs);
        Ok(Self { sgd })
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
    fn apply(&mut self, drawing: &Bound<PyDrawing>, eta: FloatType) {
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

    /// Creates an default scheduler from this SGD instance
    ///
    /// :param t_max: The maximum number of iterations
    /// :type t_max: int
    /// :param epsilon: The minimum learning rate
    /// :type epsilon: float
    /// :return: An exponential scheduler
    /// :rtype: SchedulerExponential
    pub fn scheduler(&self, t_max: usize, epsilon: FloatType) -> PySchedulerExponential {
        self.scheduler_exponential(t_max, epsilon)
    }

    /// Creates a constant scheduler from this SGD instance
    ///
    /// :param t_max: The maximum number of iterations
    /// :type t_max: int
    /// :param epsilon: The minimum learning rate
    /// :type epsilon: float
    /// :return: A constant scheduler
    /// :rtype: SchedulerConstant
    pub fn scheduler_constant(&self, t_max: usize, epsilon: FloatType) -> PySchedulerConstant {
        PySchedulerConstant::new_with_scheduler(
            self.sgd.scheduler::<SchedulerConstant<_>>(t_max, epsilon),
        )
    }

    /// Creates a linear scheduler from this SGD instance
    ///
    /// :param t_max: The maximum number of iterations
    /// :type t_max: int
    /// :param epsilon: The minimum learning rate
    /// :type epsilon: float
    /// :return: A linear scheduler
    /// :rtype: SchedulerLinear
    pub fn scheduler_linear(&self, t_max: usize, epsilon: FloatType) -> PySchedulerLinear {
        PySchedulerLinear::new_with_scheduler(
            self.sgd.scheduler::<SchedulerLinear<_>>(t_max, epsilon),
        )
    }

    /// Creates an exponential scheduler from this SGD instance
    ///
    /// :param t_max: The maximum number of iterations
    /// :type t_max: int
    /// :param epsilon: The minimum learning rate
    /// :type epsilon: float
    /// :return: An exponential scheduler
    /// :rtype: SchedulerExponential
    pub fn scheduler_exponential(
        &self,
        t_max: usize,
        epsilon: FloatType,
    ) -> PySchedulerExponential {
        PySchedulerExponential::new_with_scheduler(
            self.sgd
                .scheduler::<SchedulerExponential<_>>(t_max, epsilon),
        )
    }

    /// Creates a quadratic scheduler from this SGD instance
    ///
    /// :param t_max: The maximum number of iterations
    /// :type t_max: int
    /// :param epsilon: The minimum learning rate
    /// :type epsilon: float
    /// :return: A quadratic scheduler
    /// :rtype: SchedulerQuadratic
    pub fn scheduler_quadratic(&self, t_max: usize, epsilon: FloatType) -> PySchedulerQuadratic {
        PySchedulerQuadratic::new_with_scheduler(
            self.sgd.scheduler::<SchedulerQuadratic<_>>(t_max, epsilon),
        )
    }

    /// Creates a reciprocal scheduler from this SGD instance
    ///
    /// :param t_max: The maximum number of iterations
    /// :type t_max: int
    /// :param epsilon: The minimum learning rate
    /// :type epsilon: float
    /// :return: A reciprocal scheduler
    /// :rtype: SchedulerReciprocal
    pub fn scheduler_reciprocal(&self, t_max: usize, epsilon: FloatType) -> PySchedulerReciprocal {
        PySchedulerReciprocal::new_with_scheduler(
            self.sgd.scheduler::<SchedulerReciprocal<_>>(t_max, epsilon),
        )
    }
}
