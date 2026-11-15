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
        Python::attach(|_py| match drawing_type {
            DrawingType::Euclidean2d => {
                let mut drawing = drawing
                    .clone()
                    .cast::<PyDrawingEuclidean2d>()
                    .unwrap()
                    .borrow_mut();
                self.sgd.apply(drawing.drawing_mut(), eta)
            }
            DrawingType::Euclidean => {
                let mut drawing = drawing
                    .clone()
                    .cast::<PyDrawingEuclidean>()
                    .unwrap()
                    .borrow_mut();
                self.sgd.apply(drawing.drawing_mut(), eta)
            }
            DrawingType::Hyperbolic2d => {
                let mut drawing = drawing
                    .clone()
                    .cast::<PyDrawingHyperbolic2d>()
                    .unwrap()
                    .borrow_mut();
                self.sgd.apply(drawing.drawing_mut(), eta)
            }
            DrawingType::Spherical2d => {
                let mut drawing = drawing
                    .clone()
                    .cast::<PyDrawingSpherical2d>()
                    .unwrap()
                    .borrow_mut();
                self.sgd.apply(drawing.drawing_mut(), eta)
            }
            DrawingType::Torus2d => {
                let mut drawing = drawing
                    .clone()
                    .cast::<PyDrawingTorus2d>()
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

    /// Build an `Sgd` instance from parallel 1D NumPy arrays `(sources, targets, distances, weights)`.
    #[staticmethod]
    pub fn from_numpy_pairs(
        sources: &Bound<'_, PyAny>,
        targets: &Bound<'_, PyAny>,
        distances: numpy::PyReadonlyArray1<'_, crate::FloatType>,
        weights: numpy::PyReadonlyArray1<'_, crate::FloatType>,
    ) -> PyResult<PySgd> {
        let src = crate::graph::extract_index_array_1d(sources, "sources")?;
        let tgt = crate::graph::extract_index_array_1d(targets, "targets")?;
        let d_view = distances.as_array();
        let w_view = weights.as_array();
        let len = src.len();
        if tgt.len() != len || d_view.len() != len || w_view.len() != len {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "array lengths must match: sources={}, targets={}, distances={}, weights={}",
                len,
                tgt.len(),
                d_view.len(),
                w_view.len()
            )));
        }
        let mut pairs = Vec::with_capacity(len);
        for (((&u, &v), &d), &w) in src
            .iter()
            .zip(tgt.iter())
            .zip(d_view.iter())
            .zip(w_view.iter())
        {
            pairs.push((u, v, d, d, w, w));
        }
        Ok(PySgd::new_with_sgd(
            petgraph_layout_sgd::Sgd::from_node_pairs(pairs),
        ))
    }

    /// Update SGD pair weights from an `(N, N)` float64 NumPy matrix.
    pub fn update_weight_from_numpy(
        &mut self,
        w: numpy::PyReadonlyArray2<'_, crate::FloatType>,
    ) -> PyResult<()> {
        let w_view = w.as_array();
        if w_view.nrows() != w_view.ncols() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "weight matrix must be square, got {:?}",
                w_view.dim()
            )));
        }
        self.sgd.update_weight_from_array(w_view);
        Ok(())
    }

    /// Update SGD pair distances (and optionally weights) from `(N, N)` float64 NumPy matrices.
    #[pyo3(signature = (d, w=None))]
    pub fn update_distance_from_numpy(
        &mut self,
        d: numpy::PyReadonlyArray2<'_, crate::FloatType>,
        w: Option<numpy::PyReadonlyArray2<'_, crate::FloatType>>,
    ) -> PyResult<()> {
        let d_view = d.as_array();
        if d_view.nrows() != d_view.ncols() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "distance matrix must be square, got {:?}",
                d_view.dim()
            )));
        }
        let w_view = match &w {
            Some(w_arr) => {
                let view = w_arr.as_array();
                if view.dim() != d_view.dim() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "weight matrix shape {:?} does not match distance matrix shape {:?}",
                        view.dim(),
                        d_view.dim()
                    )));
                }
                Some(view)
            }
            None => None,
        };
        self.sgd.update_distance_from_array(d_view, w_view);
        Ok(())
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
