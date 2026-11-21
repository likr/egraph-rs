use crate::drawing::{
    DrawingType, PyDrawing, PyDrawingEuclidean, PyDrawingEuclidean2d, PyDrawingHyperbolic2d,
    PyDrawingSpherical2d, PyDrawingTorus2d,
};
use crate::layout::knn::{extract_f64_array_1d, extract_index_array_1d_with_isize};
use crate::layout::sgd::schedulers::{
    PySchedulerConstant, PySchedulerExponential, PySchedulerLinear, PySchedulerQuadratic,
    PySchedulerReciprocal,
};
use crate::FloatType;
use petgraph_drawing::Drawing;
use petgraph_layout_sgd::{
    SchedulerConstant, SchedulerExponential, SchedulerLinear, SchedulerQuadratic,
    SchedulerReciprocal, Sgd,
};
use pyo3::prelude::*;

type SgdPairTuple = (usize, usize, FloatType, FloatType, FloatType, FloatType);

fn extract_validated_pairs_1d(
    sources: &Bound<'_, PyAny>,
    targets: &Bound<'_, PyAny>,
    distances: &Bound<'_, PyAny>,
    weights: &Bound<'_, PyAny>,
    n_nodes: Option<usize>,
    prefix: &str,
) -> PyResult<Vec<SgdPairTuple>> {
    let src_name = if prefix.is_empty() {
        "sources".to_string()
    } else {
        format!("{prefix}_sources")
    };
    let tgt_name = if prefix.is_empty() {
        "targets".to_string()
    } else {
        format!("{prefix}_targets")
    };
    let dist_name = if prefix.is_empty() {
        "distances".to_string()
    } else {
        format!("{prefix}_distances")
    };
    let wt_name = if prefix.is_empty() {
        "weights".to_string()
    } else {
        format!("{prefix}_weights")
    };

    let src = extract_index_array_1d_with_isize(sources, &src_name)?;
    let tgt = extract_index_array_1d_with_isize(targets, &tgt_name)?;
    let d_arr = extract_f64_array_1d(distances, &dist_name)?;
    let w_arr = extract_f64_array_1d(weights, &wt_name)?;
    let d_view = d_arr.as_array();
    let w_view = w_arr.as_array();

    let len = src.len();
    if tgt.len() != len || d_view.len() != len || w_view.len() != len {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "array lengths must match: {src_name}={}, {tgt_name}={}, {dist_name}={}, {wt_name}={}",
            len,
            tgt.len(),
            d_view.len(),
            w_view.len()
        )));
    }

    let mut pairs = Vec::with_capacity(len);
    for (idx, (((&u, &v), &d), &w)) in src
        .iter()
        .zip(tgt.iter())
        .zip(d_view.iter())
        .zip(w_view.iter())
        .enumerate()
    {
        if let Some(n) = n_nodes {
            if u >= n || v >= n {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "pair index ({u}, {v}) at position {idx} is out of bounds for n_nodes={n}"
                )));
            }
        }
        if u == v {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "self-loop pair ({u}, {u}) at position {idx} is not allowed"
            )));
        }
        if !d.is_finite() || d <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{dist_name}[{idx}] must be finite and > 0, got {d}"
            )));
        }
        if !w.is_finite() || w <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{wt_name}[{idx}] must be finite and > 0, got {w}"
            )));
        }
        pairs.push((u, v, d, d, w, w));
    }
    Ok(pairs)
}

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
    fn apply(&mut self, drawing: &Bound<PyDrawing>, eta: FloatType) -> PyResult<()> {
        let drawing_type = drawing.borrow().drawing_type();
        Python::attach(|_py| match drawing_type {
            DrawingType::Euclidean2d => {
                let mut drawing = drawing
                    .clone()
                    .cast::<PyDrawingEuclidean2d>()
                    .map_err(|_| {
                        pyo3::exceptions::PyTypeError::new_err("invalid DrawingEuclidean2d")
                    })?
                    .borrow_mut();
                self.sgd
                    .try_validate_bounds(drawing.drawing().len())
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
                self.sgd.apply(drawing.drawing_mut(), eta);
                Ok(())
            }
            DrawingType::Euclidean => {
                let mut drawing = drawing
                    .clone()
                    .cast::<PyDrawingEuclidean>()
                    .map_err(|_| {
                        pyo3::exceptions::PyTypeError::new_err("invalid DrawingEuclidean")
                    })?
                    .borrow_mut();
                self.sgd
                    .try_validate_bounds(drawing.drawing().len())
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
                self.sgd.apply(drawing.drawing_mut(), eta);
                Ok(())
            }
            DrawingType::Hyperbolic2d => {
                let mut drawing = drawing
                    .clone()
                    .cast::<PyDrawingHyperbolic2d>()
                    .map_err(|_| {
                        pyo3::exceptions::PyTypeError::new_err("invalid DrawingHyperbolic2d")
                    })?
                    .borrow_mut();
                self.sgd
                    .try_validate_bounds(drawing.drawing().len())
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
                self.sgd.apply(drawing.drawing_mut(), eta);
                Ok(())
            }
            DrawingType::Spherical2d => {
                let mut drawing = drawing
                    .clone()
                    .cast::<PyDrawingSpherical2d>()
                    .map_err(|_| {
                        pyo3::exceptions::PyTypeError::new_err("invalid DrawingSpherical2d")
                    })?
                    .borrow_mut();
                self.sgd
                    .try_validate_bounds(drawing.drawing().len())
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
                self.sgd.apply(drawing.drawing_mut(), eta);
                Ok(())
            }
            DrawingType::Torus2d => {
                let mut drawing = drawing
                    .clone()
                    .cast::<PyDrawingTorus2d>()
                    .map_err(|_| pyo3::exceptions::PyTypeError::new_err("invalid DrawingTorus2d"))?
                    .borrow_mut();
                self.sgd
                    .try_validate_bounds(drawing.drawing().len())
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
                self.sgd.apply(drawing.drawing_mut(), eta);
                Ok(())
            }
        })
    }

    /// Updates the distance matrix using a Python function
    ///
    /// :param f: A Python function that takes (i, j, distance, weight) and returns a new distance value
    /// :type f: callable
    /// :return: None
    /// :rtype: None
    pub fn update_distance(&mut self, f: &Bound<PyAny>) -> PyResult<()> {
        if !f.is_callable() {
            return Err(pyo3::exceptions::PyTypeError::new_err("f must be callable"));
        }
        self.sgd
            .try_update_distance(|i, j, dij, wij| f.call1((i, j, dij, wij))?.extract::<FloatType>())
    }

    /// Updates the weight matrix using a Python function
    ///
    /// :param f: A Python function that takes (i, j, distance, weight) and returns a new weight value
    /// :type f: callable
    /// :return: None
    /// :rtype: None
    pub fn update_weight(&mut self, f: &Bound<PyAny>) -> PyResult<()> {
        if !f.is_callable() {
            return Err(pyo3::exceptions::PyTypeError::new_err("f must be callable"));
        }
        self.sgd
            .try_update_weight(|i, j, dij, wij| f.call1((i, j, dij, wij))?.extract::<FloatType>())
    }

    /// Build an `Sgd` instance from parallel 1D NumPy arrays `(sources, targets, distances, weights)`.
    #[staticmethod]
    #[pyo3(signature = (sources, targets, distances, weights, n_nodes=None))]
    pub fn from_numpy_pairs(
        sources: &Bound<'_, PyAny>,
        targets: &Bound<'_, PyAny>,
        distances: &Bound<'_, PyAny>,
        weights: &Bound<'_, PyAny>,
        n_nodes: Option<usize>,
    ) -> PyResult<PySgd> {
        let pairs = extract_validated_pairs_1d(sources, targets, distances, weights, n_nodes, "")?;
        Ok(PySgd::new_with_sgd(
            petgraph_layout_sgd::Sgd::from_node_pairs(pairs),
        ))
    }

    /// Build an `Sgd` instance containing only one-sided hinge repulsion pairs from NumPy arrays.
    ///
    /// Hinge pairs exert repulsion strictly when `0 < ||x_i - x_j|| < distance` and zero
    /// attractive force when `||x_i - x_j|| >= distance`.
    ///
    /// :param sources: 1D integer array of source node indices
    /// :type sources: numpy.ndarray
    /// :param targets: 1D integer array of target node indices
    /// :type targets: numpy.ndarray
    /// :param distances: 1D float64 array of target barrier distances (`> 0`, finite)
    /// :type distances: numpy.ndarray
    /// :param weights: 1D float64 array of pair weights (`> 0`, finite)
    /// :type weights: numpy.ndarray
    /// :param n_nodes: Optional node count for bounds checking
    /// :type n_nodes: int or None
    /// :return: A new `Sgd` instance with one-sided hinge pairs
    /// :rtype: Sgd
    #[staticmethod]
    #[pyo3(signature = (sources, targets, distances, weights, n_nodes=None))]
    pub fn from_numpy_hinge_pairs(
        sources: &Bound<'_, PyAny>,
        targets: &Bound<'_, PyAny>,
        distances: &Bound<'_, PyAny>,
        weights: &Bound<'_, PyAny>,
        n_nodes: Option<usize>,
    ) -> PyResult<PySgd> {
        let hinges = extract_validated_pairs_1d(sources, targets, distances, weights, n_nodes, "")?;
        Ok(PySgd::new_with_sgd(Sgd::new_with_hinge_pairs(
            Vec::new(),
            hinges,
        )))
    }

    /// Build an `Sgd` instance combining two-sided spring pairs and one-sided hinge pairs.
    ///
    /// :param sources: 1D integer array of spring source node indices
    /// :type sources: numpy.ndarray
    /// :param targets: 1D integer array of spring target node indices
    /// :type targets: numpy.ndarray
    /// :param distances: 1D float64 array of spring target distances (`> 0`, finite)
    /// :type distances: numpy.ndarray
    /// :param weights: 1D float64 array of spring weights (`> 0`, finite)
    /// :type weights: numpy.ndarray
    /// :param hinge_sources: 1D integer array of hinge source node indices
    /// :type hinge_sources: numpy.ndarray
    /// :param hinge_targets: 1D integer array of hinge target node indices
    /// :type hinge_targets: numpy.ndarray
    /// :param hinge_distances: 1D float64 array of hinge barrier distances (`> 0`, finite)
    /// :type hinge_distances: numpy.ndarray
    /// :param hinge_weights: 1D float64 array of hinge weights (`> 0`, finite)
    /// :type hinge_weights: numpy.ndarray
    /// :param n_nodes: Optional node count for bounds checking
    /// :type n_nodes: int or None
    /// :return: A new `Sgd` instance with spring and hinge pairs
    /// :rtype: Sgd
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        sources,
        targets,
        distances,
        weights,
        hinge_sources,
        hinge_targets,
        hinge_distances,
        hinge_weights,
        n_nodes=None
    ))]
    pub fn from_numpy_pairs_with_hinges(
        sources: &Bound<'_, PyAny>,
        targets: &Bound<'_, PyAny>,
        distances: &Bound<'_, PyAny>,
        weights: &Bound<'_, PyAny>,
        hinge_sources: &Bound<'_, PyAny>,
        hinge_targets: &Bound<'_, PyAny>,
        hinge_distances: &Bound<'_, PyAny>,
        hinge_weights: &Bound<'_, PyAny>,
        n_nodes: Option<usize>,
    ) -> PyResult<PySgd> {
        let springs =
            extract_validated_pairs_1d(sources, targets, distances, weights, n_nodes, "")?;
        let hinges = extract_validated_pairs_1d(
            hinge_sources,
            hinge_targets,
            hinge_distances,
            hinge_weights,
            n_nodes,
            "hinge",
        )?;
        Ok(PySgd::new_with_sgd(Sgd::new_with_hinge_pairs(
            springs, hinges,
        )))
    }

    /// Append validated one-sided hinge repulsion pairs from 1D NumPy arrays.
    ///
    /// :param sources: 1D integer array of source node indices
    /// :type sources: numpy.ndarray
    /// :param targets: 1D integer array of target node indices
    /// :type targets: numpy.ndarray
    /// :param distances: 1D float64 array of barrier target distances (`> 0`, finite)
    /// :type distances: numpy.ndarray
    /// :param weights: 1D float64 array of hinge pair weights (`> 0`, finite)
    /// :type weights: numpy.ndarray
    /// :param n_nodes: Optional node count for bounds checking
    /// :type n_nodes: int or None
    /// :return: None
    /// :rtype: None
    #[pyo3(signature = (sources, targets, distances, weights, n_nodes=None))]
    pub fn add_hinge_pairs_from_numpy(
        &mut self,
        sources: &Bound<'_, PyAny>,
        targets: &Bound<'_, PyAny>,
        distances: &Bound<'_, PyAny>,
        weights: &Bound<'_, PyAny>,
        n_nodes: Option<usize>,
    ) -> PyResult<()> {
        let hinges = extract_validated_pairs_1d(sources, targets, distances, weights, n_nodes, "")?;
        self.sgd.add_hinge_pairs(hinges);
        Ok(())
    }

    /// Return the number of two-sided spring pairs in this `Sgd` instance.
    ///
    /// :return: Number of two-sided spring pairs
    /// :rtype: int
    pub fn node_pair_count(&self) -> usize {
        self.sgd.node_pairs().len()
    }

    /// Return the number of one-sided hinge repulsion pairs in this `Sgd` instance.
    ///
    /// :return: Number of one-sided hinge pairs
    /// :rtype: int
    pub fn hinge_pair_count(&self) -> usize {
        self.sgd.hinge_pairs().len()
    }

    /// Update SGD pair weights from an `(N, N)` float64 NumPy matrix.
    pub fn update_weight_from_numpy(
        &mut self,
        w: numpy::PyReadonlyArray2<'_, crate::FloatType>,
    ) -> PyResult<()> {
        let w_view = w.as_array();
        self.sgd
            .try_update_weight_from_array(w_view)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Update SGD pair distances (and optionally weights) from `(N, N)` float64 NumPy matrices.
    #[pyo3(signature = (d, w=None))]
    pub fn update_distance_from_numpy(
        &mut self,
        d: numpy::PyReadonlyArray2<'_, crate::FloatType>,
        w: Option<numpy::PyReadonlyArray2<'_, crate::FloatType>>,
    ) -> PyResult<()> {
        let d_view = d.as_array();
        let w_view = w.as_ref().map(|arr| arr.as_array());
        self.sgd
            .try_update_distance_from_array(d_view, w_view)
            .map_err(pyo3::exceptions::PyValueError::new_err)
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
