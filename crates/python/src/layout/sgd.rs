use crate::{
    distance_matrix::{DistanceMatrixType, PyDistanceMatrix},
    drawing::{
        DrawingType, PyDrawing, PyDrawingEuclidean, PyDrawingEuclidean2d, PyDrawingHyperbolic2d,
        PyDrawingSpherical2d, PyDrawingTorus2d,
    },
    graph::{GraphType, PyGraphAdapter},
    rng::PyRng,
};
use petgraph::visit::{EdgeRef, IntoNodeIdentifiers};
use petgraph_layout_sgd::{
    DistanceAdjustedSgd, FullSgd, Scheduler, SchedulerConstant, SchedulerExponential,
    SchedulerLinear, SchedulerQuadratic, SchedulerReciprocal, Sgd, SparseSgd,
};
use pyo3::prelude::*;
/// Python class that implements a constant learning rate scheduler
///
/// This scheduler maintains a constant learning rate throughout the optimization process.
/// It's the simplest scheduler but may not converge as effectively as decreasing schedules.
#[pyclass]
#[pyo3(name = "SchedulerConstant")]
struct PySchedulerConstant {
    scheduler: SchedulerConstant<f32>,
}

#[pymethods]
impl PySchedulerConstant {
    /// Runs the complete schedule, calling the provided function with each learning rate
    ///
    /// # Parameters
    /// * `f` - A Python function that takes the current learning rate as a parameter
    pub fn run(&mut self, f: &Bound<PyAny>) {
        self.scheduler.run(&mut |eta| {
            f.call1((eta as f64,)).ok();
        })
    }

    /// Advances the schedule by one step and calls the provided function with the current learning rate
    ///
    /// # Parameters
    /// * `f` - A Python function that takes the current learning rate as a parameter
    pub fn step(&mut self, f: &Bound<PyAny>) {
        self.scheduler.step(&mut |eta| {
            f.call1((eta as f64,)).ok();
        })
    }

    /// Checks if the schedule has completed all steps
    ///
    /// # Returns
    /// `true` if the schedule is finished, `false` otherwise
    pub fn is_finished(&self) -> bool {
        self.scheduler.is_finished()
    }
}

/// Python class that implements a linear decay learning rate scheduler
///
/// This scheduler decreases the learning rate linearly from the initial value
/// to the final value over the specified number of steps.
#[pyclass]
#[pyo3(name = "SchedulerLinear")]
struct PySchedulerLinear {
    scheduler: SchedulerLinear<f32>,
}

#[pymethods]
impl PySchedulerLinear {
    /// Runs the complete schedule, calling the provided function with each learning rate
    ///
    /// # Parameters
    /// * `f` - A Python function that takes the current learning rate as a parameter
    pub fn run(&mut self, f: &Bound<PyAny>) {
        self.scheduler.run(&mut |eta| {
            f.call1((eta as f64,)).ok();
        })
    }

    /// Advances the schedule by one step and calls the provided function with the current learning rate
    ///
    /// # Parameters
    /// * `f` - A Python function that takes the current learning rate as a parameter
    pub fn step(&mut self, f: &Bound<PyAny>) {
        self.scheduler.step(&mut |eta| {
            f.call1((eta as f64,)).ok();
        })
    }

    /// Checks if the schedule has completed all steps
    ///
    /// # Returns
    /// `true` if the schedule is finished, `false` otherwise
    pub fn is_finished(&self) -> bool {
        self.scheduler.is_finished()
    }
}

/// Python class that implements a quadratic decay learning rate scheduler
///
/// This scheduler decreases the learning rate according to a quadratic function
/// from the initial value to the final value over the specified number of steps.
#[pyclass]
#[pyo3(name = "SchedulerQuadratic")]
struct PySchedulerQuadratic {
    scheduler: SchedulerQuadratic<f32>,
}

#[pymethods]
impl PySchedulerQuadratic {
    /// Runs the complete schedule, calling the provided function with each learning rate
    ///
    /// # Parameters
    /// * `f` - A Python function that takes the current learning rate as a parameter
    pub fn run(&mut self, f: &Bound<PyAny>) {
        self.scheduler.run(&mut |eta| {
            f.call1((eta as f64,)).ok();
        })
    }

    /// Advances the schedule by one step and calls the provided function with the current learning rate
    ///
    /// # Parameters
    /// * `f` - A Python function that takes the current learning rate as a parameter
    pub fn step(&mut self, f: &Bound<PyAny>) {
        self.scheduler.step(&mut |eta| {
            f.call1((eta as f64,)).ok();
        })
    }

    /// Checks if the schedule has completed all steps
    ///
    /// # Returns
    /// `true` if the schedule is finished, `false` otherwise
    pub fn is_finished(&self) -> bool {
        self.scheduler.is_finished()
    }
}

/// Python class that implements an exponential decay learning rate scheduler
///
/// This scheduler decreases the learning rate exponentially from the initial value
/// to the final value over the specified number of steps. This is often the most
/// effective scheduler for graph layout algorithms.
#[pyclass]
#[pyo3(name = "SchedulerExponential")]
struct PySchedulerExponential {
    scheduler: SchedulerExponential<f32>,
}

#[pymethods]
impl PySchedulerExponential {
    /// Runs the complete schedule, calling the provided function with each learning rate
    ///
    /// # Parameters
    /// * `f` - A Python function that takes the current learning rate as a parameter
    pub fn run(&mut self, f: &Bound<PyAny>) {
        self.scheduler.run(&mut |eta| {
            f.call1((eta as f64,)).ok();
        })
    }

    /// Advances the schedule by one step and calls the provided function with the current learning rate
    ///
    /// # Parameters
    /// * `f` - A Python function that takes the current learning rate as a parameter
    pub fn step(&mut self, f: &Bound<PyAny>) {
        self.scheduler.step(&mut |eta| {
            f.call1((eta as f64,)).ok();
        })
    }

    /// Checks if the schedule has completed all steps
    ///
    /// # Returns
    /// `true` if the schedule is finished, `false` otherwise
    pub fn is_finished(&self) -> bool {
        self.scheduler.is_finished()
    }
}

/// Python class that implements a reciprocal decay learning rate scheduler
///
/// This scheduler decreases the learning rate according to a reciprocal function (1/t)
/// from the initial value to the final value over the specified number of steps.
#[pyclass]
#[pyo3(name = "SchedulerReciprocal")]
struct PySchedulerReciprocal {
    scheduler: SchedulerReciprocal<f32>,
}

#[pymethods]
impl PySchedulerReciprocal {
    /// Runs the complete schedule, calling the provided function with each learning rate
    ///
    /// # Parameters
    /// * `f` - A Python function that takes the current learning rate as a parameter
    pub fn run(&mut self, f: &Bound<PyAny>) {
        self.scheduler.run(&mut |eta| {
            f.call1((eta as f64,)).ok();
        })
    }

    /// Advances the schedule by one step and calls the provided function with the current learning rate
    ///
    /// # Parameters
    /// * `f` - A Python function that takes the current learning rate as a parameter
    pub fn step(&mut self, f: &Bound<PyAny>) {
        self.scheduler.step(&mut |eta| {
            f.call1((eta as f64,)).ok();
        })
    }

    /// Checks if the schedule has completed all steps
    ///
    /// # Returns
    /// `true` if the schedule is finished, `false` otherwise
    pub fn is_finished(&self) -> bool {
        self.scheduler.is_finished()
    }
}

/// Python class for sparse stochastic gradient descent (SGD) layout algorithm
///
/// This class implements SGD with sparse distance approximation, which is more
/// efficient for large graphs. It uses a subset of "pivot" nodes to approximate
/// distances, reducing computational complexity from O(n²) to O(nh) where h is
/// the number of pivot nodes.
#[pyclass]
#[pyo3(name = "SparseSgd")]
struct PySparseSgd {
    sgd: SparseSgd<f32>,
}

#[pymethods]
impl PySparseSgd {
    /// Creates a new sparse SGD instance with randomly selected pivot nodes
    ///
    /// # Parameters
    /// * `graph` - The graph to layout
    /// * `f` - A Python function that takes an edge index and returns its weight
    /// * `h` - The number of pivot nodes to use
    /// * `rng` - Random number generator for selecting pivot nodes
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
    /// # Parameters
    /// * `graph` - The graph to layout
    /// * `f` - A Python function that takes an edge index and returns its weight
    /// * `pivot` - A vector of node indices to use as pivot nodes
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
    /// # Parameters
    /// * `graph` - The graph to layout
    /// * `f` - A Python function that takes an edge index and returns its weight
    /// * `pivot` - A vector of node indices to use as pivot nodes
    /// * `d` - A pre-computed distance matrix
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
    /// # Parameters
    /// * `rng` - Random number generator for shuffling
    fn shuffle(&mut self, rng: &mut PyRng) {
        self.sgd.shuffle(rng.get_mut())
    }

    /// Applies one iteration of the SGD algorithm to the drawing
    ///
    /// # Parameters
    /// * `drawing` - The drawing to modify
    /// * `eta` - The learning rate for this iteration
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
    /// # Parameters
    /// * `t_max` - The maximum number of iterations
    /// * `epsilon` - The final learning rate (initial rate is 1.0)
    ///
    /// # Returns
    /// An exponential scheduler
    pub fn scheduler(&self, t_max: usize, epsilon: f32) -> PySchedulerExponential {
        self.scheduler_exponential(t_max, epsilon)
    }

    /// Creates a constant scheduler for this SGD algorithm
    ///
    /// # Parameters
    /// * `t_max` - The maximum number of iterations
    /// * `epsilon` - The constant learning rate
    ///
    /// # Returns
    /// A constant scheduler
    pub fn scheduler_constant(&self, t_max: usize, epsilon: f32) -> PySchedulerConstant {
        PySchedulerConstant {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    /// Creates a linear scheduler for this SGD algorithm
    ///
    /// # Parameters
    /// * `t_max` - The maximum number of iterations
    /// * `epsilon` - The final learning rate (initial rate is 1.0)
    ///
    /// # Returns
    /// A linear scheduler
    pub fn scheduler_linear(&self, t_max: usize, epsilon: f32) -> PySchedulerLinear {
        PySchedulerLinear {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    /// Creates a quadratic scheduler for this SGD algorithm
    ///
    /// # Parameters
    /// * `t_max` - The maximum number of iterations
    /// * `epsilon` - The final learning rate (initial rate is 1.0)
    ///
    /// # Returns
    /// A quadratic scheduler
    pub fn scheduler_quadratic(&self, t_max: usize, epsilon: f32) -> PySchedulerQuadratic {
        PySchedulerQuadratic {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    /// Creates an exponential scheduler for this SGD algorithm
    ///
    /// # Parameters
    /// * `t_max` - The maximum number of iterations
    /// * `epsilon` - The final learning rate (initial rate is 1.0)
    ///
    /// # Returns
    /// An exponential scheduler
    pub fn scheduler_exponential(&self, t_max: usize, epsilon: f32) -> PySchedulerExponential {
        PySchedulerExponential {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    /// Creates a reciprocal scheduler for this SGD algorithm
    ///
    /// # Parameters
    /// * `t_max` - The maximum number of iterations
    /// * `epsilon` - The final learning rate (initial rate is 1.0)
    ///
    /// # Returns
    /// A reciprocal scheduler
    pub fn scheduler_reciprocal(&self, t_max: usize, epsilon: f32) -> PySchedulerReciprocal {
        PySchedulerReciprocal {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    /// Updates the distance matrix using a Python function
    ///
    /// # Parameters
    /// * `f` - A Python function that takes (i, j, distance, weight) and returns a new distance value
    pub fn update_distance(&mut self, f: &Bound<PyAny>) {
        self.sgd
            .update_distance(|i, j, dij, wij| f.call1((i, j, dij, wij)).unwrap().extract().unwrap())
    }

    /// Updates the weight matrix using a Python function
    ///
    /// # Parameters
    /// * `f` - A Python function that takes (i, j, distance, weight) and returns a new weight value
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
    /// # Parameters
    /// * `graph` - The graph to select pivots from
    /// * `f` - A Python function that takes an edge index and returns its weight
    /// * `h` - The number of pivot nodes to select
    /// * `rng` - Random number generator for selecting pivots
    ///
    /// # Returns
    /// A tuple containing the pivot node indices and a distance matrix
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

/// Python class for full stochastic gradient descent (SGD) layout algorithm
///
/// This class implements the standard SGD algorithm that uses all pairs of nodes
/// to compute the layout. It's suitable for small to medium-sized graphs but
/// becomes computationally expensive for large graphs (where SparseSgd is preferred).
///
/// The algorithm iteratively adjusts node positions by minimizing the difference
/// between the Euclidean distances in the layout and the graph-theoretic distances.
#[pyclass]
#[pyo3(name = "FullSgd")]
struct PyFullSgd {
    sgd: FullSgd<f32>,
}

#[pymethods]
impl PyFullSgd {
    #[new]
    fn new(graph: &PyGraphAdapter, f: &Bound<PyAny>) -> PyFullSgd {
        PyFullSgd {
            sgd: match graph.graph() {
                GraphType::Graph(native_graph) => FullSgd::new(native_graph, |e| {
                    f.call1((e.id().index(),)).unwrap().extract().unwrap()
                }),
                _ => panic!("unsupported graph type"),
            },
        }
    }

    #[staticmethod]
    fn new_with_distance_matrix(d: &PyDistanceMatrix) -> PyFullSgd {
        match d.distance_matrix() {
            DistanceMatrixType::Full(d) => PyFullSgd {
                sgd: FullSgd::new_with_distance_matrix(d),
            },
            _ => panic!("unsupported distance matrix type"),
        }
    }

    fn shuffle(&mut self, rng: &mut PyRng) {
        self.sgd.shuffle(rng.get_mut())
    }

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

    pub fn scheduler(&self, t_max: usize, epsilon: f32) -> PySchedulerExponential {
        self.scheduler_exponential(t_max, epsilon)
    }

    pub fn scheduler_constant(&self, t_max: usize, epsilon: f32) -> PySchedulerConstant {
        PySchedulerConstant {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    pub fn scheduler_linear(&self, t_max: usize, epsilon: f32) -> PySchedulerLinear {
        PySchedulerLinear {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    pub fn scheduler_quadratic(&self, t_max: usize, epsilon: f32) -> PySchedulerQuadratic {
        PySchedulerQuadratic {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    pub fn scheduler_exponential(&self, t_max: usize, epsilon: f32) -> PySchedulerExponential {
        PySchedulerExponential {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    pub fn scheduler_reciprocal(&self, t_max: usize, epsilon: f32) -> PySchedulerReciprocal {
        PySchedulerReciprocal {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    pub fn update_distance(&mut self, f: &Bound<PyAny>) {
        self.sgd
            .update_distance(|i, j, dij, wij| f.call1((i, j, dij, wij)).unwrap().extract().unwrap())
    }

    pub fn update_weight(&mut self, f: &Bound<PyAny>) {
        self.sgd
            .update_weight(|i, j, dij, wij| f.call1((i, j, dij, wij)).unwrap().extract().unwrap())
    }
}

/// Python class for distance-adjusted sparse SGD layout algorithm
///
/// This class combines the sparse SGD algorithm with distance adjustments to
/// create more aesthetically pleasing layouts. It adjusts the forces between
/// nodes based on their current distances in the layout, helping to avoid
/// the "hair ball" effect common in force-directed layouts.
///
/// Distance adjustment applies stronger forces between nodes that are already
/// close to each other, and weaker forces between distant nodes. This helps
/// preserve local structure while allowing the global structure to develop.
#[pyclass]
#[pyo3(name = "DistanceAdjustedSparseSgd")]
struct PyDistanceAdjustedSparseSgd {
    sgd: DistanceAdjustedSgd<SparseSgd<f32>, f32>,
}

#[pymethods]
impl PyDistanceAdjustedSparseSgd {
    #[new]
    fn new(graph: &PyGraphAdapter, f: &Bound<PyAny>, h: usize, rng: &mut PyRng) -> Self {
        Self {
            sgd: DistanceAdjustedSgd::new(match graph.graph() {
                GraphType::Graph(native_graph) => SparseSgd::new_with_rng(
                    native_graph,
                    |e| f.call1((e.id().index(),)).unwrap().extract().unwrap(),
                    h,
                    rng.get_mut(),
                ),
                _ => panic!("unsupported graph type"),
            }),
        }
    }

    #[staticmethod]
    pub fn new_with_pivot(graph: &PyGraphAdapter, f: &Bound<PyAny>, pivot: Vec<usize>) -> Self {
        Self {
            sgd: DistanceAdjustedSgd::new(match graph.graph() {
                GraphType::Graph(native_graph) => {
                    let nodes = native_graph.node_identifiers().collect::<Vec<_>>();
                    SparseSgd::new_with_pivot(
                        native_graph,
                        |e| f.call1((e.id().index(),)).unwrap().extract().unwrap(),
                        &pivot.iter().map(|&i| nodes[i]).collect::<Vec<_>>(),
                    )
                }
                _ => panic!("unsupported graph type"),
            }),
        }
    }

    fn shuffle(&mut self, rng: &mut PyRng) {
        self.sgd.shuffle(rng.get_mut())
    }

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

    pub fn scheduler(&self, t_max: usize, epsilon: f32) -> PySchedulerExponential {
        self.scheduler_exponential(t_max, epsilon)
    }

    pub fn scheduler_constant(&self, t_max: usize, epsilon: f32) -> PySchedulerConstant {
        PySchedulerConstant {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    pub fn scheduler_linear(&self, t_max: usize, epsilon: f32) -> PySchedulerLinear {
        PySchedulerLinear {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    pub fn scheduler_quadratic(&self, t_max: usize, epsilon: f32) -> PySchedulerQuadratic {
        PySchedulerQuadratic {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    pub fn scheduler_exponential(&self, t_max: usize, epsilon: f32) -> PySchedulerExponential {
        PySchedulerExponential {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    pub fn scheduler_reciprocal(&self, t_max: usize, epsilon: f32) -> PySchedulerReciprocal {
        PySchedulerReciprocal {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    pub fn update_distance(&mut self, f: &Bound<PyAny>) {
        self.sgd
            .update_distance(|i, j, dij, wij| f.call1((i, j, dij, wij)).unwrap().extract().unwrap())
    }

    pub fn update_weight(&mut self, f: &Bound<PyAny>) {
        self.sgd
            .update_weight(|i, j, dij, wij| f.call1((i, j, dij, wij)).unwrap().extract().unwrap())
    }

    #[getter]
    fn alpha(&self) -> f32 {
        self.sgd.alpha
    }

    #[setter]
    fn set_alpha(&mut self, value: f32) {
        self.sgd.alpha = value;
    }

    #[getter]
    fn minimum_distance(&self) -> f32 {
        self.sgd.minimum_distance
    }

    #[setter]
    fn set_minimum_distance(&mut self, value: f32) {
        self.sgd.minimum_distance = value;
    }
}

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
#[pyclass]
#[pyo3(name = "DistanceAdjustedFullSgd")]
struct PyDistanceAdjustedFullSgd {
    sgd: DistanceAdjustedSgd<FullSgd<f32>, f32>,
}

#[pymethods]
impl PyDistanceAdjustedFullSgd {
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

    #[staticmethod]
    fn new_with_distance_matrix(d: &PyDistanceMatrix) -> Self {
        match d.distance_matrix() {
            DistanceMatrixType::Full(d) => Self {
                sgd: DistanceAdjustedSgd::new(FullSgd::new_with_distance_matrix(d)),
            },
            _ => panic!("unsupported distance matrix type"),
        }
    }

    fn shuffle(&mut self, rng: &mut PyRng) {
        self.sgd.shuffle(rng.get_mut())
    }

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

    pub fn scheduler(&self, t_max: usize, epsilon: f32) -> PySchedulerExponential {
        self.scheduler_exponential(t_max, epsilon)
    }

    pub fn scheduler_constant(&self, t_max: usize, epsilon: f32) -> PySchedulerConstant {
        PySchedulerConstant {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    pub fn scheduler_linear(&self, t_max: usize, epsilon: f32) -> PySchedulerLinear {
        PySchedulerLinear {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    pub fn scheduler_quadratic(&self, t_max: usize, epsilon: f32) -> PySchedulerQuadratic {
        PySchedulerQuadratic {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    pub fn scheduler_exponential(&self, t_max: usize, epsilon: f32) -> PySchedulerExponential {
        PySchedulerExponential {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    pub fn scheduler_reciprocal(&self, t_max: usize, epsilon: f32) -> PySchedulerReciprocal {
        PySchedulerReciprocal {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    pub fn update_distance(&mut self, f: &Bound<PyAny>) {
        self.sgd
            .update_distance(|i, j, dij, wij| f.call1((i, j, dij, wij)).unwrap().extract().unwrap())
    }

    pub fn update_weight(&mut self, f: &Bound<PyAny>) {
        self.sgd
            .update_weight(|i, j, dij, wij| f.call1((i, j, dij, wij)).unwrap().extract().unwrap())
    }

    #[getter]
    fn alpha(&self) -> f32 {
        self.sgd.alpha
    }

    #[setter]
    fn set_alpha(&mut self, value: f32) {
        self.sgd.alpha = value;
    }

    #[getter]
    fn minimum_distance(&self) -> f32 {
        self.sgd.minimum_distance
    }

    #[setter]
    fn set_minimum_distance(&mut self, value: f32) {
        self.sgd.minimum_distance = value;
    }
}

pub fn register(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PySchedulerConstant>()?;
    m.add_class::<PySchedulerLinear>()?;
    m.add_class::<PySchedulerQuadratic>()?;
    m.add_class::<PySchedulerExponential>()?;
    m.add_class::<PySchedulerReciprocal>()?;
    m.add_class::<PyFullSgd>()?;
    m.add_class::<PySparseSgd>()?;
    m.add_class::<PyDistanceAdjustedFullSgd>()?;
    m.add_class::<PyDistanceAdjustedSparseSgd>()?;
    Ok(())
}
