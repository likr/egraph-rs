use crate::{
    distance_matrix::{DistanceMatrixType, PyDistanceMatrix},
    drawing::{DrawingType, PyDrawing, PyDrawingEuclidean2d, PyDrawingTorus2d},
    graph::{GraphType, PyGraphAdapter},
    rng::PyRng,
};
use petgraph::visit::{EdgeRef, IntoNodeIdentifiers};
use petgraph_layout_sgd::{DistanceAdjustedSgd, FullSgd, Sgd, SgdScheduler, SparseSgd};
use pyo3::prelude::*;

#[pyclass]
#[pyo3(name = "SgdScheduler")]
struct PySgdScheduler {
    scheduler: SgdScheduler,
}

#[pymethods]
impl PySgdScheduler {
    pub fn run(&mut self, f: &Bound<PyAny>) {
        self.scheduler.run(&mut |eta| {
            f.call1((eta as f64,)).ok();
        })
    }

    pub fn step(&mut self, f: &Bound<PyAny>) {
        self.scheduler.step(&mut |eta| {
            f.call1((eta as f64,)).ok();
        })
    }

    pub fn is_finished(&self) -> bool {
        self.scheduler.is_finished()
    }
}

#[pyclass]
#[pyo3(name = "SparseSgd")]
struct PySparseSgd {
    sgd: SparseSgd,
}

#[pymethods]
impl PySparseSgd {
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
            DrawingType::Torus2d => {
                let mut drawing = drawing
                    .into_py(py)
                    .downcast_bound::<PyDrawingTorus2d>(py)
                    .unwrap()
                    .borrow_mut();
                self.sgd.apply(drawing.drawing_mut(), eta)
            }
            _ => unimplemented!(),
        })
    }

    pub fn scheduler(&self, t_max: usize, epsilon: f32) -> PySgdScheduler {
        PySgdScheduler {
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

#[pyclass]
#[pyo3(name = "FullSgd")]
struct PyFullSgd {
    sgd: FullSgd,
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
            DrawingType::Torus2d => {
                let mut drawing = drawing
                    .into_py(py)
                    .downcast_bound::<PyDrawingTorus2d>(py)
                    .unwrap()
                    .borrow_mut();
                self.sgd.apply(drawing.drawing_mut(), eta)
            }
            _ => unimplemented!(),
        })
    }

    pub fn scheduler(&self, t_max: usize, epsilon: f32) -> PySgdScheduler {
        PySgdScheduler {
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

#[pyclass]
#[pyo3(name = "DistanceAdjustedSparseSgd")]
struct PyDistanceAdjustedSparseSgd {
    sgd: DistanceAdjustedSgd<SparseSgd>,
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
            DrawingType::Torus2d => {
                let mut drawing = drawing
                    .into_py(py)
                    .downcast_bound::<PyDrawingTorus2d>(py)
                    .unwrap()
                    .borrow_mut();
                self.sgd.apply(drawing.drawing_mut(), eta)
            }
            _ => unimplemented!(),
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
            DrawingType::Torus2d => {
                let mut drawing = drawing
                    .into_py(py)
                    .downcast_bound::<PyDrawingTorus2d>(py)
                    .unwrap()
                    .borrow_mut();
                self.sgd
                    .apply_with_distance_adjustment(drawing.drawing_mut(), eta)
            }
            _ => unimplemented!(),
        })
    }

    pub fn scheduler(&self, t_max: usize, epsilon: f32) -> PySgdScheduler {
        PySgdScheduler {
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

#[pyclass]
#[pyo3(name = "DistanceAdjustedFullSgd")]
struct PyDistanceAdjustedFullSgd {
    sgd: DistanceAdjustedSgd<FullSgd>,
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
            DrawingType::Torus2d => {
                let mut drawing = drawing
                    .into_py(py)
                    .downcast_bound::<PyDrawingTorus2d>(py)
                    .unwrap()
                    .borrow_mut();
                self.sgd.apply(drawing.drawing_mut(), eta)
            }
            _ => unimplemented!(),
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
            DrawingType::Torus2d => {
                let mut drawing = drawing
                    .into_py(py)
                    .downcast_bound::<PyDrawingTorus2d>(py)
                    .unwrap()
                    .borrow_mut();
                self.sgd
                    .apply_with_distance_adjustment(drawing.drawing_mut(), eta)
            }
            _ => unimplemented!(),
        })
    }

    pub fn scheduler(&self, t_max: usize, epsilon: f32) -> PySgdScheduler {
        PySgdScheduler {
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
    m.add_class::<PySgdScheduler>()?;
    m.add_class::<PyFullSgd>()?;
    m.add_class::<PySparseSgd>()?;
    m.add_class::<PyDistanceAdjustedFullSgd>()?;
    m.add_class::<PyDistanceAdjustedSparseSgd>()?;
    Ok(())
}
