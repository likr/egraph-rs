use crate::{
    distance_matrix::PyDistanceMatrix,
    drawing::PyDrawing,
    graph::{GraphType, PyGraphAdapter},
    rng::PyRng,
};
use petgraph::visit::{EdgeRef, IntoNodeIdentifiers};
use petgraph_layout_sgd::{FullSgd, Sgd, SgdScheduler, SparseSgd};
use pyo3::prelude::*;

#[pyclass]
#[pyo3(name = "SgdScheduler")]
struct PySgdScheduler {
    scheduler: SgdScheduler,
}

#[pymethods]
impl PySgdScheduler {
    pub fn run(&mut self, f: &PyAny) {
        self.scheduler.run(&mut |eta| {
            f.call1((eta as f64,)).ok();
        })
    }

    pub fn step(&mut self, f: &PyAny) {
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
    fn new(graph: &PyGraphAdapter, f: &PyAny, h: usize, rng: &mut PyRng) -> PySparseSgd {
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
    pub fn new_with_pivot(graph: &PyGraphAdapter, f: &PyAny, pivot: Vec<usize>) -> Self {
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

    fn shuffle(&mut self, rng: &mut PyRng) {
        self.sgd.shuffle(rng.get_mut())
    }

    fn apply(&self, drawing: &mut PyDrawing, eta: f32) {
        self.sgd.apply(drawing.drawing_mut(), eta);
    }

    pub fn scheduler(&self, t_max: usize, epsilon: f32) -> PySgdScheduler {
        PySgdScheduler {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    pub fn update_weight(&mut self, f: &PyAny) {
        self.sgd
            .update_weight(|i, j, dij, wij| f.call1((i, j, dij, wij)).unwrap().extract().unwrap())
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
    fn new(graph: &PyGraphAdapter, f: &PyAny) -> PyFullSgd {
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
        PyFullSgd {
            sgd: FullSgd::new_with_distance_matrix(d.distance_matrix()),
        }
    }

    fn shuffle(&mut self, rng: &mut PyRng) {
        self.sgd.shuffle(rng.get_mut())
    }

    fn apply(&self, drawing: &mut PyDrawing, eta: f32) {
        self.sgd.apply(drawing.drawing_mut(), eta);
    }

    pub fn scheduler(&self, t_max: usize, epsilon: f32) -> PySgdScheduler {
        PySgdScheduler {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }

    pub fn update_weight(&mut self, f: &PyAny) {
        self.sgd
            .update_weight(|i, j, dij, wij| f.call1((i, j, dij, wij)).unwrap().extract().unwrap())
    }
}

pub fn register(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySgdScheduler>()?;
    m.add_class::<PyFullSgd>()?;
    m.add_class::<PySparseSgd>()?;
    Ok(())
}
