use crate::{coordinates::PyCoordinates, graph::PyGraph, rng::PyRng};
use petgraph::visit::EdgeRef;
use petgraph_layout_sgd::{Sgd, SgdScheduler, SparseSgd};
use pyo3::prelude::*;
use std::collections::HashMap;

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
    fn new(graph: &PyGraph, f: &PyAny, h: usize, rng: &mut PyRng) -> PySparseSgd {
        let mut distance = HashMap::new();
        for e in graph.graph().edge_indices() {
            let v = f.call1((e.index(),)).unwrap().extract().unwrap();
            distance.insert(e, v);
        }
        PySparseSgd {
            sgd: SparseSgd::new_with_rng(
                graph.graph(),
                &mut |e| distance[&e.id()],
                h,
                rng.get_mut(),
            ),
        }
    }

    fn shuffle(&mut self, rng: &mut PyRng) {
        self.sgd.shuffle(rng.get_mut())
    }

    fn apply(&self, coordinates: &mut PyCoordinates, eta: f32) {
        self.sgd.apply(coordinates.coordinates_mut(), eta);
    }

    pub fn scheduler(&self, t_max: usize, epsilon: f32) -> PySgdScheduler {
        PySgdScheduler {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }
}

pub fn register(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySgdScheduler>()?;
    m.add_class::<PySparseSgd>()?;
    Ok(())
}
