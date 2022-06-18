use crate::coordinates::PyCoordinates;
use crate::graph::PyGraph;
use petgraph::visit::EdgeRef;
use petgraph_layout_kamada_kawai::KamadaKawai;
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass]
#[pyo3(name = "KamadaKawai")]
struct PyKamadaKawai {
    kamada_kawai: KamadaKawai,
}

#[pymethods]
impl PyKamadaKawai {
    #[new]
    fn new(graph: &PyGraph, f: &PyAny) -> PyKamadaKawai {
        let mut distance = HashMap::new();
        for e in graph.graph().edge_indices() {
            let result = f.call1((e.index(),)).unwrap();
            let d = result.get_item("distance").unwrap().extract().unwrap();
            distance.insert(e, d);
        }
        PyKamadaKawai {
            kamada_kawai: KamadaKawai::new(graph.graph(), &mut |e| distance[&e.id()]),
        }
    }

    fn select_node(&self, coordinates: &PyCoordinates) -> Option<usize> {
        self.kamada_kawai.select_node(coordinates.coordinates())
    }

    fn apply_to_node(&self, m: usize, coordinates: &mut PyCoordinates) {
        self.kamada_kawai
            .apply_to_node(m, coordinates.coordinates_mut());
    }

    fn run(&self, coordinates: &mut PyCoordinates) {
        self.kamada_kawai.run(coordinates.coordinates_mut());
    }

    #[getter]
    fn eps(&self) -> f32 {
        self.kamada_kawai.eps
    }

    #[setter]
    fn set_eps(&mut self, value: f32) {
        self.kamada_kawai.eps = value;
    }
}

pub fn register(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyKamadaKawai>()?;
    Ok(())
}
