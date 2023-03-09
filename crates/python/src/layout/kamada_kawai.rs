use crate::{
    coordinates::PyCoordinates,
    graph::{GraphType, PyGraphAdapter},
};
use petgraph::visit::EdgeRef;
use petgraph_layout_kamada_kawai::KamadaKawai;
use pyo3::prelude::*;

#[pyclass]
#[pyo3(name = "KamadaKawai")]
struct PyKamadaKawai {
    kamada_kawai: KamadaKawai,
}

#[pymethods]
impl PyKamadaKawai {
    #[new]
    fn new(graph: &PyGraphAdapter, f: &PyAny) -> PyKamadaKawai {
        PyKamadaKawai {
            kamada_kawai: match graph.graph() {
                GraphType::Graph(native_graph) => KamadaKawai::new(native_graph, &mut |e| {
                    f.call1((e.id().index(),)).unwrap().extract().unwrap()
                }),
                _ => panic!("unsupported graph type"),
            },
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
