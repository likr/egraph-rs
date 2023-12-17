use crate::{
    drawing::{DrawingType, PyDrawing},
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
                GraphType::Graph(native_graph) => KamadaKawai::new(native_graph, |e| {
                    f.call1((e.id().index(),)).unwrap().extract().unwrap()
                }),
                _ => panic!("unsupported graph type"),
            },
        }
    }

    fn select_node(&self, drawing: &PyDrawing) -> Option<usize> {
        match drawing.drawing() {
            DrawingType::Drawing2D(drawing) => self.kamada_kawai.select_node(drawing),
            _ => unimplemented!(),
        }
    }

    fn apply_to_node(&self, m: usize, drawing: &mut PyDrawing) {
        match drawing.drawing_mut() {
            DrawingType::Drawing2D(drawing) => self.kamada_kawai.apply_to_node(m, drawing),
            _ => unimplemented!(),
        };
    }

    fn run(&self, drawing: &mut PyDrawing) {
        match drawing.drawing_mut() {
            DrawingType::Drawing2D(drawing) => self.kamada_kawai.run(drawing),
            _ => unimplemented!(),
        };
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
