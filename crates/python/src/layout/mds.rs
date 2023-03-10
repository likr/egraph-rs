use crate::{
    drawing::PyDrawing,
    graph::{GraphType, PyGraphAdapter},
};
use petgraph::visit::EdgeRef;
use petgraph_layout_mds::ClassicalMds;
use pyo3::prelude::*;

#[pyclass]
#[pyo3(name = "ClassicalMds")]
struct PyClassicalMds {
    mds: ClassicalMds,
}

#[pymethods]
impl PyClassicalMds {
    #[new]
    fn new() -> PyClassicalMds {
        PyClassicalMds {
            mds: ClassicalMds::new(),
        }
    }

    fn run(&self, graph: &PyGraphAdapter, f: &PyAny) -> PyDrawing {
        PyDrawing::new(match graph.graph() {
            GraphType::Graph(native_graph) => self.mds.run(native_graph, |e| {
                f.call1((e.id().index(),)).unwrap().extract().unwrap()
            }),
            _ => panic!("unsupported graph type"),
        })
    }
}

pub fn register(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyClassicalMds>()?;
    Ok(())
}
