use crate::{
    distance_matrix::with_distance,
    graph::{GraphType, PyGraphAdapter},
    layout::sgd::PySgd,
};
use petgraph::visit::EdgeRef;
use petgraph_layout_sgd::FullSgd;
use pyo3::prelude::*;

#[pyclass]
#[pyo3(name = "FullSgd")]
pub struct PyFullSgd {
    builder: FullSgd,
}

#[pymethods]
impl PyFullSgd {
    #[new]
    fn new() -> Self {
        Self {
            builder: FullSgd::new(),
        }
    }

    fn build(&self, graph: &PyGraphAdapter, f: &Bound<PyAny>) -> PySgd {
        PySgd::new_with_sgd(match graph.graph() {
            GraphType::Graph(native_graph) => self.builder.build(native_graph, |e| {
                f.call1((e.id().index(),)).unwrap().extract().unwrap()
            }),
            _ => panic!("unsupported graph type"),
        })
    }

    fn build_with_distance_matrix(&self, d: &Bound<PyAny>) -> PyResult<PySgd> {
        let sgd = with_distance(d, |distance| {
            self.builder.build_with_distance_matrix(distance)
        })?;
        Ok(PySgd::new_with_sgd(sgd))
    }
}
