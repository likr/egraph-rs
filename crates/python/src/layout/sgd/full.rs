use crate::{
    distance_matrix::{DistanceMatrixType, PyDistanceMatrix},
    graph::{GraphType, PyGraphAdapter},
    layout::sgd::PySgd,
};
use petgraph::visit::EdgeRef;
use petgraph_layout_sgd::FullSgd;
use pyo3::prelude::*;

#[pyclass]
#[pyo3(name = "FullSgd")]
pub struct PyFullSgd {
    builder: FullSgd<f32>,
}

#[pymethods]
impl PyFullSgd {
    #[new]
    fn new() -> Self {
        Self {
            builder: FullSgd::new(),
        }
    }

    fn eps(mut slf: PyRefMut<Self>, eps: f32) -> Py<Self> {
        slf.builder.eps(eps);
        slf.into()
    }

    fn build(&self, graph: &PyGraphAdapter, f: &Bound<PyAny>) -> PySgd {
        PySgd::new_with_sgd(match graph.graph() {
            GraphType::Graph(native_graph) => self.builder.build(native_graph, |e| {
                f.call1((e.id().index(),)).unwrap().extract().unwrap()
            }),
            _ => panic!("unsupported graph type"),
        })
    }

    fn build_with_distance_matrix(&self, d: &PyDistanceMatrix) -> PySgd {
        PySgd::new_with_sgd(match d.distance_matrix() {
            DistanceMatrixType::Full(d) => self.builder.build_with_distance_matrix(d),
            _ => panic!("unsupported distance matrix type"),
        })
    }
}
