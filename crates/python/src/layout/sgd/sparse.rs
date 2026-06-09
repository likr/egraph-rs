use crate::{
    distance_matrix::with_distance,
    graph::{GraphType, PyGraphAdapter},
    layout::sgd::PySgd,
    rng::PyRng,
};
use petgraph::visit::{EdgeRef, IntoNodeIdentifiers};
use petgraph_layout_sgd::SparseSgd;
use pyo3::prelude::*;

#[pyclass]
#[pyo3(name = "SparseSgd")]
pub struct PySparseSgd {
    builder: SparseSgd,
}

#[pymethods]
impl PySparseSgd {
    #[new]
    fn new() -> Self {
        Self {
            builder: SparseSgd::new(),
        }
    }

    fn h(mut slf: PyRefMut<Self>, h: usize) -> Py<Self> {
        slf.builder.h(h);
        slf.into()
    }

    fn build(&self, graph: &PyGraphAdapter, f: &Bound<PyAny>, rng: &mut PyRng) -> PySgd {
        PySgd::new_with_sgd(match graph.graph() {
            GraphType::Graph(native_graph) => self.builder.build(
                native_graph,
                |e| f.call1((e.id().index(),)).unwrap().extract().unwrap(),
                rng.get_mut(),
            ),
            _ => panic!("unsupported graph type"),
        })
    }

    pub fn new_with_pivot(
        &self,
        graph: &PyGraphAdapter,
        f: &Bound<PyAny>,
        pivot: Vec<usize>,
    ) -> PySgd {
        PySgd::new_with_sgd(match graph.graph() {
            GraphType::Graph(native_graph) => {
                let nodes = native_graph.node_identifiers().collect::<Vec<_>>();
                let pivot_nodes = pivot.iter().map(|&i| nodes[i]).collect::<Vec<_>>();
                self.builder.build_with_pivot(
                    native_graph,
                    |e| f.call1((e.id().index(),)).unwrap().extract().unwrap(),
                    &pivot_nodes,
                )
            }
            _ => panic!("unsupported graph type"),
        })
    }

    pub fn build_with_pivot_and_distance_matrix(
        &self,
        graph: &PyGraphAdapter,
        f: &Bound<PyAny>,
        pivot: Vec<usize>,
        d: &Bound<PyAny>,
    ) -> PyResult<PySgd> {
        match graph.graph() {
            GraphType::Graph(native_graph) => {
                let nodes = native_graph.node_identifiers().collect::<Vec<_>>();
                let pivot_nodes = pivot.iter().map(|&i| nodes[i]).collect::<Vec<_>>();
                let sgd = with_distance(d, |distance| {
                    self.builder.build_with_pivot_and_distance_matrix(
                        native_graph,
                        |e| f.call1((e.id().index(),)).unwrap().extract().unwrap(),
                        &pivot_nodes,
                        distance,
                    )
                })?;
                Ok(PySgd::new_with_sgd(sgd))
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                "unsupported graph type",
            )),
        }
    }
}
