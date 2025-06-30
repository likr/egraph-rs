use crate::{
    distance_matrix::{DistanceMatrixType, PyDistanceMatrix},
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
    builder: SparseSgd<f32>,
}

#[pymethods]
impl PySparseSgd {
    #[new]
    fn new() -> Self {
        Self {
            builder: SparseSgd::new(),
        }
    }

    fn eps(mut slf: PyRefMut<Self>, eps: f32) -> Py<Self> {
        slf.builder.eps(eps);
        slf.into()
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
        d: &PyDistanceMatrix,
    ) -> PySgd {
        PySgd::new_with_sgd(match graph.graph() {
            GraphType::Graph(native_graph) => {
                let nodes = native_graph.node_identifiers().collect::<Vec<_>>();
                match d.distance_matrix() {
                    DistanceMatrixType::Full(d) => {
                        self.builder.build_with_pivot_and_distance_matrix(
                            native_graph,
                            |e| f.call1((e.id().index(),)).unwrap().extract().unwrap(),
                            &pivot.iter().map(|&i| nodes[i]).collect::<Vec<_>>(),
                            d,
                        )
                    }
                    DistanceMatrixType::Sub(d) => {
                        self.builder.build_with_pivot_and_distance_matrix(
                            native_graph,
                            |e| f.call1((e.id().index(),)).unwrap().extract().unwrap(),
                            &pivot.iter().map(|&i| nodes[i]).collect::<Vec<_>>(),
                            d,
                        )
                    }
                }
            }
            _ => panic!("unsupported graph type"),
        })
    }
}
