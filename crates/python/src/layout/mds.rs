use crate::{
    drawing::PyDrawing,
    graph::{GraphType, PyGraphAdapter},
};
use petgraph::{graph::node_index, visit::EdgeRef};
use petgraph_layout_mds::{ClassicalMds, PivotMds};
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
        PyDrawing::new_drawing_2d(match graph.graph() {
            GraphType::Graph(native_graph) => self.mds.run(native_graph, |e| {
                f.call1((e.id().index(),)).unwrap().extract().unwrap()
            }),
            _ => panic!("unsupported graph type"),
        })
    }

    #[getter]
    fn eps(&self) -> f32 {
        self.mds.eps
    }

    #[setter]
    fn set_eps(&mut self, value: f32) {
        self.mds.eps = value;
    }
}

#[pyclass]
#[pyo3(name = "PivotMds")]
struct PyPivotMds {
    mds: PivotMds,
}

#[pymethods]
impl PyPivotMds {
    #[new]
    fn new() -> PyPivotMds {
        PyPivotMds {
            mds: PivotMds::new(),
        }
    }

    fn run(&self, graph: &PyGraphAdapter, f: &PyAny, pivot: Vec<usize>) -> PyDrawing {
        PyDrawing::new_drawing_2d(match graph.graph() {
            GraphType::Graph(native_graph) => {
                let pivot = pivot.into_iter().map(|u| node_index(u)).collect::<Vec<_>>();
                self.mds.run(
                    native_graph,
                    |e| f.call1((e.id().index(),)).unwrap().extract().unwrap(),
                    &pivot,
                )
            }
            _ => panic!("unsupported graph type"),
        })
    }

    #[getter]
    fn eps(&self) -> f32 {
        self.mds.eps
    }

    #[setter]
    fn set_eps(&mut self, value: f32) {
        self.mds.eps = value;
    }
}

pub fn register(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyClassicalMds>()?;
    m.add_class::<PyPivotMds>()?;
    Ok(())
}
