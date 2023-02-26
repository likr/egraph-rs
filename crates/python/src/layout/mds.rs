use crate::{coordinates::PyCoordinates, graph::PyGraph};
use petgraph::visit::EdgeRef;
use petgraph_layout_mds::ClassicalMds;
use pyo3::prelude::*;
use std::collections::HashMap;

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

    fn run(&self, graph: &PyGraph, f: &PyAny) -> PyCoordinates {
        let mut distance = HashMap::new();
        for e in graph.graph().edge_indices() {
            let v = f.call1((e.index(),)).unwrap().extract().unwrap();
            distance.insert(e, v);
        }
        PyCoordinates::new(self.mds.run(graph.graph(), &mut |e| distance[&e.id()]))
    }
}

pub fn register(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyClassicalMds>()?;
    Ok(())
}
