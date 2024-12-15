use crate::{
    distance_matrix::{DistanceMatrixType, PyDistanceMatrix},
    drawing::PyDrawing,
    graph::{GraphType, PyGraphAdapter},
};
use petgraph::{graph::node_index, stable_graph::NodeIndex, visit::EdgeRef};
use petgraph_layout_mds::{ClassicalMds, PivotMds};
use pyo3::prelude::*;

#[pyclass]
#[pyo3(name = "ClassicalMds")]
struct PyClassicalMds {
    mds: ClassicalMds<NodeIndex>,
}

#[pymethods]
impl PyClassicalMds {
    #[new]
    fn new(graph: &PyGraphAdapter, f: &Bound<PyAny>) -> PyClassicalMds {
        match graph.graph() {
            GraphType::Graph(native_graph) => PyClassicalMds {
                mds: ClassicalMds::new(native_graph, |e| {
                    f.call1((e.id().index(),)).unwrap().extract().unwrap()
                }),
            },
            _ => panic!("unsupported graph type"),
        }
    }

    #[staticmethod]
    fn new_with_distance_matrix(d: &PyDistanceMatrix) -> Self {
        match d.distance_matrix() {
            DistanceMatrixType::Full(d) => Self {
                mds: ClassicalMds::new_with_distance_matrix(d),
            },
            _ => panic!("unsupported distance matrix type"),
        }
    }

    fn run(&self, d: usize) -> PyObject {
        PyDrawing::new_drawing_euclidean(self.mds.run(d))
    }

    fn run_2d(&self) -> PyObject {
        PyDrawing::new_drawing_euclidean_2d(self.mds.run_2d())
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
    mds: PivotMds<NodeIndex>,
}

#[pymethods]
impl PyPivotMds {
    #[new]
    fn new(graph: &PyGraphAdapter, f: &Bound<PyAny>, pivot: Vec<usize>) -> PyPivotMds {
        match graph.graph() {
            GraphType::Graph(native_graph) => {
                let pivot = pivot.into_iter().map(|u| node_index(u)).collect::<Vec<_>>();
                PyPivotMds {
                    mds: PivotMds::new(
                        native_graph,
                        |e| f.call1((e.id().index(),)).unwrap().extract().unwrap(),
                        &pivot,
                    ),
                }
            }
            _ => panic!("unsupported graph type"),
        }
    }

    #[staticmethod]
    fn new_with_distance_matrix(d: &PyDistanceMatrix) -> Self {
        match d.distance_matrix() {
            DistanceMatrixType::Full(d) => Self {
                mds: PivotMds::new_with_distance_matrix(d),
            },
            DistanceMatrixType::Sub(d) => Self {
                mds: PivotMds::new_with_distance_matrix(d),
            },
        }
    }

    fn run(&self, d: usize) -> PyObject {
        PyDrawing::new_drawing_euclidean(self.mds.run(d))
    }

    fn run_2d(&self) -> PyObject {
        PyDrawing::new_drawing_euclidean_2d(self.mds.run_2d())
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

pub fn register(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyClassicalMds>()?;
    m.add_class::<PyPivotMds>()?;
    Ok(())
}
