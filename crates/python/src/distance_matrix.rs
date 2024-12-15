use crate::graph::{GraphType, IndexType, PyGraphAdapter};
use petgraph::{graph::NodeIndex, stable_graph::node_index};
use petgraph_algorithm_shortest_path::{DistanceMatrix, FullDistanceMatrix, SubDistanceMatrix};
use pyo3::prelude::*;

pub enum DistanceMatrixType {
    Full(FullDistanceMatrix<NodeIndex<IndexType>, f32>),
    Sub(SubDistanceMatrix<NodeIndex<IndexType>, f32>),
}

#[pyclass]
#[pyo3(name = "DistanceMatrix")]
pub struct PyDistanceMatrix {
    distance_matrix: DistanceMatrixType,
}

impl PyDistanceMatrix {
    pub fn new_with_full_distance_matrix(
        distance_matrix: FullDistanceMatrix<NodeIndex<IndexType>, f32>,
    ) -> Self {
        PyDistanceMatrix {
            distance_matrix: DistanceMatrixType::Full(distance_matrix),
        }
    }

    pub fn new_with_sub_distance_matrix(
        distance_matrix: SubDistanceMatrix<NodeIndex<IndexType>, f32>,
    ) -> Self {
        PyDistanceMatrix {
            distance_matrix: DistanceMatrixType::Sub(distance_matrix),
        }
    }

    pub fn distance_matrix(&self) -> &DistanceMatrixType {
        &self.distance_matrix
    }

    pub fn distance_matrix_mut(&mut self) -> &mut DistanceMatrixType {
        &mut self.distance_matrix
    }
}

#[pymethods]
impl PyDistanceMatrix {
    #[new]
    pub fn new(graph: &PyGraphAdapter) -> PyDistanceMatrix {
        match graph.graph() {
            GraphType::Graph(g) => Self::new_with_full_distance_matrix(FullDistanceMatrix::new(g)),
            GraphType::DiGraph(g) => {
                Self::new_with_full_distance_matrix(FullDistanceMatrix::new(g))
            }
        }
    }

    pub fn get(&self, u: usize, v: usize) -> Option<f32> {
        match self.distance_matrix() {
            DistanceMatrixType::Full(distance_matrix) => {
                distance_matrix.get(node_index(u), node_index(v))
            }
            DistanceMatrixType::Sub(distance_matrix) => {
                distance_matrix.get(node_index(u), node_index(v))
            }
        }
    }

    pub fn set(&mut self, u: usize, v: usize, d: f32) -> Option<()> {
        match self.distance_matrix_mut() {
            DistanceMatrixType::Full(distance_matrix) => {
                distance_matrix.set(node_index(u), node_index(v), d)
            }
            DistanceMatrixType::Sub(distance_matrix) => {
                distance_matrix.set(node_index(u), node_index(v), d)
            }
        }
    }
}

pub fn register(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyDistanceMatrix>()?;
    Ok(())
}
