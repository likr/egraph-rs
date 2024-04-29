use crate::{
    distance_matrix::{DistanceMatrixType, PyDistanceMatrix},
    drawing::{DrawingType, PyDrawing},
    graph::{GraphType, PyGraphAdapter},
};
use petgraph::visit::EdgeRef;
use petgraph_layout_stress_majorization::StressMajorization;
use pyo3::{prelude::*, types::PyType};

#[pyclass]
#[pyo3(name = "StressMajorization")]
struct PyStressMajorization {
    stress_majorization: StressMajorization,
}

#[pymethods]
impl PyStressMajorization {
    #[new]
    fn new(graph: &PyGraphAdapter, drawing: &PyDrawing, f: &PyAny) -> PyStressMajorization {
        match drawing.drawing() {
            DrawingType::Euclidean2d(drawing) => PyStressMajorization {
                stress_majorization: match graph.graph() {
                    GraphType::Graph(native_graph) => {
                        StressMajorization::new(native_graph, drawing, |e| {
                            f.call1((e.id().index(),)).unwrap().extract().unwrap()
                        })
                    }
                    _ => panic!("unsupported graph type"),
                },
            },
            _ => unimplemented!(),
        }
    }

    #[classmethod]
    fn with_distance_matrix(
        _cls: &PyType,
        drawing: &PyDrawing,
        distance_matrix: &PyDistanceMatrix,
    ) -> PyStressMajorization {
        match distance_matrix.distance_matrix() {
            DistanceMatrixType::Full(distance_matrix) => match drawing.drawing() {
                DrawingType::Euclidean2d(drawing) => PyStressMajorization {
                    stress_majorization: StressMajorization::new_with_distance_matrix(
                        drawing,
                        distance_matrix,
                    ),
                },
                _ => unimplemented!(),
            },
            _ => unimplemented!(),
        }
    }

    fn apply(&mut self, drawing: &mut PyDrawing) -> f32 {
        match drawing.drawing_mut() {
            DrawingType::Euclidean2d(drawing) => self.stress_majorization.apply(drawing),
            _ => unimplemented!(),
        }
    }

    pub fn run(&mut self, drawing: &mut PyDrawing) {
        match drawing.drawing_mut() {
            DrawingType::Euclidean2d(drawing) => self.stress_majorization.run(drawing),
            _ => unimplemented!(),
        }
    }

    pub fn update_weight(&mut self, f: &PyAny) {
        self.stress_majorization
            .update_weight(|i, j, dij, wij| f.call1((i, j, dij, wij)).unwrap().extract().unwrap())
    }
}

pub fn register(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyStressMajorization>()?;
    Ok(())
}
