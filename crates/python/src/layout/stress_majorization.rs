use crate::{
    distance_matrix::{DistanceMatrixType, PyDistanceMatrix},
    drawing::PyDrawingEuclidean2d,
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
    fn new(
        graph: &PyGraphAdapter,
        drawing: &PyDrawingEuclidean2d,
        f: &Bound<PyAny>,
    ) -> PyStressMajorization {
        PyStressMajorization {
            stress_majorization: match graph.graph() {
                GraphType::Graph(native_graph) => {
                    StressMajorization::new(native_graph, drawing.drawing(), |e| {
                        f.call1((e.id().index(),)).unwrap().extract().unwrap()
                    })
                }
                _ => panic!("unsupported graph type"),
            },
        }
    }

    #[classmethod]
    fn with_distance_matrix(
        _cls: &Bound<PyType>,
        drawing: &PyDrawingEuclidean2d,
        distance_matrix: &PyDistanceMatrix,
    ) -> PyStressMajorization {
        match distance_matrix.distance_matrix() {
            DistanceMatrixType::Full(distance_matrix) => PyStressMajorization {
                stress_majorization: StressMajorization::new_with_distance_matrix(
                    drawing.drawing(),
                    distance_matrix,
                ),
            },
            _ => unimplemented!(),
        }
    }

    fn apply(&mut self, drawing: &mut PyDrawingEuclidean2d) -> f32 {
        self.stress_majorization.apply(drawing.drawing_mut())
    }

    pub fn run(&mut self, drawing: &mut PyDrawingEuclidean2d) {
        self.stress_majorization.run(drawing.drawing_mut())
    }

    pub fn update_weight(&mut self, f: &Bound<PyAny>) {
        self.stress_majorization
            .update_weight(|i, j, dij, wij| f.call1((i, j, dij, wij)).unwrap().extract().unwrap())
    }
}

pub fn register(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyStressMajorization>()?;
    Ok(())
}
