use crate::{
    coordinates::PyCoordinates,
    distance_matrix::PyDistanceMatrix,
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
    fn new(graph: &PyGraphAdapter, coordinates: &PyCoordinates, f: &PyAny) -> PyStressMajorization {
        PyStressMajorization {
            stress_majorization: match graph.graph() {
                GraphType::Graph(native_graph) => {
                    StressMajorization::new(native_graph, coordinates.coordinates(), &mut |e| {
                        f.call1((e.id().index(),)).unwrap().extract().unwrap()
                    })
                }
                _ => panic!("unsupported graph type"),
            },
        }
    }

    #[classmethod]
    fn with_distance_matrix(
        _cls: &PyType,
        coordinates: &PyCoordinates,
        distance_matrix: &PyDistanceMatrix,
    ) -> PyStressMajorization {
        PyStressMajorization {
            stress_majorization: StressMajorization::new_with_distance_matrix(
                coordinates.coordinates(),
                distance_matrix.distance_matrix(),
            ),
        }
    }

    fn apply(&mut self, coordinates: &mut PyCoordinates) -> f32 {
        self.stress_majorization
            .apply(coordinates.coordinates_mut())
    }

    pub fn run(&mut self, coordinates: &mut PyCoordinates) {
        self.stress_majorization.run(coordinates.coordinates_mut());
    }
}

pub fn register(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyStressMajorization>()?;
    Ok(())
}
