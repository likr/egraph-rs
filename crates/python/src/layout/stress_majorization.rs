use crate::{coordinates::PyCoordinates, distance_matrix::PyDistanceMatrix, graph::PyGraph};
use petgraph::visit::EdgeRef;
use petgraph_layout_stress_majorization::StressMajorization;
use pyo3::{prelude::*, types::PyType};
use std::collections::HashMap;

#[pyclass]
#[pyo3(name = "StressMajorization")]
struct PyStressMajorization {
    stress_majorization: StressMajorization,
}

#[pymethods]
impl PyStressMajorization {
    #[new]
    fn new(graph: &PyGraph, coordinates: &PyCoordinates, f: &PyAny) -> PyStressMajorization {
        let mut distance = HashMap::new();
        for e in graph.graph().edge_indices() {
            let v = f.call1((e.index(),)).unwrap().extract().unwrap();
            distance.insert(e, v);
        }
        PyStressMajorization {
            stress_majorization: StressMajorization::new(
                graph.graph(),
                coordinates.coordinates(),
                &mut |e| distance[&e.id()],
            ),
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
