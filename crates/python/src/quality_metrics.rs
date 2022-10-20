use crate::{coordinates::PyCoordinates, distance_matrix::PyDistanceMatrix};
use petgraph_quality_metrics::stress;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(name = "stress")]
fn py_stress(coordinates: &PyCoordinates, distance_matrix: &PyDistanceMatrix) -> f32 {
    stress(
        &coordinates.coordinates(),
        distance_matrix.distance_matrix(),
    )
}

pub fn register(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_stress, m)?)?;
    Ok(())
}
