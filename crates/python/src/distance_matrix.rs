use ndarray::prelude::*;
use pyo3::prelude::*;

#[pyclass]
#[pyo3(name = "DistanceMatrix")]
pub struct PyDistanceMatrix {
    distance_matrix: Array2<f32>,
}

impl PyDistanceMatrix {
    pub fn new(distance_matrix: Array2<f32>) -> PyDistanceMatrix {
        PyDistanceMatrix { distance_matrix }
    }

    pub fn distance_matrix(&self) -> &Array2<f32> {
        &self.distance_matrix
    }
}

pub fn register(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyDistanceMatrix>()?;
    Ok(())
}
