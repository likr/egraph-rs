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

#[pymethods]
impl PyDistanceMatrix {
    pub fn get(&self, u: usize, v: usize) -> f32 {
        self.distance_matrix[[u, v]]
    }

    pub fn set(&mut self, u: usize, v: usize, d: f32) {
        self.distance_matrix[[u, v]] = d
    }
}

pub fn register(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyDistanceMatrix>()?;
    Ok(())
}
