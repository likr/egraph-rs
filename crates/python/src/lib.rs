use pyo3::prelude::*;

mod algorithm;
mod distance_matrix;
mod drawing;
mod drawing_torus;
mod graph;
mod layout;
mod quality_metrics;
mod rng;

#[pymodule]
fn egraph(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    graph::register(py, m)?;
    drawing::register(py, m)?;
    drawing_torus::register(py, m)?;
    distance_matrix::register(py, m)?;
    rng::register(py, m)?;
    layout::register(py, m)?;
    algorithm::register(py, m)?;
    quality_metrics::register(py, m)?;
    Ok(())
}
