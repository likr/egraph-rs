use pyo3::prelude::*;

mod algorithm;
mod coordinates;
mod distance_matrix;
mod drawing;
mod graph;
mod layout;
mod quality_metrics;
mod rng;

#[pymodule]
fn egraph(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    algorithm::register(py, m)?;
    coordinates::register(py, m)?;
    distance_matrix::register(py, m)?;
    drawing::register(py, m)?;
    graph::register(py, m)?;
    layout::register(py, m)?;
    quality_metrics::register(py, m)?;
    rng::register(py, m)?;
    Ok(())
}
