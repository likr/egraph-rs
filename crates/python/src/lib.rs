use pyo3::prelude::*;

mod algorithm;
mod distance_matrix;
mod drawing;
mod graph;
mod layout;
mod quality_metrics;
mod rng;

#[pymodule]
fn egraph(py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    graph::register(py, m)?;
    drawing::register(py, m)?;
    distance_matrix::register(py, m)?;
    rng::register(py, m)?;
    layout::register(py, m)?;
    algorithm::register(py, m)?;
    quality_metrics::register(py, m)?;
    Ok(())
}
