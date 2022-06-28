use pyo3::prelude::*;

mod coordinates;
mod graph;
mod layout;
mod rng;

#[pymodule]
fn egraph(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    coordinates::register(py, m)?;
    graph::register(py, m)?;
    layout::register(py, m)?;
    rng::register(py, m)?;
    Ok(())
}
