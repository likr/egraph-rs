use pyo3::prelude::*;
mod kamada_kawai;

pub fn register(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    kamada_kawai::register(py, m)?;
    Ok(())
}
