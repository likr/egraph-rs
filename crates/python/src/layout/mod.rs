use pyo3::prelude::*;
mod kamada_kawai;
mod sgd;
mod stress_majorization;

pub fn register(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    kamada_kawai::register(py, m)?;
    sgd::register(py, m)?;
    stress_majorization::register(py, m)?;
    Ok(())
}
