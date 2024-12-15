mod shortest_path;
use pyo3::prelude::*;

pub fn register(py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    shortest_path::register(py, m)?;
    Ok(())
}
