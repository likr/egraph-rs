use pyo3::prelude::*;
mod kamada_kawai;

pub fn register(py: Python<'_>, p: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "layout")?;
    kamada_kawai::register(py, m)?;
    p.add_submodule(m)?;
    Ok(())
}
