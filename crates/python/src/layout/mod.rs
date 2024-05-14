mod kamada_kawai;
mod mds;
mod overwrap_removal;
mod sgd;
mod stress_majorization;

use pyo3::prelude::*;

pub fn register(py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    mds::register(py, m)?;
    kamada_kawai::register(py, m)?;
    overwrap_removal::register(py, m)?;
    stress_majorization::register(py, m)?;
    sgd::register(py, m)?;
    Ok(())
}
