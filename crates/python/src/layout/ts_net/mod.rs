//! tsNET and accelerated tsNET layout algorithm bindings for Python.

mod barnes_hut;
mod exact;
mod fit;
mod linear_fft;

use pyo3::prelude::*;

/// Registers TsNet, TsNetBuilder, BhTsNet, BhTsNetBuilder, FitTsNet, FitTsNetBuilder, LTsNet, and LTsNetBuilder classes with the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    exact::register(m)?;
    barnes_hut::register(m)?;
    fit::register(m)?;
    linear_fft::register(m)?;
    Ok(())
}
