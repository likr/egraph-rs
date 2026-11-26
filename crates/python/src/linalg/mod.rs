//! Linear algebra, spectral initialization, Chebyshev diffusion, and AMG-PCG bindings.

mod amg;
mod amg_helpers;
mod chebyshev;
mod curvature_flow;
mod hde;
mod matrix;
mod resistance;
mod spectral;

#[cfg(test)]
mod tests;

pub(crate) use matrix::PyCsrMatrix;

use pyo3::prelude::*;

/// Register linear algebra, spectral, Chebyshev, and AMG-PCG bindings on `egraph._core`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    matrix::register(m)?;
    amg::register(m)?;
    resistance::register(m)?;
    curvature_flow::register(m)?;
    spectral::register(m)?;
    chebyshev::register(m)?;
    hde::register(m)?;
    Ok(())
}
