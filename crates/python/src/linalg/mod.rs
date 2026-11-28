//! Linear algebra, spectral initialization, Chebyshev diffusion, shortest-path, and AMG-PCG bindings.

mod amg;
mod amg_helpers;
mod amg_results;
mod chebyshev;
mod chebyshev_kernel;
mod chebyshev_types;
mod curvature_flow;
mod hde;
mod matrix;
mod mds;
mod pivot_mds;
mod resistance;
mod shortest_path;
mod spectral;
mod spectral_generators;

#[cfg(test)]
mod tests;

pub(crate) use matrix::PyCsrMatrix;

use pyo3::prelude::*;

/// Register linear algebra, spectral, Chebyshev, shortest-path, and AMG-PCG bindings on `egraph._core`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    matrix::register(m)?;
    amg::register(m)?;
    amg_results::register(m)?;
    resistance::register(m)?;
    curvature_flow::register(m)?;
    spectral::register(m)?;
    spectral_generators::register(m)?;
    mds::register(m)?;
    pivot_mds::register(m)?;
    chebyshev::register(m)?;
    chebyshev_kernel::register(m)?;
    chebyshev_types::register(m)?;
    shortest_path::register(m)?;
    hde::register(m)?;
    Ok(())
}
