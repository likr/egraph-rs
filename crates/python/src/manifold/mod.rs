//! Non-Euclidean manifold layout (`ManifoldLayout2d`) and geodesic helper bindings.

mod geodesic;
mod layout;

#[cfg(test)]
mod tests;

use pyo3::prelude::*;

pub(crate) use geodesic::{ExtractedManifold, extract_manifold_for_solver, write_back_manifold};
pub(crate) use layout::PyManifoldLayout2d;

/// Register `ManifoldLayout2d` and geodesic metric functions on `egraph._core`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyManifoldLayout2d>()?;
    geodesic::register(m)?;
    Ok(())
}
