//! Non-Euclidean manifold layout (`ManifoldLayout2d`) and geodesic helper bindings.

mod crossings;
mod geodesic;
mod layout;
mod layout_ops;

#[cfg(test)]
mod tests;

use pyo3::prelude::*;

pub(crate) use crossings::TorusLayoutInput;
pub(crate) use geodesic::{
    ExtractedManifold, build_core_f32, build_core_f64, extract_manifold_for_solver,
    manifold_kind_str, write_back_manifold,
};
pub(crate) use layout::{ManifoldLayout2dInner, PyManifoldLayout2d};

/// Register `ManifoldLayout2d`, geodesic math, and manifold crossing functions on `egraph._core`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyManifoldLayout2d>()?;
    geodesic::register(m)?;
    crossings::register(m)?;
    Ok(())
}
