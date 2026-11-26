//! Layout quality metrics bindings backed by `egraph-metrics`.

mod cluster_baselines;
mod cluster_eval;
mod dispatcher;
mod neighborhood_geometry;
mod stress_crossings;

#[cfg(test)]
mod tests;

use pyo3::prelude::*;

/// Register layout quality metric functions on the Python extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    stress_crossings::register(m)?;
    neighborhood_geometry::register(m)?;
    cluster_eval::register(m)?;
    cluster_baselines::register(m)?;
    dispatcher::register(m)?;
    Ok(())
}
