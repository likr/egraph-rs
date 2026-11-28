//! Layout quality metrics bindings backed by `egraph-metrics`.

mod cluster_baselines;
mod cluster_baselines_nd;
mod cluster_eval;
mod cluster_eval_types;
mod cluster_result_types;
mod dispatcher;
mod geometry_eval;
mod nd_manifold_helpers;
mod neighborhood_geometry;
mod report;
mod report_fields;
mod result_types;
mod rta;
mod stress_crossings;
mod stress_eval;

#[cfg(test)]
mod tests;

use pyo3::prelude::*;

/// Register layout quality metric functions on the Python extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    result_types::register(m)?;
    cluster_result_types::register(m)?;
    stress_eval::register(m)?;
    stress_crossings::register(m)?;
    neighborhood_geometry::register(m)?;
    geometry_eval::register(m)?;
    cluster_eval::register(m)?;
    cluster_eval_types::register(m)?;
    rta::register(m)?;
    cluster_baselines::register(m)?;
    dispatcher::register(m)?;
    Ok(())
}
