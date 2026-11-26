//! Layout quality metrics bindings backed by `egraph-metrics`.

mod cluster_baselines;
mod cluster_eval;
mod dispatcher;
mod neighborhood_geometry;
mod stress_crossings;

#[cfg(test)]
mod tests;

pub(crate) use neighborhood_geometry::{
    csr_angular_resolution_impl, csr_aspect_ratio_impl, csr_gabriel_graph_property_impl,
    csr_ideal_edge_lengths_impl, csr_neighborhood_preservation_1hop_impl,
    csr_neighborhood_preservation_2hop_impl, csr_node_resolution_impl,
};
pub(crate) use stress_crossings::{
    csr_crossing_angle_impl, csr_crossing_edges_impl, csr_crossing_number_impl,
};

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
