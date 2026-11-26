//! Sweep-line edge-crossing detection and non-Euclidean geodesic crossing geometry for `egraph-rs`.

pub mod crossings;

pub use crossings::{
    CrossingPair, CrossingSummary, SweepLineCrossings, TorusCrossing, TorusLayoutView,
    crossing_edges_hyperbolic, crossing_edges_spherical, hyperbolic_crossing_number,
    sampled_crossing_number, sampled_crossing_number_edges, spherical_crossing_number,
    torus_crossing_edges, torus_crossing_edges_from_edges, torus_crossing_number,
    torus_edge_segments, torus_layout_edge_segments,
};

/// Returns the crate version.
pub fn crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::crate_version;

    #[test]
    fn spatial_crossings_crate_version_non_empty() {
        assert!(!crate_version().is_empty());
    }
}
