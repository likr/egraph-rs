//! Spatial indexing trees, rectangle overlap sweeps, and optional Delaunay triangulation for `egraph-rs`.

/// Returns the crate version.
pub fn crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::crate_version;

    #[test]
    fn spatial_trees_crate_version_non_empty() {
        assert!(!crate_version().is_empty());
    }
}
