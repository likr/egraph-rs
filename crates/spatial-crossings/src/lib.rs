//! Sweep-line edge-crossing detection and torus segment geometry for `egraph-rs`.

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
