//! Classical, Pivot, Landmark, Harel-Koren HDE, and Koren spectral multidimensional scaling for `egraph-rs`.

#![warn(missing_docs)]

pub mod mds;

pub use mds::{
    HdeConfig, PivotMdsConfig, RdMds, classical_mds, classical_mds_2d,
    classical_mds_from_distances, hde, hde_nd, hde_nd_with_pivots, hde_with_pivots,
    koren_spectral_init, normalize_layout_2d, pivot_mds, pivot_mds_from_distances,
    pivot_mds_with_pivots, rd_mds, representative_mds, robust_normalize_layout_2d,
};

/// Returns the crate version string.
pub fn crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::crate_version;

    #[test]
    fn linalg_mds_crate_version_non_empty() {
        assert!(!crate_version().is_empty());
    }
}
