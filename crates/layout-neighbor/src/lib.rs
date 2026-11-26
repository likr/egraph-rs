//! Neighborhood-preserving tsNET, GuMap, and FFT-accelerated field layout solvers for `egraph-rs`.

pub mod field_layout;
pub mod gumap;
pub mod ts_net;
mod validation;

pub use field_layout::{
    FieldLayout, FieldLayoutSolver, FieldRepulsionBackend, IbUmapConfig, RhoSchedule, TFdpConfig,
};
pub use gumap::{GuMapConfig, GuMapSolver};
pub use ts_net::{TsNetBackend, TsNetConfig, TsNetSolver, TsNetStage};

/// Returns the crate version.
pub fn crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::crate_version;

    #[test]
    fn layout_neighbor_crate_version_non_empty() {
        assert!(!crate_version().is_empty());
    }
}
