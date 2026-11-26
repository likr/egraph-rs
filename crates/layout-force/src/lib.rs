//! Force-directed placement and edge bundling solvers for `egraph-rs`.

pub mod continuous_fdp;
pub mod fdeb;
#[cfg(feature = "baselines")]
pub mod fr;
#[cfg(feature = "baselines")]
pub mod gem;
#[cfg(feature = "baselines")]
pub mod linlog;
#[cfg(feature = "baselines")]
mod manifold_utils;

pub use continuous_fdp::{
    ContinuousFdp, ForceAtlas2Config, ForceAtlas2Solver, YifanHuConfig, YifanHuSolver,
};
pub use fdeb::{FdebConfig, FdebSolver};
#[cfg(feature = "baselines")]
pub use fr::{FrBackend, FruchtermanReingoldConfig, FruchtermanReingoldSolver};
#[cfg(feature = "baselines")]
pub use gem::{GemConfig, GemSolver};
#[cfg(feature = "baselines")]
pub use linlog::{LinLogConfig, LinLogRepulsion, LinLogSolver};

/// Returns the crate version.
pub fn crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::crate_version;

    #[test]
    fn layout_force_crate_version_non_empty() {
        assert!(!crate_version().is_empty());
    }
}
