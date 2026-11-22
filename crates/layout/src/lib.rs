//! Single-threaded, pure-Rust graph layout solvers and spatial refiners for `egraph-rs`.
//!
//! `egraph-layout` builds on `egraph-core`, `egraph-spatial`, `egraph-linalg`, and
//! `egraph-neighborhood` to provide cooperative `step()`/`run_steps(k)` layout execution across
//! Native Rust, Python (PyO3), and WebAssembly (`wasm32-unknown-unknown`) targets.

pub mod gumap;
pub mod kamada_kawai;
pub mod refiner;
pub mod sgd;
pub mod stress_majorization;
pub mod ts_net;

pub use gumap::{GuMapConfig, GuMapSolver};
pub use kamada_kawai::{KamadaKawaiConfig, KamadaKawaiSolver};
pub use refiner::{
    SpatialIntruderRefiner, SpatialRefinerConfig, SpatialRefinerRoundStats, SpatialRefinerStats,
};
pub use sgd::{
    BarrierSymmetry, SgdScheduleConfig, SgdScheduleKind, SgdSolver, SparseBarrierConfig,
    SparseBarrierStats, StreamingNegativeConfig,
};
pub use stress_majorization::{PcgWorkspace, StressMajorizationConfig, StressMajorizationSolver};
pub use ts_net::{TsNetBackend, TsNetConfig, TsNetSolver, TsNetStage};

/// Returns the crate version string.
pub fn crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::crate_version;

    #[test]
    fn m3_scaffold_ok() {
        assert!(!crate_version().is_empty());
    }
}
