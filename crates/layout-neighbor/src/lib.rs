//! Neighborhood-preserving tsNET, GuMap, t-FDP, and ibUMAP layout solvers for `egraph-rs`.

mod field_layout;
mod gumap;
mod manifold_utils;
mod ts_net;
mod validation;

pub use field_layout::{
    DEFAULT_MAX_2HOP_PER_NODE, FieldFftStats, FieldRepulsionBackend, IbUmapConfig, IbUmapSolver,
    RhoSchedule, TFdpConfig, TFdpSolver,
};
pub use gumap::{GuMapConfig, GuMapSolver};
pub use ts_net::{TsNetBackend, TsNetConfig, TsNetSolver, TsNetStage};
