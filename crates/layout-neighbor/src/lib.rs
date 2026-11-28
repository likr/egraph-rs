//! Neighborhood-preserving tsNET, GuMap, t-FDP, and ibUMAP graph layout solvers for `egraph-rs`.

mod backends;
mod field_layout;
mod gumap;
mod manifold_utils;
mod ts_net;
mod validation;

pub use backends::{
    Auto2d, Auto3d, BarnesHut, DEFAULT_AUTO_EXACT_THRESHOLD, Exact, FitFft, IbFft, LinearFft,
    TFdpBackend2d, TFdpBackend3d, TsNetBackend2d, TsNetBackend3d,
};
pub use field_layout::{
    DEFAULT_MAX_2HOP_PER_NODE, DefaultTFdpBackend, FieldFftStats, FieldRepulsionBackend,
    IbUmapConfig, IbUmapSolver, RhoSchedule, TFdpConfig, TFdpRepulsion, TFdpSolver,
};
#[doc(hidden)]
pub use field_layout::{
    ExactTFdpState, TFdpBackend2dState, TFdpBackend3dState, TFdpBarnesHut2dState,
    TFdpBarnesHut3dState, TFdpCore, TFdpIbFft2dState, TFdpWorkspace,
};
pub use gumap::{GuMapConfig, GuMapSolver, GuMapTarget};
#[doc(hidden)]
pub use gumap::{GuMapCore, GuMapWorkspace};
#[doc(hidden)]
pub use ts_net::{
    Auto2dState, Auto3dState, Backend2dState, Backend3dState, BarnesHut2dState, BarnesHut3dState,
    ExactTsNetState, FitFft2dState, TsNetCore, TsNetWorkspace,
};
pub use ts_net::{DefaultTsNetBackend, TsNetConfig, TsNetRepulsion, TsNetSolver, TsNetStage};
