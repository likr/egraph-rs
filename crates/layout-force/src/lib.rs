//! Force-directed graph layout solvers for `egraph-rs`.

mod backend;
mod coarsening;
mod force_atlas2;
#[cfg(feature = "baselines")]
mod fr;
#[cfg(feature = "baselines")]
mod gem;
mod graph_edges;
#[cfg(feature = "baselines")]
mod linlog;
mod yifan_hu;

pub use backend::{BarnesHut, DefaultForceBackend, Exact, ForceBackend};
#[cfg(feature = "baselines")]
pub use backend::{DefaultFrBackend, FrBackend, Grid};
pub use force_atlas2::{
    ForceAtlas2Config, ForceAtlas2Solver, ForceRepulsion, ForceRepulsionParams,
};
#[cfg(feature = "baselines")]
pub use fr::{FrRepulsion, FruchtermanReingoldConfig, FruchtermanReingoldSolver};
#[cfg(feature = "baselines")]
pub use gem::{GemConfig, GemSolver};
#[cfg(feature = "baselines")]
pub use linlog::{LinLogConfig, LinLogRepulsion, LinLogRepulsionBackend, LinLogSolver};
pub use yifan_hu::{YifanHuConfig, YifanHuLevelCoords, YifanHuRepulsion, YifanHuSolver};
