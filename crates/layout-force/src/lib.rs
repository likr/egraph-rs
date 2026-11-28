//! Force-directed graph layout solvers for `egraph-rs`.

mod coarsening;
mod force_atlas2;
#[cfg(feature = "baselines")]
mod fr;
#[cfg(feature = "baselines")]
mod gem;
mod graph_edges;
#[cfg(feature = "baselines")]
mod linlog;
#[cfg(feature = "baselines")]
mod manifold_utils;
mod repulsion;
mod yifan_hu;

pub use force_atlas2::{ForceAtlas2Config, ForceAtlas2Solver};
#[cfg(feature = "baselines")]
pub use fr::{FrBackend, FruchtermanReingoldConfig, FruchtermanReingoldSolver};
#[cfg(feature = "baselines")]
pub use gem::{GemConfig, GemSolver};
#[cfg(feature = "baselines")]
pub use linlog::{LinLogConfig, LinLogRepulsion, LinLogSolver};
pub use repulsion::CoulombRepulsionBackend;
pub use yifan_hu::{YifanHuConfig, YifanHuSolver};
