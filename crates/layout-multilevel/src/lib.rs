//! Multilevel coarse backbone, GRIP, and space-filling curve graph layout algorithms for `egraph-rs`.

#![warn(missing_docs)]

mod coarse_backbone;
#[cfg(feature = "baselines")]
mod grip;
#[cfg(feature = "baselines")]
mod sfc;

pub use coarse_backbone::{
    CoarseBackbone, CoarseBackboneConfig, CoarseBackboneResult, CoarseBackboneStats,
    CoarseDistanceBackend, CoarseEdgeLengthMode,
};
#[cfg(feature = "baselines")]
pub use grip::{GripConfig, GripSolver};
#[cfg(feature = "baselines")]
pub use sfc::{SfcConfig, SfcCurveKind, SfcLayout};
