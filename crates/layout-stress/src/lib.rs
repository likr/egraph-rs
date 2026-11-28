//! Stress majorization, Kamada-Kawai, and maxent-stress graph layout solvers for `egraph-rs`.

mod kamada_kawai;
mod manifold_utils;
#[cfg(feature = "baselines")]
mod maxent;
mod stress_majorization;

pub use kamada_kawai::{KamadaKawaiConfig, KamadaKawaiSolver};
#[cfg(feature = "baselines")]
pub use maxent::{MaxentStressConfig, MaxentStressSolver};
pub use stress_majorization::{PcgWorkspace, StressMajorizationConfig, StressMajorizationSolver};
