//! Stress majorization, Kamada-Kawai, and maxent-stress graph layout solvers for `egraph-rs`.

#![cfg_attr(docsrs, feature(doc_cfg))]

mod kamada_kawai;
#[cfg(feature = "baselines")]
mod maxent;
mod stress_majorization;

pub use kamada_kawai::{KamadaKawaiConfig, KamadaKawaiSolver};
#[cfg(feature = "baselines")]
#[cfg_attr(docsrs, doc(cfg(feature = "baselines")))]
pub use maxent::{
    BarnesHut, DefaultMaxentBackend, Exact, MaxentRepulsion, MaxentStressConfig, MaxentStressSolver,
};
pub use stress_majorization::{
    DefaultStressStrategy, EuclideanPcg, GeodesicGradient, PcgWorkspace, StressMajorizationConfig,
    StressMajorizationSolver, StressMajorizationStrategy,
};
