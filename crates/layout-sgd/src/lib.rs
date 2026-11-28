//! Stochastic gradient descent stress and sparse barrier graph layout solvers for `egraph-rs`.

mod sgd;

pub use sgd::{
    BarrierSymmetry, OrtmannPivotStrategy, OrtmannSparseStressConfig, Sgd2Config, Sgd2Solver,
    SgdPairBuffer, SgdScheduleConfig, SgdScheduleKind, SgdSolver, SparseBarrierConfig,
    SparseBarrierStats, StreamingNegativeConfig,
};
