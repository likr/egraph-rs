//! Stochastic gradient descent stress and sparse barrier graph layout solvers for `egraph-rs`.

mod sgd;

pub use sgd::{
    BarrierSymmetry, OrtmannPivotStrategy, OrtmannSparseProblem, OrtmannSparseStressConfig,
    Sgd2Config, Sgd2Solver, SgdPairBuffer, SgdPairKind, SgdPairRecord, SgdScheduleConfig,
    SgdScheduleKind, SgdSolver, SparseBarrierConfig, SparseBarrierStats, StreamingNegativeConfig,
    build_ortmann_sparse_problem,
};
