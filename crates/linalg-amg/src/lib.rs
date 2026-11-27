//! Algebraic multigrid preconditioned conjugate gradient solvers and effective resistance for `egraph-rs`.

#![warn(missing_docs)]

mod amg_pcg;
mod resistance_curvature;

pub use amg_pcg::{
    AmgPcgBatchResult, AmgPcgConfig, AmgPcgSolveResult, AmgPcgSolver, AmgPcgWorkspace,
};
pub use resistance_curvature::{
    EffectiveResistanceConfig, EffectiveResistanceResult, ResistanceCurvatureConfig,
    ResistanceCurvatureFlowResult, ResistanceCurvatureStep, SuggestedGeometry,
    classify_geometry_from_curvature, classify_geometry_from_curvature_with_sampling,
    compute_effective_resistances, compute_effective_resistances_with_operator,
    resistance_curvature_flow,
};
