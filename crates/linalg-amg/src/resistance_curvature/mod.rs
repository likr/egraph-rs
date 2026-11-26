//! Devriendt-Lambiotte effective resistance curvature and resistance curvature flow.
//!
//! Implements single-threaded, deterministic effective resistance estimation via
//! Spielman-Srivastava Johnson-Lindenstrauss projection solved with [`AmgPcgSolver`](crate::AmgPcgSolver)
//! (or exact Moore-Penrose pseudoinverse via [`symmetric_eigen_jacobi`](egraph_linalg::symmetric_eigen_jacobi) on small graphs),
//! Foster's theorem diagnostics, vertex and link resistance curvatures, geometry regime
//! classification, and multi-step resistance curvature flow.

mod config;
mod evaluation;
mod flow;
mod graph_ops;

pub use config::{
    EffectiveResistanceConfig, EffectiveResistanceResult, ResistanceCurvatureConfig,
    ResistanceCurvatureFlowResult, ResistanceCurvatureStep, SuggestedGeometry,
    classify_geometry_from_curvature,
};
pub use evaluation::compute_effective_resistances;
pub use flow::resistance_curvature_flow;

/// Hard upper bound on dense exact Jacobi eigendecomposition to avoid OOM.
pub(super) const MAX_EXACT_NODES: usize = 512;
pub(super) const EXACT_JACOBI_MAX_SWEEPS: usize = 120;
pub(super) const EXACT_JACOBI_TOL: f64 = 1e-12;
pub(super) const LINK_RESISTANCE_EPS: f64 = 1e-15;
pub(super) const TORUS_DEG_CV_THRESHOLD: f64 = 0.15;

#[cfg(test)]
mod tests;
