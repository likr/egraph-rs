//! Internal target-and-strategy dispatch helpers for `StressMajorizationSolver`.

use egraph::core::{
    CoreResult, IterativeSolver, Layout2d, Layout3d, LayoutNd, ManifoldLayout2d, SolverStatus,
};
use egraph::layout::{EuclideanPcg, GeodesicGradient, StressMajorizationSolver};
use pyo3::prelude::*;

use super::layout_dispatch::{DualTargetSolver, SolverTargetKind, TargetLayout};
use crate::errors::{invalid_input_err, unsupported_operation_err};
use crate::ffi_convert::{ScalarDtype, normalize_enum_key};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum StressStrategyChoice {
    Auto,
    EuclideanPcg,
    GeodesicGradient,
}

fn parse_stress_strategy_token(raw: &str) -> PyResult<StressStrategyChoice> {
    match normalize_enum_key(raw).as_str() {
        "auto" => Ok(StressStrategyChoice::Auto),
        "euclideanpcg" | "pcg" | "euclidean" => Ok(StressStrategyChoice::EuclideanPcg),
        "geodesicgradient" | "geodesic" | "riemannian" => {
            Ok(StressStrategyChoice::GeodesicGradient)
        }
        _ => Err(unsupported_operation_err(format!(
            "unknown StressMajorizationSolver backend/strategy {raw:?}; expected 'auto', 'euclidean_pcg', or 'geodesic_gradient'"
        ))),
    }
}

pub(super) fn resolve_stress_strategy_pair(
    target: SolverTargetKind,
    strategy: Option<&str>,
    backend: Option<&str>,
) -> PyResult<StressStrategyChoice> {
    let s_choice = strategy.map(parse_stress_strategy_token).transpose()?;
    let b_choice = backend.map(parse_stress_strategy_token).transpose()?;
    let parsed = match (s_choice, b_choice) {
        (Some(s), Some(b)) if s != b => {
            return Err(invalid_input_err(format!(
                "conflicting options: strategy={:?} cannot be combined with backend={:?}",
                strategy.unwrap_or_default(),
                backend.unwrap_or_default(),
            )));
        }
        (Some(s), _) => s,
        (None, Some(b)) => b,
        (None, None) => StressStrategyChoice::Auto,
    };
    match (target, parsed) {
        (SolverTargetKind::Manifold2d, StressStrategyChoice::EuclideanPcg) => {
            Err(unsupported_operation_err(
                "EuclideanPcg backend does not support target='manifold2d'; use backend='geodesic_gradient' (or 'auto') or target='layout2d'/'layout3d'/'layout_nd'",
            ))
        }
        (SolverTargetKind::Manifold2d, StressStrategyChoice::Auto) => {
            Ok(StressStrategyChoice::GeodesicGradient)
        }
        (_, StressStrategyChoice::Auto) => Ok(StressStrategyChoice::EuclideanPcg),
        (_, choice) => Ok(choice),
    }
}

pub(super) enum StressMajorizationInner {
    L2PcgF64(StressMajorizationSolver<Layout2d<f64>, EuclideanPcg>),
    L2PcgF32(StressMajorizationSolver<Layout2d<f32>, EuclideanPcg>),
    L3PcgF64(StressMajorizationSolver<Layout3d<f64>, EuclideanPcg>),
    L3PcgF32(StressMajorizationSolver<Layout3d<f32>, EuclideanPcg>),
    NdPcgF64(StressMajorizationSolver<LayoutNd<f64>, EuclideanPcg>),
    NdPcgF32(StressMajorizationSolver<LayoutNd<f32>, EuclideanPcg>),
    L2GeoF64(StressMajorizationSolver<Layout2d<f64>, GeodesicGradient>),
    L2GeoF32(StressMajorizationSolver<Layout2d<f32>, GeodesicGradient>),
    L3GeoF64(StressMajorizationSolver<Layout3d<f64>, GeodesicGradient>),
    L3GeoF32(StressMajorizationSolver<Layout3d<f32>, GeodesicGradient>),
    NdGeoF64(StressMajorizationSolver<LayoutNd<f64>, GeodesicGradient>),
    NdGeoF32(StressMajorizationSolver<LayoutNd<f32>, GeodesicGradient>),
    M2GeoF64(StressMajorizationSolver<ManifoldLayout2d<f64>, GeodesicGradient>),
    M2GeoF32(StressMajorizationSolver<ManifoldLayout2d<f32>, GeodesicGradient>),
}

macro_rules! match_stress_inner {
    ($self:expr, $s:ident => $body:expr) => {
        match $self {
            StressMajorizationInner::L2PcgF64($s) => $body,
            StressMajorizationInner::L2PcgF32($s) => $body,
            StressMajorizationInner::L3PcgF64($s) => $body,
            StressMajorizationInner::L3PcgF32($s) => $body,
            StressMajorizationInner::NdPcgF64($s) => $body,
            StressMajorizationInner::NdPcgF32($s) => $body,
            StressMajorizationInner::L2GeoF64($s) => $body,
            StressMajorizationInner::L2GeoF32($s) => $body,
            StressMajorizationInner::L3GeoF64($s) => $body,
            StressMajorizationInner::L3GeoF32($s) => $body,
            StressMajorizationInner::NdGeoF64($s) => $body,
            StressMajorizationInner::NdGeoF32($s) => $body,
            StressMajorizationInner::M2GeoF64($s) => $body,
            StressMajorizationInner::M2GeoF32($s) => $body,
        }
    };
}

macro_rules! match_stress_target {
    ($self:expr, $layout:expr, ($s:ident, $l:ident) => $body:expr) => {
        match ($self, $layout) {
            (
                StressMajorizationInner::L2PcgF64($s),
                $crate::solvers::target_layout::TargetLayout::L2F64($l),
            ) => $body,
            (
                StressMajorizationInner::L2PcgF32($s),
                $crate::solvers::target_layout::TargetLayout::L2F32($l),
            ) => $body,
            (
                StressMajorizationInner::L3PcgF64($s),
                $crate::solvers::target_layout::TargetLayout::L3F64($l),
            ) => $body,
            (
                StressMajorizationInner::L3PcgF32($s),
                $crate::solvers::target_layout::TargetLayout::L3F32($l),
            ) => $body,
            (
                StressMajorizationInner::NdPcgF64($s),
                $crate::solvers::target_layout::TargetLayout::NdF64($l),
            ) => $body,
            (
                StressMajorizationInner::NdPcgF32($s),
                $crate::solvers::target_layout::TargetLayout::NdF32($l),
            ) => $body,
            (
                StressMajorizationInner::L2GeoF64($s),
                $crate::solvers::target_layout::TargetLayout::L2F64($l),
            ) => $body,
            (
                StressMajorizationInner::L2GeoF32($s),
                $crate::solvers::target_layout::TargetLayout::L2F32($l),
            ) => $body,
            (
                StressMajorizationInner::L3GeoF64($s),
                $crate::solvers::target_layout::TargetLayout::L3F64($l),
            ) => $body,
            (
                StressMajorizationInner::L3GeoF32($s),
                $crate::solvers::target_layout::TargetLayout::L3F32($l),
            ) => $body,
            (
                StressMajorizationInner::NdGeoF64($s),
                $crate::solvers::target_layout::TargetLayout::NdF64($l),
            ) => $body,
            (
                StressMajorizationInner::NdGeoF32($s),
                $crate::solvers::target_layout::TargetLayout::NdF32($l),
            ) => $body,
            (
                StressMajorizationInner::M2GeoF64($s),
                $crate::solvers::target_layout::TargetLayout::M2F64($l),
            ) => $body,
            (
                StressMajorizationInner::M2GeoF32($s),
                $crate::solvers::target_layout::TargetLayout::M2F32($l),
            ) => $body,
            _ => Err(egraph::core::CoreError::invalid_input(
                "internal solver target/dtype mismatch",
            )),
        }
    };
}

pub(super) use {match_stress_inner, match_stress_target};

impl DualTargetSolver for StressMajorizationInner {
    fn target_kind(&self) -> SolverTargetKind {
        match self {
            Self::L2PcgF64(_) | Self::L2PcgF32(_) | Self::L2GeoF64(_) | Self::L2GeoF32(_) => {
                SolverTargetKind::Layout2d
            }
            Self::L3PcgF64(_) | Self::L3PcgF32(_) | Self::L3GeoF64(_) | Self::L3GeoF32(_) => {
                SolverTargetKind::Layout3d
            }
            Self::NdPcgF64(_) | Self::NdPcgF32(_) | Self::NdGeoF64(_) | Self::NdGeoF32(_) => {
                SolverTargetKind::LayoutNd
            }
            Self::M2GeoF64(_) | Self::M2GeoF32(_) => SolverTargetKind::Manifold2d,
        }
    }

    fn dtype(&self) -> ScalarDtype {
        match self {
            Self::L2PcgF64(_)
            | Self::L3PcgF64(_)
            | Self::NdPcgF64(_)
            | Self::L2GeoF64(_)
            | Self::L3GeoF64(_)
            | Self::NdGeoF64(_)
            | Self::M2GeoF64(_) => ScalarDtype::F64,
            _ => ScalarDtype::F32,
        }
    }

    fn solver_status(&self) -> SolverStatus {
        match_stress_inner!(self, s => IterativeSolver::status(s))
    }

    fn solver_iteration(&self) -> usize {
        match_stress_inner!(self, s => IterativeSolver::iteration(s))
    }

    fn solver_total_iterations(&self) -> Option<usize> {
        match_stress_inner!(self, s => IterativeSolver::total_iterations(s))
    }

    fn validate_raw(&self, layout: &TargetLayout) -> CoreResult<()> {
        match_stress_target!(self, layout, (s, l) => IterativeSolver::validate_target(s, l))
    }

    fn step_raw(&mut self, layout: &mut TargetLayout) -> CoreResult<SolverStatus> {
        match_stress_target!(self, layout, (s, l) => IterativeSolver::step(s, l))
    }

    fn run_steps_raw(
        &mut self,
        layout: &mut TargetLayout,
        steps: usize,
    ) -> CoreResult<SolverStatus> {
        match_stress_target!(self, layout, (s, l) => IterativeSolver::run_steps(s, l, steps))
    }

    fn run_raw(&mut self, layout: &mut TargetLayout) -> CoreResult<SolverStatus> {
        match_stress_target!(self, layout, (s, l) => IterativeSolver::run(s, l))
    }
}
