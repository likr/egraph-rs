//! Backend & configuration builders for `TFdpSolver`.

use egraph::core::{
    CoreError, CoreResult, IterativeSolver, Layout2d, Layout3d, LayoutNd, SolverStatus,
};
use egraph::layout::neighbor::{BarnesHut as NeighborBarnesHut, Exact as NeighborExact};
use egraph::layout::{IbFft, RhoSchedule, TFdpBackend2d, TFdpBackend3d, TFdpConfig, TFdpSolver};
use egraph::spatial::IbFftConfig;
use pyo3::prelude::*;

use super::fit_field_stats::build_ibfft_config;
use super::layout_dispatch::{DualTargetSolver, SolverTargetKind, TargetLayout};
use crate::errors::{core_error_to_py, unsupported_operation_err};
use crate::ffi_convert::{ScalarDtype, normalize_enum_key};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum TFdpBackendChoice {
    Auto,
    IbFft,
    BarnesHut,
    Exact,
}

impl TFdpBackendChoice {
    pub(super) fn parse_str(raw: &str) -> Result<Self, String> {
        match normalize_enum_key(raw).as_str() {
            "auto" => Ok(Self::Auto),
            "ibfft" | "fft" => Ok(Self::IbFft),
            "barneshut" | "bh" => Ok(Self::BarnesHut),
            "exact" => Ok(Self::Exact),
            _ => Err(format!(
                "unknown TFdpSolver backend {raw:?}; expected 'auto', 'ib_fft', 'barnes_hut', or 'exact'"
            )),
        }
    }

    pub(super) fn parse(raw: &str) -> PyResult<Self> {
        Self::parse_str(raw).map_err(unsupported_operation_err)
    }
}

pub(super) fn build_t_fdp_backend_2d(
    choice: TFdpBackendChoice,
    theta: f64,
    ibfft: IbFftConfig,
) -> PyResult<TFdpBackend2d> {
    NeighborBarnesHut::new(theta)
        .validate()
        .map_err(core_error_to_py)?;
    ibfft.validate().map_err(core_error_to_py)?;
    Ok(match choice {
        TFdpBackendChoice::Auto | TFdpBackendChoice::IbFft => {
            TFdpBackend2d::IbFft(IbFft::new().with_ibfft(ibfft))
        }
        TFdpBackendChoice::BarnesHut => TFdpBackend2d::BarnesHut(NeighborBarnesHut::new(theta)),
        TFdpBackendChoice::Exact => TFdpBackend2d::Exact(NeighborExact::new()),
    })
}

pub(super) enum ResolvedTFdpBackend {
    L2(TFdpBackend2d),
    L3(TFdpBackend3d),
    Nd(NeighborExact),
}

pub(super) enum TFdpInner {
    L2F64(TFdpSolver<Layout2d<f64>, TFdpBackend2d>),
    L2F32(TFdpSolver<Layout2d<f32>, TFdpBackend2d>),
    L3F64(TFdpSolver<Layout3d<f64>, TFdpBackend3d>),
    L3F32(TFdpSolver<Layout3d<f32>, TFdpBackend3d>),
    NdF64(TFdpSolver<LayoutNd<f64>, NeighborExact>),
    NdF32(TFdpSolver<LayoutNd<f32>, NeighborExact>),
}

impl DualTargetSolver for TFdpInner {
    fn target_kind(&self) -> SolverTargetKind {
        match self {
            Self::L2F64(_) | Self::L2F32(_) => SolverTargetKind::Layout2d,
            Self::L3F64(_) | Self::L3F32(_) => SolverTargetKind::Layout3d,
            Self::NdF64(_) | Self::NdF32(_) => SolverTargetKind::LayoutNd,
        }
    }

    fn dtype(&self) -> ScalarDtype {
        match self {
            Self::L2F64(_) | Self::L3F64(_) | Self::NdF64(_) => ScalarDtype::F64,
            Self::L2F32(_) | Self::L3F32(_) | Self::NdF32(_) => ScalarDtype::F32,
        }
    }

    fn solver_status(&self) -> SolverStatus {
        match self {
            Self::L2F64(s) => IterativeSolver::status(s),
            Self::L2F32(s) => IterativeSolver::status(s),
            Self::L3F64(s) => IterativeSolver::status(s),
            Self::L3F32(s) => IterativeSolver::status(s),
            Self::NdF64(s) => IterativeSolver::status(s),
            Self::NdF32(s) => IterativeSolver::status(s),
        }
    }

    fn solver_iteration(&self) -> usize {
        match self {
            Self::L2F64(s) => IterativeSolver::iteration(s),
            Self::L2F32(s) => IterativeSolver::iteration(s),
            Self::L3F64(s) => IterativeSolver::iteration(s),
            Self::L3F32(s) => IterativeSolver::iteration(s),
            Self::NdF64(s) => IterativeSolver::iteration(s),
            Self::NdF32(s) => IterativeSolver::iteration(s),
        }
    }

    fn solver_total_iterations(&self) -> Option<usize> {
        match self {
            Self::L2F64(s) => IterativeSolver::total_iterations(s),
            Self::L2F32(s) => IterativeSolver::total_iterations(s),
            Self::L3F64(s) => IterativeSolver::total_iterations(s),
            Self::L3F32(s) => IterativeSolver::total_iterations(s),
            Self::NdF64(s) => IterativeSolver::total_iterations(s),
            Self::NdF32(s) => IterativeSolver::total_iterations(s),
        }
    }

    fn validate_raw(&self, layout: &TargetLayout) -> CoreResult<()> {
        match (self, layout) {
            (Self::L2F64(s), TargetLayout::L2F64(l)) => IterativeSolver::validate_target(s, l),
            (Self::L2F32(s), TargetLayout::L2F32(l)) => IterativeSolver::validate_target(s, l),
            (Self::L3F64(s), TargetLayout::L3F64(l)) => IterativeSolver::validate_target(s, l),
            (Self::L3F32(s), TargetLayout::L3F32(l)) => IterativeSolver::validate_target(s, l),
            (Self::NdF64(s), TargetLayout::NdF64(l)) => IterativeSolver::validate_target(s, l),
            (Self::NdF32(s), TargetLayout::NdF32(l)) => IterativeSolver::validate_target(s, l),
            _ => Err(CoreError::invalid_input(
                "internal solver target/dtype mismatch",
            )),
        }
    }

    fn step_raw(&mut self, layout: &mut TargetLayout) -> CoreResult<SolverStatus> {
        match (self, layout) {
            (Self::L2F64(s), TargetLayout::L2F64(l)) => IterativeSolver::step(s, l),
            (Self::L2F32(s), TargetLayout::L2F32(l)) => IterativeSolver::step(s, l),
            (Self::L3F64(s), TargetLayout::L3F64(l)) => IterativeSolver::step(s, l),
            (Self::L3F32(s), TargetLayout::L3F32(l)) => IterativeSolver::step(s, l),
            (Self::NdF64(s), TargetLayout::NdF64(l)) => IterativeSolver::step(s, l),
            (Self::NdF32(s), TargetLayout::NdF32(l)) => IterativeSolver::step(s, l),
            _ => Err(CoreError::invalid_input(
                "internal solver target/dtype mismatch",
            )),
        }
    }

    fn run_steps_raw(
        &mut self,
        layout: &mut TargetLayout,
        steps: usize,
    ) -> CoreResult<SolverStatus> {
        match (self, layout) {
            (Self::L2F64(s), TargetLayout::L2F64(l)) => IterativeSolver::run_steps(s, l, steps),
            (Self::L2F32(s), TargetLayout::L2F32(l)) => IterativeSolver::run_steps(s, l, steps),
            (Self::L3F64(s), TargetLayout::L3F64(l)) => IterativeSolver::run_steps(s, l, steps),
            (Self::L3F32(s), TargetLayout::L3F32(l)) => IterativeSolver::run_steps(s, l, steps),
            (Self::NdF64(s), TargetLayout::NdF64(l)) => IterativeSolver::run_steps(s, l, steps),
            (Self::NdF32(s), TargetLayout::NdF32(l)) => IterativeSolver::run_steps(s, l, steps),
            _ => Err(CoreError::invalid_input(
                "internal solver target/dtype mismatch",
            )),
        }
    }

    fn run_raw(&mut self, layout: &mut TargetLayout) -> CoreResult<SolverStatus> {
        match (self, layout) {
            (Self::L2F64(s), TargetLayout::L2F64(l)) => IterativeSolver::run(s, l),
            (Self::L2F32(s), TargetLayout::L2F32(l)) => IterativeSolver::run(s, l),
            (Self::L3F64(s), TargetLayout::L3F64(l)) => IterativeSolver::run(s, l),
            (Self::L3F32(s), TargetLayout::L3F32(l)) => IterativeSolver::run(s, l),
            (Self::NdF64(s), TargetLayout::NdF64(l)) => IterativeSolver::run(s, l),
            (Self::NdF32(s), TargetLayout::NdF32(l)) => IterativeSolver::run(s, l),
            _ => Err(CoreError::invalid_input(
                "internal solver target/dtype mismatch",
            )),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_t_fdp_config_and_backend(
    target: &str,
    backend: &str,
    iterations: usize,
    learning_rate: f64,
    momentum: f64,
    theta: f64,
    alpha: f64,
    gamma: f64,
    rho_start: f64,
    rho_end: f64,
    rho_power: f64,
    interpolation_order: usize,
    intervals: usize,
    min_grid_dim: usize,
    max_grid_dim: usize,
    seed: u64,
) -> PyResult<(TFdpConfig, ResolvedTFdpBackend)> {
    let target_kind = SolverTargetKind::from_str(target)?;
    let choice = TFdpBackendChoice::parse(backend)?;
    NeighborBarnesHut::new(theta)
        .validate()
        .map_err(core_error_to_py)?;
    let ibfft = build_ibfft_config(interpolation_order, intervals, min_grid_dim, max_grid_dim)?;
    let resolved = match target_kind {
        SolverTargetKind::Layout2d => match choice {
            TFdpBackendChoice::Auto | TFdpBackendChoice::IbFft => {
                ResolvedTFdpBackend::L2(TFdpBackend2d::IbFft(IbFft::new().with_ibfft(ibfft)))
            }
            TFdpBackendChoice::BarnesHut => {
                ResolvedTFdpBackend::L2(TFdpBackend2d::BarnesHut(NeighborBarnesHut::new(theta)))
            }
            TFdpBackendChoice::Exact => {
                ResolvedTFdpBackend::L2(TFdpBackend2d::Exact(NeighborExact::new()))
            }
        },
        SolverTargetKind::Layout3d => match choice {
            TFdpBackendChoice::Auto | TFdpBackendChoice::BarnesHut => {
                ResolvedTFdpBackend::L3(TFdpBackend3d::BarnesHut(NeighborBarnesHut::new(theta)))
            }
            TFdpBackendChoice::Exact => {
                ResolvedTFdpBackend::L3(TFdpBackend3d::Exact(NeighborExact::new()))
            }
            TFdpBackendChoice::IbFft => {
                return Err(unsupported_operation_err(
                    "TFdpSolver backend='ib_fft' only supports target='layout2d'; use 'auto', 'barnes_hut', or 'exact' for target='layout3d'",
                ));
            }
        },
        SolverTargetKind::LayoutNd => match choice {
            TFdpBackendChoice::Auto | TFdpBackendChoice::Exact => {
                ResolvedTFdpBackend::Nd(NeighborExact::new())
            }
            TFdpBackendChoice::IbFft | TFdpBackendChoice::BarnesHut => {
                return Err(unsupported_operation_err(
                    "TFdpSolver with target='layout_nd' only supports 'exact' (or 'auto') repulsion; use target='layout2d' or target='layout3d' for spatial/FFT backends",
                ));
            }
        },
        SolverTargetKind::Manifold2d => {
            return Err(unsupported_operation_err(
                "TFdpSolver only supports Euclidean targets ('layout2d', 'layout3d', 'layout_nd'), not 'manifold2d'",
            ));
        }
    };
    let rho = RhoSchedule::new()
        .with_start_weight(rho_start)
        .with_end_weight(rho_end)
        .with_power(rho_power);
    let cfg = TFdpConfig::new()
        .with_iterations(iterations)
        .with_learning_rate(learning_rate)
        .with_momentum(momentum)
        .with_alpha(alpha)
        .with_gamma(gamma)
        .with_rho(rho)
        .with_seed(seed);
    cfg.validate().map_err(core_error_to_py)?;
    Ok((cfg, resolved))
}
