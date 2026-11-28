//! Shared Option 1 target-backend dispatch, `FitFieldStats`, `TsNetStage`, and config builder for `TsNetSolver`.

use egraph::core::{
    CoreError, CoreResult, IterativeSolver, Layout2d, Layout3d, LayoutNd, ManifoldLayout2d,
    SolverStatus,
};
use egraph::layout::neighbor::{BarnesHut as NeighborBarnesHut, Exact as NeighborExact};
use egraph::layout::{
    Auto2d, Auto3d, FitFft, LinearFft, TsNetBackend2d, TsNetBackend3d, TsNetConfig, TsNetSolver,
    TsNetStage,
};
use egraph::neighborhood::{SymmetrizationMode, TwoHopSamplingMode};
use egraph::spatial::{IbFftConfig, InterpolationOrder};
use pyo3::prelude::*;

use crate::errors::{core_error_to_py, unsupported_operation_err};
use crate::ffi_convert::{ScalarDtype, normalize_enum_key};

use super::fit_field_stats::PyFitFieldStats;
use super::layout_dispatch::{DualTargetSolver, SolverTargetKind, TargetLayout};
use super::tsnet_builders::parse_stage_iterations;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum TsNetBackendChoice {
    Auto,
    Exact,
    BarnesHut,
    FitFft,
    LinearFft,
}

impl TsNetBackendChoice {
    pub(super) fn parse_str(raw: &str) -> Result<Self, String> {
        match normalize_enum_key(raw).as_str() {
            "auto" => Ok(Self::Auto),
            "exact" => Ok(Self::Exact),
            "barneshut" | "bh" => Ok(Self::BarnesHut),
            "fitfft" | "fft" | "fftinterpolation" | "fit" => Ok(Self::FitFft),
            "linearfft" | "l" => Ok(Self::LinearFft),
            _ => Err(format!(
                "unknown tsNET backend {raw:?}; expected 'auto', 'exact', 'barnes_hut', 'fit_fft', or 'linear_fft'"
            )),
        }
    }

    pub(super) fn parse(raw: &str) -> PyResult<Self> {
        Self::parse_str(raw).map_err(unsupported_operation_err)
    }
}

pub(super) enum ResolvedTsNetBackend {
    L2(TsNetBackend2d),
    L3(TsNetBackend3d),
    Nd(NeighborExact),
    M2(NeighborExact),
}

pub(super) fn validate_and_resolve_tsnet_backend(
    target: SolverTargetKind,
    choice: TsNetBackendChoice,
    theta: f64,
    ibfft: IbFftConfig,
) -> PyResult<ResolvedTsNetBackend> {
    NeighborBarnesHut::new(theta)
        .validate()
        .map_err(core_error_to_py)?;
    ibfft.validate().map_err(core_error_to_py)?;
    match target {
        SolverTargetKind::Layout2d => {
            let b = match choice {
                TsNetBackendChoice::Auto => TsNetBackend2d::Auto(Auto2d::new().with_ibfft(ibfft)),
                TsNetBackendChoice::Exact => TsNetBackend2d::Exact(NeighborExact::new()),
                TsNetBackendChoice::BarnesHut => {
                    TsNetBackend2d::BarnesHut(NeighborBarnesHut::new(theta))
                }
                TsNetBackendChoice::FitFft => {
                    TsNetBackend2d::FitFft(FitFft::new().with_ibfft(ibfft))
                }
                TsNetBackendChoice::LinearFft => {
                    TsNetBackend2d::LinearFft(LinearFft::new().with_ibfft(ibfft))
                }
            };
            Ok(ResolvedTsNetBackend::L2(b))
        }
        SolverTargetKind::Layout3d => {
            let b = match choice {
                TsNetBackendChoice::Auto => TsNetBackend3d::Auto(Auto3d::new().with_theta(theta)),
                TsNetBackendChoice::Exact => TsNetBackend3d::Exact(NeighborExact::new()),
                TsNetBackendChoice::BarnesHut => {
                    TsNetBackend3d::BarnesHut(NeighborBarnesHut::new(theta))
                }
                TsNetBackendChoice::FitFft | TsNetBackendChoice::LinearFft => {
                    return Err(unsupported_operation_err(
                        "FFT tsNET backends ('fit_fft'/'linear_fft') only support target='layout2d'; use 'auto', 'barnes_hut', or 'exact' for target='layout3d'",
                    ));
                }
            };
            Ok(ResolvedTsNetBackend::L3(b))
        }
        SolverTargetKind::LayoutNd => match choice {
            TsNetBackendChoice::Auto | TsNetBackendChoice::Exact => {
                Ok(ResolvedTsNetBackend::Nd(NeighborExact::new()))
            }
            TsNetBackendChoice::BarnesHut
            | TsNetBackendChoice::FitFft
            | TsNetBackendChoice::LinearFft => Err(unsupported_operation_err(
                "tsNET with target='layout_nd' only supports 'exact' (or 'auto') repulsion; use target='layout2d' or target='layout3d' for spatial/FFT approximations",
            )),
        },
        SolverTargetKind::Manifold2d => match choice {
            TsNetBackendChoice::Auto | TsNetBackendChoice::Exact => {
                Ok(ResolvedTsNetBackend::M2(NeighborExact::new()))
            }
            TsNetBackendChoice::BarnesHut
            | TsNetBackendChoice::FitFft
            | TsNetBackendChoice::LinearFft => Err(unsupported_operation_err(
                "tsNET with target='manifold2d' only supports 'exact' (or 'auto') geodesic repulsion; use target='layout2d' or target='layout3d' for spatial/FFT approximations",
            )),
        },
    }
}

pub(crate) enum TsNetInner {
    L2F64(TsNetSolver<Layout2d<f64>, TsNetBackend2d>),
    L2F32(TsNetSolver<Layout2d<f32>, TsNetBackend2d>),
    L3F64(TsNetSolver<Layout3d<f64>, TsNetBackend3d>),
    L3F32(TsNetSolver<Layout3d<f32>, TsNetBackend3d>),
    NdF64(TsNetSolver<LayoutNd<f64>, NeighborExact>),
    NdF32(TsNetSolver<LayoutNd<f32>, NeighborExact>),
    M2F64(TsNetSolver<ManifoldLayout2d<f64>, NeighborExact>),
    M2F32(TsNetSolver<ManifoldLayout2d<f32>, NeighborExact>),
}

impl DualTargetSolver for TsNetInner {
    fn target_kind(&self) -> SolverTargetKind {
        match self {
            Self::L2F64(_) | Self::L2F32(_) => SolverTargetKind::Layout2d,
            Self::L3F64(_) | Self::L3F32(_) => SolverTargetKind::Layout3d,
            Self::NdF64(_) | Self::NdF32(_) => SolverTargetKind::LayoutNd,
            Self::M2F64(_) | Self::M2F32(_) => SolverTargetKind::Manifold2d,
        }
    }
    fn dtype(&self) -> ScalarDtype {
        match self {
            Self::L2F64(_) | Self::L3F64(_) | Self::NdF64(_) | Self::M2F64(_) => ScalarDtype::F64,
            Self::L2F32(_) | Self::L3F32(_) | Self::NdF32(_) | Self::M2F32(_) => ScalarDtype::F32,
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
            Self::M2F64(s) => IterativeSolver::status(s),
            Self::M2F32(s) => IterativeSolver::status(s),
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
            Self::M2F64(s) => IterativeSolver::iteration(s),
            Self::M2F32(s) => IterativeSolver::iteration(s),
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
            Self::M2F64(s) => IterativeSolver::total_iterations(s),
            Self::M2F32(s) => IterativeSolver::total_iterations(s),
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
            (Self::M2F64(s), TargetLayout::M2F64(l)) => IterativeSolver::validate_target(s, l),
            (Self::M2F32(s), TargetLayout::M2F32(l)) => IterativeSolver::validate_target(s, l),
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
            (Self::M2F64(s), TargetLayout::M2F64(l)) => IterativeSolver::step(s, l),
            (Self::M2F32(s), TargetLayout::M2F32(l)) => IterativeSolver::step(s, l),
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
            (Self::M2F64(s), TargetLayout::M2F64(l)) => IterativeSolver::run_steps(s, l, steps),
            (Self::M2F32(s), TargetLayout::M2F32(l)) => IterativeSolver::run_steps(s, l, steps),
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
            (Self::M2F64(s), TargetLayout::M2F64(l)) => IterativeSolver::run(s, l),
            (Self::M2F32(s), TargetLayout::M2F32(l)) => IterativeSolver::run(s, l),
            _ => Err(CoreError::invalid_input(
                "internal solver target/dtype mismatch",
            )),
        }
    }
}

impl TsNetInner {
    pub(super) fn total_fft_2d_calls(&self) -> u64 {
        match self {
            Self::L2F64(s) => s.total_fft_2d_calls(),
            Self::L2F32(s) => s.total_fft_2d_calls(),
            _ => 0,
        }
    }
    pub(super) fn num_nodes(&self) -> usize {
        match self {
            Self::L2F64(s) => s.num_nodes(),
            Self::L2F32(s) => s.num_nodes(),
            Self::L3F64(s) => s.num_nodes(),
            Self::L3F32(s) => s.num_nodes(),
            Self::NdF64(s) => s.num_nodes(),
            Self::NdF32(s) => s.num_nodes(),
            Self::M2F64(s) => s.num_nodes(),
            Self::M2F32(s) => s.num_nodes(),
        }
    }
    pub(super) fn stage_index(&self) -> usize {
        match self {
            Self::L2F64(s) => s.stage_index(),
            Self::L2F32(s) => s.stage_index(),
            Self::L3F64(s) => s.stage_index(),
            Self::L3F32(s) => s.stage_index(),
            Self::NdF64(s) => s.stage_index(),
            Self::NdF32(s) => s.stage_index(),
            Self::M2F64(s) => s.stage_index(),
            Self::M2F32(s) => s.stage_index(),
        }
    }
    pub(super) fn current_stage_lr(&self) -> f64 {
        match self {
            Self::L2F64(s) => s.current_stage_lr(),
            Self::L2F32(s) => s.current_stage_lr(),
            Self::L3F64(s) => s.current_stage_lr(),
            Self::L3F32(s) => s.current_stage_lr(),
            Self::NdF64(s) => s.current_stage_lr(),
            Self::NdF32(s) => s.current_stage_lr(),
            Self::M2F64(s) => s.current_stage_lr(),
            Self::M2F32(s) => s.current_stage_lr(),
        }
    }
    pub(super) fn resolved_backend(&self) -> &'static str {
        let n = self.num_nodes();
        match self {
            Self::L2F64(s) => match s.backend() {
                TsNetBackend2d::Exact(_) => "exact",
                TsNetBackend2d::BarnesHut(_) => "barnes_hut",
                TsNetBackend2d::FitFft(_) => "fit_fft",
                TsNetBackend2d::LinearFft(_) => "linear_fft",
                TsNetBackend2d::Auto(a) => {
                    if n <= a.exact_threshold() {
                        "exact"
                    } else {
                        "linear_fft"
                    }
                }
                _ => "exact",
            },
            Self::L2F32(s) => match s.backend() {
                TsNetBackend2d::Exact(_) => "exact",
                TsNetBackend2d::BarnesHut(_) => "barnes_hut",
                TsNetBackend2d::FitFft(_) => "fit_fft",
                TsNetBackend2d::LinearFft(_) => "linear_fft",
                TsNetBackend2d::Auto(a) => {
                    if n <= a.exact_threshold() {
                        "exact"
                    } else {
                        "linear_fft"
                    }
                }
                _ => "exact",
            },
            Self::L3F64(s) => match s.backend() {
                TsNetBackend3d::Exact(_) => "exact",
                TsNetBackend3d::BarnesHut(_) => "barnes_hut",
                TsNetBackend3d::Auto(a) => {
                    if n <= a.exact_threshold() {
                        "exact"
                    } else {
                        "barnes_hut"
                    }
                }
                _ => "exact",
            },
            Self::L3F32(s) => match s.backend() {
                TsNetBackend3d::Exact(_) => "exact",
                TsNetBackend3d::BarnesHut(_) => "barnes_hut",
                TsNetBackend3d::Auto(a) => {
                    if n <= a.exact_threshold() {
                        "exact"
                    } else {
                        "barnes_hut"
                    }
                }
                _ => "exact",
            },
            Self::NdF64(_) | Self::NdF32(_) | Self::M2F64(_) | Self::M2F32(_) => "exact",
        }
    }
    pub(super) fn last_fft_stats(&self) -> Option<PyFitFieldStats> {
        match self {
            Self::L2F64(s) => s.last_fft_stats().map(PyFitFieldStats::from_fit),
            Self::L2F32(s) => s.last_fft_stats().map(PyFitFieldStats::from_fit),
            _ => None,
        }
    }
}

pub(super) fn parse_tsnet_sampling_mode(raw: &str) -> PyResult<TwoHopSamplingMode> {
    match normalize_enum_key(raw).as_str() {
        "degreeorderedhubsuppressed" | "degreeordered" | "hubsuppressed" => {
            Ok(TwoHopSamplingMode::DegreeOrderedHubSuppressed)
        }
        "twosteprandomwalk" | "randomwalk" | "rw" => Ok(TwoHopSamplingMode::TwoStepRandomWalk),
        _ => Err(unsupported_operation_err(format!(
            "unknown 2-hop sampling mode {raw:?}; expected 'degree_ordered_hub_suppressed' or 'two_step_random_walk'"
        ))),
    }
}

pub(super) struct TsNetStageOpts<'a, 'py> {
    pub(super) stage_iterations: Option<&'a Bound<'py, PyAny>>,
    pub(super) exaggeration: f64,
    pub(super) lambda_c_stage1: f64,
    pub(super) lambda_c_stage2: f64,
    pub(super) lambda_c_stage3: f64,
    pub(super) lambda_r_stage3: f64,
    pub(super) learning_rate: Option<f64>,
    pub(super) learning_rate_divisor: f64,
    pub(super) jacobi_precondition: bool,
    pub(super) momentum: f64,
    pub(super) epsilon_r: f64,
    pub(super) theta: f64,
    pub(super) intervals: usize,
    pub(super) interpolation_order: usize,
    pub(super) min_grid_dim: usize,
    pub(super) max_grid_dim: usize,
    pub(super) seed: u64,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_tsnet_config_and_backend(
    target: SolverTargetKind,
    backend: &str,
    perplexity_c: f64,
    min_perplexity: f64,
    max_perplexity: f64,
    power: f64,
    sigma_iters: usize,
    sigma_tolerance: f64,
    row_normalize: bool,
    opts: TsNetStageOpts<'_, '_>,
) -> PyResult<(TsNetConfig, ResolvedTsNetBackend)> {
    let choice = TsNetBackendChoice::parse(backend)?;
    let order =
        InterpolationOrder::from_order(opts.interpolation_order).map_err(core_error_to_py)?;
    let ibfft = IbFftConfig::default()
        .with_order(order)
        .with_intervals(opts.intervals)
        .with_min_grid_dim(opts.min_grid_dim)
        .with_max_grid_dim(opts.max_grid_dim);
    let resolved = validate_and_resolve_tsnet_backend(target, choice, opts.theta, ibfft)?;
    let stage_iterations = match opts.stage_iterations {
        Some(obj) if !obj.is_none() => parse_stage_iterations(obj)?,
        _ => [250, 250, 500],
    };
    let stages = vec![
        TsNetStage::new()
            .with_iterations(stage_iterations[0])
            .with_exaggeration(opts.exaggeration)
            .with_lambda_kl(1.0)
            .with_lambda_c(opts.lambda_c_stage1)
            .with_lambda_r(0.0),
        TsNetStage::new()
            .with_iterations(stage_iterations[1])
            .with_exaggeration(1.0)
            .with_lambda_kl(1.0)
            .with_lambda_c(opts.lambda_c_stage2)
            .with_lambda_r(0.0),
        TsNetStage::new()
            .with_iterations(stage_iterations[2])
            .with_exaggeration(1.0)
            .with_lambda_kl(1.0)
            .with_lambda_c(opts.lambda_c_stage3)
            .with_lambda_r(opts.lambda_r_stage3),
    ];
    let sym = if row_normalize {
        SymmetrizationMode::RowNormalized
    } else {
        SymmetrizationMode::StandardTsne
    };
    let cfg = TsNetConfig::new()
        .with_perplexity_c(perplexity_c)
        .with_min_perplexity(min_perplexity)
        .with_max_perplexity(max_perplexity)
        .with_power(power)
        .with_sigma_iters(sigma_iters)
        .with_sigma_tolerance(sigma_tolerance)
        .with_symmetrization(sym)
        .with_stages(stages)
        .with_learning_rate(opts.learning_rate)
        .with_learning_rate_divisor(opts.learning_rate_divisor)
        .with_jacobi_precondition(opts.jacobi_precondition)
        .with_momentum(opts.momentum)
        .with_epsilon_r(opts.epsilon_r)
        .with_seed(opts.seed);
    cfg.validate().map_err(core_error_to_py)?;
    Ok((cfg, resolved))
}
