//! Constructor builders for `IbUmapSolver`.

use egraph::core::{CoreError, CoreResult, IterativeSolver, Layout2d, SolverStatus};
use egraph::layout::{
    IbFft, IbUmapBackend2d, IbUmapConfig, IbUmapSolver,
    neighbor::{BarnesHut as NeighborBarnesHut, Exact as NeighborExact},
};
use egraph::neighborhood::DogVaradhanConfig;
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::errors::{core_error_to_py, unsupported_operation_err};
use crate::ffi_convert::{ScalarDtype, StrictF64, StrictU64, StrictUsize, extract_u32_1d};
use crate::neighborhood::{PyDogVaradhanLandmarks, PyKnnGraph, PySparseProbabilities};
use crate::solver_common::{DualIterativeSolver, SolverLayout};

use super::fit_field_stats::build_ibfft_config;
use super::layout_dispatch::SolverTargetKind;
use super::t_fdp::TFdpBackendChoice;

pub(super) enum IbUmapInner {
    F64(IbUmapSolver<Layout2d<f64>, IbUmapBackend2d>),
    F32(IbUmapSolver<Layout2d<f32>, IbUmapBackend2d>),
}

impl DualIterativeSolver for IbUmapInner {
    fn dtype(&self) -> ScalarDtype {
        match self {
            Self::F64(_) => ScalarDtype::F64,
            Self::F32(_) => ScalarDtype::F32,
        }
    }

    fn step_layout(&mut self, layout: &mut SolverLayout) -> CoreResult<SolverStatus> {
        match (self, layout) {
            (Self::F64(s), SolverLayout::F64(l)) => IterativeSolver::step(s, l),
            (Self::F32(s), SolverLayout::F32(l)) => IterativeSolver::step(s, l),
            _ => Err(CoreError::invalid_input(
                "internal solver layout dtype mismatch",
            )),
        }
    }

    fn run_steps_layout(
        &mut self,
        layout: &mut SolverLayout,
        steps: usize,
    ) -> CoreResult<SolverStatus> {
        match (self, layout) {
            (Self::F64(s), SolverLayout::F64(l)) => IterativeSolver::run_steps(s, l, steps),
            (Self::F32(s), SolverLayout::F32(l)) => IterativeSolver::run_steps(s, l, steps),
            _ => Err(CoreError::invalid_input(
                "internal solver layout dtype mismatch",
            )),
        }
    }

    fn run_layout(&mut self, layout: &mut SolverLayout) -> CoreResult<SolverStatus> {
        match (self, layout) {
            (Self::F64(s), SolverLayout::F64(l)) => IterativeSolver::run(s, l),
            (Self::F32(s), SolverLayout::F32(l)) => IterativeSolver::run(s, l),
            _ => Err(CoreError::invalid_input(
                "internal solver layout dtype mismatch",
            )),
        }
    }

    fn solver_status(&self) -> SolverStatus {
        match self {
            Self::F64(s) => IterativeSolver::status(s),
            Self::F32(s) => IterativeSolver::status(s),
        }
    }

    fn solver_iteration(&self) -> usize {
        match self {
            Self::F64(s) => IterativeSolver::iteration(s),
            Self::F32(s) => IterativeSolver::iteration(s),
        }
    }

    fn solver_total_iterations(&self) -> Option<usize> {
        match self {
            Self::F64(s) => IterativeSolver::total_iterations(s),
            Self::F32(s) => IterativeSolver::total_iterations(s),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_ib_umap_config(
    target: &str,
    backend: &str,
    iterations: usize,
    learning_rate: f64,
    min_dist: f64,
    spread: f64,
    a: Option<f64>,
    b: Option<f64>,
    c_k: f64,
    tau_scale: f64,
    hub_damping: bool,
    theta: f64,
    interpolation_order: usize,
    intervals: usize,
    min_grid_dim: usize,
    max_grid_dim: usize,
    seed: u64,
) -> PyResult<(IbUmapConfig, IbUmapBackend2d)> {
    let target_kind = SolverTargetKind::from_str(target)?;
    if target_kind != SolverTargetKind::Layout2d {
        return Err(unsupported_operation_err(format!(
            "IbUmapSolver only supports target='layout2d', got '{}'",
            target_kind.as_str()
        )));
    }
    let choice = TFdpBackendChoice::parse(backend)?;
    NeighborBarnesHut::new(theta)
        .validate()
        .map_err(core_error_to_py)?;
    let ibfft = build_ibfft_config(interpolation_order, intervals, min_grid_dim, max_grid_dim)?;
    ibfft.validate().map_err(core_error_to_py)?;
    let backend_2d = match choice {
        TFdpBackendChoice::Auto | TFdpBackendChoice::IbFft => {
            IbUmapBackend2d::IbFft(IbFft::new().with_ibfft(ibfft))
        }
        TFdpBackendChoice::BarnesHut => IbUmapBackend2d::BarnesHut(NeighborBarnesHut::new(theta)),
        TFdpBackendChoice::Exact => IbUmapBackend2d::Exact(NeighborExact::new()),
    };
    backend_2d.validate().map_err(core_error_to_py)?;

    let cfg = IbUmapConfig::new()
        .with_iterations(iterations)
        .with_learning_rate(learning_rate)
        .with_min_dist(min_dist)
        .with_spread(spread)
        .with_a(a)
        .with_b(b)
        .with_c_k(c_k)
        .with_tau_scale(tau_scale)
        .with_hub_damping(hub_damping)
        .with_seed(seed);
    cfg.validate().map_err(core_error_to_py)?;
    Ok((cfg, backend_2d))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_ib_umap_new(
    py: Python<'_>,
    graph: &PyCsrGraph,
    target: &str,
    max_2hop_per_node: StrictUsize,
    degree: Option<StrictUsize>,
    num_landmarks: StrictUsize,
    min_degree: StrictUsize,
    jl_dim: StrictUsize,
    degree_normalize: bool,
    p_exp: StrictF64,
    eps_psd: StrictF64,
    t_coarse: StrictF64,
    t1: Option<StrictF64>,
    t2: Option<StrictF64>,
    alpha: Option<StrictF64>,
    beta_floor: Option<StrictF64>,
    backend: &str,
    iterations: StrictUsize,
    learning_rate: StrictF64,
    min_dist: StrictF64,
    spread: StrictF64,
    a: Option<StrictF64>,
    b: Option<StrictF64>,
    c_k: StrictF64,
    tau_scale: StrictF64,
    hub_damping: bool,
    theta: StrictF64,
    interpolation_order: StrictUsize,
    intervals: StrictUsize,
    min_grid_dim: StrictUsize,
    max_grid_dim: StrictUsize,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<IbUmapInner> {
    let (cfg, backend_2d) = build_ib_umap_config(
        target,
        backend,
        iterations.0,
        learning_rate.0,
        min_dist.0,
        spread.0,
        a.map(|v| v.0),
        b.map(|v| v.0),
        c_k.0,
        tau_scale.0,
        hub_damping,
        theta.0,
        interpolation_order.0,
        intervals.0,
        min_grid_dim.0,
        max_grid_dim.0,
        seed.0,
    )?;
    let dog_cfg = DogVaradhanConfig::new()
        .with_t1(t1.map(|v| v.0))
        .with_t2(t2.map(|v| v.0))
        .with_alpha(alpha.map(|v| v.0))
        .with_beta_floor(beta_floor.map(|v| v.0))
        .with_degree(degree.map(|v| v.0))
        .with_n_landmarks(num_landmarks.0)
        .with_min_degree(min_degree.0)
        .with_jl_dim(jl_dim.0)
        .with_eps_psd(eps_psd.0)
        .with_degree_normalize(degree_normalize)
        .with_p_exp(p_exp.0)
        .with_t_coarse(t_coarse.0)
        .with_seed(seed.0);
    let g = graph.inner();
    py.detach(|| match dtype {
        ScalarDtype::F64 => {
            IbUmapSolver::<Layout2d<f64>, IbUmapBackend2d>::from_graph_dog_with_backend(
                &*g,
                max_2hop_per_node.0,
                &dog_cfg,
                cfg,
                backend_2d,
            )
            .map(IbUmapInner::F64)
        }
        ScalarDtype::F32 => {
            IbUmapSolver::<Layout2d<f32>, IbUmapBackend2d>::from_graph_dog_with_backend(
                &*g,
                max_2hop_per_node.0,
                &dog_cfg,
                cfg,
                backend_2d,
            )
            .map(IbUmapInner::F32)
        }
    })
    .map_err(core_error_to_py)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_ib_umap_from_landmarks(
    py: Python<'_>,
    graph: &PyCsrGraph,
    landmarks: &PyDogVaradhanLandmarks,
    max_2hop_per_node: StrictUsize,
    target: &str,
    backend: &str,
    iterations: StrictUsize,
    learning_rate: StrictF64,
    min_dist: StrictF64,
    spread: StrictF64,
    a: Option<StrictF64>,
    b: Option<StrictF64>,
    c_k: StrictF64,
    tau_scale: StrictF64,
    hub_damping: bool,
    theta: StrictF64,
    interpolation_order: StrictUsize,
    intervals: StrictUsize,
    min_grid_dim: StrictUsize,
    max_grid_dim: StrictUsize,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<IbUmapInner> {
    let (cfg, backend_2d) = build_ib_umap_config(
        target,
        backend,
        iterations.0,
        learning_rate.0,
        min_dist.0,
        spread.0,
        a.map(|v| v.0),
        b.map(|v| v.0),
        c_k.0,
        tau_scale.0,
        hub_damping,
        theta.0,
        interpolation_order.0,
        intervals.0,
        min_grid_dim.0,
        max_grid_dim.0,
        seed.0,
    )?;
    let g = graph.inner();
    let lm = landmarks.inner();
    py.detach(|| match dtype {
        ScalarDtype::F64 => {
            IbUmapSolver::<Layout2d<f64>, IbUmapBackend2d>::from_landmarks_with_backend(
                &*g,
                lm,
                max_2hop_per_node.0,
                cfg,
                backend_2d,
            )
            .map(IbUmapInner::F64)
        }
        ScalarDtype::F32 => {
            IbUmapSolver::<Layout2d<f32>, IbUmapBackend2d>::from_landmarks_with_backend(
                &*g,
                lm,
                max_2hop_per_node.0,
                cfg,
                backend_2d,
            )
            .map(IbUmapInner::F32)
        }
    })
    .map_err(core_error_to_py)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_ib_umap_from_knn(
    py: Python<'_>,
    knn: &PyKnnGraph,
    sigma_iters: StrictUsize,
    sigma_tolerance: StrictF64,
    target: &str,
    backend: &str,
    iterations: StrictUsize,
    learning_rate: StrictF64,
    min_dist: StrictF64,
    spread: StrictF64,
    a: Option<StrictF64>,
    b: Option<StrictF64>,
    c_k: StrictF64,
    tau_scale: StrictF64,
    hub_damping: bool,
    theta: StrictF64,
    interpolation_order: StrictUsize,
    intervals: StrictUsize,
    min_grid_dim: StrictUsize,
    max_grid_dim: StrictUsize,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<IbUmapInner> {
    let (cfg, backend_2d) = build_ib_umap_config(
        target,
        backend,
        iterations.0,
        learning_rate.0,
        min_dist.0,
        spread.0,
        a.map(|v| v.0),
        b.map(|v| v.0),
        c_k.0,
        tau_scale.0,
        hub_damping,
        theta.0,
        interpolation_order.0,
        intervals.0,
        min_grid_dim.0,
        max_grid_dim.0,
        seed.0,
    )?;
    let knn_ref = knn.inner();
    py.detach(|| match dtype {
        ScalarDtype::F64 => IbUmapSolver::<Layout2d<f64>, IbUmapBackend2d>::from_knn_with_backend(
            knn_ref,
            sigma_iters.0,
            sigma_tolerance.0,
            cfg,
            backend_2d,
        )
        .map(IbUmapInner::F64),
        ScalarDtype::F32 => IbUmapSolver::<Layout2d<f32>, IbUmapBackend2d>::from_knn_with_backend(
            knn_ref,
            sigma_iters.0,
            sigma_tolerance.0,
            cfg,
            backend_2d,
        )
        .map(IbUmapInner::F32),
    })
    .map_err(core_error_to_py)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_ib_umap_from_fuzzy_set(
    py: Python<'_>,
    probs: &PySparseProbabilities,
    degrees: Option<&Bound<'_, PyAny>>,
    target: &str,
    backend: &str,
    iterations: StrictUsize,
    learning_rate: StrictF64,
    min_dist: StrictF64,
    spread: StrictF64,
    a: Option<StrictF64>,
    b: Option<StrictF64>,
    c_k: StrictF64,
    tau_scale: StrictF64,
    hub_damping: bool,
    theta: StrictF64,
    interpolation_order: StrictUsize,
    intervals: StrictUsize,
    min_grid_dim: StrictUsize,
    max_grid_dim: StrictUsize,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<IbUmapInner> {
    let (cfg, backend_2d) = build_ib_umap_config(
        target,
        backend,
        iterations.0,
        learning_rate.0,
        min_dist.0,
        spread.0,
        a.map(|v| v.0),
        b.map(|v| v.0),
        c_k.0,
        tau_scale.0,
        hub_damping,
        theta.0,
        interpolation_order.0,
        intervals.0,
        min_grid_dim.0,
        max_grid_dim.0,
        seed.0,
    )?;
    let degs_opt = match degrees {
        Some(obj) if !obj.is_none() => Some(extract_u32_1d(obj, "degrees")?),
        _ => None,
    };
    let probs_ref = probs.inner();
    py.detach(|| match dtype {
        ScalarDtype::F64 => {
            IbUmapSolver::<Layout2d<f64>, IbUmapBackend2d>::from_fuzzy_set_ref_with_backend(
                probs_ref,
                degs_opt.as_deref(),
                cfg,
                backend_2d,
            )
            .map(IbUmapInner::F64)
        }
        ScalarDtype::F32 => {
            IbUmapSolver::<Layout2d<f32>, IbUmapBackend2d>::from_fuzzy_set_ref_with_backend(
                probs_ref,
                degs_opt.as_deref(),
                cfg,
                backend_2d,
            )
            .map(IbUmapInner::F32)
        }
    })
    .map_err(core_error_to_py)
}
