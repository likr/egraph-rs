//! Constructor builders for `TsNetSolver` enforcing Policy 1A target & Option 1 backend rules.

use egraph::core::{Layout2d, Layout3d, LayoutNd, ManifoldLayout2d};
use egraph::layout::{TsNetSolver, TsNetStage};
use numpy::{PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::array_extract::extract_perplexities;
use crate::csr_graph::PyCsrGraph;
use crate::errors::{core_error_to_py, dimension_mismatch_err, invalid_input_err};
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, extract_non_negative_usize,
    is_non_sequence_container,
};

pub(super) fn parse_stage_iterations(obj: &Bound<'_, PyAny>) -> PyResult<[usize; 3]> {
    if is_non_sequence_container(obj) {
        return Err(PyTypeError::new_err(
            "stage_iterations must be a sequence of 3 positive integers",
        ));
    }
    if let Ok(arr) = obj.cast::<PyUntypedArray>()
        && arr.ndim() != 1
    {
        return Err(dimension_mismatch_err(format!(
            "stage_iterations must be a 1D sequence of 3 positive integers, got {}D",
            arr.ndim()
        )));
    }
    let iter = obj.try_iter().map_err(|_| {
        PyTypeError::new_err("stage_iterations must be a sequence of 3 positive integers")
    })?;
    let mut vals = Vec::with_capacity(3);
    for item_res in iter {
        let item = item_res?;
        let val = extract_non_negative_usize(&item, "stage_iterations entry")?;
        if val == 0 {
            return Err(invalid_input_err(
                "stage_iterations entries must be positive (>= 1)",
            ));
        }
        vals.push(val);
    }
    if vals.len() != 3 {
        return Err(dimension_mismatch_err(format!(
            "stage_iterations must have length 3, got {}",
            vals.len()
        )));
    }
    Ok([vals[0], vals[1], vals[2]])
}
use crate::neighborhood::{PyKnnGraph, PySparseProbabilities};

use super::fit_field_stats::PyTsNetStage;
use super::layout_dispatch::SolverTargetKind;
use super::tsnet::PyTsNetSolver;
use super::tsnet_config::{
    ResolvedTsNetBackend, TsNetInner, TsNetStageOpts, build_tsnet_config_and_backend,
    parse_tsnet_sampling_mode,
};

macro_rules! dispatch_tsnet_ctor {
    ($py:expr, $resolved:expr, $dtype:expr, |$b:ident, $solver_ty:ident| $call:expr) => {
        $py.detach(|| match ($resolved, $dtype) {
            (ResolvedTsNetBackend::L2($b), ScalarDtype::F64) => {
                type $solver_ty = TsNetSolver<Layout2d<f64>, egraph::layout::TsNetBackend2d>;
                $call.map(TsNetInner::L2F64)
            }
            (ResolvedTsNetBackend::L2($b), ScalarDtype::F32) => {
                type $solver_ty = TsNetSolver<Layout2d<f32>, egraph::layout::TsNetBackend2d>;
                $call.map(TsNetInner::L2F32)
            }
            (ResolvedTsNetBackend::L3($b), ScalarDtype::F64) => {
                type $solver_ty = TsNetSolver<Layout3d<f64>, egraph::layout::TsNetBackend3d>;
                $call.map(TsNetInner::L3F64)
            }
            (ResolvedTsNetBackend::L3($b), ScalarDtype::F32) => {
                type $solver_ty = TsNetSolver<Layout3d<f32>, egraph::layout::TsNetBackend3d>;
                $call.map(TsNetInner::L3F32)
            }
            (ResolvedTsNetBackend::Nd($b), ScalarDtype::F64) => {
                type $solver_ty = TsNetSolver<LayoutNd<f64>, egraph::layout::neighbor::Exact>;
                $call.map(TsNetInner::NdF64)
            }
            (ResolvedTsNetBackend::Nd($b), ScalarDtype::F32) => {
                type $solver_ty = TsNetSolver<LayoutNd<f32>, egraph::layout::neighbor::Exact>;
                $call.map(TsNetInner::NdF32)
            }
            (ResolvedTsNetBackend::M2($b), ScalarDtype::F64) => {
                type $solver_ty =
                    TsNetSolver<ManifoldLayout2d<f64>, egraph::layout::neighbor::Exact>;
                $call.map(TsNetInner::M2F64)
            }
            (ResolvedTsNetBackend::M2($b), ScalarDtype::F32) => {
                type $solver_ty =
                    TsNetSolver<ManifoldLayout2d<f32>, egraph::layout::neighbor::Exact>;
                $call.map(TsNetInner::M2F32)
            }
        })
        .map_err(core_error_to_py)
    };
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_pure_2hop(
    py: Python<'_>,
    graph: &PyCsrGraph,
    target: &str,
    backend: &str,
    perplexity_c: StrictF64,
    min_perplexity: StrictF64,
    max_perplexity: StrictF64,
    power: StrictF64,
    sigma_iters: StrictUsize,
    sigma_tolerance: StrictF64,
    row_normalize: bool,
    opts: TsNetStageOpts<'_, '_>,
    dtype: ScalarDtype,
) -> PyResult<PyTsNetSolver> {
    let target_kind = SolverTargetKind::from_str(target)?;
    let (cfg, resolved) = build_tsnet_config_and_backend(
        target_kind,
        backend,
        perplexity_c.0,
        min_perplexity.0,
        max_perplexity.0,
        power.0,
        sigma_iters.0,
        sigma_tolerance.0,
        row_normalize,
        opts,
    )?;
    let g = graph.inner();
    let inner = dispatch_tsnet_ctor!(py, resolved, dtype, |b, S| {
        S::from_graph_pure_2hop_with_backend(&*g, cfg, b)
    })?;
    Ok(PyTsNetSolver::from_inner(inner))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_from_knn(
    py: Python<'_>,
    knn: &PyKnnGraph,
    perplexities: Option<&Bound<'_, PyAny>>,
    uniform_perplexity: StrictF64,
    target: &str,
    backend: &str,
    power: StrictF64,
    sigma_iters: StrictUsize,
    sigma_tolerance: StrictF64,
    row_normalize: bool,
    opts: TsNetStageOpts<'_, '_>,
    dtype: ScalarDtype,
) -> PyResult<PyTsNetSolver> {
    let target_kind = SolverTargetKind::from_str(target)?;
    let (cfg, resolved) = build_tsnet_config_and_backend(
        target_kind,
        backend,
        0.9,
        5.0,
        1200.0,
        power.0,
        sigma_iters.0,
        sigma_tolerance.0,
        row_normalize,
        opts,
    )?;
    let knn_ref = knn.inner();
    let perps_opt = match perplexities {
        Some(obj) if !obj.is_none() => Some(extract_perplexities(obj, knn_ref.num_nodes())?),
        _ => None,
    };
    let u_perp = uniform_perplexity.0;
    let inner = dispatch_tsnet_ctor!(py, resolved, dtype, |b, S| {
        S::from_knn_with_backend(knn_ref, perps_opt.as_deref(), u_perp, cfg, b)
    })?;
    Ok(PyTsNetSolver::from_inner(inner))
}

pub(super) fn build_from_probabilities(
    py: Python<'_>,
    probs: &PySparseProbabilities,
    target: &str,
    backend: &str,
    opts: TsNetStageOpts<'_, '_>,
    dtype: ScalarDtype,
) -> PyResult<PyTsNetSolver> {
    let target_kind = SolverTargetKind::from_str(target)?;
    let (cfg, resolved) = build_tsnet_config_and_backend(
        target_kind,
        backend,
        0.9,
        5.0,
        1200.0,
        2.0,
        50,
        1e-5,
        false,
        opts,
    )?;
    let probs_owned = probs.inner().clone();
    let inner = dispatch_tsnet_ctor!(py, resolved, dtype, |b, S| {
        S::from_probabilities_with_backend(probs_owned, cfg, b)
    })?;
    Ok(PyTsNetSolver::from_inner(inner))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_from_probabilities_with_stages(
    py: Python<'_>,
    probs: &PySparseProbabilities,
    stages: &Bound<'_, PyAny>,
    target: &str,
    backend: &str,
    learning_rate: Option<StrictF64>,
    learning_rate_divisor: StrictF64,
    jacobi_precondition: bool,
    momentum: StrictF64,
    epsilon_r: StrictF64,
    theta: StrictF64,
    intervals: StrictUsize,
    interpolation_order: StrictUsize,
    min_grid_dim: StrictUsize,
    max_grid_dim: StrictUsize,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<PyTsNetSolver> {
    let target_kind = SolverTargetKind::from_str(target)?;
    if is_non_sequence_container(stages) {
        return Err(PyTypeError::new_err(
            "stages must be a sequence or iterable of TsNetStage",
        ));
    }
    let iter = stages
        .try_iter()
        .map_err(|_| PyTypeError::new_err("stages must be a sequence or iterable of TsNetStage"))?;
    let mut stage_vec: Vec<TsNetStage> = Vec::new();
    for item_res in iter {
        let item = item_res?;
        let st = item
            .extract::<PyTsNetStage>()
            .map_err(|_| PyTypeError::new_err("stages entries must be TsNetStage instances"))?;
        stage_vec.push(st.inner);
    }
    let opts = TsNetStageOpts {
        stage_iterations: None,
        exaggeration: 1.0,
        lambda_c_stage1: 0.0,
        lambda_c_stage2: 0.0,
        lambda_c_stage3: 0.0,
        lambda_r_stage3: 0.0,
        learning_rate: learning_rate.map(|v| v.0),
        learning_rate_divisor: learning_rate_divisor.0,
        jacobi_precondition,
        momentum: momentum.0,
        epsilon_r: epsilon_r.0,
        theta: theta.0,
        intervals: intervals.0,
        interpolation_order: interpolation_order.0,
        min_grid_dim: min_grid_dim.0,
        max_grid_dim: max_grid_dim.0,
        seed: seed.0,
    };
    let (cfg, resolved) = build_tsnet_config_and_backend(
        target_kind,
        backend,
        0.9,
        5.0,
        1200.0,
        2.0,
        50,
        1e-5,
        false,
        opts,
    )?;
    let cfg = cfg.with_stages(stage_vec);
    cfg.validate().map_err(core_error_to_py)?;
    let probs_owned = probs.inner().clone();
    let inner = dispatch_tsnet_ctor!(py, resolved, dtype, |b, S| {
        S::from_probabilities_with_backend(probs_owned, cfg, b)
    })?;
    Ok(PyTsNetSolver::from_inner(inner))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_from_graph_partial_bfs(
    py: Python<'_>,
    graph: &PyCsrGraph,
    k: StrictUsize,
    uniform_perplexity: StrictF64,
    target: &str,
    backend: &str,
    power: StrictF64,
    sigma_iters: StrictUsize,
    sigma_tolerance: StrictF64,
    row_normalize: bool,
    opts: TsNetStageOpts<'_, '_>,
    dtype: ScalarDtype,
) -> PyResult<PyTsNetSolver> {
    let target_kind = SolverTargetKind::from_str(target)?;
    let (cfg, resolved) = build_tsnet_config_and_backend(
        target_kind,
        backend,
        0.9,
        5.0,
        1200.0,
        power.0,
        sigma_iters.0,
        sigma_tolerance.0,
        row_normalize,
        opts,
    )?;
    let g = graph.inner();
    let (k_val, u_perp) = (k.0, uniform_perplexity.0);
    let inner = dispatch_tsnet_ctor!(py, resolved, dtype, |b, S| {
        S::from_graph_partial_bfs_with_backend(&*g, k_val, u_perp, cfg, b)
    })?;
    Ok(PyTsNetSolver::from_inner(inner))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_from_graph_row_normalized(
    py: Python<'_>,
    graph: &PyCsrGraph,
    w2: StrictF64,
    max_2hop_per_node: Option<StrictUsize>,
    degree_beta: StrictF64,
    mode: &str,
    max_bridge_degree: Option<StrictUsize>,
    target: &str,
    backend: &str,
    opts: TsNetStageOpts<'_, '_>,
    dtype: ScalarDtype,
) -> PyResult<PyTsNetSolver> {
    let target_kind = SolverTargetKind::from_str(target)?;
    let sampling_mode = parse_tsnet_sampling_mode(mode)?;
    let (cfg, resolved) = build_tsnet_config_and_backend(
        target_kind,
        backend,
        0.9,
        5.0,
        1200.0,
        2.0,
        50,
        1e-5,
        false,
        opts,
    )?;
    let g = graph.inner();
    let (w2_val, max_2hop_val, beta_val, bridge_val) = (
        w2.0,
        max_2hop_per_node.map(|v| v.0),
        degree_beta.0,
        max_bridge_degree.map(|v| v.0),
    );
    let inner = dispatch_tsnet_ctor!(py, resolved, dtype, |b, S| {
        S::from_graph_row_normalized_with_options_with_backend(
            &*g,
            w2_val,
            max_2hop_val,
            beta_val,
            sampling_mode,
            bridge_val,
            cfg,
            b,
        )
    })?;
    Ok(PyTsNetSolver::from_inner(inner))
}
