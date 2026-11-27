//! Constructor builders for `TsNetSolver`.

use egraph::layout as egraph_layout;
use egraph_layout::{TsNetSolver, TsNetStage};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::array_extract::extract_perplexities;
use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py, is_non_sequence_container,
};
use crate::neighborhood::{PyKnnGraph, PySparseProbabilities};

use super::tsnet::PyTsNetSolver;
use super::tsnet_config::{
    PyTsNetStage, TsNetInner, TsNetStageOpts, build_tsnet_config, parse_tsnet_sampling_mode,
};

#[allow(clippy::too_many_arguments)]
pub(super) fn build_pure_2hop(
    py: Python<'_>,
    graph: &PyCsrGraph,
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
    let cfg = build_tsnet_config(
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
    let inner = py
        .detach(|| match dtype {
            ScalarDtype::F64 => {
                TsNetSolver::<f64>::from_graph_pure_2hop(&*g, cfg).map(TsNetInner::F64)
            }
            ScalarDtype::F32 => {
                TsNetSolver::<f32>::from_graph_pure_2hop(&*g, cfg).map(TsNetInner::F32)
            }
        })
        .map_err(core_error_to_py)?;
    Ok(PyTsNetSolver::from_inner(inner))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_from_knn(
    py: Python<'_>,
    knn: &PyKnnGraph,
    perplexities: Option<&Bound<'_, PyAny>>,
    uniform_perplexity: StrictF64,
    backend: &str,
    power: StrictF64,
    sigma_iters: StrictUsize,
    sigma_tolerance: StrictF64,
    row_normalize: bool,
    opts: TsNetStageOpts<'_, '_>,
    dtype: ScalarDtype,
) -> PyResult<PyTsNetSolver> {
    let cfg = build_tsnet_config(
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
    let inner = py
        .detach(|| match dtype {
            ScalarDtype::F64 => {
                TsNetSolver::<f64>::from_knn(knn_ref, perps_opt.as_deref(), u_perp, cfg)
                    .map(TsNetInner::F64)
            }
            ScalarDtype::F32 => {
                TsNetSolver::<f32>::from_knn(knn_ref, perps_opt.as_deref(), u_perp, cfg)
                    .map(TsNetInner::F32)
            }
        })
        .map_err(core_error_to_py)?;
    Ok(PyTsNetSolver::from_inner(inner))
}

pub(super) fn build_from_probabilities(
    py: Python<'_>,
    probs: &PySparseProbabilities,
    backend: &str,
    opts: TsNetStageOpts<'_, '_>,
    dtype: ScalarDtype,
) -> PyResult<PyTsNetSolver> {
    let cfg = build_tsnet_config(backend, 0.9, 5.0, 1200.0, 2.0, 50, 1e-5, false, opts)?;
    let probs_owned = probs.inner().clone();
    let inner = py
        .detach(|| match dtype {
            ScalarDtype::F64 => {
                TsNetSolver::<f64>::from_probabilities(probs_owned, cfg).map(TsNetInner::F64)
            }
            ScalarDtype::F32 => {
                TsNetSolver::<f32>::from_probabilities(probs_owned, cfg).map(TsNetInner::F32)
            }
        })
        .map_err(core_error_to_py)?;
    Ok(PyTsNetSolver::from_inner(inner))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_from_probabilities_with_stages(
    py: Python<'_>,
    probs: &PySparseProbabilities,
    stages: &Bound<'_, PyAny>,
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
        iterations_stage1: 1,
        iterations_stage2: 1,
        iterations_stage3: 1,
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
    let cfg = build_tsnet_config(backend, 0.9, 5.0, 1200.0, 2.0, 50, 1e-5, false, opts)?;
    let probs_owned = probs.inner().clone();
    let inner = py
        .detach(|| match dtype {
            ScalarDtype::F64 => {
                TsNetSolver::<f64>::from_probabilities_with_stages(probs_owned, cfg, stage_vec)
                    .map(TsNetInner::F64)
            }
            ScalarDtype::F32 => {
                TsNetSolver::<f32>::from_probabilities_with_stages(probs_owned, cfg, stage_vec)
                    .map(TsNetInner::F32)
            }
        })
        .map_err(core_error_to_py)?;
    Ok(PyTsNetSolver::from_inner(inner))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_from_graph_partial_bfs(
    py: Python<'_>,
    graph: &PyCsrGraph,
    k: StrictUsize,
    uniform_perplexity: StrictF64,
    backend: &str,
    power: StrictF64,
    sigma_iters: StrictUsize,
    sigma_tolerance: StrictF64,
    row_normalize: bool,
    opts: TsNetStageOpts<'_, '_>,
    dtype: ScalarDtype,
) -> PyResult<PyTsNetSolver> {
    let cfg = build_tsnet_config(
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
    let k_val = k.0;
    let u_perp = uniform_perplexity.0;
    let inner = py
        .detach(|| match dtype {
            ScalarDtype::F64 => TsNetSolver::<f64>::from_graph_partial_bfs(&*g, k_val, u_perp, cfg)
                .map(TsNetInner::F64),
            ScalarDtype::F32 => TsNetSolver::<f32>::from_graph_partial_bfs(&*g, k_val, u_perp, cfg)
                .map(TsNetInner::F32),
        })
        .map_err(core_error_to_py)?;
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
    backend: &str,
    opts: TsNetStageOpts<'_, '_>,
    dtype: ScalarDtype,
) -> PyResult<PyTsNetSolver> {
    let sampling_mode = parse_tsnet_sampling_mode(mode).map_err(PyValueError::new_err)?;
    let cfg = build_tsnet_config(backend, 0.9, 5.0, 1200.0, 2.0, 50, 1e-5, false, opts)?;
    let g = graph.inner();
    let w2_val = w2.0;
    let max_2hop_val = max_2hop_per_node.map(|v| v.0);
    let beta_val = degree_beta.0;
    let bridge_val = max_bridge_degree.map(|v| v.0);
    let inner = py
        .detach(|| match dtype {
            ScalarDtype::F64 => TsNetSolver::<f64>::from_graph_row_normalized_with_options(
                &*g,
                w2_val,
                max_2hop_val,
                beta_val,
                sampling_mode,
                bridge_val,
                cfg,
            )
            .map(TsNetInner::F64),
            ScalarDtype::F32 => TsNetSolver::<f32>::from_graph_row_normalized_with_options(
                &*g,
                w2_val,
                max_2hop_val,
                beta_val,
                sampling_mode,
                bridge_val,
                cfg,
            )
            .map(TsNetInner::F32),
        })
        .map_err(core_error_to_py)?;
    Ok(PyTsNetSolver::from_inner(inner))
}
