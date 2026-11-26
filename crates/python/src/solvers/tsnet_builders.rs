//! Constructor builders for `TsNetSolver`.

use egraph::layout as egraph_layout;
use egraph_layout::TsNetSolver;
use pyo3::prelude::*;

use crate::array_extract::extract_perplexities;
use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py};
use crate::neighborhood::{PyKnnGraph, PySparseProbabilities};

use super::tsnet_config::{TsNetInner, build_tsnet_config};

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
    iterations_stage1: StrictUsize,
    iterations_stage2: StrictUsize,
    iterations_stage3: StrictUsize,
    exaggeration: StrictF64,
    lambda_c_stage1: StrictF64,
    lambda_c_stage2: StrictF64,
    lambda_c_stage3: StrictF64,
    lambda_r_stage3: StrictF64,
    learning_rate: Option<StrictF64>,
    momentum: StrictF64,
    epsilon_r: StrictF64,
    theta: StrictF64,
    intervals: StrictUsize,
    interpolation_order: StrictUsize,
    min_grid_dim: StrictUsize,
    max_grid_dim: StrictUsize,
    jacobi_precondition: bool,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<TsNetInner> {
    let cfg = build_tsnet_config(
        backend,
        perplexity_c.0,
        min_perplexity.0,
        max_perplexity.0,
        power.0,
        sigma_iters.0,
        sigma_tolerance.0,
        row_normalize,
        iterations_stage1.0,
        iterations_stage2.0,
        iterations_stage3.0,
        exaggeration.0,
        lambda_c_stage1.0,
        lambda_c_stage2.0,
        lambda_c_stage3.0,
        lambda_r_stage3.0,
        learning_rate.map(|v| v.0),
        jacobi_precondition,
        momentum.0,
        epsilon_r.0,
        theta.0,
        intervals.0,
        interpolation_order.0,
        min_grid_dim.0,
        max_grid_dim.0,
        seed.0,
    )?;
    let g = graph.inner();
    py.detach(|| match dtype {
        ScalarDtype::F64 => TsNetSolver::<f64>::from_graph_pure_2hop(g, cfg).map(TsNetInner::F64),
        ScalarDtype::F32 => TsNetSolver::<f32>::from_graph_pure_2hop(g, cfg).map(TsNetInner::F32),
    })
    .map_err(core_error_to_py)
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
    iterations_stage1: StrictUsize,
    iterations_stage2: StrictUsize,
    iterations_stage3: StrictUsize,
    exaggeration: StrictF64,
    lambda_c_stage1: StrictF64,
    lambda_c_stage2: StrictF64,
    lambda_c_stage3: StrictF64,
    lambda_r_stage3: StrictF64,
    learning_rate: Option<StrictF64>,
    momentum: StrictF64,
    epsilon_r: StrictF64,
    theta: StrictF64,
    intervals: StrictUsize,
    interpolation_order: StrictUsize,
    min_grid_dim: StrictUsize,
    max_grid_dim: StrictUsize,
    jacobi_precondition: bool,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<TsNetInner> {
    let cfg = build_tsnet_config(
        backend,
        0.9,
        5.0,
        1200.0,
        power.0,
        sigma_iters.0,
        sigma_tolerance.0,
        row_normalize,
        iterations_stage1.0,
        iterations_stage2.0,
        iterations_stage3.0,
        exaggeration.0,
        lambda_c_stage1.0,
        lambda_c_stage2.0,
        lambda_c_stage3.0,
        lambda_r_stage3.0,
        learning_rate.map(|v| v.0),
        jacobi_precondition,
        momentum.0,
        epsilon_r.0,
        theta.0,
        intervals.0,
        interpolation_order.0,
        min_grid_dim.0,
        max_grid_dim.0,
        seed.0,
    )?;
    let knn_ref = knn.inner();
    let perps_opt = match perplexities {
        Some(obj) if !obj.is_none() => Some(extract_perplexities(obj, knn_ref.num_nodes())?),
        _ => None,
    };
    let u_perp = uniform_perplexity.0;
    py.detach(|| match dtype {
        ScalarDtype::F64 => {
            TsNetSolver::<f64>::from_knn(knn_ref, perps_opt.as_deref(), u_perp, cfg)
                .map(TsNetInner::F64)
        }
        ScalarDtype::F32 => {
            TsNetSolver::<f32>::from_knn(knn_ref, perps_opt.as_deref(), u_perp, cfg)
                .map(TsNetInner::F32)
        }
    })
    .map_err(core_error_to_py)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_from_probabilities(
    py: Python<'_>,
    probs: &PySparseProbabilities,
    backend: &str,
    iterations_stage1: StrictUsize,
    iterations_stage2: StrictUsize,
    iterations_stage3: StrictUsize,
    exaggeration: StrictF64,
    lambda_c_stage1: StrictF64,
    lambda_c_stage2: StrictF64,
    lambda_c_stage3: StrictF64,
    lambda_r_stage3: StrictF64,
    learning_rate: Option<StrictF64>,
    momentum: StrictF64,
    epsilon_r: StrictF64,
    theta: StrictF64,
    intervals: StrictUsize,
    interpolation_order: StrictUsize,
    min_grid_dim: StrictUsize,
    max_grid_dim: StrictUsize,
    jacobi_precondition: bool,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<TsNetInner> {
    let cfg = build_tsnet_config(
        backend,
        0.9,
        5.0,
        1200.0,
        2.0,
        50,
        1e-5,
        false,
        iterations_stage1.0,
        iterations_stage2.0,
        iterations_stage3.0,
        exaggeration.0,
        lambda_c_stage1.0,
        lambda_c_stage2.0,
        lambda_c_stage3.0,
        lambda_r_stage3.0,
        learning_rate.map(|v| v.0),
        jacobi_precondition,
        momentum.0,
        epsilon_r.0,
        theta.0,
        intervals.0,
        interpolation_order.0,
        min_grid_dim.0,
        max_grid_dim.0,
        seed.0,
    )?;
    let probs_owned = probs.inner().clone();
    py.detach(|| match dtype {
        ScalarDtype::F64 => {
            TsNetSolver::<f64>::from_probabilities(probs_owned, cfg).map(TsNetInner::F64)
        }
        ScalarDtype::F32 => {
            TsNetSolver::<f32>::from_probabilities(probs_owned, cfg).map(TsNetInner::F32)
        }
    })
    .map_err(core_error_to_py)
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
    iterations_stage1: StrictUsize,
    iterations_stage2: StrictUsize,
    iterations_stage3: StrictUsize,
    exaggeration: StrictF64,
    lambda_c_stage1: StrictF64,
    lambda_c_stage2: StrictF64,
    lambda_c_stage3: StrictF64,
    lambda_r_stage3: StrictF64,
    learning_rate: Option<StrictF64>,
    momentum: StrictF64,
    epsilon_r: StrictF64,
    theta: StrictF64,
    intervals: StrictUsize,
    interpolation_order: StrictUsize,
    min_grid_dim: StrictUsize,
    max_grid_dim: StrictUsize,
    jacobi_precondition: bool,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<TsNetInner> {
    let cfg = build_tsnet_config(
        backend,
        0.9,
        5.0,
        1200.0,
        power.0,
        sigma_iters.0,
        sigma_tolerance.0,
        row_normalize,
        iterations_stage1.0,
        iterations_stage2.0,
        iterations_stage3.0,
        exaggeration.0,
        lambda_c_stage1.0,
        lambda_c_stage2.0,
        lambda_c_stage3.0,
        lambda_r_stage3.0,
        learning_rate.map(|v| v.0),
        jacobi_precondition,
        momentum.0,
        epsilon_r.0,
        theta.0,
        intervals.0,
        interpolation_order.0,
        min_grid_dim.0,
        max_grid_dim.0,
        seed.0,
    )?;
    let g = graph.inner();
    let k_val = k.0;
    let u_perp = uniform_perplexity.0;
    py.detach(|| match dtype {
        ScalarDtype::F64 => {
            TsNetSolver::<f64>::from_graph_partial_bfs(g, k_val, u_perp, cfg).map(TsNetInner::F64)
        }
        ScalarDtype::F32 => {
            TsNetSolver::<f32>::from_graph_partial_bfs(g, k_val, u_perp, cfg).map(TsNetInner::F32)
        }
    })
    .map_err(core_error_to_py)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_from_graph_row_normalized(
    py: Python<'_>,
    graph: &PyCsrGraph,
    w2: StrictF64,
    max_2hop_per_node: Option<StrictUsize>,
    degree_beta: StrictF64,
    backend: &str,
    iterations_stage1: StrictUsize,
    iterations_stage2: StrictUsize,
    iterations_stage3: StrictUsize,
    exaggeration: StrictF64,
    lambda_c_stage1: StrictF64,
    lambda_c_stage2: StrictF64,
    lambda_c_stage3: StrictF64,
    lambda_r_stage3: StrictF64,
    learning_rate: Option<StrictF64>,
    momentum: StrictF64,
    epsilon_r: StrictF64,
    theta: StrictF64,
    intervals: StrictUsize,
    interpolation_order: StrictUsize,
    min_grid_dim: StrictUsize,
    max_grid_dim: StrictUsize,
    jacobi_precondition: bool,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<TsNetInner> {
    let cfg = build_tsnet_config(
        backend,
        0.9,
        5.0,
        1200.0,
        2.0,
        50,
        1e-5,
        false,
        iterations_stage1.0,
        iterations_stage2.0,
        iterations_stage3.0,
        exaggeration.0,
        lambda_c_stage1.0,
        lambda_c_stage2.0,
        lambda_c_stage3.0,
        lambda_r_stage3.0,
        learning_rate.map(|v| v.0),
        jacobi_precondition,
        momentum.0,
        epsilon_r.0,
        theta.0,
        intervals.0,
        interpolation_order.0,
        min_grid_dim.0,
        max_grid_dim.0,
        seed.0,
    )?;
    let g = graph.inner();
    let w2_val = w2.0;
    let max_2hop_val = max_2hop_per_node.map(|v| v.0);
    let beta_val = degree_beta.0;
    py.detach(|| match dtype {
        ScalarDtype::F64 => TsNetSolver::<f64>::from_graph_row_normalized_with_degree_decay(
            g,
            w2_val,
            max_2hop_val,
            beta_val,
            cfg,
        )
        .map(TsNetInner::F64),
        ScalarDtype::F32 => TsNetSolver::<f32>::from_graph_row_normalized_with_degree_decay(
            g,
            w2_val,
            max_2hop_val,
            beta_val,
            cfg,
        )
        .map(TsNetInner::F32),
    })
    .map_err(core_error_to_py)
}
