//! Constructor builders for `GuMapSolver`.

use egraph::layout::{GuMapConfig, GuMapSolver};
use egraph::neighborhood::DogVaradhanConfig;
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::errors::core_error_to_py;
use crate::ffi_convert::{ScalarDtype, StrictF64, StrictU64, StrictUsize};
use crate::neighborhood::{PyDogVaradhanLandmarks, PyKnnGraph, PySparseProbabilities};

use super::gumap::{GuMapInner, PyGuMapSolver};
use super::layout_dispatch::{SolverTargetKind, dispatch_riemannian_target};

#[allow(clippy::too_many_arguments)]
fn build_gumap_config(
    iterations: usize,
    negative_sample_rate: usize,
    learning_rate: f64,
    min_dist: f64,
    spread: f64,
    a: Option<f64>,
    b: Option<f64>,
    sigma_iters: usize,
    sigma_tolerance: f64,
    symmetric_negative: bool,
    seed: u64,
) -> PyResult<GuMapConfig> {
    let cfg = GuMapConfig::new()
        .with_iterations(iterations)
        .with_negative_sample_rate(negative_sample_rate)
        .with_learning_rate(learning_rate)
        .with_min_dist(min_dist)
        .with_spread(spread)
        .with_a(a)
        .with_b(b)
        .with_sigma_iters(sigma_iters)
        .with_sigma_tolerance(sigma_tolerance)
        .with_symmetric_negative(symmetric_negative)
        .with_seed(seed);
    cfg.validate().map_err(core_error_to_py)?;
    Ok(cfg)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_gumap_from_fuzzy_set(
    py: Python<'_>,
    probs: &PySparseProbabilities,
    target: SolverTargetKind,
    iterations: StrictUsize,
    negative_sample_rate: StrictUsize,
    learning_rate: StrictF64,
    min_dist: StrictF64,
    spread: StrictF64,
    a: Option<StrictF64>,
    b: Option<StrictF64>,
    sigma_iters: StrictUsize,
    sigma_tolerance: StrictF64,
    symmetric_negative: bool,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<PyGuMapSolver> {
    let cfg = build_gumap_config(
        iterations.0,
        negative_sample_rate.0,
        learning_rate.0,
        min_dist.0,
        spread.0,
        a.map(|v| v.0),
        b.map(|v| v.0),
        sigma_iters.0,
        sigma_tolerance.0,
        symmetric_negative,
        seed.0,
    )?;
    let probs_ref = probs.inner();
    let inner = py
        .detach(|| {
            dispatch_riemannian_target!(
                GuMapInner,
                target,
                dtype,
                GuMapSolver::from_fuzzy_set_ref(probs_ref, cfg)
            )
        })
        .map_err(core_error_to_py)?;
    Ok(PyGuMapSolver::from_inner(inner))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_gumap_from_knn(
    py: Python<'_>,
    knn: &PyKnnGraph,
    target: SolverTargetKind,
    iterations: StrictUsize,
    negative_sample_rate: StrictUsize,
    learning_rate: StrictF64,
    min_dist: StrictF64,
    spread: StrictF64,
    a: Option<StrictF64>,
    b: Option<StrictF64>,
    sigma_iters: StrictUsize,
    sigma_tolerance: StrictF64,
    symmetric_negative: bool,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<PyGuMapSolver> {
    let cfg = build_gumap_config(
        iterations.0,
        negative_sample_rate.0,
        learning_rate.0,
        min_dist.0,
        spread.0,
        a.map(|v| v.0),
        b.map(|v| v.0),
        sigma_iters.0,
        sigma_tolerance.0,
        symmetric_negative,
        seed.0,
    )?;
    let knn_ref = knn.inner();
    let inner = py
        .detach(|| {
            dispatch_riemannian_target!(
                GuMapInner,
                target,
                dtype,
                GuMapSolver::from_knn(knn_ref, cfg)
            )
        })
        .map_err(core_error_to_py)?;
    Ok(PyGuMapSolver::from_inner(inner))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_gumap_from_graph_dog(
    py: Python<'_>,
    graph: &PyCsrGraph,
    max_2hop_per_node: StrictUsize,
    target: SolverTargetKind,
    degree: Option<StrictUsize>,
    num_landmarks: StrictUsize,
    degree_normalize: bool,
    p_exp: StrictF64,
    eps_psd: StrictF64,
    t_coarse: StrictF64,
    t1: Option<StrictF64>,
    t2: Option<StrictF64>,
    alpha: Option<StrictF64>,
    beta_floor: Option<StrictF64>,
    min_degree: StrictUsize,
    jl_dim: StrictUsize,
    iterations: StrictUsize,
    negative_sample_rate: StrictUsize,
    learning_rate: StrictF64,
    min_dist: StrictF64,
    spread: StrictF64,
    a: Option<StrictF64>,
    b: Option<StrictF64>,
    sigma_iters: StrictUsize,
    sigma_tolerance: StrictF64,
    symmetric_negative: bool,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<PyGuMapSolver> {
    let cfg = build_gumap_config(
        iterations.0,
        negative_sample_rate.0,
        learning_rate.0,
        min_dist.0,
        spread.0,
        a.map(|v| v.0),
        b.map(|v| v.0),
        sigma_iters.0,
        sigma_tolerance.0,
        symmetric_negative,
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
    let inner = py
        .detach(|| {
            dispatch_riemannian_target!(
                GuMapInner,
                target,
                dtype,
                GuMapSolver::from_graph_dog(&*g, max_2hop_per_node.0, &dog_cfg, cfg)
            )
        })
        .map_err(core_error_to_py)?;
    Ok(PyGuMapSolver::from_inner(inner))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_gumap_from_landmarks(
    py: Python<'_>,
    graph: &PyCsrGraph,
    landmarks: &PyDogVaradhanLandmarks,
    max_2hop_per_node: StrictUsize,
    target: SolverTargetKind,
    iterations: StrictUsize,
    negative_sample_rate: StrictUsize,
    learning_rate: StrictF64,
    min_dist: StrictF64,
    spread: StrictF64,
    a: Option<StrictF64>,
    b: Option<StrictF64>,
    sigma_iters: StrictUsize,
    sigma_tolerance: StrictF64,
    symmetric_negative: bool,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<PyGuMapSolver> {
    let cfg = build_gumap_config(
        iterations.0,
        negative_sample_rate.0,
        learning_rate.0,
        min_dist.0,
        spread.0,
        a.map(|v| v.0),
        b.map(|v| v.0),
        sigma_iters.0,
        sigma_tolerance.0,
        symmetric_negative,
        seed.0,
    )?;
    let g = graph.inner();
    let lm = landmarks.inner();
    let inner = py
        .detach(|| {
            dispatch_riemannian_target!(
                GuMapInner,
                target,
                dtype,
                GuMapSolver::from_landmarks(&*g, lm, max_2hop_per_node.0, cfg)
            )
        })
        .map_err(core_error_to_py)?;
    Ok(PyGuMapSolver::from_inner(inner))
}
