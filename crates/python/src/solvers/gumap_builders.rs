//! Constructor builders for `GuMapSolver`.

use egraph::{layout as egraph_layout, neighborhood as egraph_neighborhood};
use egraph_layout::{GuMapConfig, GuMapSolver};
use egraph_neighborhood::DogVaradhanConfig;
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py};
use crate::neighborhood::{PyDogVaradhanLandmarks, PyKnnGraph, PySparseProbabilities};

use super::gumap::{GuMapInner, PyGuMapSolver};

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
    let cfg = GuMapConfig {
        iterations,
        negative_sample_rate,
        learning_rate,
        min_dist,
        spread,
        a,
        b,
        sigma_iters,
        sigma_tolerance,
        symmetric_negative,
        seed,
    };
    cfg.validate().map_err(core_error_to_py)?;
    Ok(cfg)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_gumap_from_fuzzy_set(
    py: Python<'_>,
    probs: &PySparseProbabilities,
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
        .detach(|| match dtype {
            ScalarDtype::F64 => {
                GuMapSolver::<f64>::from_fuzzy_set_ref(probs_ref, cfg).map(GuMapInner::F64)
            }
            ScalarDtype::F32 => {
                GuMapSolver::<f32>::from_fuzzy_set_ref(probs_ref, cfg).map(GuMapInner::F32)
            }
        })
        .map_err(core_error_to_py)?;
    Ok(PyGuMapSolver::from_inner(inner))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_gumap_from_knn(
    py: Python<'_>,
    knn: &PyKnnGraph,
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
        .detach(|| match dtype {
            ScalarDtype::F64 => GuMapSolver::<f64>::from_knn(knn_ref, cfg).map(GuMapInner::F64),
            ScalarDtype::F32 => GuMapSolver::<f32>::from_knn(knn_ref, cfg).map(GuMapInner::F32),
        })
        .map_err(core_error_to_py)?;
    Ok(PyGuMapSolver::from_inner(inner))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_gumap_from_graph_dog(
    py: Python<'_>,
    graph: &PyCsrGraph,
    max_2hop_per_node: StrictUsize,
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
    let dog_cfg = DogVaradhanConfig {
        t1: t1.map(|v| v.0),
        t2: t2.map(|v| v.0),
        alpha: alpha.map(|v| v.0),
        beta_floor: beta_floor.map(|v| v.0),
        degree: degree.map(|v| v.0),
        n_landmarks: num_landmarks.0,
        min_degree: min_degree.0,
        jl_dim: jl_dim.0,
        eps_psd: eps_psd.0,
        degree_normalize,
        p_exp: p_exp.0,
        t_coarse: t_coarse.0,
        seed: seed.0,
        ..DogVaradhanConfig::default()
    };
    let g = graph.inner();
    let inner = py
        .detach(|| match dtype {
            ScalarDtype::F64 => {
                GuMapSolver::<f64>::from_graph_dog(&*g, max_2hop_per_node.0, &dog_cfg, cfg)
                    .map(GuMapInner::F64)
            }
            ScalarDtype::F32 => {
                GuMapSolver::<f32>::from_graph_dog(&*g, max_2hop_per_node.0, &dog_cfg, cfg)
                    .map(GuMapInner::F32)
            }
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
        .detach(|| match dtype {
            ScalarDtype::F64 => {
                GuMapSolver::<f64>::from_landmarks(&*g, lm, max_2hop_per_node.0, cfg)
                    .map(GuMapInner::F64)
            }
            ScalarDtype::F32 => {
                GuMapSolver::<f32>::from_landmarks(&*g, lm, max_2hop_per_node.0, cfg)
                    .map(GuMapInner::F32)
            }
        })
        .map_err(core_error_to_py)?;
    Ok(PyGuMapSolver::from_inner(inner))
}
