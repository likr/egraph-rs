//! Constructor builders for `IbUmapSolver`.

use egraph::{layout as egraph_layout, neighborhood as egraph_neighborhood};
use egraph_layout::{FieldLayoutSolver, IbUmapConfig};
use egraph_neighborhood::DogVaradhanConfig;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py, extract_u32_1d,
};
use crate::neighborhood::{PyDogVaradhanLandmarks, PyKnnGraph, PySparseProbabilities};

use super::t_fdp::{FieldLayoutInner, build_ibfft_config, parse_field_backend};

#[allow(clippy::too_many_arguments)]
fn build_ib_umap_config(
    backend: &str,
    n_epochs: usize,
    learning_rate: f64,
    min_dist: f64,
    spread: f64,
    a: Option<f64>,
    b: Option<f64>,
    c_k: f64,
    tau_scale: f64,
    hub_damping: bool,
    theta: f64,
    ibfft_order: usize,
    ibfft_intervals: usize,
    ibfft_min_grid_dim: usize,
    ibfft_max_grid_dim: usize,
    seed: u64,
) -> PyResult<IbUmapConfig> {
    let parsed_backend = parse_field_backend(backend).map_err(PyValueError::new_err)?;
    let ibfft = build_ibfft_config(
        ibfft_order,
        ibfft_intervals,
        ibfft_min_grid_dim,
        ibfft_max_grid_dim,
    )?;
    let cfg = IbUmapConfig {
        backend: parsed_backend,
        iterations: n_epochs,
        learning_rate,
        min_dist,
        spread,
        a,
        b,
        c_k,
        tau_scale,
        hub_damping,
        theta,
        ibfft,
        seed,
    };
    cfg.validate().map_err(core_error_to_py)?;
    Ok(cfg)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_ib_umap_new(
    py: Python<'_>,
    graph: &PyCsrGraph,
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
    n_epochs: StrictUsize,
    learning_rate: StrictF64,
    min_dist: StrictF64,
    spread: StrictF64,
    a: Option<StrictF64>,
    b: Option<StrictF64>,
    c_k: StrictF64,
    tau_scale: StrictF64,
    hub_damping: bool,
    theta: StrictF64,
    ibfft_order: StrictUsize,
    ibfft_intervals: StrictUsize,
    ibfft_min_grid_dim: StrictUsize,
    ibfft_max_grid_dim: StrictUsize,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<FieldLayoutInner> {
    let cfg = build_ib_umap_config(
        backend,
        n_epochs.0,
        learning_rate.0,
        min_dist.0,
        spread.0,
        a.map(|v| v.0),
        b.map(|v| v.0),
        c_k.0,
        tau_scale.0,
        hub_damping,
        theta.0,
        ibfft_order.0,
        ibfft_intervals.0,
        ibfft_min_grid_dim.0,
        ibfft_max_grid_dim.0,
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
    py.detach(|| match dtype {
        ScalarDtype::F64 => {
            FieldLayoutSolver::<f64>::ib_umap(g, max_2hop_per_node.0, &dog_cfg, cfg)
                .map(FieldLayoutInner::F64)
        }
        ScalarDtype::F32 => {
            FieldLayoutSolver::<f32>::ib_umap(g, max_2hop_per_node.0, &dog_cfg, cfg)
                .map(FieldLayoutInner::F32)
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
    backend: &str,
    n_epochs: StrictUsize,
    learning_rate: StrictF64,
    min_dist: StrictF64,
    spread: StrictF64,
    a: Option<StrictF64>,
    b: Option<StrictF64>,
    c_k: StrictF64,
    tau_scale: StrictF64,
    hub_damping: bool,
    theta: StrictF64,
    ibfft_order: StrictUsize,
    ibfft_intervals: StrictUsize,
    ibfft_min_grid_dim: StrictUsize,
    ibfft_max_grid_dim: StrictUsize,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<FieldLayoutInner> {
    let cfg = build_ib_umap_config(
        backend,
        n_epochs.0,
        learning_rate.0,
        min_dist.0,
        spread.0,
        a.map(|v| v.0),
        b.map(|v| v.0),
        c_k.0,
        tau_scale.0,
        hub_damping,
        theta.0,
        ibfft_order.0,
        ibfft_intervals.0,
        ibfft_min_grid_dim.0,
        ibfft_max_grid_dim.0,
        seed.0,
    )?;
    let g = graph.inner();
    let lm = landmarks.inner();
    py.detach(|| match dtype {
        ScalarDtype::F64 => {
            FieldLayoutSolver::<f64>::ib_umap_from_landmarks(g, lm, max_2hop_per_node.0, cfg)
                .map(FieldLayoutInner::F64)
        }
        ScalarDtype::F32 => {
            FieldLayoutSolver::<f32>::ib_umap_from_landmarks(g, lm, max_2hop_per_node.0, cfg)
                .map(FieldLayoutInner::F32)
        }
    })
    .map_err(core_error_to_py)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_ib_umap_from_knn(
    py: Python<'_>,
    knn: &PyKnnGraph,
    sigma_iters: StrictUsize,
    sigma_tol: StrictF64,
    backend: &str,
    n_epochs: StrictUsize,
    learning_rate: StrictF64,
    min_dist: StrictF64,
    spread: StrictF64,
    a: Option<StrictF64>,
    b: Option<StrictF64>,
    c_k: StrictF64,
    tau_scale: StrictF64,
    hub_damping: bool,
    theta: StrictF64,
    ibfft_order: StrictUsize,
    ibfft_intervals: StrictUsize,
    ibfft_min_grid_dim: StrictUsize,
    ibfft_max_grid_dim: StrictUsize,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<FieldLayoutInner> {
    let cfg = build_ib_umap_config(
        backend,
        n_epochs.0,
        learning_rate.0,
        min_dist.0,
        spread.0,
        a.map(|v| v.0),
        b.map(|v| v.0),
        c_k.0,
        tau_scale.0,
        hub_damping,
        theta.0,
        ibfft_order.0,
        ibfft_intervals.0,
        ibfft_min_grid_dim.0,
        ibfft_max_grid_dim.0,
        seed.0,
    )?;
    let knn_ref = knn.inner();
    py.detach(|| match dtype {
        ScalarDtype::F64 => {
            FieldLayoutSolver::<f64>::ib_umap_from_knn(knn_ref, sigma_iters.0, sigma_tol.0, cfg)
                .map(FieldLayoutInner::F64)
        }
        ScalarDtype::F32 => {
            FieldLayoutSolver::<f32>::ib_umap_from_knn(knn_ref, sigma_iters.0, sigma_tol.0, cfg)
                .map(FieldLayoutInner::F32)
        }
    })
    .map_err(core_error_to_py)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_ib_umap_from_fuzzy_set(
    py: Python<'_>,
    probs: &PySparseProbabilities,
    degrees: Option<&Bound<'_, PyAny>>,
    backend: &str,
    n_epochs: StrictUsize,
    learning_rate: StrictF64,
    min_dist: StrictF64,
    spread: StrictF64,
    a: Option<StrictF64>,
    b: Option<StrictF64>,
    c_k: StrictF64,
    tau_scale: StrictF64,
    hub_damping: bool,
    theta: StrictF64,
    ibfft_order: StrictUsize,
    ibfft_intervals: StrictUsize,
    ibfft_min_grid_dim: StrictUsize,
    ibfft_max_grid_dim: StrictUsize,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<FieldLayoutInner> {
    let cfg = build_ib_umap_config(
        backend,
        n_epochs.0,
        learning_rate.0,
        min_dist.0,
        spread.0,
        a.map(|v| v.0),
        b.map(|v| v.0),
        c_k.0,
        tau_scale.0,
        hub_damping,
        theta.0,
        ibfft_order.0,
        ibfft_intervals.0,
        ibfft_min_grid_dim.0,
        ibfft_max_grid_dim.0,
        seed.0,
    )?;
    let degs_opt = match degrees {
        Some(obj) if !obj.is_none() => Some(extract_u32_1d(obj, "degrees")?),
        _ => None,
    };
    let probs_ref = probs.inner();
    py.detach(|| match dtype {
        ScalarDtype::F64 => FieldLayoutSolver::<f64>::ib_umap_from_fuzzy_set_ref(
            probs_ref,
            degs_opt.as_deref(),
            cfg,
        )
        .map(FieldLayoutInner::F64),
        ScalarDtype::F32 => FieldLayoutSolver::<f32>::ib_umap_from_fuzzy_set_ref(
            probs_ref,
            degs_opt.as_deref(),
            cfg,
        )
        .map(FieldLayoutInner::F32),
    })
    .map_err(core_error_to_py)
}
