//! Sparse Barrier SGD constructors and diagnostic dictionary converters.

use egraph::{layout as egraph_layout, neighborhood as egraph_neighborhood};
use egraph_layout::{BarrierSymmetry, SgdSolver, SparseBarrierConfig, SparseBarrierStats};
use egraph_neighborhood::{DogVaradhanConfig, DogVaradhanStats};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py, extract_u32_1d,
    normalize_enum_key,
};
use crate::neighborhood::PyDogVaradhanLandmarks;

use super::sgd::PySgdSolver;
use super::sgd_core::{SgdInner, build_schedule_config};

pub(super) fn parse_barrier_symmetry(raw: &str) -> Result<BarrierSymmetry, String> {
    match normalize_enum_key(raw).as_str() {
        "symmax" => Ok(BarrierSymmetry::SymMax),
        "symavg" => Ok(BarrierSymmetry::SymAvg),
        _ => Err(format!(
            "unknown barrier_sym {raw:?}; expected 'sym_max' or 'sym_avg'"
        )),
    }
}

pub(super) fn sparse_barrier_stats_to_py_dict(
    py: Python<'_>,
    stats: &SparseBarrierStats,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("n_pairs_1hop", stats.n_pairs_1hop)?;
    dict.set_item("n_pairs_2hop", stats.n_pairs_2hop)?;
    dict.set_item("n_pairs_non2", stats.n_pairs_non2)?;
    dict.set_item("n_pairs_hinge", stats.n_pairs_hinge)?;
    dict.set_item("n_pairs_two_sided", stats.n_pairs_two_sided)?;
    dict.set_item("perplexity_val", stats.perplexity_val)?;
    dict.set_item("per_node_neg_used", stats.per_node_neg_used)?;
    dict.set_item("perplexity_source", stats.perplexity_source)?;
    dict.set_item("d_floor", stats.d_floor)?;
    dict.set_item("gamma_eff", stats.gamma_eff)?;
    Ok(dict.unbind())
}

pub(super) fn dog_stats_to_py_dict(
    py: Python<'_>,
    stats: &DogVaradhanStats,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("t1", stats.t1)?;
    dict.set_item("t2", stats.t2)?;
    dict.set_item("alpha", stats.alpha)?;
    dict.set_item("degree", stats.degree)?;
    dict.set_item("delta_psd", stats.delta_psd)?;
    dict.set_item("h_min", stats.h_min)?;
    dict.set_item("k_min", stats.k_min)?;
    dict.set_item("k_med", stats.k_med)?;
    dict.set_item("beta_auto", stats.beta_auto)?;
    dict.set_item("beta_eff", stats.beta_eff)?;
    dict.set_item("n_landmarks", stats.n_landmarks)?;
    dict.set_item("t_coarse", stats.t_coarse)?;
    dict.set_item("used_coarse_scaffold", stats.used_coarse_scaffold)?;
    Ok(dict.unbind())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_sparse_barrier_from_landmarks(
    py: Python<'_>,
    graph: &PyCsrGraph,
    landmarks: &PyDogVaradhanLandmarks,
    pivots: Option<&Bound<'_, PyAny>>,
    max_2hop_per_node: StrictUsize,
    hinge_repulsion: bool,
    delta_rel: StrictF64,
    barrier_sym: &str,
    omega_1: StrictF64,
    omega_2: StrictF64,
    gamma_non2: StrictF64,
    ht_repulsion: bool,
    min_ratio: StrictF64,
    eps_dist: StrictF64,
    neg_per_node: Option<StrictUsize>,
    num_pivots: StrictUsize,
    core_pivots: bool,
    degree_weight_beta: StrictF64,
    iterations: StrictUsize,
    eps: StrictF64,
    schedule: &str,
    step_scale: StrictF64,
    max_mu: Option<StrictF64>,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<PySgdSolver> {
    let sched = build_schedule_config(
        iterations.0,
        eps.0,
        schedule,
        step_scale.0,
        max_mu.map(|v| v.0),
    )?;
    let sym = parse_barrier_symmetry(barrier_sym).map_err(PyValueError::new_err)?;
    let pivots_vec = match pivots {
        Some(obj) if !obj.is_none() => Some(extract_u32_1d(obj, "pivots")?),
        _ => None,
    };
    let cfg = SparseBarrierConfig {
        max_2hop_per_node: max_2hop_per_node.0,
        hinge_repulsion,
        delta_rel: delta_rel.0,
        barrier_sym: sym,
        omega_1: omega_1.0,
        omega_2: omega_2.0,
        gamma_non2: gamma_non2.0,
        ht_repulsion,
        min_ratio: min_ratio.0,
        eps_dist: eps_dist.0,
        neg_per_node: neg_per_node.map(|v| v.0),
        num_pivots: num_pivots.0,
        core_pivots,
        degree_weight_beta: degree_weight_beta.0,
        schedule: sched,
        seed: seed.0,
    };
    let g = graph.inner();
    let lm = landmarks.inner();
    let (inner, b_stats) = py
        .detach(|| match dtype {
            ScalarDtype::F64 => SgdSolver::<f64>::sparse_barrier_from_landmarks(
                &*g,
                lm,
                pivots_vec.as_deref(),
                &cfg,
            )
            .map(|(s, st)| (SgdInner::F64(s), st)),
            ScalarDtype::F32 => SgdSolver::<f32>::sparse_barrier_from_landmarks(
                &*g,
                lm,
                pivots_vec.as_deref(),
                &cfg,
            )
            .map(|(s, st)| (SgdInner::F32(s), st)),
        })
        .map_err(core_error_to_py)?;
    Ok(PySgdSolver::with_stats(inner, Some(b_stats), None))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_sparse_barrier_dog(
    py: Python<'_>,
    graph: &PyCsrGraph,
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
    max_2hop_per_node: StrictUsize,
    hinge_repulsion: bool,
    delta_rel: StrictF64,
    barrier_sym: &str,
    omega_1: StrictF64,
    omega_2: StrictF64,
    gamma_non2: StrictF64,
    ht_repulsion: bool,
    min_ratio: StrictF64,
    eps_dist: StrictF64,
    neg_per_node: Option<StrictUsize>,
    num_pivots: StrictUsize,
    core_pivots: bool,
    degree_weight_beta: StrictF64,
    iterations: StrictUsize,
    eps: StrictF64,
    schedule: &str,
    step_scale: StrictF64,
    max_mu: Option<StrictF64>,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<PySgdSolver> {
    let sched = build_schedule_config(
        iterations.0,
        eps.0,
        schedule,
        step_scale.0,
        max_mu.map(|v| v.0),
    )?;
    let sym = parse_barrier_symmetry(barrier_sym).map_err(PyValueError::new_err)?;
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
    let barrier_cfg = SparseBarrierConfig {
        max_2hop_per_node: max_2hop_per_node.0,
        hinge_repulsion,
        delta_rel: delta_rel.0,
        barrier_sym: sym,
        omega_1: omega_1.0,
        omega_2: omega_2.0,
        gamma_non2: gamma_non2.0,
        ht_repulsion,
        min_ratio: min_ratio.0,
        eps_dist: eps_dist.0,
        neg_per_node: neg_per_node.map(|v| v.0),
        num_pivots: num_pivots.0,
        core_pivots,
        degree_weight_beta: degree_weight_beta.0,
        schedule: sched,
        seed: seed.0,
    };
    let g = graph.inner();
    let (inner, b_stats, d_stats) = py
        .detach(|| match dtype {
            ScalarDtype::F64 => SgdSolver::<f64>::sparse_barrier_dog(&*g, &dog_cfg, &barrier_cfg)
                .map(|(s, b, d)| (SgdInner::F64(s), b, d)),
            ScalarDtype::F32 => SgdSolver::<f32>::sparse_barrier_dog(&*g, &dog_cfg, &barrier_cfg)
                .map(|(s, b, d)| (SgdInner::F32(s), b, d)),
        })
        .map_err(core_error_to_py)?;
    Ok(PySgdSolver::with_stats(inner, Some(b_stats), Some(d_stats)))
}
