//! Sparse Barrier SGD constructors and config builders.

use egraph::layout::{BarrierSymmetry, SgdSolver, SparseBarrierConfig};
use egraph::neighborhood::DogVaradhanConfig;
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::errors::{core_error_to_py, unsupported_operation_err};
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, extract_u32_1d, normalize_enum_key,
};
use crate::neighborhood::PyDogVaradhanLandmarks;

use super::layout_dispatch::SolverTargetKind;
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

#[allow(clippy::too_many_arguments)]
pub(super) fn build_sparse_barrier_from_landmarks(
    py: Python<'_>,
    graph: &PyCsrGraph,
    landmarks: &PyDogVaradhanLandmarks,
    pivots: Option<&Bound<'_, PyAny>>,
    target: SolverTargetKind,
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
    let sym = parse_barrier_symmetry(barrier_sym).map_err(unsupported_operation_err)?;
    let pivots_vec = match pivots {
        Some(obj) if !obj.is_none() => Some(extract_u32_1d(obj, "pivots")?),
        _ => None,
    };
    let cfg = SparseBarrierConfig::new()
        .with_max_2hop_per_node(max_2hop_per_node.0)
        .with_hinge_repulsion(hinge_repulsion)
        .with_delta_rel(delta_rel.0)
        .with_barrier_sym(sym)
        .with_omega_1(omega_1.0)
        .with_omega_2(omega_2.0)
        .with_gamma_non2(gamma_non2.0)
        .with_ht_repulsion(ht_repulsion)
        .with_min_ratio(min_ratio.0)
        .with_eps_dist(eps_dist.0)
        .with_neg_per_node(neg_per_node.map(|v| v.0))
        .with_num_pivots(num_pivots.0)
        .with_core_pivots(core_pivots)
        .with_degree_weight_beta(degree_weight_beta.0)
        .with_schedule(sched)
        .with_seed(seed.0);
    let g = graph.inner();
    let lm = landmarks.inner();
    let (inner, b_stats) = py
        .detach(|| {
            let piv = pivots_vec.as_deref();
            match (target, dtype) {
                (SolverTargetKind::Layout2d, ScalarDtype::F64) => {
                    SgdSolver::sparse_barrier_from_landmarks(&*g, lm, piv, &cfg)
                        .map(|(s, st)| (SgdInner::L2F64(s), st))
                }
                (SolverTargetKind::Layout2d, ScalarDtype::F32) => {
                    SgdSolver::sparse_barrier_from_landmarks(&*g, lm, piv, &cfg)
                        .map(|(s, st)| (SgdInner::L2F32(s), st))
                }
                (SolverTargetKind::Layout3d, ScalarDtype::F64) => {
                    SgdSolver::sparse_barrier_from_landmarks(&*g, lm, piv, &cfg)
                        .map(|(s, st)| (SgdInner::L3F64(s), st))
                }
                (SolverTargetKind::Layout3d, ScalarDtype::F32) => {
                    SgdSolver::sparse_barrier_from_landmarks(&*g, lm, piv, &cfg)
                        .map(|(s, st)| (SgdInner::L3F32(s), st))
                }
                (SolverTargetKind::LayoutNd, ScalarDtype::F64) => {
                    SgdSolver::sparse_barrier_from_landmarks(&*g, lm, piv, &cfg)
                        .map(|(s, st)| (SgdInner::NdF64(s), st))
                }
                (SolverTargetKind::LayoutNd, ScalarDtype::F32) => {
                    SgdSolver::sparse_barrier_from_landmarks(&*g, lm, piv, &cfg)
                        .map(|(s, st)| (SgdInner::NdF32(s), st))
                }
                (SolverTargetKind::Manifold2d, ScalarDtype::F64) => {
                    SgdSolver::sparse_barrier_from_landmarks(&*g, lm, piv, &cfg)
                        .map(|(s, st)| (SgdInner::M2F64(s), st))
                }
                (SolverTargetKind::Manifold2d, ScalarDtype::F32) => {
                    SgdSolver::sparse_barrier_from_landmarks(&*g, lm, piv, &cfg)
                        .map(|(s, st)| (SgdInner::M2F32(s), st))
                }
            }
        })
        .map_err(core_error_to_py)?;
    Ok(PySgdSolver::with_stats(inner, Some(b_stats), None))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_sparse_barrier_dog(
    py: Python<'_>,
    graph: &PyCsrGraph,
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
    let sym = parse_barrier_symmetry(barrier_sym).map_err(unsupported_operation_err)?;
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
    let barrier_cfg = SparseBarrierConfig::new()
        .with_max_2hop_per_node(max_2hop_per_node.0)
        .with_hinge_repulsion(hinge_repulsion)
        .with_delta_rel(delta_rel.0)
        .with_barrier_sym(sym)
        .with_omega_1(omega_1.0)
        .with_omega_2(omega_2.0)
        .with_gamma_non2(gamma_non2.0)
        .with_ht_repulsion(ht_repulsion)
        .with_min_ratio(min_ratio.0)
        .with_eps_dist(eps_dist.0)
        .with_neg_per_node(neg_per_node.map(|v| v.0))
        .with_num_pivots(num_pivots.0)
        .with_core_pivots(core_pivots)
        .with_degree_weight_beta(degree_weight_beta.0)
        .with_schedule(sched)
        .with_seed(seed.0);
    let g = graph.inner();
    let (inner, b_stats, d_stats) = py
        .detach(|| match (target, dtype) {
            (SolverTargetKind::Layout2d, ScalarDtype::F64) => {
                SgdSolver::sparse_barrier_dog(&*g, &dog_cfg, &barrier_cfg)
                    .map(|(s, b, d)| (SgdInner::L2F64(s), b, d))
            }
            (SolverTargetKind::Layout2d, ScalarDtype::F32) => {
                SgdSolver::sparse_barrier_dog(&*g, &dog_cfg, &barrier_cfg)
                    .map(|(s, b, d)| (SgdInner::L2F32(s), b, d))
            }
            (SolverTargetKind::Layout3d, ScalarDtype::F64) => {
                SgdSolver::sparse_barrier_dog(&*g, &dog_cfg, &barrier_cfg)
                    .map(|(s, b, d)| (SgdInner::L3F64(s), b, d))
            }
            (SolverTargetKind::Layout3d, ScalarDtype::F32) => {
                SgdSolver::sparse_barrier_dog(&*g, &dog_cfg, &barrier_cfg)
                    .map(|(s, b, d)| (SgdInner::L3F32(s), b, d))
            }
            (SolverTargetKind::LayoutNd, ScalarDtype::F64) => {
                SgdSolver::sparse_barrier_dog(&*g, &dog_cfg, &barrier_cfg)
                    .map(|(s, b, d)| (SgdInner::NdF64(s), b, d))
            }
            (SolverTargetKind::LayoutNd, ScalarDtype::F32) => {
                SgdSolver::sparse_barrier_dog(&*g, &dog_cfg, &barrier_cfg)
                    .map(|(s, b, d)| (SgdInner::NdF32(s), b, d))
            }
            (SolverTargetKind::Manifold2d, ScalarDtype::F64) => {
                SgdSolver::sparse_barrier_dog(&*g, &dog_cfg, &barrier_cfg)
                    .map(|(s, b, d)| (SgdInner::M2F64(s), b, d))
            }
            (SolverTargetKind::Manifold2d, ScalarDtype::F32) => {
                SgdSolver::sparse_barrier_dog(&*g, &dog_cfg, &barrier_cfg)
                    .map(|(s, b, d)| (SgdInner::M2F32(s), b, d))
            }
        })
        .map_err(core_error_to_py)?;
    Ok(PySgdSolver::with_stats(inner, Some(b_stats), Some(d_stats)))
}
