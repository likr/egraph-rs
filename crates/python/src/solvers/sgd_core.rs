//! Core SGD constructors, schedule helpers, and polymorphic trait implementations.

use egraph::{core as egraph_core, layout as egraph_layout};
use egraph_layout::{SgdScheduleConfig, SgdScheduleKind, SgdSolver, StreamingNegativeConfig};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::array_extract::extract_f64_1d;
use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py, extract_coords_any,
    extract_u32_1d, normalize_enum_key,
};
use crate::solver_common::{impl_dual_solver, vec_f64_to_f32_checked};

use super::layout_dispatch::{impl_dual_manifold_solver, impl_dual_nd_solver};
use super::sgd::PySgdSolver;

impl_dual_solver!(SgdInner, SgdSolver);
impl_dual_nd_solver!(SgdInner, SgdSolver);
impl_dual_manifold_solver!(SgdInner, SgdSolver);

impl SgdInner {
    pub(super) fn num_nodes(&self) -> usize {
        match self {
            Self::F64(s) => s.num_nodes(),
            Self::F32(s) => s.num_nodes(),
        }
    }

    pub(super) fn num_springs(&self) -> usize {
        match self {
            Self::F64(s) => s.num_springs(),
            Self::F32(s) => s.num_springs(),
        }
    }

    pub(super) fn num_hinges(&self) -> usize {
        match self {
            Self::F64(s) => s.num_hinges(),
            Self::F32(s) => s.num_hinges(),
        }
    }

    pub(super) fn current_eta(&self) -> f64 {
        match self {
            Self::F64(s) => s.current_eta(),
            Self::F32(s) => s.current_eta(),
        }
    }

    pub(super) fn supports_manifold(&self) -> bool {
        match self {
            Self::F64(s) => s.supports_manifold(),
            Self::F32(s) => s.supports_manifold(),
        }
    }
}

pub(super) fn parse_sgd_schedule_kind(raw: &str) -> Result<SgdScheduleKind, String> {
    match normalize_enum_key(raw).as_str() {
        "exponential" => Ok(SgdScheduleKind::Exponential),
        "linear" => Ok(SgdScheduleKind::Linear),
        "quadratic" => Ok(SgdScheduleKind::Quadratic),
        "reciprocal" => Ok(SgdScheduleKind::Reciprocal),
        "constant" => Ok(SgdScheduleKind::Constant),
        "squared" => Ok(SgdScheduleKind::Squared),
        _ => Err(format!(
            "unknown SGD schedule {raw:?}; expected 'exponential', 'linear', 'quadratic', 'reciprocal', 'constant', or 'squared'"
        )),
    }
}

pub(super) fn build_schedule_config(
    iterations: usize,
    eps: f64,
    schedule: &str,
    step_scale: f64,
    max_mu: Option<f64>,
) -> PyResult<SgdScheduleConfig> {
    let kind = parse_sgd_schedule_kind(schedule).map_err(PyValueError::new_err)?;
    Ok(SgdScheduleConfig::new(iterations)
        .with_eps(eps)
        .with_kind(kind)
        .with_step_scale(step_scale)
        .with_max_mu(max_mu))
}

pub(super) fn default_inv_sq_weights(dists: &[f64]) -> Vec<f64> {
    dists
        .iter()
        .map(|&d| {
            if d.is_finite() && d > 0.0 {
                1.0 / (d * d)
            } else {
                0.0
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_from_pairs(
    py: Python<'_>,
    num_nodes: StrictUsize,
    src: &Bound<'_, PyAny>,
    tgt: &Bound<'_, PyAny>,
    dij: &Bound<'_, PyAny>,
    dji: Option<&Bound<'_, PyAny>>,
    wij: Option<&Bound<'_, PyAny>>,
    wji: Option<&Bound<'_, PyAny>>,
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
    let src_u32 = extract_u32_1d(src, "src")?;
    let tgt_u32 = extract_u32_1d(tgt, "tgt")?;
    let dij_f64 = extract_f64_1d(dij, "dij")?;
    let dji_f64 = match dji {
        Some(obj) if !obj.is_none() => extract_f64_1d(obj, "dji")?,
        _ => dij_f64.clone(),
    };
    let wij_f64 = match wij {
        Some(obj) if !obj.is_none() => extract_f64_1d(obj, "wij")?,
        _ => default_inv_sq_weights(&dij_f64),
    };
    let wji_f64 = match wji {
        Some(obj) if !obj.is_none() => extract_f64_1d(obj, "wji")?,
        _ if wij.is_none_or(|o| o.is_none()) => default_inv_sq_weights(&dji_f64),
        _ => wij_f64.clone(),
    };

    let inner = match dtype {
        ScalarDtype::F64 => py
            .detach(|| {
                SgdSolver::<f64>::from_pairs(
                    num_nodes.0,
                    src_u32,
                    tgt_u32,
                    dij_f64,
                    dji_f64,
                    wij_f64,
                    wji_f64,
                    sched,
                    seed.0,
                )
                .map(SgdInner::F64)
            })
            .map_err(core_error_to_py)?,
        ScalarDtype::F32 => {
            let dij_f32 = vec_f64_to_f32_checked(&dij_f64, "dij")?;
            let dji_f32 = vec_f64_to_f32_checked(&dji_f64, "dji")?;
            let wij_f32 = vec_f64_to_f32_checked(&wij_f64, "wij")?;
            let wji_f32 = vec_f64_to_f32_checked(&wji_f64, "wji")?;
            py.detach(|| {
                SgdSolver::<f32>::from_pairs(
                    num_nodes.0,
                    src_u32,
                    tgt_u32,
                    dij_f32,
                    dji_f32,
                    wij_f32,
                    wji_f32,
                    sched,
                    seed.0,
                )
                .map(SgdInner::F32)
            })
            .map_err(core_error_to_py)?
        }
    };
    Ok(PySgdSolver::from_inner(inner))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_from_pairs_with_hinges(
    py: Python<'_>,
    num_nodes: StrictUsize,
    spring_src: &Bound<'_, PyAny>,
    spring_tgt: &Bound<'_, PyAny>,
    spring_dij: &Bound<'_, PyAny>,
    spring_dji: Option<&Bound<'_, PyAny>>,
    spring_wij: Option<&Bound<'_, PyAny>>,
    spring_wji: Option<&Bound<'_, PyAny>>,
    hinge_src: &Bound<'_, PyAny>,
    hinge_tgt: &Bound<'_, PyAny>,
    hinge_dij: &Bound<'_, PyAny>,
    hinge_dji: Option<&Bound<'_, PyAny>>,
    hinge_wij: Option<&Bound<'_, PyAny>>,
    hinge_wji: Option<&Bound<'_, PyAny>>,
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
    let s_src = extract_u32_1d(spring_src, "spring_src")?;
    let s_tgt = extract_u32_1d(spring_tgt, "spring_tgt")?;
    let s_dij = extract_f64_1d(spring_dij, "spring_dij")?;
    let s_dji = match spring_dji {
        Some(obj) if !obj.is_none() => extract_f64_1d(obj, "spring_dji")?,
        _ => s_dij.clone(),
    };
    let s_wij = match spring_wij {
        Some(obj) if !obj.is_none() => extract_f64_1d(obj, "spring_wij")?,
        _ => default_inv_sq_weights(&s_dij),
    };
    let s_wji = match spring_wji {
        Some(obj) if !obj.is_none() => extract_f64_1d(obj, "spring_wji")?,
        _ if spring_wij.is_none_or(|o| o.is_none()) => default_inv_sq_weights(&s_dji),
        _ => s_wij.clone(),
    };

    let h_src = extract_u32_1d(hinge_src, "hinge_src")?;
    let h_tgt = extract_u32_1d(hinge_tgt, "hinge_tgt")?;
    let h_dij = extract_f64_1d(hinge_dij, "hinge_dij")?;
    let h_dji = match hinge_dji {
        Some(obj) if !obj.is_none() => extract_f64_1d(obj, "hinge_dji")?,
        _ => h_dij.clone(),
    };
    let h_wij = match hinge_wij {
        Some(obj) if !obj.is_none() => extract_f64_1d(obj, "hinge_wij")?,
        _ => default_inv_sq_weights(&h_dij),
    };
    let h_wji = match hinge_wji {
        Some(obj) if !obj.is_none() => extract_f64_1d(obj, "hinge_wji")?,
        _ if hinge_wij.is_none_or(|o| o.is_none()) => default_inv_sq_weights(&h_dji),
        _ => h_wij.clone(),
    };

    let inner = match dtype {
        ScalarDtype::F64 => py
            .detach(|| {
                SgdSolver::<f64>::from_pairs_with_hinges(
                    num_nodes.0,
                    s_src,
                    s_tgt,
                    s_dij,
                    s_dji,
                    s_wij,
                    s_wji,
                    h_src,
                    h_tgt,
                    h_dij,
                    h_dji,
                    h_wij,
                    h_wji,
                    sched,
                    seed.0,
                )
                .map(SgdInner::F64)
            })
            .map_err(core_error_to_py)?,
        ScalarDtype::F32 => {
            let s_dij_32 = vec_f64_to_f32_checked(&s_dij, "spring_dij")?;
            let s_dji_32 = vec_f64_to_f32_checked(&s_dji, "spring_dji")?;
            let s_wij_32 = vec_f64_to_f32_checked(&s_wij, "spring_wij")?;
            let s_wji_32 = vec_f64_to_f32_checked(&s_wji, "spring_wji")?;
            let h_dij_32 = vec_f64_to_f32_checked(&h_dij, "hinge_dij")?;
            let h_dji_32 = vec_f64_to_f32_checked(&h_dji, "hinge_dji")?;
            let h_wij_32 = vec_f64_to_f32_checked(&h_wij, "hinge_wij")?;
            let h_wji_32 = vec_f64_to_f32_checked(&h_wji, "hinge_wji")?;
            py.detach(|| {
                SgdSolver::<f32>::from_pairs_with_hinges(
                    num_nodes.0,
                    s_src,
                    s_tgt,
                    s_dij_32,
                    s_dji_32,
                    s_wij_32,
                    s_wji_32,
                    h_src,
                    h_tgt,
                    h_dij_32,
                    h_dji_32,
                    h_wij_32,
                    h_wji_32,
                    sched,
                    seed.0,
                )
                .map(SgdInner::F32)
            })
            .map_err(core_error_to_py)?
        }
    };
    Ok(PySgdSolver::from_inner(inner))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_full(
    py: Python<'_>,
    graph: &PyCsrGraph,
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
    let g = graph.inner();
    let inner = py
        .detach(|| match dtype {
            ScalarDtype::F64 => SgdSolver::<f64>::full(&*g, sched, seed.0).map(SgdInner::F64),
            ScalarDtype::F32 => SgdSolver::<f32>::full(&*g, sched, seed.0).map(SgdInner::F32),
        })
        .map_err(core_error_to_py)?;
    Ok(PySgdSolver::from_inner(inner))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_random_pair(
    py: Python<'_>,
    graph: &PyCsrGraph,
    pairs_per_node: StrictUsize,
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
    let g = graph.inner();
    let inner = py
        .detach(|| match dtype {
            ScalarDtype::F64 => SgdSolver::<f64>::random_pair(&*g, pairs_per_node.0, sched, seed.0)
                .map(SgdInner::F64),
            ScalarDtype::F32 => SgdSolver::<f32>::random_pair(&*g, pairs_per_node.0, sched, seed.0)
                .map(SgdInner::F32),
        })
        .map_err(core_error_to_py)?;
    Ok(PySgdSolver::from_inner(inner))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_omega(
    py: Python<'_>,
    graph: &PyCsrGraph,
    embedding: &Bound<'_, PyAny>,
    k: StrictUsize,
    min_dist: StrictF64,
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
    let (emb_flat, nrows, dims) = extract_coords_any(embedding, "embedding")?;
    let g = graph.inner();
    if nrows != g.num_nodes() {
        return Err(PyValueError::new_err(format!(
            "embedding row count {nrows} does not match graph.num_nodes ({})",
            g.num_nodes()
        )));
    }
    let inner = py
        .detach(|| match dtype {
            ScalarDtype::F64 => {
                SgdSolver::<f64>::omega(&*g, &emb_flat, dims, k.0, min_dist.0, sched, seed.0)
                    .map(SgdInner::F64)
            }
            ScalarDtype::F32 => {
                SgdSolver::<f32>::omega(&*g, &emb_flat, dims, k.0, min_dist.0, sched, seed.0)
                    .map(SgdInner::F32)
            }
        })
        .map_err(core_error_to_py)?;
    Ok(PySgdSolver::from_inner(inner))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_streaming_negatives(
    py: Python<'_>,
    graph: &PyCsrGraph,
    negatives_per_node: StrictUsize,
    target_distance: StrictF64,
    weight: StrictF64,
    hinge_only: bool,
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
    let neg_cfg = StreamingNegativeConfig {
        negatives_per_node: negatives_per_node.0,
        target_distance: target_distance.0,
        weight: weight.0,
        hinge_only,
    };
    let g = graph.inner();
    let inner = py
        .detach(|| match dtype {
            ScalarDtype::F64 => SgdSolver::<f64>::streaming_negatives(&*g, neg_cfg, sched, seed.0)
                .map(SgdInner::F64),
            ScalarDtype::F32 => SgdSolver::<f32>::streaming_negatives(&*g, neg_cfg, sched, seed.0)
                .map(SgdInner::F32),
        })
        .map_err(core_error_to_py)?;
    Ok(PySgdSolver::from_inner(inner))
}

#[allow(dead_code)]
fn _check_core(_: egraph_core::CoreError) {}
