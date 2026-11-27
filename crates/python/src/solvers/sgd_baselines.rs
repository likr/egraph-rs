//! Baseline SGD solver constructors (`ortmann_sparse_stress`, `largevis`, `drgraph`, `graph_cne`).

use egraph::layout as egraph_layout;
use egraph_layout::{
    DrGraphConfig, GraphCneConfig, GraphCneObjective, LargeVisConfig, OrtmannPivotStrategy,
    OrtmannSparseStressConfig, SgdSolver,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py, normalize_enum_key,
};

use super::sgd::PySgdSolver;
use super::sgd_core::SgdInner;

pub(super) fn parse_ortmann_pivot_strategy(raw: &str) -> Result<OrtmannPivotStrategy, String> {
    match normalize_enum_key(raw).as_str() {
        "kmeanssp" | "kmeans" => Ok(OrtmannPivotStrategy::KMeansSp),
        "maxminrandomsp" | "maxminrandom" => Ok(OrtmannPivotStrategy::MaxMinRandomSp),
        "maxminsp" | "maxmin" => Ok(OrtmannPivotStrategy::MaxMinSp),
        _ => Err(format!(
            "unknown Ortmann pivot_strategy {raw:?}; expected 'kmeans_sp', 'maxmin_random_sp', or 'maxmin_sp'"
        )),
    }
}

pub(super) fn parse_graph_cne_objective(raw: &str) -> Result<GraphCneObjective, String> {
    match normalize_enum_key(raw).as_str() {
        "infonce" => Ok(GraphCneObjective::InfoNce),
        "negtsne" => Ok(GraphCneObjective::NegTSne),
        _ => Err(format!(
            "unknown Graph-CNE objective {raw:?}; expected 'infonce' or 'neg_tsne'"
        )),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_ortmann_sparse_stress_solver(
    py: Python<'_>,
    graph: &PyCsrGraph,
    num_pivots: StrictUsize,
    pivot_strategy: &str,
    iterations: StrictUsize,
    eta_max: Option<StrictF64>,
    eta_min: Option<StrictF64>,
    eps: StrictF64,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<PySgdSolver> {
    let strategy = parse_ortmann_pivot_strategy(pivot_strategy).map_err(PyValueError::new_err)?;
    let cfg = OrtmannSparseStressConfig {
        num_pivots: num_pivots.0,
        pivot_strategy: strategy,
        iterations: iterations.0,
        eta_max: eta_max.map(|v| v.0),
        eta_min: eta_min.map(|v| v.0),
        eps: eps.0,
        seed: seed.0,
    };
    cfg.validate().map_err(core_error_to_py)?;
    let g = graph.inner();
    let inner = py
        .detach(|| match dtype {
            ScalarDtype::F64 => {
                SgdSolver::<f64>::ortmann_sparse_stress(&*g, cfg).map(SgdInner::F64)
            }
            ScalarDtype::F32 => {
                SgdSolver::<f32>::ortmann_sparse_stress(&*g, cfg).map(SgdInner::F32)
            }
        })
        .map_err(core_error_to_py)?;
    Ok(PySgdSolver::from_inner(inner))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_largevis_solver(
    py: Python<'_>,
    graph: &PyCsrGraph,
    iterations: StrictUsize,
    negative_samples: StrictUsize,
    gamma: StrictF64,
    initial_lr: StrictF64,
    min_lr: StrictF64,
    degree_exponent: StrictF64,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<PySgdSolver> {
    let cfg = LargeVisConfig {
        iterations: iterations.0,
        negative_samples: negative_samples.0,
        gamma: gamma.0,
        initial_lr: initial_lr.0,
        min_lr: min_lr.0,
        degree_exponent: degree_exponent.0,
        seed: seed.0,
    };
    cfg.validate().map_err(core_error_to_py)?;
    let g = graph.inner();
    let inner = py
        .detach(|| match dtype {
            ScalarDtype::F64 => SgdSolver::<f64>::largevis(&*g, cfg).map(SgdInner::F64),
            ScalarDtype::F32 => SgdSolver::<f32>::largevis(&*g, cfg).map(SgdInner::F32),
        })
        .map_err(core_error_to_py)?;
    Ok(PySgdSolver::from_inner(inner))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_drgraph_solver(
    py: Python<'_>,
    graph: &PyCsrGraph,
    k_order: StrictUsize,
    negative_samples: StrictUsize,
    b: StrictF64,
    gamma_coarse: StrictF64,
    gamma_fine: StrictF64,
    coarsen_stop_ratio: StrictF64,
    iterations: StrictUsize,
    initial_lr: StrictF64,
    min_lr: StrictF64,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<PySgdSolver> {
    let cfg = DrGraphConfig {
        k_order: k_order.0,
        negative_samples: negative_samples.0,
        b: b.0,
        gamma_coarse: gamma_coarse.0,
        gamma_fine: gamma_fine.0,
        coarsen_stop_ratio: coarsen_stop_ratio.0,
        iterations: iterations.0,
        initial_lr: initial_lr.0,
        min_lr: min_lr.0,
        seed: seed.0,
    };
    cfg.validate().map_err(core_error_to_py)?;
    let g = graph.inner();
    let inner = py
        .detach(|| match dtype {
            ScalarDtype::F64 => SgdSolver::<f64>::drgraph(&*g, cfg).map(SgdInner::F64),
            ScalarDtype::F32 => SgdSolver::<f32>::drgraph(&*g, cfg).map(SgdInner::F32),
        })
        .map_err(core_error_to_py)?;
    Ok(PySgdSolver::from_inner(inner))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_graph_cne_solver(
    py: Python<'_>,
    graph: &PyCsrGraph,
    objective: &str,
    negative_samples: StrictUsize,
    iterations: StrictUsize,
    initial_lr: StrictF64,
    min_lr: StrictF64,
    repulsion_weight: StrictF64,
    row_normalize: bool,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<PySgdSolver> {
    let parsed_obj = parse_graph_cne_objective(objective).map_err(PyValueError::new_err)?;
    let cfg = GraphCneConfig {
        objective: parsed_obj,
        negative_samples: negative_samples.0,
        iterations: iterations.0,
        initial_lr: initial_lr.0,
        min_lr: min_lr.0,
        repulsion_weight: repulsion_weight.0,
        row_normalize,
        seed: seed.0,
    };
    cfg.validate().map_err(core_error_to_py)?;
    let g = graph.inner();
    let inner = py
        .detach(|| match dtype {
            ScalarDtype::F64 => SgdSolver::<f64>::graph_cne(&*g, cfg).map(SgdInner::F64),
            ScalarDtype::F32 => SgdSolver::<f32>::graph_cne(&*g, cfg).map(SgdInner::F32),
        })
        .map_err(core_error_to_py)?;
    Ok(PySgdSolver::from_inner(inner))
}
