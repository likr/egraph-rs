//! Ortmann et al. (2016) Sparse Stress SGD builder for `SgdSolver`.

use egraph::layout::{OrtmannPivotStrategy, OrtmannSparseStressConfig, SgdSolver};
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::errors::{core_error_to_py, unsupported_operation_err};
use crate::ffi_convert::{ScalarDtype, StrictF64, StrictU64, StrictUsize, normalize_enum_key};

use super::layout_dispatch::{SolverTargetKind, dispatch_riemannian_target};
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

#[allow(clippy::too_many_arguments)]
pub(super) fn build_ortmann_sparse_stress_solver(
    py: Python<'_>,
    graph: &PyCsrGraph,
    target: SolverTargetKind,
    num_pivots: StrictUsize,
    pivot_strategy: &str,
    iterations: StrictUsize,
    eta_max: Option<StrictF64>,
    eta_min: Option<StrictF64>,
    eps: StrictF64,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<PySgdSolver> {
    let strategy =
        parse_ortmann_pivot_strategy(pivot_strategy).map_err(unsupported_operation_err)?;
    let cfg = OrtmannSparseStressConfig::new()
        .with_num_pivots(num_pivots.0)
        .with_pivot_strategy(strategy)
        .with_iterations(iterations.0)
        .with_eta_max(eta_max.map(|v| v.0))
        .with_eta_min(eta_min.map(|v| v.0))
        .with_eps(eps.0)
        .with_seed(seed.0);
    cfg.validate().map_err(core_error_to_py)?;
    let g = graph.inner();
    let inner = py
        .detach(|| {
            dispatch_riemannian_target!(
                SgdInner,
                target,
                dtype,
                SgdSolver::ortmann_sparse_stress(&*g, cfg)
            )
        })
        .map_err(core_error_to_py)?;
    Ok(PySgdSolver::from_inner(inner))
}
