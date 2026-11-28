//! Helper functions for AMG-PCG solver construction and result marshalling.

use egraph::core::{CoreError, CoreResult};
use egraph::linalg::{
    AmgPcgConfig, AmgPcgSolveResult, AmgPcgSolver, AmgPcgWorkspace, CsrMatrix, LaplacianKind,
};
use pyo3::prelude::*;

use super::amg_results::{PyAmgBatchSolveResult, PyAmgSolveResult};
use super::matrix::PyCsrMatrix;
use crate::array_extract::{extract_f64_1d, extract_f64_2d};
use crate::csr_graph::PyCsrGraph;
use crate::errors::invalid_input_err;
use crate::ffi_convert::{
    StrictF64, StrictU64, StrictUsize, core_error_to_py, extract_optional_weights_1d,
    extract_u32_1d, normalize_enum_key, solver_status_str,
};

pub(super) fn parse_amg_laplacian_kind(raw: &str) -> Result<LaplacianKind, String> {
    match normalize_enum_key(raw).as_str() {
        "combinatorial" | "unnormalized" | "standard" => Ok(LaplacianKind::Combinatorial),
        "symmetricnormalized" | "symnorm" | "normalized" | "sym" => {
            Ok(LaplacianKind::SymmetricNormalized)
        }
        _ => Err(format!(
            "unknown Laplacian kind {raw:?}; expected 'combinatorial' or 'symmetric_normalized'"
        )),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_amg_pcg_config(
    rel_tol: StrictF64,
    abs_tol: StrictF64,
    max_iterations: StrictUsize,
    theta: StrictF64,
    max_levels: StrictUsize,
    max_coarse_size: StrictUsize,
    pre_sweeps: StrictUsize,
    post_sweeps: StrictUsize,
    max_agg_size: StrictUsize,
    diag_eps: StrictF64,
    eig_eps: StrictF64,
    sym_tol: StrictF64,
    seed: StrictU64,
    randomize_ties: bool,
) -> PyResult<AmgPcgConfig> {
    let cfg = AmgPcgConfig::new()
        .with_rel_tol(rel_tol.0)
        .with_abs_tol(abs_tol.0)
        .with_max_iterations(max_iterations.0)
        .with_theta(theta.0)
        .with_max_levels(max_levels.0)
        .with_max_coarse_size(max_coarse_size.0)
        .with_pre_sweeps(pre_sweeps.0)
        .with_post_sweeps(post_sweeps.0)
        .with_max_agg_size(max_agg_size.0)
        .with_diag_eps(diag_eps.0)
        .with_eig_eps(eig_eps.0)
        .with_sym_tol(sym_tol.0)
        .with_seed(seed.0)
        .with_randomize_ties(randomize_ties);
    cfg.validate().map_err(core_error_to_py)?;
    Ok(cfg)
}

pub(super) fn create_amg_from_csr(
    py: Python<'_>,
    num_nodes: StrictUsize,
    indptr: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    values: &Bound<'_, PyAny>,
    cfg: AmgPcgConfig,
    near_nullspace: Option<&Bound<'_, PyAny>>,
) -> PyResult<(AmgPcgSolver, AmgPcgWorkspace)> {
    let indptr_v = extract_u32_1d(indptr, "indptr")?;
    let indices_v = extract_u32_1d(indices, "indices")?;
    let values_v = extract_f64_1d(values, "values")?;
    let ns_v = extract_optional_weights_1d(near_nullspace, "near_nullspace")?;
    let n = num_nodes.0;
    let inner = py
        .detach(|| -> CoreResult<AmgPcgSolver> {
            let mat = CsrMatrix::from_csr(n, indptr_v, indices_v, values_v)?;
            AmgPcgSolver::with_near_nullspace(&mat, ns_v.as_deref(), cfg)
        })
        .map_err(core_error_to_py)?;
    let ws = inner.new_workspace();
    Ok((inner, ws))
}

pub(super) fn create_amg_from_csr_matrix(
    py: Python<'_>,
    matrix: &PyCsrMatrix,
    cfg: AmgPcgConfig,
    near_nullspace: Option<&Bound<'_, PyAny>>,
) -> PyResult<(AmgPcgSolver, AmgPcgWorkspace)> {
    let ns_v = extract_optional_weights_1d(near_nullspace, "near_nullspace")?;
    let mat = matrix.inner();
    let inner = py
        .detach(|| AmgPcgSolver::with_near_nullspace(mat, ns_v.as_deref(), cfg))
        .map_err(core_error_to_py)?;
    let ws = inner.new_workspace();
    Ok((inner, ws))
}

pub(super) fn create_amg_from_regularized_laplacian(
    py: Python<'_>,
    graph: &PyCsrGraph,
    gamma: StrictF64,
    cfg: AmgPcgConfig,
) -> PyResult<(AmgPcgSolver, AmgPcgWorkspace)> {
    let g = graph.inner();
    let gamma_val = gamma.0;
    let inner = py
        .detach(|| AmgPcgSolver::from_regularized_sym_laplacian(&*g, gamma_val, cfg))
        .map_err(core_error_to_py)?;
    let ws = inner.new_workspace();
    Ok((inner, ws))
}

pub(super) fn create_amg_from_laplacian(
    py: Python<'_>,
    graph: &PyCsrGraph,
    kind: &str,
    shift: StrictF64,
    cfg: AmgPcgConfig,
) -> PyResult<(AmgPcgSolver, AmgPcgWorkspace)> {
    let lap_kind = parse_amg_laplacian_kind(kind).map_err(invalid_input_err)?;
    let g = graph.inner();
    let shift_val = shift.0;
    let inner = py
        .detach(|| AmgPcgSolver::from_laplacian(&*g, lap_kind, shift_val, cfg))
        .map_err(core_error_to_py)?;
    let ws = inner.new_workspace();
    Ok((inner, ws))
}

pub(super) fn solve_amg(
    inner: &AmgPcgSolver,
    ws: &mut AmgPcgWorkspace,
    py: Python<'_>,
    b: &Bound<'_, PyAny>,
    x0: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyAmgSolveResult> {
    let b_vec = extract_f64_1d(b, "b")?;
    let x0_vec = extract_optional_weights_1d(x0, "x0")?;
    let res = py
        .detach(|| -> CoreResult<AmgPcgSolveResult> {
            inner.init_solve(&b_vec, x0_vec.as_deref(), ws)?;
            inner.run(ws)?;
            Ok(ws.result())
        })
        .map_err(core_error_to_py)?;
    Ok(PyAmgSolveResult::new(ws.x().to_vec(), res))
}

pub(super) fn solve_batch_amg(
    inner: &AmgPcgSolver,
    ws: &mut AmgPcgWorkspace,
    py: Python<'_>,
    b: &Bound<'_, PyAny>,
    x0: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyAmgBatchSolveResult> {
    let (b_flat, nrows, ncols) = extract_f64_2d(b, "b")?;
    if nrows != inner.num_nodes() {
        return Err(core_error_to_py(CoreError::dimension_mismatch(
            inner.num_nodes(),
            nrows,
            "AmgPcgSolver.solve_batch::b",
        )));
    }
    let mut x_flat = match x0 {
        Some(obj) if !obj.is_none() => {
            let (xf, xr, xc) = extract_f64_2d(obj, "x0")?;
            if xr != nrows || xc != ncols {
                return Err(core_error_to_py(CoreError::dimension_mismatch(
                    nrows * ncols,
                    xr * xc,
                    "AmgPcgSolver.solve_batch::x0",
                )));
            }
            xf
        }
        _ => vec![0.0_f64; nrows * ncols],
    };
    let res = py
        .detach(|| inner.solve_batch(&b_flat, &mut x_flat, ncols, ws))
        .map_err(core_error_to_py)?;
    Ok(PyAmgBatchSolveResult::new(x_flat, nrows, ncols, res))
}

pub(super) fn start_solve_amg(
    inner: &AmgPcgSolver,
    ws: &mut AmgPcgWorkspace,
    py: Python<'_>,
    b: &Bound<'_, PyAny>,
    x0: Option<&Bound<'_, PyAny>>,
) -> PyResult<&'static str> {
    let b_vec = extract_f64_1d(b, "b")?;
    let x0_vec = extract_optional_weights_1d(x0, "x0")?;
    let status = py
        .detach(|| inner.init_solve(&b_vec, x0_vec.as_deref(), ws))
        .map_err(core_error_to_py)?;
    Ok(solver_status_str(status))
}
