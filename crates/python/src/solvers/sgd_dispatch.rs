//! Additional `SgdSolver` dispatch helpers (`from_packed_symmetric`, schedule window/reset, `phase_iterations`).

use egraph::core::{CoreError, Layout2d, ManifoldLayout2d, SolverStatus};
use egraph::layout::sgd::PackedSymPair;
use egraph::layout::{SgdScheduleWindowLayout, SgdSolver};
use numpy::{PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::array_extract::extract_f64_1d;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py, extract_u32_1d,
    is_non_sequence_container, solver_status_str,
};
use crate::layout_buffers::{Layout2dInner, PyLayout2d};
use crate::manifold::{
    ExtractedManifold, PyManifoldLayout2d, extract_manifold_for_solver, write_back_manifold,
};
use crate::solver_common::{
    DualIterativeSolver, SolverLayout, extract_solver_layout, vec_f64_to_f32_checked,
    write_back_solver_layout,
};

use super::sgd::PySgdSolver;
use super::sgd_core::{SgdInner, build_schedule_config};

fn extract_bool_1d(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<bool>> {
    if is_non_sequence_container(obj) {
        return Err(PyTypeError::new_err(format!(
            "{name} must be a 1D array or sequence of booleans"
        )));
    }
    if let Ok(untyped) = obj.cast::<PyUntypedArray>()
        && untyped.ndim() != 1
    {
        return Err(PyValueError::new_err(format!(
            "{name} must be a 1D array, got {}D",
            untyped.ndim()
        )));
    }
    let iter = obj.try_iter().map_err(|_| {
        PyTypeError::new_err(format!("{name} must be a 1D array or sequence of booleans"))
    })?;
    let mut out = Vec::new();
    for item_res in iter {
        let item = item_res?;
        if let Ok(b) = item.extract::<bool>() {
            out.push(b);
        } else if let Ok(v) = item.extract::<i64>() {
            if v == 0 {
                out.push(false);
            } else if v == 1 {
                out.push(true);
            } else {
                return Err(PyValueError::new_err(format!(
                    "{name} integer entries must be 0 or 1, got {v}"
                )));
            }
        } else {
            return Err(PyTypeError::new_err(format!(
                "{name} entries must be booleans or 0/1 integers"
            )));
        }
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_from_packed_symmetric(
    py: Python<'_>,
    num_nodes: StrictUsize,
    src: &Bound<'_, PyAny>,
    tgt: &Bound<'_, PyAny>,
    dists: &Bound<'_, PyAny>,
    weights: &Bound<'_, PyAny>,
    is_hinge: &Bound<'_, PyAny>,
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
    let d_f64 = extract_f64_1d(dists, "dists")?;
    let w_f64 = extract_f64_1d(weights, "weights")?;
    let hinge_vec = extract_bool_1d(is_hinge, "is_hinge")?;

    let len = src_u32.len();
    for (name, actual) in [
        ("tgt", tgt_u32.len()),
        ("dists", d_f64.len()),
        ("weights", w_f64.len()),
        ("is_hinge", hinge_vec.len()),
    ] {
        if actual != len {
            return Err(PyValueError::new_err(format!(
                "{name} length ({actual}) must match src length ({len})"
            )));
        }
    }
    let num_nodes_val = num_nodes.0;
    for (idx, (&u, &v)) in src_u32.iter().zip(tgt_u32.iter()).enumerate() {
        if (u as usize) >= num_nodes_val || (v as usize) >= num_nodes_val {
            return Err(PyValueError::new_err(format!(
                "packed pair ({u}, {v}) at index {idx} out of bounds for num_nodes={num_nodes_val}"
            )));
        }
    }
    let num_hinges = hinge_vec.iter().filter(|&&h| h).count();
    let num_springs = len - num_hinges;

    let inner = match dtype {
        ScalarDtype::F64 => py
            .detach(|| {
                let mut pairs = Vec::with_capacity(len);
                for idx in 0..len {
                    pairs.push(PackedSymPair::new(
                        src_u32[idx],
                        tgt_u32[idx],
                        hinge_vec[idx],
                        d_f64[idx],
                        w_f64[idx],
                    ));
                }
                SgdSolver::<f64>::from_packed_symmetric(
                    num_nodes_val,
                    pairs,
                    num_springs,
                    num_hinges,
                    sched,
                    seed.0,
                )
                .map(SgdInner::F64)
            })
            .map_err(core_error_to_py)?,
        ScalarDtype::F32 => {
            let d_f32 = vec_f64_to_f32_checked(&d_f64, "dists")?;
            let w_f32 = vec_f64_to_f32_checked(&w_f64, "weights")?;
            py.detach(|| {
                let mut pairs = Vec::with_capacity(len);
                for idx in 0..len {
                    pairs.push(PackedSymPair::new(
                        src_u32[idx],
                        tgt_u32[idx],
                        hinge_vec[idx],
                        d_f32[idx],
                        w_f32[idx],
                    ));
                }
                SgdSolver::<f32>::from_packed_symmetric(
                    num_nodes_val,
                    pairs,
                    num_springs,
                    num_hinges,
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

pub(super) fn execute_reset_schedule(
    inner: &mut SgdInner,
    iterations: StrictUsize,
    eps: StrictF64,
    schedule: &str,
    step_scale: StrictF64,
    max_mu: Option<StrictF64>,
    seed: StrictU64,
) -> PyResult<()> {
    let sched = build_schedule_config(
        iterations.0,
        eps.0,
        schedule,
        step_scale.0,
        max_mu.map(|v| v.0),
    )?;
    match inner {
        SgdInner::F64(s) => s.reset_schedule(sched, seed.0),
        SgdInner::F32(s) => s.reset_schedule(sched, seed.0),
    }
    .map_err(core_error_to_py)
}

fn run_window_on<S: egraph::core::Scalar, L: SgdScheduleWindowLayout<S> + ?Sized>(
    solver: &mut SgdSolver<S>,
    layout: &mut L,
    start_iter: usize,
    count: usize,
) -> egraph::core::CoreResult<SolverStatus> {
    solver.run_schedule_window(layout, start_iter, count)
}

pub(super) fn execute_schedule_window(
    inner: &mut SgdInner,
    py: Python<'_>,
    layout: &Bound<'_, PyAny>,
    start_iter: StrictUsize,
    count: StrictUsize,
) -> PyResult<&'static str> {
    let target_dtype = inner.dtype();
    if let Ok(l2) = layout.cast::<PyLayout2d>() {
        let mut soa = extract_solver_layout(l2, target_dtype)?;
        let status = py
            .detach(|| match (inner, &mut soa) {
                (SgdInner::F64(s), SolverLayout::F64(l)) => {
                    run_window_on::<f64, Layout2d<f64>>(s, l, start_iter.0, count.0)
                }
                (SgdInner::F32(s), SolverLayout::F32(l)) => {
                    run_window_on::<f32, Layout2d<f32>>(s, l, start_iter.0, count.0)
                }
                _ => Err(CoreError::invalid_input(
                    "internal solver layout dtype mismatch",
                )),
            })
            .map_err(core_error_to_py)?;
        write_back_solver_layout(l2, &soa)?;
        return Ok(solver_status_str(status));
    }
    if let Ok(lm) = layout.cast::<PyManifoldLayout2d>() {
        let mut extracted = extract_manifold_for_solver(lm, target_dtype)?;
        let status = py
            .detach(|| match (inner, &mut extracted) {
                (SgdInner::F64(s), ExtractedManifold::F64(m)) => {
                    run_window_on::<f64, ManifoldLayout2d<f64>>(s, m, start_iter.0, count.0)
                }
                (SgdInner::F32(s), ExtractedManifold::F32(m)) => {
                    run_window_on::<f32, ManifoldLayout2d<f32>>(s, m, start_iter.0, count.0)
                }
                _ => Err(CoreError::invalid_input(
                    "internal solver manifold dtype mismatch",
                )),
            })
            .map_err(core_error_to_py)?;
        write_back_manifold(lm, extracted)?;
        return Ok(solver_status_str(status));
    }
    Err(PyTypeError::new_err(
        "layout must be a Layout2d or ManifoldLayout2d instance",
    ))
}

#[allow(dead_code)]
fn _check_layout2d_inner(_: &Layout2dInner) {}
