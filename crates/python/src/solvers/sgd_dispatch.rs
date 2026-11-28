//! Additional `SgdSolver` dispatch helpers (`from_packed_symmetric`, schedule window/reset).

use egraph::core::{CoreError, ManifoldKind};
use egraph::layout::{SgdPairBuffer, SgdSolver};
use numpy::{PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::array_extract::extract_f64_1d;
use crate::errors::{
    core_error_to_py, dimension_mismatch_err, index_out_of_bounds_err, invalid_input_err,
    unsupported_operation_err,
};
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, extract_u32_1d, is_non_sequence_container,
    solver_status_str,
};
use crate::manifold::manifold_kind_str;
use crate::solver_common::vec_f64_to_f32_checked;

use super::layout_dispatch::{
    DualTargetSolver, SolverTargetKind, TargetLayout, dispatch_riemannian_target,
    extract_target_layout, match_riemannian_target, write_back_target_layout,
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
        return Err(dimension_mismatch_err(format!(
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
                return Err(invalid_input_err(format!(
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
    target: SolverTargetKind,
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
            return Err(dimension_mismatch_err(format!(
                "{name} length ({actual}) must match src length ({len})"
            )));
        }
    }
    let num_nodes_val = num_nodes.0;
    for (idx, (&u, &v)) in src_u32.iter().zip(tgt_u32.iter()).enumerate() {
        if (u as usize) >= num_nodes_val || (v as usize) >= num_nodes_val {
            return Err(index_out_of_bounds_err(format!(
                "packed pair ({u}, {v}) at index {idx} out of bounds for num_nodes={num_nodes_val}"
            )));
        }
    }

    let inner = match dtype {
        ScalarDtype::F64 => {
            let mut buffer = SgdPairBuffer::<f64>::with_capacity(len);
            for idx in 0..len {
                if hinge_vec[idx] {
                    buffer
                        .push_hinge(src_u32[idx], tgt_u32[idx], d_f64[idx], w_f64[idx])
                        .map_err(core_error_to_py)?;
                } else {
                    buffer
                        .push_spring(src_u32[idx], tgt_u32[idx], d_f64[idx], w_f64[idx])
                        .map_err(core_error_to_py)?;
                }
            }
            py.detach(|| {
                dispatch_riemannian_target!(
                    @f64
                    SgdInner,
                    target,
                    SgdSolver::from_pair_buffer(num_nodes_val, buffer, sched, seed.0)
                )
            })
            .map_err(core_error_to_py)?
        }
        ScalarDtype::F32 => {
            let d_f32 = vec_f64_to_f32_checked(&d_f64, "dists")?;
            let w_f32 = vec_f64_to_f32_checked(&w_f64, "weights")?;
            let mut buffer = SgdPairBuffer::<f32>::with_capacity(len);
            for idx in 0..len {
                if hinge_vec[idx] {
                    buffer
                        .push_hinge(src_u32[idx], tgt_u32[idx], d_f32[idx], w_f32[idx])
                        .map_err(core_error_to_py)?;
                } else {
                    buffer
                        .push_spring(src_u32[idx], tgt_u32[idx], d_f32[idx], w_f32[idx])
                        .map_err(core_error_to_py)?;
                }
            }
            py.detach(|| {
                dispatch_riemannian_target!(
                    @f32
                    SgdInner,
                    target,
                    SgdSolver::from_pair_buffer(num_nodes_val, buffer, sched, seed.0)
                )
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
    match_riemannian_target!(SgdInner, inner, s => s.reset_schedule(sched, seed.0))
        .map_err(core_error_to_py)
}

pub(super) fn execute_schedule_window(
    inner: &mut SgdInner,
    locked_manifold: &mut Option<ManifoldKind>,
    py: Python<'_>,
    layout: &Bound<'_, PyAny>,
    start_iter: StrictUsize,
    count: StrictUsize,
) -> PyResult<&'static str> {
    let mut extracted = extract_target_layout(layout, inner.target_kind(), inner.dtype())?;
    if let (Some(prev), Some(curr)) = (*locked_manifold, extracted.manifold_kind())
        && prev != curr
    {
        return Err(unsupported_operation_err(format!(
            "manifold solver is locked to '{}', got '{}'",
            manifold_kind_str(prev),
            manifold_kind_str(curr)
        )));
    }
    let kind_opt = extracted.manifold_kind();
    let status = py
        .detach(|| match (inner, &mut extracted) {
            (SgdInner::L2F64(s), TargetLayout::L2F64(l)) => {
                s.run_schedule_window(l, start_iter.0, count.0)
            }
            (SgdInner::L2F32(s), TargetLayout::L2F32(l)) => {
                s.run_schedule_window(l, start_iter.0, count.0)
            }
            (SgdInner::L3F64(s), TargetLayout::L3F64(l)) => {
                s.run_schedule_window(l, start_iter.0, count.0)
            }
            (SgdInner::L3F32(s), TargetLayout::L3F32(l)) => {
                s.run_schedule_window(l, start_iter.0, count.0)
            }
            (SgdInner::NdF64(s), TargetLayout::NdF64(l)) => {
                s.run_schedule_window(l, start_iter.0, count.0)
            }
            (SgdInner::NdF32(s), TargetLayout::NdF32(l)) => {
                s.run_schedule_window(l, start_iter.0, count.0)
            }
            (SgdInner::M2F64(s), TargetLayout::M2F64(l)) => {
                s.run_schedule_window(l, start_iter.0, count.0)
            }
            (SgdInner::M2F32(s), TargetLayout::M2F32(l)) => {
                s.run_schedule_window(l, start_iter.0, count.0)
            }
            _ => Err(CoreError::invalid_input(
                "internal solver target/dtype mismatch",
            )),
        })
        .map_err(core_error_to_py)?;
    if locked_manifold.is_none()
        && let Some(curr) = kind_opt
    {
        *locked_manifold = Some(curr);
    }
    write_back_target_layout(layout, extracted)?;
    Ok(solver_status_str(status))
}
