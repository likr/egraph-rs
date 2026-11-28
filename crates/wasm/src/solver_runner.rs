//! Shared in-place cooperative execution helpers for `IterativeSolver` over
//! `Layout2d<f32>`, `Layout3d<f32>`, `LayoutNd<f32>`, and `ManifoldLayout2d<f32>`.

use egraph::core::{
    CoreError, IterativeSolver, Layout2d, Layout3d, LayoutNd, ManifoldKind, ManifoldLayout2d,
};
use wasm_bindgen::prelude::*;

use crate::js_convert::{
    core_err_to_js, ffi_err_to_js, invalid_input_err, parse_non_negative_usize,
};
use crate::js_options::solver_status_str;
use crate::layout_buffers::{JsLayout2d, JsLayoutNd};
use crate::layout3d::JsLayout3d;
use crate::manifold::JsManifoldLayout2d;

fn validate_layout2d_for_solver(
    layout: &JsLayout2d,
    expected_nodes: usize,
    ctx: &'static str,
) -> Result<(), JsValue> {
    if layout.len() != expected_nodes {
        return Err(core_err_to_js(CoreError::dimension_mismatch(
            expected_nodes,
            layout.len(),
            ctx,
        )));
    }
    if !layout.inner().is_all_finite() {
        return Err(core_err_to_js(CoreError::non_finite(ctx)));
    }
    Ok(())
}

pub(crate) fn step_layout_in_place<S: IterativeSolver<Target = Layout2d<f32>>>(
    solver: &mut S,
    layout: &mut JsLayout2d,
) -> Result<String, JsValue> {
    let mut soa = layout.inner().to_layout2d();
    let status = solver.step(&mut soa).map_err(core_err_to_js)?;
    layout
        .inner_mut()
        .copy_from_layout2d(&soa)
        .map_err(ffi_err_to_js)?;
    Ok(solver_status_str(status).to_owned())
}

pub(crate) fn run_steps_layout_in_place<S: IterativeSolver<Target = Layout2d<f32>>>(
    expected_nodes: usize,
    solver: &mut S,
    layout: &mut JsLayout2d,
    steps: f64,
    ctx: &'static str,
) -> Result<String, JsValue> {
    let n_steps = parse_non_negative_usize(steps, "steps")?;
    validate_layout2d_for_solver(layout, expected_nodes, ctx)?;
    let mut soa = layout.inner().to_layout2d();
    let status = solver
        .run_steps(&mut soa, n_steps)
        .map_err(core_err_to_js)?;
    layout
        .inner_mut()
        .copy_from_layout2d(&soa)
        .map_err(ffi_err_to_js)?;
    Ok(solver_status_str(status).to_owned())
}

pub(crate) fn run_layout_in_place<S: IterativeSolver<Target = Layout2d<f32>>>(
    expected_nodes: usize,
    solver: &mut S,
    layout: &mut JsLayout2d,
    ctx: &'static str,
) -> Result<String, JsValue> {
    validate_layout2d_for_solver(layout, expected_nodes, ctx)?;
    let mut soa = layout.inner().to_layout2d();
    let status = solver.run(&mut soa).map_err(core_err_to_js)?;
    layout
        .inner_mut()
        .copy_from_layout2d(&soa)
        .map_err(ffi_err_to_js)?;
    Ok(solver_status_str(status).to_owned())
}

fn validate_layout3d_for_solver(
    layout: &JsLayout3d,
    expected_nodes: usize,
    ctx: &'static str,
) -> Result<(), JsValue> {
    if layout.len() != expected_nodes {
        return Err(core_err_to_js(CoreError::dimension_mismatch(
            expected_nodes,
            layout.len(),
            ctx,
        )));
    }
    if !layout.inner().is_all_finite() {
        return Err(core_err_to_js(CoreError::non_finite(ctx)));
    }
    Ok(())
}

pub(crate) fn step_layout3d_in_place<S: IterativeSolver<Target = Layout3d<f32>>>(
    solver: &mut S,
    layout: &mut JsLayout3d,
) -> Result<String, JsValue> {
    let mut soa = layout.inner().to_layout3d();
    let status = solver.step(&mut soa).map_err(core_err_to_js)?;
    layout
        .inner_mut()
        .copy_from_layout3d(&soa)
        .map_err(ffi_err_to_js)?;
    Ok(solver_status_str(status).to_owned())
}

pub(crate) fn run_steps_layout3d_in_place<S: IterativeSolver<Target = Layout3d<f32>>>(
    expected_nodes: usize,
    solver: &mut S,
    layout: &mut JsLayout3d,
    steps: f64,
    ctx: &'static str,
) -> Result<String, JsValue> {
    let n_steps = parse_non_negative_usize(steps, "steps")?;
    validate_layout3d_for_solver(layout, expected_nodes, ctx)?;
    let mut soa = layout.inner().to_layout3d();
    let status = solver
        .run_steps(&mut soa, n_steps)
        .map_err(core_err_to_js)?;
    layout
        .inner_mut()
        .copy_from_layout3d(&soa)
        .map_err(ffi_err_to_js)?;
    Ok(solver_status_str(status).to_owned())
}

pub(crate) fn run_layout3d_in_place<S: IterativeSolver<Target = Layout3d<f32>>>(
    expected_nodes: usize,
    solver: &mut S,
    layout: &mut JsLayout3d,
    ctx: &'static str,
) -> Result<String, JsValue> {
    validate_layout3d_for_solver(layout, expected_nodes, ctx)?;
    let mut soa = layout.inner().to_layout3d();
    let status = solver.run(&mut soa).map_err(core_err_to_js)?;
    layout
        .inner_mut()
        .copy_from_layout3d(&soa)
        .map_err(ffi_err_to_js)?;
    Ok(solver_status_str(status).to_owned())
}

fn validate_nd_for_solver(
    layout: &JsLayoutNd,
    expected_nodes: usize,
    ctx: &'static str,
) -> Result<(), JsValue> {
    if layout.len() != expected_nodes {
        return Err(core_err_to_js(CoreError::dimension_mismatch(
            expected_nodes,
            layout.len(),
            ctx,
        )));
    }
    if !layout.inner().is_all_finite() {
        return Err(core_err_to_js(CoreError::non_finite(ctx)));
    }
    Ok(())
}

pub(crate) fn step_nd_in_place<S: IterativeSolver<Target = LayoutNd<f32>>>(
    solver: &mut S,
    layout: &mut JsLayoutNd,
) -> Result<String, JsValue> {
    let status = solver.step(layout.inner_mut()).map_err(core_err_to_js)?;
    Ok(solver_status_str(status).to_owned())
}

pub(crate) fn run_steps_nd_in_place<S: IterativeSolver<Target = LayoutNd<f32>>>(
    expected_nodes: usize,
    solver: &mut S,
    layout: &mut JsLayoutNd,
    steps: f64,
    ctx: &'static str,
) -> Result<String, JsValue> {
    let n_steps = parse_non_negative_usize(steps, "steps")?;
    validate_nd_for_solver(layout, expected_nodes, ctx)?;
    let status = solver
        .run_steps(layout.inner_mut(), n_steps)
        .map_err(core_err_to_js)?;
    Ok(solver_status_str(status).to_owned())
}

pub(crate) fn run_nd_in_place<S: IterativeSolver<Target = LayoutNd<f32>>>(
    expected_nodes: usize,
    solver: &mut S,
    layout: &mut JsLayoutNd,
    ctx: &'static str,
) -> Result<String, JsValue> {
    validate_nd_for_solver(layout, expected_nodes, ctx)?;
    let status = solver.run(layout.inner_mut()).map_err(core_err_to_js)?;
    Ok(solver_status_str(status).to_owned())
}

pub(crate) fn check_and_lock_manifold(
    locked: &mut Option<ManifoldKind>,
    layout: &JsManifoldLayout2d,
    expected_nodes: usize,
    ctx: &'static str,
) -> Result<(), JsValue> {
    if layout.len() != expected_nodes {
        return Err(core_err_to_js(CoreError::dimension_mismatch(
            expected_nodes,
            layout.len(),
            ctx,
        )));
    }
    if !layout.inner().as_layout().is_all_finite() {
        return Err(core_err_to_js(CoreError::non_finite(ctx)));
    }
    let kind = layout.inner().kind();
    match *locked {
        Some(prev) if prev != kind => {
            return Err(invalid_input_err(format!(
                "{ctx}: manifold kind locked to {prev:?}, cannot switch to {kind:?}"
            )));
        }
        None => *locked = Some(kind),
        Some(_) => {}
    }
    Ok(())
}

pub(crate) fn step_manifold_in_place<S: IterativeSolver<Target = ManifoldLayout2d<f32>>>(
    solver: &mut S,
    layout: &mut JsManifoldLayout2d,
    locked: &mut Option<ManifoldKind>,
    expected_nodes: usize,
    ctx: &'static str,
) -> Result<String, JsValue> {
    check_and_lock_manifold(locked, layout, expected_nodes, ctx)?;
    let status = solver.step(layout.inner_mut()).map_err(core_err_to_js)?;
    layout.sync_shadow()?;
    Ok(solver_status_str(status).to_owned())
}

pub(crate) fn run_steps_manifold_in_place<S: IterativeSolver<Target = ManifoldLayout2d<f32>>>(
    expected_nodes: usize,
    solver: &mut S,
    layout: &mut JsManifoldLayout2d,
    steps: f64,
    locked: &mut Option<ManifoldKind>,
    ctx: &'static str,
) -> Result<String, JsValue> {
    let n_steps = parse_non_negative_usize(steps, "steps")?;
    check_and_lock_manifold(locked, layout, expected_nodes, ctx)?;
    let status = solver
        .run_steps(layout.inner_mut(), n_steps)
        .map_err(core_err_to_js)?;
    layout.sync_shadow()?;
    Ok(solver_status_str(status).to_owned())
}

pub(crate) fn run_manifold_in_place<S: IterativeSolver<Target = ManifoldLayout2d<f32>>>(
    expected_nodes: usize,
    solver: &mut S,
    layout: &mut JsManifoldLayout2d,
    locked: &mut Option<ManifoldKind>,
    ctx: &'static str,
) -> Result<String, JsValue> {
    check_and_lock_manifold(locked, layout, expected_nodes, ctx)?;
    let status = solver.run(layout.inner_mut()).map_err(core_err_to_js)?;
    layout.sync_shadow()?;
    Ok(solver_status_str(status).to_owned())
}
