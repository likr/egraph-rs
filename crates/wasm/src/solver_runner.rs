//! Shared in-place cooperative execution helpers for `IterativeSolver<f32>` over
//! `Layout2d`, `LayoutNd`, and `ManifoldLayout2d`.

use egraph::core::{CoreError, IterativeSolver, LayoutNd, ManifoldLayout2d};
use wasm_bindgen::prelude::*;

use crate::js_convert::{core_err_to_js, ffi_err_to_js, parse_non_negative_usize};
use crate::js_options::solver_status_str;
use crate::layout_buffers::{JsLayout2d, JsLayoutNd};
use crate::manifold::JsManifoldLayout2d;

fn validate_layout_for_solver(
    layout: &JsLayout2d,
    expected_nodes: usize,
    ctx: &'static str,
) -> Result<(), JsError> {
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

pub(crate) fn step_layout_in_place<S: IterativeSolver<f32>>(
    solver: &mut S,
    layout: &mut JsLayout2d,
) -> Result<String, JsError> {
    let mut soa = layout.inner().to_layout2d();
    let status = solver.step(&mut soa).map_err(core_err_to_js)?;
    layout
        .inner_mut()
        .copy_from_layout2d(&soa)
        .map_err(ffi_err_to_js)?;
    Ok(solver_status_str(status).to_owned())
}

pub(crate) fn run_steps_layout_in_place<S: IterativeSolver<f32>>(
    expected_nodes: usize,
    solver: &mut S,
    layout: &mut JsLayout2d,
    steps: f64,
    ctx: &'static str,
) -> Result<String, JsError> {
    let n_steps = parse_non_negative_usize(steps, "steps")?;
    validate_layout_for_solver(layout, expected_nodes, ctx)?;
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

pub(crate) fn run_layout_in_place<S: IterativeSolver<f32>>(
    expected_nodes: usize,
    solver: &mut S,
    layout: &mut JsLayout2d,
    ctx: &'static str,
) -> Result<String, JsError> {
    validate_layout_for_solver(layout, expected_nodes, ctx)?;
    let mut soa = layout.inner().to_layout2d();
    let status = solver.run(&mut soa).map_err(core_err_to_js)?;
    layout
        .inner_mut()
        .copy_from_layout2d(&soa)
        .map_err(ffi_err_to_js)?;
    Ok(solver_status_str(status).to_owned())
}

fn validate_nd_for_solver(
    layout: &JsLayoutNd,
    expected_nodes: usize,
    ctx: &'static str,
) -> Result<(), JsError> {
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

pub(crate) fn step_nd_in_place<S: IterativeSolver<f32, LayoutNd<f32>>>(
    solver: &mut S,
    layout: &mut JsLayoutNd,
) -> Result<String, JsError> {
    let status = solver.step(layout.inner_mut()).map_err(core_err_to_js)?;
    Ok(solver_status_str(status).to_owned())
}

pub(crate) fn run_steps_nd_in_place<S: IterativeSolver<f32, LayoutNd<f32>>>(
    expected_nodes: usize,
    solver: &mut S,
    layout: &mut JsLayoutNd,
    steps: f64,
    ctx: &'static str,
) -> Result<String, JsError> {
    let n_steps = parse_non_negative_usize(steps, "steps")?;
    validate_nd_for_solver(layout, expected_nodes, ctx)?;
    let status = solver
        .run_steps(layout.inner_mut(), n_steps)
        .map_err(core_err_to_js)?;
    Ok(solver_status_str(status).to_owned())
}

pub(crate) fn run_nd_in_place<S: IterativeSolver<f32, LayoutNd<f32>>>(
    expected_nodes: usize,
    solver: &mut S,
    layout: &mut JsLayoutNd,
    ctx: &'static str,
) -> Result<String, JsError> {
    validate_nd_for_solver(layout, expected_nodes, ctx)?;
    let status = solver.run(layout.inner_mut()).map_err(core_err_to_js)?;
    Ok(solver_status_str(status).to_owned())
}

fn validate_manifold_for_solver(
    layout: &JsManifoldLayout2d,
    expected_nodes: usize,
    ctx: &'static str,
) -> Result<(), JsError> {
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
    Ok(())
}

pub(crate) fn step_manifold_in_place<S: IterativeSolver<f32, ManifoldLayout2d<f32>>>(
    solver: &mut S,
    layout: &mut JsManifoldLayout2d,
) -> Result<String, JsError> {
    let status = solver.step(layout.inner_mut()).map_err(core_err_to_js)?;
    layout.sync_shadow()?;
    Ok(solver_status_str(status).to_owned())
}

pub(crate) fn run_steps_manifold_in_place<S: IterativeSolver<f32, ManifoldLayout2d<f32>>>(
    expected_nodes: usize,
    solver: &mut S,
    layout: &mut JsManifoldLayout2d,
    steps: f64,
    ctx: &'static str,
) -> Result<String, JsError> {
    let n_steps = parse_non_negative_usize(steps, "steps")?;
    validate_manifold_for_solver(layout, expected_nodes, ctx)?;
    let status = solver
        .run_steps(layout.inner_mut(), n_steps)
        .map_err(core_err_to_js)?;
    layout.sync_shadow()?;
    Ok(solver_status_str(status).to_owned())
}

pub(crate) fn run_manifold_in_place<S: IterativeSolver<f32, ManifoldLayout2d<f32>>>(
    expected_nodes: usize,
    solver: &mut S,
    layout: &mut JsManifoldLayout2d,
    ctx: &'static str,
) -> Result<String, JsError> {
    validate_manifold_for_solver(layout, expected_nodes, ctx)?;
    let status = solver.run(layout.inner_mut()).map_err(core_err_to_js)?;
    layout.sync_shadow()?;
    Ok(solver_status_str(status).to_owned())
}
