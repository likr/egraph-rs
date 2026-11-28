//! Separation constraints, VPSC projection, cycle removal, and hierarchical layering bindings.

use egraph::core::{CoreError, CsrGraph, NodeExtents2d};
use egraph::layout::{
    ConstraintProjector, LayeringMethod, SeparationConstraint, assign_layers, cycle_edges,
    generate_layered_constraints, project_clustered_rectangle_no_overlap_2d,
    project_rectangle_no_overlap_2d, remove_cycles, transitive_reduction,
};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{
    core_err_to_js, ffi_err_to_js, invalid_input_err, parse_non_negative_u32,
    parse_non_negative_usize,
};
use crate::js_options::{get_prop, normalize_enum_token};
use crate::layout_buffers::JsLayout2d;

const DEFAULT_MIN_LAYER_DISTANCE: f32 = 1.0;

/// One-dimensional separation constraint enforcing `coord[left] + gap <= coord[right]`.
#[wasm_bindgen(js_name = SeparationConstraint)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsSeparationConstraint {
    inner: SeparationConstraint<f32>,
}

#[wasm_bindgen(js_class = SeparationConstraint)]
impl JsSeparationConstraint {
    /// Creates a new `SeparationConstraint(left, right, gap)`.
    #[wasm_bindgen(constructor)]
    pub fn new(left: f64, right: f64, gap: f32) -> Result<JsSeparationConstraint, JsValue> {
        let l = parse_non_negative_u32(left, "left")?;
        let r = parse_non_negative_u32(right, "right")?;
        if !gap.is_finite() {
            return Err(core_err_to_js(CoreError::non_finite(
                "SeparationConstraint::gap",
            )));
        }
        let inner = SeparationConstraint::new(l, r, gap).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Returns the constrained left vertex index.
    #[wasm_bindgen(getter)]
    pub fn left(&self) -> u32 {
        self.inner.left
    }

    /// Returns the constrained right vertex index.
    #[wasm_bindgen(getter)]
    pub fn right(&self) -> u32 {
        self.inner.right
    }

    /// Returns the required minimum separation gap.
    #[wasm_bindgen(getter)]
    pub fn gap(&self) -> f32 {
        self.inner.gap
    }
}

fn parse_constraints(constraints: &JsValue) -> Result<Vec<SeparationConstraint<f32>>, JsValue> {
    if constraints.is_null()
        || constraints.is_undefined()
        || constraints.is_string()
        || !constraints.is_object()
    {
        return Err(invalid_input_err(
            "constraints must be an array or iterable of SeparationConstraint objects",
        ));
    }
    let iter = js_sys::try_iter(constraints)
        .map_err(|err| invalid_input_err(format!("failed to iterate constraints: {err:?}")))?
        .ok_or_else(|| {
            invalid_input_err("constraints must be an array or iterable of SeparationConstraint")
        })?;
    let mut out = Vec::new();
    for item_res in iter {
        let item = item_res
            .map_err(|err| invalid_input_err(format!("failed to read constraint item: {err:?}")))?;
        if item.is_null() || !item.is_object() {
            return Err(invalid_input_err(
                "each constraint entry must be a SeparationConstraint or { left, right, gap } object",
            ));
        }
        let left_val = get_prop(&item, "left")?;
        let right_val = get_prop(&item, "right")?;
        let gap_val = get_prop(&item, "gap")?;

        let left_f64 = left_val
            .as_f64()
            .ok_or_else(|| invalid_input_err("constraint.left must be a non-negative integer"))?;
        let right_f64 = right_val
            .as_f64()
            .ok_or_else(|| invalid_input_err("constraint.right must be a non-negative integer"))?;
        let gap_f64 = gap_val
            .as_f64()
            .ok_or_else(|| invalid_input_err("constraint.gap must be a finite number"))?;

        let left = parse_non_negative_u32(left_f64, "constraint.left")?;
        let right = parse_non_negative_u32(right_f64, "constraint.right")?;
        let gap = gap_f64 as f32;
        if !gap.is_finite() {
            return Err(core_err_to_js(CoreError::non_finite("constraint.gap")));
        }
        out.push(SeparationConstraint::new(left, right, gap).map_err(core_err_to_js)?);
    }
    Ok(out)
}

fn parse_axis(axis: &JsValue) -> Result<usize, JsValue> {
    if let Some(num) = axis.as_f64() {
        return parse_non_negative_usize(num, "axis");
    }
    if let Some(raw) = axis.as_string() {
        return match normalize_enum_token(&raw)?.as_str() {
            "x" | "0" => Ok(0),
            "y" | "1" => Ok(1),
            _ => Err(invalid_input_err(format!(
                "layout axis must be 0 ('x') or 1 ('y'), got {raw:?}"
            ))),
        };
    }
    Err(invalid_input_err("layout axis must be 0 ('x') or 1 ('y')"))
}

/// Projects `coords` (`Float32Array`) in place onto the feasible region defined by `constraints`.
#[wasm_bindgen(js_name = project1d)]
pub fn js_project_1d(coords: &mut [f32], constraints: &JsValue) -> Result<(), JsValue> {
    let parsed = parse_constraints(constraints)?;
    let mut projector = ConstraintProjector::<f32>::new(coords.len());
    projector
        .project_1d(coords, &parsed)
        .map_err(core_err_to_js)
}

/// Projects axis `axis` (`0` / `"x"` or `1` / `"y"`) of `layout` in place onto `constraints`.
#[wasm_bindgen(js_name = projectLayoutAxis)]
pub fn js_project_layout_axis(
    layout: &mut JsLayout2d,
    axis: &JsValue,
    constraints: &JsValue,
) -> Result<(), JsValue> {
    let axis_idx = parse_axis(axis)?;
    let parsed = parse_constraints(constraints)?;
    let mut projector = ConstraintProjector::<f32>::new(layout.len());
    let mut soa = layout.inner().to_layout2d();
    projector
        .project_layout_axis(&mut soa, axis_idx, &parsed)
        .map_err(core_err_to_js)?;
    layout
        .inner_mut()
        .copy_from_layout2d(&soa)
        .map_err(ffi_err_to_js)?;
    Ok(())
}

/// Generates hierarchical separation constraints `y[u] + gap <= y[v]` for directed edges
/// `(u, v)` spanning `layers[u] < layers[v]`.
#[wasm_bindgen(js_name = generateLayeredConstraints)]
pub fn js_generate_layered_constraints(
    graph: &JsCsrGraph,
    layers: &[u32],
    min_layer_distance: Option<f32>,
) -> Result<Vec<JsSeparationConstraint>, JsValue> {
    let dist = min_layer_distance.unwrap_or(DEFAULT_MIN_LAYER_DISTANCE);
    let raw = generate_layered_constraints(graph.inner(), layers, dist).map_err(core_err_to_js)?;
    Ok(raw
        .into_iter()
        .map(|inner| JsSeparationConstraint { inner })
        .collect())
}

/// Projects 2D rectangle positions in `layout` in place to eliminate all axis-aligned overlaps
/// while minimizing quadratic coordinate displacement.
#[wasm_bindgen(js_name = projectRectangleNoOverlap2d)]
pub fn js_project_rectangle_no_overlap_2d(
    layout: &mut JsLayout2d,
    half_widths: &[f32],
    half_heights: &[f32],
) -> Result<(), JsValue> {
    let extents = NodeExtents2d::new(half_widths, half_heights).map_err(core_err_to_js)?;
    let mut soa = layout.inner().to_layout2d();
    project_rectangle_no_overlap_2d(&mut soa, extents).map_err(core_err_to_js)?;
    layout
        .inner_mut()
        .copy_from_layout2d(&soa)
        .map_err(ffi_err_to_js)?;
    Ok(())
}

/// One-shot post-processing pass that eliminates overlaps between axis-aligned cluster bounding
/// boxes via rigid AABB VPSC in `layout` in place while preserving relative coordinates within
/// each cluster.
///
/// Note: This rigid AABB VPSC pass is distinct from Dwyer et al. (2006) iterative constrained
/// stress minimization (`IPSep-CoLa`).
#[wasm_bindgen(js_name = projectClusteredRectangleNoOverlap2d)]
pub fn js_project_clustered_rectangle_no_overlap_2d(
    layout: &mut JsLayout2d,
    labels: &[u32],
    half_widths: &[f32],
    half_heights: &[f32],
) -> Result<(), JsValue> {
    let extents = NodeExtents2d::new(half_widths, half_heights).map_err(core_err_to_js)?;
    let mut soa = layout.inner().to_layout2d();
    project_clustered_rectangle_no_overlap_2d(&mut soa, labels, extents).map_err(core_err_to_js)?;
    layout
        .inner_mut()
        .copy_from_layout2d(&soa)
        .map_err(ffi_err_to_js)?;
    Ok(())
}

fn csr_unit_to_f32(unit: &CsrGraph<()>) -> Result<CsrGraph<f32>, JsValue> {
    CsrGraph::<f32>::from_csr(
        unit.indptr(),
        unit.indices(),
        None,
        unit.is_directed(),
        false,
    )
    .map_err(core_err_to_js)
}

/// Computes the exact transitive reduction of a directed acyclic `CsrGraph`.
#[wasm_bindgen(js_name = transitiveReduction)]
pub fn js_transitive_reduction(graph: &JsCsrGraph) -> Result<JsCsrGraph, JsValue> {
    let reduced = transitive_reduction(graph.inner()).map_err(core_err_to_js)?;
    Ok(JsCsrGraph::from_inner(csr_unit_to_f32(&reduced)?))
}

fn parse_layering_method(method: Option<String>) -> Result<LayeringMethod, JsValue> {
    match method {
        None => Ok(LayeringMethod::LongestPath),
        Some(raw) => match normalize_enum_token(&raw)?.as_str() {
            "longestpath" | "lp" => Ok(LayeringMethod::LongestPath),
            "networksimplex" | "ns" => Ok(LayeringMethod::NetworkSimplex),
            _ => Err(invalid_input_err(format!(
                "unknown LayeringMethod {raw:?}; expected 'longestPath' or 'networkSimplex'"
            ))),
        },
    }
}

/// Identifies back-edges `(u, v)` whose reversal eliminates all directed cycles in `graph`,
/// returned as a flat interleaved `[u0, v0, u1, v1, ...]` `Uint32Array`.
#[wasm_bindgen(js_name = cycleEdges)]
pub fn js_cycle_edges(graph: &JsCsrGraph) -> Result<Vec<u32>, JsValue> {
    let pairs = cycle_edges(graph.inner()).map_err(core_err_to_js)?;
    let mut flat = Vec::with_capacity(pairs.len() * 2);
    for (u, v) in pairs {
        flat.push(u);
        flat.push(v);
    }
    Ok(flat)
}

/// Constructs a directed acyclic `CsrGraph` by reversing cycle-closing back-edges.
#[wasm_bindgen(js_name = removeCycles)]
pub fn js_remove_cycles(graph: &JsCsrGraph) -> Result<JsCsrGraph, JsValue> {
    let dag = remove_cycles(graph.inner()).map_err(core_err_to_js)?;
    Ok(JsCsrGraph::from_inner(csr_unit_to_f32(&dag)?))
}

/// Assigns hierarchical layers (`0, 1, ...`) to all vertices of a directed `CsrGraph`.
#[wasm_bindgen(js_name = assignLayers)]
pub fn js_assign_layers(graph: &JsCsrGraph, method: Option<String>) -> Result<Vec<u32>, JsValue> {
    let m = parse_layering_method(method)?;
    assign_layers(graph.inner(), m).map_err(core_err_to_js)
}
