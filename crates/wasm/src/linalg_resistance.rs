//! WebAssembly bindings for effective edge resistances and Devriendt-Lambiotte
//! resistance curvature flow.

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::core_err_to_js;
use crate::js_options::{parse_effective_resistance_options, parse_resistance_curvature_options};
use egraph::linalg::{
    EffectiveResistanceResult, ResistanceCurvatureFlowResult, ResistanceCurvatureStep,
    compute_effective_resistances, resistance_curvature_flow,
};
use wasm_bindgen::prelude::*;

/// Effective edge resistances and Foster's theorem diagnostics for canonical simple undirected edges.
#[wasm_bindgen(js_name = EffectiveResistanceResult)]
#[derive(Debug, Clone)]
pub struct JsEffectiveResistanceResult {
    inner: EffectiveResistanceResult,
}

#[wasm_bindgen(js_class = EffectiveResistanceResult)]
impl JsEffectiveResistanceResult {
    /// Returns the number of vertices in the graph.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes
    }

    /// Returns the number of canonical undirected simple edges (`u < v`).
    #[wasm_bindgen(getter, js_name = numEdges)]
    pub fn num_edges(&self) -> usize {
        self.inner.num_edges
    }

    /// Returns the number of connected components (including isolated vertices).
    #[wasm_bindgen(getter, js_name = numComponents)]
    pub fn num_components(&self) -> usize {
        self.inner.num_components
    }

    /// Returns the observed Foster sum `sum_e w_e * R_e`.
    #[wasm_bindgen(getter, js_name = fosterSum)]
    pub fn foster_sum(&self) -> f64 {
        self.inner.foster_sum
    }

    /// Returns the exact theoretical Foster sum `numNodes - numComponents`.
    #[wasm_bindgen(getter, js_name = expectedFoster)]
    pub fn expected_foster(&self) -> f64 {
        self.inner.expected_foster
    }

    /// Returns the relative Foster error.
    #[wasm_bindgen(getter, js_name = fosterRelErr)]
    pub fn foster_rel_err(&self) -> f64 {
        self.inner.foster_rel_err
    }

    /// Returns `true` if the exact dense pseudoinverse solver path was used.
    #[wasm_bindgen(getter, js_name = usedExact)]
    pub fn used_exact(&self) -> bool {
        self.inner.used_exact
    }

    /// Returns total AMG-PCG iterations executed across all JL RHS columns.
    #[wasm_bindgen(getter, js_name = amgTotalIterations)]
    pub fn amg_total_iterations(&self) -> usize {
        self.inner.amg_total_iterations
    }

    /// Returns maximum relative PCG residual across all JL RHS columns.
    #[wasm_bindgen(getter, js_name = amgMaxRelativeResidual)]
    pub fn amg_max_relative_residual(&self) -> f64 {
        self.inner.amg_max_relative_residual
    }

    /// Returns `true` if every JL PCG column converged within tolerance.
    #[wasm_bindgen(getter, js_name = amgAllConverged)]
    pub fn amg_all_converged(&self) -> bool {
        self.inner.amg_all_converged
    }

    /// Returns canonical edge source endpoints `u < v` (`Uint32Array`).
    #[wasm_bindgen(js_name = edgeU)]
    pub fn edge_u(&self) -> Vec<u32> {
        self.inner.edge_u.clone()
    }

    /// Returns canonical edge target endpoints `v > u` (`Uint32Array`).
    #[wasm_bindgen(js_name = edgeV)]
    pub fn edge_v(&self) -> Vec<u32> {
        self.inner.edge_v.clone()
    }

    /// Returns canonical edge lengths `l_e` (`Float64Array`).
    #[wasm_bindgen(js_name = edgeLengths)]
    pub fn edge_lengths(&self) -> Vec<f64> {
        self.inner.edge_lengths.clone()
    }

    /// Returns canonical edge conductances `w_e = 1 / l_e` (`Float64Array`).
    pub fn conductances(&self) -> Vec<f64> {
        self.inner.conductances.clone()
    }

    /// Returns effective resistances `R_e` (`Float64Array`).
    #[wasm_bindgen(js_name = effectiveResistances)]
    pub fn effective_resistances(&self) -> Vec<f64> {
        self.inner.effective_resistances.clone()
    }
}

/// Per-iteration summary recorded during `resistanceCurvatureFlow()`.
#[wasm_bindgen(js_name = ResistanceCurvatureStep)]
#[derive(Debug, Clone, Copy)]
pub struct JsResistanceCurvatureStep {
    inner: ResistanceCurvatureStep,
}

#[wasm_bindgen(js_class = ResistanceCurvatureStep)]
impl JsResistanceCurvatureStep {
    /// Flow step index (`0..steps`, plus final evaluation step `steps`).
    #[wasm_bindgen(getter)]
    pub fn step(&self) -> usize {
        self.inner.step
    }

    /// Observed Foster sum `sum_e w_e * R_e`.
    #[wasm_bindgen(getter, js_name = fosterSum)]
    pub fn foster_sum(&self) -> f64 {
        self.inner.foster_sum
    }

    /// Exact theoretical Foster sum `numNodes - numComponents`.
    #[wasm_bindgen(getter, js_name = expectedFoster)]
    pub fn expected_foster(&self) -> f64 {
        self.inner.expected_foster
    }

    /// Relative Foster error.
    #[wasm_bindgen(getter, js_name = fosterRelErr)]
    pub fn foster_rel_err(&self) -> f64 {
        self.inner.foster_rel_err
    }

    /// Mean link resistance curvature.
    #[wasm_bindgen(getter, js_name = meanLinkCurvature)]
    pub fn mean_link_curvature(&self) -> f64 {
        self.inner.mean_link_curvature
    }

    /// Population standard deviation of link resistance curvatures.
    #[wasm_bindgen(getter, js_name = stdLinkCurvature)]
    pub fn std_link_curvature(&self) -> f64 {
        self.inner.std_link_curvature
    }

    /// Minimum link resistance curvature.
    #[wasm_bindgen(getter, js_name = minLinkCurvature)]
    pub fn min_link_curvature(&self) -> f64 {
        self.inner.min_link_curvature
    }

    /// Maximum link resistance curvature.
    #[wasm_bindgen(getter, js_name = maxLinkCurvature)]
    pub fn max_link_curvature(&self) -> f64 {
        self.inner.max_link_curvature
    }

    /// Mean vertex resistance curvature.
    #[wasm_bindgen(getter, js_name = meanVertexCurvature)]
    pub fn mean_vertex_curvature(&self) -> f64 {
        self.inner.mean_vertex_curvature
    }

    /// Minimum vertex resistance curvature.
    #[wasm_bindgen(getter, js_name = minVertexCurvature)]
    pub fn min_vertex_curvature(&self) -> f64 {
        self.inner.min_vertex_curvature
    }

    /// Maximum vertex resistance curvature.
    #[wasm_bindgen(getter, js_name = maxVertexCurvature)]
    pub fn max_vertex_curvature(&self) -> f64 {
        self.inner.max_vertex_curvature
    }

    /// Whether the exact dense pseudoinverse solver was used.
    #[wasm_bindgen(getter, js_name = usedExact)]
    pub fn used_exact(&self) -> bool {
        self.inner.used_exact
    }

    /// Total AMG-PCG iterations executed in this step.
    #[wasm_bindgen(getter, js_name = amgTotalIterations)]
    pub fn amg_total_iterations(&self) -> usize {
        self.inner.amg_total_iterations
    }

    /// Maximum relative PCG residual in this step.
    #[wasm_bindgen(getter, js_name = amgMaxRelativeResidual)]
    pub fn amg_max_relative_residual(&self) -> f64 {
        self.inner.amg_max_relative_residual
    }
}

/// Devriendt-Lambiotte resistance curvature flow result with reweighted graphs and geometry classification.
#[wasm_bindgen(js_name = ResistanceCurvatureFlowResult)]
#[derive(Debug, Clone)]
pub struct JsResistanceCurvatureFlowResult {
    inner: ResistanceCurvatureFlowResult,
}

#[wasm_bindgen(js_class = ResistanceCurvatureFlowResult)]
impl JsResistanceCurvatureFlowResult {
    /// Returns the number of vertices in the graph.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes
    }

    /// Returns the number of canonical undirected simple edges (`u < v`).
    #[wasm_bindgen(getter, js_name = numEdges)]
    pub fn num_edges(&self) -> usize {
        self.inner.num_edges
    }

    /// Returns the number of connected components (including isolated vertices).
    #[wasm_bindgen(getter, js_name = numComponents)]
    pub fn num_components(&self) -> usize {
        self.inner.num_components
    }

    /// Returns the final Foster sum `sum_e w_e * R_e`.
    #[wasm_bindgen(getter, js_name = fosterSum)]
    pub fn foster_sum(&self) -> f64 {
        self.inner.foster_sum
    }

    /// Returns the exact theoretical Foster sum `numNodes - numComponents`.
    #[wasm_bindgen(getter, js_name = expectedFoster)]
    pub fn expected_foster(&self) -> f64 {
        self.inner.expected_foster
    }

    /// Returns the final relative Foster error.
    #[wasm_bindgen(getter, js_name = fosterRelErr)]
    pub fn foster_rel_err(&self) -> f64 {
        self.inner.foster_rel_err
    }

    /// Returns mean final link resistance curvature.
    #[wasm_bindgen(getter, js_name = meanLinkCurvature)]
    pub fn mean_link_curvature(&self) -> f64 {
        self.inner.mean_link_curvature
    }

    /// Returns population standard deviation of final link resistance curvatures.
    #[wasm_bindgen(getter, js_name = stdLinkCurvature)]
    pub fn std_link_curvature(&self) -> f64 {
        self.inner.std_link_curvature
    }

    /// Returns minimum final link resistance curvature.
    #[wasm_bindgen(getter, js_name = minLinkCurvature)]
    pub fn min_link_curvature(&self) -> f64 {
        self.inner.min_link_curvature
    }

    /// Returns maximum final link resistance curvature.
    #[wasm_bindgen(getter, js_name = maxLinkCurvature)]
    pub fn max_link_curvature(&self) -> f64 {
        self.inner.max_link_curvature
    }

    /// Returns mean final vertex resistance curvature.
    #[wasm_bindgen(getter, js_name = meanVertexCurvature)]
    pub fn mean_vertex_curvature(&self) -> f64 {
        self.inner.mean_vertex_curvature
    }

    /// Returns minimum final vertex resistance curvature.
    #[wasm_bindgen(getter, js_name = minVertexCurvature)]
    pub fn min_vertex_curvature(&self) -> f64 {
        self.inner.min_vertex_curvature
    }

    /// Returns maximum final vertex resistance curvature.
    #[wasm_bindgen(getter, js_name = maxVertexCurvature)]
    pub fn max_vertex_curvature(&self) -> f64 {
        self.inner.max_vertex_curvature
    }

    /// Returns `true` if the final evaluation used the exact dense pseudoinverse solver.
    #[wasm_bindgen(getter, js_name = usedExact)]
    pub fn used_exact(&self) -> bool {
        self.inner.used_exact
    }

    /// Returns total AMG-PCG iterations executed in the final evaluation.
    #[wasm_bindgen(getter, js_name = amgTotalIterations)]
    pub fn amg_total_iterations(&self) -> usize {
        self.inner.amg_total_iterations
    }

    /// Returns maximum relative PCG residual in the final evaluation.
    #[wasm_bindgen(getter, js_name = amgMaxRelativeResidual)]
    pub fn amg_max_relative_residual(&self) -> f64 {
        self.inner.amg_max_relative_residual
    }

    /// Returns `true` if every JL PCG column converged in the final evaluation.
    #[wasm_bindgen(getter, js_name = amgAllConverged)]
    pub fn amg_all_converged(&self) -> bool {
        self.inner.amg_all_converged
    }

    /// Returns the suggested 2D geometry (`"hyperbolic"`, `"spherical"`, `"torus"`, or `"euclidean"`).
    #[wasm_bindgen(getter, js_name = suggestedGeometry)]
    pub fn suggested_geometry(&self) -> String {
        self.inner.suggested_geometry.as_str().to_owned()
    }

    /// Returns the suggested `ManifoldKind` (`"hyperbolic"`, `"spherical"`, `"torus"`), or `undefined`.
    #[wasm_bindgen(getter, js_name = suggestedManifold)]
    pub fn suggested_manifold(&self) -> Option<String> {
        self.inner
            .suggested_manifold_kind()
            .map(|_| self.inner.suggested_geometry.as_str().to_owned())
    }

    /// Returns canonical edge source endpoints `u < v` (`Uint32Array`).
    #[wasm_bindgen(js_name = edgeU)]
    pub fn edge_u(&self) -> Vec<u32> {
        self.inner.edge_u.clone()
    }

    /// Returns canonical edge target endpoints `v > u` (`Uint32Array`).
    #[wasm_bindgen(js_name = edgeV)]
    pub fn edge_v(&self) -> Vec<u32> {
        self.inner.edge_v.clone()
    }

    /// Returns final normalized edge lengths `l_e` (`Float64Array`).
    #[wasm_bindgen(js_name = edgeLengths)]
    pub fn edge_lengths(&self) -> Vec<f64> {
        self.inner.edge_lengths.clone()
    }

    /// Returns final edge conductances `w_e = 1 / l_e` (`Float64Array`).
    pub fn conductances(&self) -> Vec<f64> {
        self.inner.conductances.clone()
    }

    /// Returns final effective resistances `R_e` (`Float64Array`).
    #[wasm_bindgen(js_name = effectiveResistances)]
    pub fn effective_resistances(&self) -> Vec<f64> {
        self.inner.effective_resistances.clone()
    }

    /// Returns final vertex resistance curvatures `p_i` (`Float64Array`).
    #[wasm_bindgen(js_name = vertexCurvatures)]
    pub fn vertex_curvatures(&self) -> Vec<f64> {
        self.inner.vertex_curvatures.clone()
    }

    /// Returns final link resistance curvatures `kappa_res(u, v)` (`Float64Array`).
    #[wasm_bindgen(js_name = linkCurvatures)]
    pub fn link_curvatures(&self) -> Vec<f64> {
        self.inner.link_curvatures.clone()
    }

    /// Returns per-step diagnostics across `0..=steps`.
    #[wasm_bindgen(js_name = stepHistory)]
    pub fn step_history(&self) -> Vec<JsResistanceCurvatureStep> {
        self.inner
            .step_history
            .iter()
            .copied()
            .map(|inner| JsResistanceCurvatureStep { inner })
            .collect()
    }

    /// Builds a weighted undirected `CsrGraph` (`f32`) storing final edge lengths `l_e`.
    #[wasm_bindgen(js_name = lengthGraph)]
    pub fn length_graph(&self) -> Result<JsCsrGraph, JsError> {
        self.inner
            .to_length_graph_f32()
            .map(JsCsrGraph::from_inner)
            .map_err(core_err_to_js)
    }

    /// Builds a weighted undirected `CsrGraph` (`f32`) storing final conductances `w_e = 1 / l_e`.
    #[wasm_bindgen(js_name = conductanceGraph)]
    pub fn conductance_graph(&self) -> Result<JsCsrGraph, JsError> {
        self.inner
            .to_conductance_graph_f32()
            .map(JsCsrGraph::from_inner)
            .map_err(core_err_to_js)
    }
}

/// Computes effective edge resistances and Foster's theorem diagnostics via exact pseudoinverse
/// (for small graphs) or Spielman-Srivastava JL projection solved with `AmgPcgSolver`.
#[wasm_bindgen(js_name = computeEffectiveResistances)]
pub fn js_compute_effective_resistances(
    graph: &JsCsrGraph,
    options: &JsValue,
) -> Result<JsEffectiveResistanceResult, JsError> {
    let cfg = parse_effective_resistance_options(options)?;
    let inner = compute_effective_resistances(graph.inner(), &cfg).map_err(core_err_to_js)?;
    Ok(JsEffectiveResistanceResult { inner })
}

/// Runs Devriendt-Lambiotte resistance curvature flow on `graph` and classifies target geometry.
#[wasm_bindgen(js_name = resistanceCurvatureFlow)]
pub fn js_resistance_curvature_flow(
    graph: &JsCsrGraph,
    options: &JsValue,
) -> Result<JsResistanceCurvatureFlowResult, JsError> {
    let cfg = parse_resistance_curvature_options(options)?;
    let inner = resistance_curvature_flow(graph.inner(), &cfg).map_err(core_err_to_js)?;
    Ok(JsResistanceCurvatureFlowResult { inner })
}
