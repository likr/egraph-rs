//! WebAssembly bindings for Quadrilateral Simmelian Backbone extraction and layout (`f32`).

use egraph::core::{CsrGraph, IterativeSolver};
use egraph::layout::{
    BackboneConfig, BackboneLayoutMode, BackboneResult, BackboneSolver,
    extract_quadrilateral_backbone,
};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, csr_f64_to_f32, set_js_prop};
use crate::js_options::{
    check_options_arg, normalize_enum_token, opt_f64_prop, opt_seed_prop, opt_string_prop,
    opt_usize_prop, solver_status_str,
};
use crate::layout_buffers::JsLayout2d;
use crate::solver_runner::{run_layout_in_place, run_steps_layout_in_place, step_layout_in_place};

fn flatten_edge_pairs(edges: &[(u32, u32)]) -> Vec<u32> {
    let mut flat = Vec::with_capacity(edges.len() * 2);
    for &(u, v) in edges {
        flat.push(u);
        flat.push(v);
    }
    flat
}

fn parse_backbone_layout_mode(raw: &str) -> Result<BackboneLayoutMode, JsError> {
    match normalize_enum_token(raw).as_str() {
        "stress" | "stressmajorization" => Ok(BackboneLayoutMode::StressMajorization),
        "sgd" => Ok(BackboneLayoutMode::Sgd),
        _ => Err(JsError::new(&format!(
            "unknown BackboneLayoutMode {raw:?}; expected 'stress' or 'sgd'"
        ))),
    }
}

fn parse_backbone_config(options: &JsValue, ctx: &str) -> Result<BackboneConfig, JsError> {
    let mut cfg = BackboneConfig::default();
    if check_options_arg(options, ctx)? {
        cfg.retain_ratio = opt_f64_prop(options, "retainRatio", cfg.retain_ratio)?;
        cfg.umst_weight_scale = opt_f64_prop(options, "umstWeightScale", cfg.umst_weight_scale)?;
        cfg.non_backbone_weight_scale = opt_f64_prop(
            options,
            "nonBackboneWeightScale",
            cfg.non_backbone_weight_scale,
        )?;
        if let Some(m_str) = opt_string_prop(options, "layoutMode")? {
            cfg.layout_mode = parse_backbone_layout_mode(&m_str)?;
        }
        cfg.iterations = opt_usize_prop(options, "iterations", cfg.iterations)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    Ok(cfg)
}

/// Quadrilateral Simmelian backbone extraction output (`Nocaj et al. 2015`).
#[wasm_bindgen(js_name = BackboneResult)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsBackboneResult {
    umst_edges: Vec<u32>,
    backbone_edges: Vec<u32>,
    simmelian_weights: Vec<f64>,
    edge_weights: Vec<f64>,
    backbone_csr: CsrGraph<f32>,
}

impl JsBackboneResult {
    fn from_inner(inner: &BackboneResult) -> Result<Self, JsError> {
        let backbone_csr = csr_f64_to_f32(&inner.backbone_csr)?;
        Ok(Self {
            umst_edges: flatten_edge_pairs(&inner.umst_edges),
            backbone_edges: flatten_edge_pairs(&inner.backbone_edges),
            simmelian_weights: inner.simmelian_weights.clone(),
            edge_weights: inner.edge_weights.clone(),
            backbone_csr,
        })
    }
}

#[wasm_bindgen(js_class = BackboneResult)]
impl JsBackboneResult {
    /// Returns flat `[u0, v0, u1, v1, ...]` Union of Maximum Spanning Trees edges (`Uint32Array`).
    #[wasm_bindgen(js_name = umstEdges)]
    pub fn umst_edges(&self) -> Vec<u32> {
        self.umst_edges.clone()
    }

    /// Returns flat `[u0, v0, u1, v1, ...]` backbone edge endpoints (`Uint32Array`).
    #[wasm_bindgen(js_name = backboneEdges)]
    pub fn backbone_edges(&self) -> Vec<u32> {
        self.backbone_edges.clone()
    }

    /// Returns per-edge Simmelian prefix-Jaccard weights (`Float64Array`).
    #[wasm_bindgen(js_name = simmelianWeights)]
    pub fn simmelian_weights(&self) -> Vec<f64> {
        self.simmelian_weights.clone()
    }

    /// Returns final reweighted edge weights corresponding to `backboneEdges` (`Float64Array`).
    #[wasm_bindgen(js_name = edgeWeights)]
    pub fn edge_weights(&self) -> Vec<f64> {
        self.edge_weights.clone()
    }

    /// Returns a cloned weighted undirected `CsrGraph` representing the reweighted backbone.
    #[wasm_bindgen(js_name = backboneCsr)]
    pub fn backbone_csr(&self) -> JsCsrGraph {
        JsCsrGraph::from_inner(self.backbone_csr.clone())
    }

    /// Converts edge and weight arrays into a plain JavaScript object without requiring `serde`.
    #[wasm_bindgen(js_name = toRecord)]
    pub fn to_record(&self) -> Result<js_sys::Object, JsError> {
        let obj = js_sys::Object::new();
        set_js_prop(
            &obj,
            "umstEdges",
            &js_sys::Uint32Array::from(self.umst_edges.as_slice()).into(),
        )?;
        set_js_prop(
            &obj,
            "backboneEdges",
            &js_sys::Uint32Array::from(self.backbone_edges.as_slice()).into(),
        )?;
        set_js_prop(
            &obj,
            "simmelianWeights",
            &js_sys::Float64Array::from(self.simmelian_weights.as_slice()).into(),
        )?;
        set_js_prop(
            &obj,
            "edgeWeights",
            &js_sys::Float64Array::from(self.edge_weights.as_slice()).into(),
        )?;
        Ok(obj)
    }
}

/// Extracts the Quadrilateral Simmelian Backbone from an undirected `CsrGraph`.
#[wasm_bindgen(js_name = extractQuadrilateralBackbone)]
pub fn js_extract_quadrilateral_backbone(
    graph: &JsCsrGraph,
    options: &JsValue,
) -> Result<JsBackboneResult, JsError> {
    let cfg = parse_backbone_config(options, "extractQuadrilateralBackbone")?;
    let res = extract_quadrilateral_backbone(graph.inner(), &cfg).map_err(core_err_to_js)?;
    JsBackboneResult::from_inner(&res)
}

/// Cooperative Quadrilateral Simmelian Backbone layout solver (`f32`).
#[wasm_bindgen(js_name = BackboneSolver)]
#[derive(Debug, Clone)]
pub struct JsBackboneSolver {
    inner: BackboneSolver<f32>,
}

#[wasm_bindgen(js_class = BackboneSolver)]
impl JsBackboneSolver {
    /// Constructs a `BackboneSolver` from `graph` and optional `BackboneOptions`.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsBackboneSolver, JsError> {
        let cfg = parse_backbone_config(options, "BackboneSolver")?;
        let inner = BackboneSolver::<f32>::new(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Returns the extracted `BackboneResult`.
    #[wasm_bindgen(js_name = backboneResult)]
    pub fn backbone_result(&self) -> Result<JsBackboneResult, JsError> {
        JsBackboneResult::from_inner(self.inner.backbone_result())
    }

    /// Executes one backbone layout iteration on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` backbone layout iterations on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "BackboneSolver::run_steps",
        )
    }

    /// Runs backbone layout optimization to completion on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "BackboneSolver::run",
        )
    }

    /// Returns the solver lifecycle status string.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        solver_status_str(self.inner.status()).to_owned()
    }

    /// Returns the number of iterations executed so far.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    /// Returns the scheduled iteration count.
    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> usize {
        self.inner.total_iterations()
    }

    /// Returns `true` when the solver has converged or completed.
    #[wasm_bindgen(getter, js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        self.inner.status().is_finished()
    }

    /// Returns the expected vertex count.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }
}
