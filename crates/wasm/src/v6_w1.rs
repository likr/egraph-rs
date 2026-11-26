//! W1 v6 WebAssembly bindings for baseline community detection, layout solvers,
//! quadrilateral Simmelian backbone extraction, cluster refinement, Group-in-a-Box,
//! GTree/FORBID overlap removal, and `eval-baselines` evaluation metrics (`f32`-backed
//! layouts with `f64` internal metric accumulation; zero `serde` / `getrandom` / threads).

use egraph::clustering::{LeidenConfig, LeidenObjective, LeidenResult, leiden};
use egraph::core::{CsrGraph, IterativeSolver};
use egraph::layout::{
    BackboneConfig, BackboneLayoutMode, BackboneResult, BackboneSolver, ClusterRefineConfig,
    ClusterRefineMethod, ClusterRefineSolver, ForbidConfig, ForbidOverlapRemoval, ForbidVariant,
    FrBackend, FruchtermanReingoldConfig, FruchtermanReingoldSolver, GTreeConfig,
    GTreeOverlapRemoval, GibBoxLayoutMode, GibIntraLayoutMode, GroupInABoxConfig,
    GroupInABoxSolver, LinLogConfig, LinLogRepulsion, LinLogSolver, extract_quadrilateral_backbone,
    refine_cluster_layout, squarified_treemap,
};
use egraph::metrics::{
    ClusterBaselineConfig, NorMetrics, PartitionAgreement, calinski_harabasz_score,
    class_density_measure, cq_hac, cq_kmeans, davies_bouldin_score, distance_consistency,
    distribution_consistency, neighborhood_hit, nor_metrics, nor_metrics_without_graph,
    pairwise_distance_overlap_error, partition_agreement, sepme_gab, sepme_gong, sepme_knng,
    sepme_rng, silhouette_score, sprawlter_nn, sprawlter_nn_radii,
};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{
    core_err_to_js, csr_f64_to_f32, ffi_err_to_js, parse_f64_sequence, parse_non_negative_usize,
    parse_optional_f64_sequence, parse_seed_u64, parse_u32_sequence,
};
use crate::js_options::{
    check_options_arg, get_prop, normalize_enum_token, opt_bool_prop, opt_f64_prop,
    opt_maybe_f64_prop, opt_seed_prop, opt_string_prop, opt_usize_prop, solver_status_str,
};
use crate::layout_buffers::JsLayout2d;
use crate::v6_m3::{run_layout_in_place, run_steps_layout_in_place, step_layout_in_place};

fn set_js_prop(obj: &js_sys::Object, key: &str, val: &JsValue) -> Result<(), JsError> {
    js_sys::Reflect::set(obj, &JsValue::from_str(key), val)
        .map(|_| ())
        .map_err(|err| JsError::new(&format!("failed to set {key}: {err:?}")))
}

fn set_f64_prop(obj: &js_sys::Object, key: &str, val: f64) -> Result<(), JsError> {
    set_js_prop(obj, key, &JsValue::from_f64(val))
}

fn opt_usize_with_default(val: Option<f64>, default: usize, name: &str) -> Result<usize, JsError> {
    match val {
        None => Ok(default),
        Some(v) => parse_non_negative_usize(v, name),
    }
}

fn flatten_edge_pairs(edges: &[(u32, u32)]) -> Vec<u32> {
    let mut flat = Vec::with_capacity(edges.len() * 2);
    for &(u, v) in edges {
        flat.push(u);
        flat.push(v);
    }
    flat
}

// ============================================================================
// W1-1. Leiden Community Detection
// ============================================================================

fn parse_leiden_objective(raw: &str) -> Result<LeidenObjective, JsError> {
    match normalize_enum_token(raw).as_str() {
        "modularity" | "mod" => Ok(LeidenObjective::Modularity),
        "cpm" | "constantpottsmodel" => Ok(LeidenObjective::Cpm),
        _ => Err(JsError::new(&format!(
            "unknown LeidenObjective {raw:?}; expected 'modularity' or 'cpm'"
        ))),
    }
}

/// Community detection output returned by `leidenCommunities`.
#[wasm_bindgen(js_name = LeidenResult)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsLeidenResult {
    inner: LeidenResult,
}

#[wasm_bindgen(js_class = LeidenResult)]
impl JsLeidenResult {
    /// Returns the community ID per vertex (`Uint32Array`).
    pub fn labels(&self) -> Vec<u32> {
        self.inner.labels.clone()
    }

    /// Returns the number of distinct communities detected.
    #[wasm_bindgen(getter, js_name = numCommunities)]
    pub fn num_communities(&self) -> usize {
        self.inner.num_communities
    }

    /// Returns the objective quality value achieved by Leiden.
    #[wasm_bindgen(getter)]
    pub fn quality(&self) -> f64 {
        self.inner.quality
    }

    /// Returns the resolution-weighted modularity $Q_\gamma$.
    #[wasm_bindgen(getter)]
    pub fn modularity(&self) -> f64 {
        self.inner.modularity
    }

    /// Returns the number of multi-level aggregation passes committed.
    #[wasm_bindgen(getter)]
    pub fn passes(&self) -> usize {
        self.inner.passes
    }

    /// Converts this result into a plain JavaScript object without requiring `serde`.
    #[wasm_bindgen(js_name = toRecord)]
    pub fn to_record(&self) -> Result<js_sys::Object, JsError> {
        let obj = js_sys::Object::new();
        set_js_prop(
            &obj,
            "labels",
            &js_sys::Uint32Array::from(self.inner.labels.as_slice()).into(),
        )?;
        set_f64_prop(&obj, "numCommunities", self.inner.num_communities as f64)?;
        set_f64_prop(&obj, "quality", self.inner.quality)?;
        set_f64_prop(&obj, "modularity", self.inner.modularity)?;
        set_f64_prop(&obj, "passes", self.inner.passes as f64)?;
        Ok(obj)
    }
}

/// Detects communities on an undirected `CsrGraph` via the Leiden algorithm (`Modularity` or `CPM`).
#[wasm_bindgen(js_name = leidenCommunities)]
pub fn js_leiden_communities(
    graph: &JsCsrGraph,
    options: &JsValue,
) -> Result<JsLeidenResult, JsError> {
    let mut cfg = LeidenConfig::default();
    if check_options_arg(options, "leidenCommunities")? {
        if let Some(obj_str) = opt_string_prop(options, "objective")? {
            cfg.objective = parse_leiden_objective(&obj_str)?;
        }
        cfg.resolution = opt_f64_prop(options, "resolution", cfg.resolution)?;
        cfg.randomness = opt_f64_prop(options, "randomness", cfg.randomness)?;
        cfg.max_passes = opt_usize_prop(options, "maxPasses", cfg.max_passes)?;
        cfg.min_quality_gain = opt_f64_prop(options, "minQualityGain", cfg.min_quality_gain)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    let inner = leiden(graph.inner(), &cfg).map_err(core_err_to_js)?;
    Ok(JsLeidenResult { inner })
}

// ============================================================================
// W1-2. Noack LinLog & Fruchterman-Reingold Layout Solvers
// ============================================================================

fn parse_linlog_repulsion(raw: &str) -> Result<LinLogRepulsion, JsError> {
    match normalize_enum_token(raw).as_str() {
        "edgerepulsion" | "edge" => Ok(LinLogRepulsion::EdgeRepulsion),
        "noderepulsion" | "node" => Ok(LinLogRepulsion::NodeRepulsion),
        _ => Err(JsError::new(&format!(
            "unknown LinLogRepulsion {raw:?}; expected 'edge_repulsion' or 'node_repulsion'"
        ))),
    }
}

/// Cooperative Noack (2007/2009) LinLog / $(a, r)$-PolyLog force-directed layout solver (`f32`).
#[wasm_bindgen(js_name = LinLogSolver)]
#[derive(Debug, Clone)]
pub struct JsLinLogSolver {
    inner: LinLogSolver<f32>,
}

#[wasm_bindgen(js_class = LinLogSolver)]
impl JsLinLogSolver {
    /// Constructs a `LinLogSolver` from `graph` and optional `LinLogOptions`.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsLinLogSolver, JsError> {
        let mut cfg = LinLogConfig::default();
        if check_options_arg(options, "LinLogSolver")? {
            cfg.iterations = opt_usize_prop(options, "iterations", cfg.iterations)?;
            if let Some(rep_str) = opt_string_prop(options, "repulsionModel")? {
                cfg.repulsion_model = parse_linlog_repulsion(&rep_str)?;
            }
            cfg.attraction_exponent =
                opt_f64_prop(options, "attractionExponent", cfg.attraction_exponent)?;
            cfg.repulsion_exponent =
                opt_f64_prop(options, "repulsionExponent", cfg.repulsion_exponent)?;
            cfg.theta = opt_f64_prop(options, "theta", cfg.theta)?;
            cfg.use_barnes_hut = opt_bool_prop(options, "useBarnesHut", cfg.use_barnes_hut)?;
            cfg.gravity = opt_f64_prop(options, "gravity", cfg.gravity)?;
            cfg.edge_weight_influence =
                opt_f64_prop(options, "edgeWeightInfluence", cfg.edge_weight_influence)?;
            cfg.max_displacement = opt_maybe_f64_prop(options, "maxDisplacement")?;
            cfg.seed = opt_seed_prop(options)?;
        }
        let inner = LinLogSolver::<f32>::new(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one LinLog iteration on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` LinLog iterations on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "LinLogSolver::run_steps",
        )
    }

    /// Runs LinLog to completion on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "LinLogSolver::run",
        )
    }

    /// Evaluates the exact LinLog energy of `layout`.
    pub fn energy(&self, layout: &JsLayout2d) -> Result<f64, JsError> {
        let soa = layout.inner().to_layout2d();
        self.inner.energy(&soa).map_err(core_err_to_js)
    }

    /// Returns the per-node repulsive masses (`Float64Array`).
    pub fn masses(&self) -> Vec<f64> {
        self.inner.masses().to_vec()
    }

    /// Returns the effective per-step vertex displacement cap.
    #[wasm_bindgen(getter, js_name = maxDisplacement)]
    pub fn max_displacement(&self) -> f64 {
        self.inner.max_displacement()
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

fn parse_fr_backend(raw: &str) -> Result<FrBackend, JsError> {
    match normalize_enum_token(raw).as_str() {
        "exact" => Ok(FrBackend::Exact),
        "barneshut" | "bh" => Ok(FrBackend::BarnesHut),
        "grid" => Ok(FrBackend::Grid),
        _ => Err(JsError::new(&format!(
            "unknown FrBackend {raw:?}; expected 'exact', 'barnes_hut', or 'grid'"
        ))),
    }
}

/// Cooperative Fruchterman-Reingold (1991) force-directed layout solver (`f32`).
#[wasm_bindgen(js_name = FruchtermanReingoldSolver)]
#[derive(Debug, Clone)]
pub struct JsFruchtermanReingoldSolver {
    inner: FruchtermanReingoldSolver<f32>,
}

#[wasm_bindgen(js_class = FruchtermanReingoldSolver)]
impl JsFruchtermanReingoldSolver {
    /// Constructs a `FruchtermanReingoldSolver` from `graph` and optional `FruchtermanReingoldOptions`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        graph: &JsCsrGraph,
        options: &JsValue,
    ) -> Result<JsFruchtermanReingoldSolver, JsError> {
        let mut cfg = FruchtermanReingoldConfig::default();
        if check_options_arg(options, "FruchtermanReingoldSolver")? {
            cfg.iterations = opt_usize_prop(options, "iterations", cfg.iterations)?;
            if let Some(b_str) = opt_string_prop(options, "backend")? {
                cfg.backend = parse_fr_backend(&b_str)?;
            }
            cfg.k = opt_maybe_f64_prop(options, "k")?;
            cfg.area = opt_maybe_f64_prop(options, "area")?;
            cfg.initial_temperature = opt_maybe_f64_prop(options, "initialTemperature")?;
            cfg.gravity = opt_f64_prop(options, "gravity", cfg.gravity)?;
            cfg.theta = opt_f64_prop(options, "theta", cfg.theta)?;
            cfg.seed = opt_seed_prop(options)?;
        }
        let inner =
            FruchtermanReingoldSolver::<f32>::new(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one Fruchterman-Reingold iteration on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` Fruchterman-Reingold iterations on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "FruchtermanReingoldSolver::run_steps",
        )
    }

    /// Runs Fruchterman-Reingold to completion on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "FruchtermanReingoldSolver::run",
        )
    }

    /// Returns the effective optimal pairwise distance $k$.
    #[wasm_bindgen(getter, js_name = optimalDistance)]
    pub fn optimal_distance(&self) -> f64 {
        self.inner.optimal_distance()
    }

    /// Returns the current cooling temperature.
    #[wasm_bindgen(getter, js_name = currentTemperature)]
    pub fn current_temperature(&self) -> f64 {
        self.inner.current_temperature()
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

// ============================================================================
// W1-3. Quadrilateral Simmelian Backbone Extraction & Solver
// ============================================================================

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

// ============================================================================
// W1-4. Cluster Refinement & Group-in-a-Box
// ============================================================================

fn parse_cluster_refine_method(raw: &str) -> Result<ClusterRefineMethod, JsError> {
    match normalize_enum_token(raw).as_str() {
        "kmeans" | "clusterkmeans" => Ok(ClusterRefineMethod::Kmeans),
        "hac" | "clusterhac" => Ok(ClusterRefineMethod::Hac),
        _ => Err(JsError::new(&format!(
            "unknown ClusterRefineMethod {raw:?}; expected 'kmeans' or 'hac'"
        ))),
    }
}

fn parse_cluster_refine_config(
    options: &JsValue,
    ctx: &str,
) -> Result<ClusterRefineConfig, JsError> {
    let mut cfg = ClusterRefineConfig::default();
    if check_options_arg(options, ctx)? {
        if let Some(m_str) = opt_string_prop(options, "method")? {
            cfg.method = parse_cluster_refine_method(&m_str)?;
        }
        cfg.max_rounds = opt_usize_prop(options, "maxRounds", cfg.max_rounds)?;
        cfg.kmeans_n_init = opt_usize_prop(options, "kmeansNInit", cfg.kmeans_n_init)?;
        cfg.kmeans_max_iters = opt_usize_prop(options, "kmeansMaxIters", cfg.kmeans_max_iters)?;
        cfg.shift_step = opt_f64_prop(options, "shiftStep", cfg.shift_step)?;
        cfg.margin = opt_f64_prop(options, "margin", cfg.margin)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    Ok(cfg)
}

/// Cooperative cluster-faithful layout refiner (`Cluster-kmeans` / `Cluster-HAC`, `f32`).
#[wasm_bindgen(js_name = ClusterRefineSolver)]
#[derive(Debug, Clone)]
pub struct JsClusterRefineSolver {
    inner: ClusterRefineSolver<f32>,
}

#[wasm_bindgen(js_class = ClusterRefineSolver)]
impl JsClusterRefineSolver {
    /// Constructs a `ClusterRefineSolver` from `clusterLabels` without graph edge lengths.
    #[wasm_bindgen(constructor)]
    pub fn new(
        cluster_labels: &JsValue,
        options: &JsValue,
    ) -> Result<JsClusterRefineSolver, JsError> {
        let labels = parse_u32_sequence(cluster_labels, "clusterLabels")?;
        let cfg = parse_cluster_refine_config(options, "ClusterRefineSolver")?;
        let inner = ClusterRefineSolver::<f32>::new(labels, cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Constructs a `ClusterRefineSolver` with graph topology for `hac` mean edge-length scaling.
    #[wasm_bindgen(js_name = fromGraph)]
    pub fn from_graph(
        graph: &JsCsrGraph,
        cluster_labels: &JsValue,
        options: &JsValue,
    ) -> Result<JsClusterRefineSolver, JsError> {
        let labels = parse_u32_sequence(cluster_labels, "clusterLabels")?;
        let cfg = parse_cluster_refine_config(options, "ClusterRefineSolver.fromGraph")?;
        let inner = ClusterRefineSolver::<f32>::from_graph(graph.inner(), labels, cfg)
            .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one refinement round on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` refinement rounds on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "ClusterRefineSolver::run_steps",
        )
    }

    /// Runs cluster refinement to completion on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "ClusterRefineSolver::run",
        )
    }

    /// Returns the number of misplaced vertices shifted in the most recent round.
    #[wasm_bindgen(getter, js_name = lastMovedCount)]
    pub fn last_moved_count(&self) -> usize {
        self.inner.last_moved_count()
    }

    /// Returns the number of distinct compacted clusters.
    #[wasm_bindgen(getter, js_name = numClusters)]
    pub fn num_clusters(&self) -> usize {
        self.inner.num_clusters()
    }

    /// Returns the solver lifecycle status string.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        solver_status_str(self.inner.status()).to_owned()
    }

    /// Returns the number of rounds executed so far.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    /// Returns the maximum scheduled rounds (`maxRounds`).
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

/// Refines a `Layout2d` in place to improve cluster spatial separation, returning total moved nodes.
#[wasm_bindgen(js_name = refineClusterLayout)]
pub fn js_refine_cluster_layout(
    layout: &mut JsLayout2d,
    cluster_labels: &JsValue,
    options: &JsValue,
) -> Result<usize, JsError> {
    let labels = parse_u32_sequence(cluster_labels, "clusterLabels")?;
    let cfg = parse_cluster_refine_config(options, "refineClusterLayout")?;
    let mut soa = layout.inner().to_layout2d();
    let moved = refine_cluster_layout(&mut soa, &labels, &cfg).map_err(core_err_to_js)?;
    layout
        .inner_mut()
        .copy_from_layout2d(&soa)
        .map_err(ffi_err_to_js)?;
    Ok(moved)
}

/// Computes a Squarified Treemap partition returning flat `[xMin0, xMax0, yMin0, yMax0, ...]` (`Float64Array`).
#[wasm_bindgen(js_name = squarifiedTreemap)]
pub fn js_squarified_treemap(
    weights: &JsValue,
    width: Option<f64>,
    height: Option<f64>,
    x: Option<f64>,
    y: Option<f64>,
) -> Result<Vec<f64>, JsError> {
    let w = parse_f64_sequence(weights, "weights")?;
    let rects = squarified_treemap(
        &w,
        x.unwrap_or(0.0),
        y.unwrap_or(0.0),
        width.unwrap_or(100.0),
        height.unwrap_or(100.0),
    )
    .map_err(core_err_to_js)?;
    let mut flat = Vec::with_capacity(rects.len() * 4);
    for r in rects {
        flat.extend_from_slice(&r);
    }
    Ok(flat)
}

fn parse_gib_box_mode(raw: &str) -> Result<GibBoxLayoutMode, JsError> {
    match normalize_enum_token(raw).as_str() {
        "treemap" | "squarifiedtreemap" | "trgib" => Ok(GibBoxLayoutMode::SquarifiedTreemap),
        "forcedirected" | "fd" | "fdgib" => Ok(GibBoxLayoutMode::ForceDirected),
        _ => Err(JsError::new(&format!(
            "unknown GibBoxLayoutMode {raw:?}; expected 'treemap' or 'force_directed'"
        ))),
    }
}

fn parse_gib_intra_mode(raw: &str) -> Result<GibIntraLayoutMode, JsError> {
    match normalize_enum_token(raw).as_str() {
        "fr" | "fruchtermanreingold" => Ok(GibIntraLayoutMode::Fr),
        "sgdfull" | "sgd" => Ok(GibIntraLayoutMode::SgdFull),
        "yifanhu" => Ok(GibIntraLayoutMode::YifanHu),
        _ => Err(JsError::new(&format!(
            "unknown GibIntraLayoutMode {raw:?}; expected 'fr', 'sgd_full', or 'yifan_hu'"
        ))),
    }
}

/// Cooperative Group-in-a-Box (`TR-GIB` / `FD-GIB`) clustered layout solver (`f32`).
///
/// Note: Intentionally omits `Debug` and `Clone` because the inner solver wraps stateful
/// boxed intra-cluster solvers.
#[wasm_bindgen(js_name = GroupInABoxSolver)]
pub struct JsGroupInABoxSolver {
    inner: GroupInABoxSolver<f32>,
}

#[wasm_bindgen(js_class = GroupInABoxSolver)]
impl JsGroupInABoxSolver {
    /// Constructs a `GroupInABoxSolver` from `graph`, `clusterLabels`, and optional `GroupInABoxOptions`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        graph: &JsCsrGraph,
        cluster_labels: &JsValue,
        options: &JsValue,
    ) -> Result<JsGroupInABoxSolver, JsError> {
        let labels = parse_u32_sequence(cluster_labels, "clusterLabels")?;
        let mut cfg = GroupInABoxConfig::default();
        let mut cluster_weights: Option<Vec<f64>> = None;
        if check_options_arg(options, "GroupInABoxSolver")? {
            if let Some(bm) = opt_string_prop(options, "boxMode")? {
                cfg.box_mode = parse_gib_box_mode(&bm)?;
            }
            if let Some(im) = opt_string_prop(options, "intraMode")? {
                cfg.intra_mode = parse_gib_intra_mode(&im)?;
            }
            cfg.total_width = opt_f64_prop(options, "totalWidth", cfg.total_width)?;
            cfg.total_height = opt_f64_prop(options, "totalHeight", cfg.total_height)?;
            cfg.box_padding = opt_f64_prop(options, "boxPadding", cfg.box_padding)?;
            cfg.intra_iterations =
                opt_usize_prop(options, "intraIterations", cfg.intra_iterations)?;
            cfg.fd_iterations = opt_usize_prop(options, "fdIterations", cfg.fd_iterations)?;
            cluster_weights = parse_optional_f64_sequence(
                &get_prop(options, "clusterWeights")?,
                "clusterWeights",
            )?;
            cfg.seed = opt_seed_prop(options)?;
        }
        let inner = match cluster_weights {
            Some(cw) => {
                GroupInABoxSolver::<f32>::with_cluster_weights(graph.inner(), labels, cw, cfg)
            }
            None => GroupInABoxSolver::<f32>::new(graph.inner(), labels, cfg),
        }
        .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Returns the computed cluster boxes ordered canonically by first appearance.
    pub fn boxes(&self) -> Result<js_sys::Array, JsError> {
        let out = js_sys::Array::new();
        for b in self.inner.boxes() {
            let obj = js_sys::Object::new();
            set_f64_prop(&obj, "clusterId", b.cluster_id as f64)?;
            set_f64_prop(&obj, "xMin", b.x_min)?;
            set_f64_prop(&obj, "xMax", b.x_max)?;
            set_f64_prop(&obj, "yMin", b.y_min)?;
            set_f64_prop(&obj, "yMax", b.y_max)?;
            set_f64_prop(&obj, "nodeCount", b.node_count as f64)?;
            set_f64_prop(&obj, "weight", b.weight)?;
            out.push(&obj);
        }
        Ok(out)
    }

    /// Executes one intra-box layout iteration across active clusters on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` intra-box layout iterations on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "GroupInABoxSolver::run_steps",
        )
    }

    /// Runs Group-in-a-Box intra-cluster optimization to completion on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "GroupInABoxSolver::run",
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

    /// Returns the scheduled intra-cluster iteration count.
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

// ============================================================================
// W1-5. GTree & FORBID Overlap Removal
// ============================================================================

fn parse_gtree_config(options: &JsValue, ctx: &str) -> Result<GTreeConfig, JsError> {
    let mut cfg = GTreeConfig::default();
    if check_options_arg(options, ctx)? {
        cfg.max_iterations = opt_usize_prop(options, "maxIterations", cfg.max_iterations)?;
        cfg.padding = opt_f64_prop(options, "padding", cfg.padding)?;
        cfg.compact_t_max = opt_maybe_f64_prop(options, "compactTMax")?;
        cfg.seed = opt_seed_prop(options)?;
    }
    Ok(cfg)
}

/// Cooperative Nachmanson et al. (2016) GTree overlap removal solver (`f32`).
///
/// Uses `Delaunay2d` when compiled with `delaunay` and falls back automatically to
/// `StaticKdTree` proximity under `--no-default-features`.
#[wasm_bindgen(js_name = GTreeOverlapRemoval)]
#[derive(Debug, Clone)]
pub struct JsGTreeOverlapRemoval {
    inner: GTreeOverlapRemoval<f32>,
}

#[wasm_bindgen(js_class = GTreeOverlapRemoval)]
impl JsGTreeOverlapRemoval {
    /// Constructs a `GTreeOverlapRemoval` solver from per-node `halfWidths` and `halfHeights`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        half_widths: &[f32],
        half_heights: &[f32],
        options: &JsValue,
    ) -> Result<JsGTreeOverlapRemoval, JsError> {
        let cfg = parse_gtree_config(options, "GTreeOverlapRemoval")?;
        let inner =
            GTreeOverlapRemoval::<f32>::new(half_widths.to_vec(), half_heights.to_vec(), cfg)
                .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Constructs a `GTreeOverlapRemoval` solver with uniform square `radii`.
    #[wasm_bindgen(js_name = fromRadii)]
    pub fn from_radii(radii: &[f32], options: &JsValue) -> Result<JsGTreeOverlapRemoval, JsError> {
        let cfg = parse_gtree_config(options, "GTreeOverlapRemoval.fromRadii")?;
        let inner =
            GTreeOverlapRemoval::<f32>::from_radii(radii.to_vec(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one GTree overlap removal pass on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` GTree overlap removal passes on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "GTreeOverlapRemoval::run_steps",
        )
    }

    /// Runs GTree overlap removal until `0` overlaps remain or `maxIterations` is reached.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "GTreeOverlapRemoval::run",
        )
    }

    /// Returns the number of overlapping rectangle pairs remaining after the most recent step.
    #[wasm_bindgen(getter, js_name = lastOverlapCount)]
    pub fn last_overlap_count(&self) -> usize {
        self.inner.last_overlap_count()
    }

    /// Returns the solver lifecycle status string.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        solver_status_str(self.inner.status()).to_owned()
    }

    /// Returns the number of passes executed so far.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    /// Returns the maximum scheduled passes (`maxIterations`).
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

fn parse_forbid_variant(raw: &str) -> Result<ForbidVariant, JsError> {
    match normalize_enum_token(raw).as_str() {
        "forbid" => Ok(ForbidVariant::Forbid),
        "forbidprime" => Ok(ForbidVariant::ForbidPrime),
        "sparseforbid" => Ok(ForbidVariant::SparseForbid),
        _ => Err(JsError::new(&format!(
            "unknown ForbidVariant {raw:?}; expected 'forbid', 'forbid_prime', or 'sparse_forbid'"
        ))),
    }
}

fn parse_forbid_config(options: &JsValue, ctx: &str) -> Result<ForbidConfig, JsError> {
    let mut cfg = ForbidConfig::default();
    if check_options_arg(options, ctx)? {
        if let Some(v_str) = opt_string_prop(options, "variant")? {
            cfg.variant = parse_forbid_variant(&v_str)?;
        }
        cfg.max_scale_iters = opt_usize_prop(options, "maxScaleIters", cfg.max_scale_iters)?;
        cfg.sgd_iterations = opt_usize_prop(options, "sgdIterations", cfg.sgd_iterations)?;
        cfg.sparse_k = opt_usize_prop(options, "sparseK", cfg.sparse_k)?;
        cfg.overlap_weight_k = opt_f64_prop(options, "overlapWeightK", cfg.overlap_weight_k)?;
        cfg.scale_step = opt_f64_prop(options, "scaleStep", cfg.scale_step)?;
        cfg.padding = opt_f64_prop(options, "padding", cfg.padding)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    Ok(cfg)
}

/// Cooperative FORBID / FORBID' / SparseFORBID overlap removal solver (`f32`).
#[wasm_bindgen(js_name = ForbidOverlapRemoval)]
#[derive(Debug, Clone)]
pub struct JsForbidOverlapRemoval {
    inner: ForbidOverlapRemoval<f32>,
}

#[wasm_bindgen(js_class = ForbidOverlapRemoval)]
impl JsForbidOverlapRemoval {
    /// Constructs a `ForbidOverlapRemoval` solver from per-node `halfWidths` and `halfHeights`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        half_widths: &[f32],
        half_heights: &[f32],
        options: &JsValue,
    ) -> Result<JsForbidOverlapRemoval, JsError> {
        let cfg = parse_forbid_config(options, "ForbidOverlapRemoval")?;
        let inner =
            ForbidOverlapRemoval::<f32>::new(half_widths.to_vec(), half_heights.to_vec(), cfg)
                .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Constructs a `ForbidOverlapRemoval` solver with uniform square `radii`.
    #[wasm_bindgen(js_name = fromRadii)]
    pub fn from_radii(radii: &[f32], options: &JsValue) -> Result<JsForbidOverlapRemoval, JsError> {
        let cfg = parse_forbid_config(options, "ForbidOverlapRemoval.fromRadii")?;
        let inner =
            ForbidOverlapRemoval::<f32>::from_radii(radii.to_vec(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one FORBID scaling + SGD round on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` FORBID rounds on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "ForbidOverlapRemoval::run_steps",
        )
    }

    /// Runs FORBID overlap removal until `0` overlaps remain or `maxScaleIters` is reached.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "ForbidOverlapRemoval::run",
        )
    }

    /// Returns the number of overlapping rectangle pairs remaining after the most recent step.
    #[wasm_bindgen(getter, js_name = lastOverlapCount)]
    pub fn last_overlap_count(&self) -> usize {
        self.inner.last_overlap_count()
    }

    /// Returns the solver lifecycle status string.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        solver_status_str(self.inner.status()).to_owned()
    }

    /// Returns the number of scaling rounds executed so far.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    /// Returns the maximum scheduled scaling rounds (`maxScaleIters`).
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

// ============================================================================
// W1-6. Cluster Separation, CVI, Sprawlter-NN, and NOR Baseline Metrics
// ============================================================================

/// Partition agreement metrics returned by `partitionAgreement`, `cqKmeans`, and `cqHac`.
#[wasm_bindgen(js_name = PartitionAgreementResult)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JsPartitionAgreementResult {
    inner: PartitionAgreement,
}

#[wasm_bindgen(js_class = PartitionAgreementResult)]
impl JsPartitionAgreementResult {
    #[wasm_bindgen(getter)]
    pub fn ari(&self) -> f64 {
        self.inner.ari
    }

    #[wasm_bindgen(getter)]
    pub fn ami(&self) -> f64 {
        self.inner.ami
    }

    #[wasm_bindgen(getter)]
    pub fn homogeneity(&self) -> f64 {
        self.inner.homogeneity
    }

    #[wasm_bindgen(getter)]
    pub fn completeness(&self) -> f64 {
        self.inner.completeness
    }

    #[wasm_bindgen(getter, js_name = vMeasure)]
    pub fn v_measure(&self) -> f64 {
        self.inner.v_measure
    }

    #[wasm_bindgen(getter)]
    pub fn fmi(&self) -> f64 {
        self.inner.fmi
    }

    #[wasm_bindgen(js_name = toRecord)]
    pub fn to_record(&self) -> Result<js_sys::Object, JsError> {
        let obj = js_sys::Object::new();
        set_f64_prop(&obj, "ari", self.inner.ari)?;
        set_f64_prop(&obj, "ami", self.inner.ami)?;
        set_f64_prop(&obj, "homogeneity", self.inner.homogeneity)?;
        set_f64_prop(&obj, "completeness", self.inner.completeness)?;
        set_f64_prop(&obj, "vMeasure", self.inner.v_measure)?;
        set_f64_prop(&obj, "fmi", self.inner.fmi)?;
        Ok(obj)
    }
}

/// Class Density Measure (`cdm`) and 2D Histogram Density Measure (`hdm`) returned by `classDensityMeasure`.
#[wasm_bindgen(js_name = ClassDensityResult)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JsClassDensityResult {
    cdm: f64,
    hdm: f64,
}

#[wasm_bindgen(js_class = ClassDensityResult)]
impl JsClassDensityResult {
    #[wasm_bindgen(getter)]
    pub fn cdm(&self) -> f64 {
        self.cdm
    }

    #[wasm_bindgen(getter)]
    pub fn hdm(&self) -> f64 {
        self.hdm
    }

    #[wasm_bindgen(js_name = toRecord)]
    pub fn to_record(&self) -> Result<js_sys::Object, JsError> {
        let obj = js_sys::Object::new();
        set_f64_prop(&obj, "cdm", self.cdm)?;
        set_f64_prop(&obj, "hdm", self.hdm)?;
        Ok(obj)
    }
}

/// Chen et al. (2020) Node Overlap Removal (NOR) 5-metric summary returned by `norMetrics` / `norMetricsWithoutGraph`.
#[wasm_bindgen(js_name = NorMetricsResult)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JsNorMetricsResult {
    inner: NorMetrics,
}

#[wasm_bindgen(js_class = NorMetricsResult)]
impl JsNorMetricsResult {
    #[wasm_bindgen(getter, js_name = ooNni)]
    pub fn oo_nni(&self) -> f64 {
        self.inner.oo_nni
    }

    #[wasm_bindgen(getter, js_name = spChA)]
    pub fn sp_ch_a(&self) -> f64 {
        self.inner.sp_ch_a
    }

    #[wasm_bindgen(getter, js_name = gsBbIar)]
    pub fn gs_bb_iar(&self) -> f64 {
        self.inner.gs_bb_iar
    }

    #[wasm_bindgen(getter, js_name = nmDmImse)]
    pub fn nm_dm_imse(&self) -> f64 {
        self.inner.nm_dm_imse
    }

    #[wasm_bindgen(getter, js_name = elRsd)]
    pub fn el_rsd(&self) -> f64 {
        self.inner.el_rsd
    }

    #[wasm_bindgen(js_name = toRecord)]
    pub fn to_record(&self) -> Result<js_sys::Object, JsError> {
        let obj = js_sys::Object::new();
        set_f64_prop(&obj, "ooNni", self.inner.oo_nni)?;
        set_f64_prop(&obj, "spChA", self.inner.sp_ch_a)?;
        set_f64_prop(&obj, "gsBbIar", self.inner.gs_bb_iar)?;
        set_f64_prop(&obj, "nmDmImse", self.inner.nm_dm_imse)?;
        set_f64_prop(&obj, "elRsd", self.inner.el_rsd)?;
        Ok(obj)
    }
}

/// Computes `ARI`, `AMI`, `homogeneity`, `completeness`, `vMeasure`, and `fmi` between two labelings.
#[wasm_bindgen(js_name = partitionAgreement)]
pub fn js_partition_agreement(
    labels_true: &JsValue,
    labels_pred: &JsValue,
) -> Result<JsPartitionAgreementResult, JsError> {
    let lt = parse_u32_sequence(labels_true, "labelsTrue")?;
    let lp = parse_u32_sequence(labels_pred, "labelsPred")?;
    let inner = partition_agreement(&lt, &lp).map_err(core_err_to_js)?;
    Ok(JsPartitionAgreementResult { inner })
}

/// Evaluates `k`-means Cluster Quality (`CQ-kmeans`) of `layout` against `communityLabels`.
#[wasm_bindgen(js_name = cqKmeans)]
pub fn js_cq_kmeans(
    layout: &JsLayout2d,
    community_labels: &JsValue,
    n_init: Option<f64>,
    max_iters: Option<f64>,
    seed: Option<f64>,
) -> Result<JsPartitionAgreementResult, JsError> {
    let defaults = ClusterBaselineConfig::default();
    let labels = parse_u32_sequence(community_labels, "communityLabels")?;
    let n_init_val = opt_usize_with_default(n_init, defaults.kmeans_n_init, "nInit")?;
    let max_iters_val = opt_usize_with_default(max_iters, defaults.kmeans_max_iters, "maxIters")?;
    let seed_u64 = parse_seed_u64(seed)?;
    let soa = layout.inner().to_layout2d();
    let inner =
        cq_kmeans(&soa, &labels, n_init_val, max_iters_val, seed_u64).map_err(core_err_to_js)?;
    Ok(JsPartitionAgreementResult { inner })
}

/// Evaluates average-linkage HAC Cluster Quality (`CQ-HAC`) of `layout` against `communityLabels`.
#[wasm_bindgen(js_name = cqHac)]
pub fn js_cq_hac(
    layout: &JsLayout2d,
    community_labels: &JsValue,
) -> Result<JsPartitionAgreementResult, JsError> {
    let labels = parse_u32_sequence(community_labels, "communityLabels")?;
    let soa = layout.inner().to_layout2d();
    let inner = cq_hac(&soa, &labels).map_err(core_err_to_js)?;
    Ok(JsPartitionAgreementResult { inner })
}

/// Computes Distance Consistency (`DSC`) in `[0.0, 1.0]`.
#[wasm_bindgen(js_name = distanceConsistency)]
pub fn js_distance_consistency(layout: &JsLayout2d, labels: &JsValue) -> Result<f64, JsError> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let soa = layout.inner().to_layout2d();
    distance_consistency(&soa, &parsed_labels).map_err(core_err_to_js)
}

/// Computes Distribution Consistency (`DC`) in `[0.0, 1.0]`.
#[wasm_bindgen(js_name = distributionConsistency)]
pub fn js_distribution_consistency(
    layout: &JsLayout2d,
    labels: &JsValue,
    grid_size: Option<f64>,
) -> Result<f64, JsError> {
    let defaults = ClusterBaselineConfig::default();
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let gs = opt_usize_with_default(grid_size, defaults.dc_grid_size, "gridSize")?;
    let soa = layout.inner().to_layout2d();
    distribution_consistency(&soa, &parsed_labels, gs).map_err(core_err_to_js)
}

/// Computes Class Density Measure (`CDM`) and 2D Histogram Density Measure (`2D-HDM`).
#[wasm_bindgen(js_name = classDensityMeasure)]
pub fn js_class_density_measure(
    layout: &JsLayout2d,
    labels: &JsValue,
    grid_size: Option<f64>,
    bandwidth: Option<f64>,
) -> Result<JsClassDensityResult, JsError> {
    let defaults = ClusterBaselineConfig::default();
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let gs = opt_usize_with_default(grid_size, defaults.cdm_grid_size, "gridSize")?;
    let soa = layout.inner().to_layout2d();
    let (cdm, hdm) =
        class_density_measure(&soa, &parsed_labels, gs, bandwidth).map_err(core_err_to_js)?;
    Ok(JsClassDensityResult { cdm, hdm })
}

/// Computes SepMe $k$-NN graph DIR CPT (`k = 2` default).
#[wasm_bindgen(js_name = sepmeKnng)]
pub fn js_sepme_knng(
    layout: &JsLayout2d,
    labels: &JsValue,
    k: Option<f64>,
) -> Result<f64, JsError> {
    let defaults = ClusterBaselineConfig::default();
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let k_val = opt_usize_with_default(k, defaults.sepme_knng_k, "k")?;
    let soa = layout.inner().to_layout2d();
    sepme_knng(&soa, &parsed_labels, k_val).map_err(core_err_to_js)
}

/// Computes SepMe Relative Neighborhood Graph (`RNG`) DIR CPT.
#[wasm_bindgen(js_name = sepmeRng)]
pub fn js_sepme_rng(layout: &JsLayout2d, labels: &JsValue) -> Result<f64, JsError> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let soa = layout.inner().to_layout2d();
    sepme_rng(&soa, &parsed_labels).map_err(core_err_to_js)
}

/// Computes SepMe Gabriel Graph (`GAB`) DIR CPT.
#[wasm_bindgen(js_name = sepmeGab)]
pub fn js_sepme_gab(layout: &JsLayout2d, labels: &JsValue) -> Result<f64, JsError> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let soa = layout.inner().to_layout2d();
    sepme_gab(&soa, &parsed_labels).map_err(core_err_to_js)
}

/// Computes SepMe $\gamma$-Observable Neighbor Graph (`GONG`) DIR CPT (`gamma = 0.35` default).
#[wasm_bindgen(js_name = sepmeGong)]
pub fn js_sepme_gong(
    layout: &JsLayout2d,
    labels: &JsValue,
    gamma: Option<f64>,
) -> Result<f64, JsError> {
    let defaults = ClusterBaselineConfig::default();
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let g = gamma.unwrap_or(defaults.gong_gamma);
    let soa = layout.inner().to_layout2d();
    sepme_gong(&soa, &parsed_labels, g).map_err(core_err_to_js)
}

/// Computes Neighborhood Hit ($M_{\mathrm{NH}}$, `k = 7` default).
#[wasm_bindgen(js_name = neighborhoodHit)]
pub fn js_neighborhood_hit(
    layout: &JsLayout2d,
    labels: &JsValue,
    k: Option<f64>,
) -> Result<f64, JsError> {
    let defaults = ClusterBaselineConfig::default();
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let k_val = opt_usize_with_default(k, defaults.neighborhood_hit_k, "k")?;
    let soa = layout.inner().to_layout2d();
    neighborhood_hit(&soa, &parsed_labels, k_val).map_err(core_err_to_js)
}

/// Computes the Euclidean Silhouette coefficient in `[-1.0, 1.0]`.
#[wasm_bindgen(js_name = silhouetteScore)]
pub fn js_silhouette_score(layout: &JsLayout2d, labels: &JsValue) -> Result<f64, JsError> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let soa = layout.inner().to_layout2d();
    silhouette_score(&soa, &parsed_labels).map_err(core_err_to_js)
}

/// Computes the Calinski-Harabasz Index.
#[wasm_bindgen(js_name = calinskiHarabaszScore)]
pub fn js_calinski_harabasz_score(layout: &JsLayout2d, labels: &JsValue) -> Result<f64, JsError> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let soa = layout.inner().to_layout2d();
    calinski_harabasz_score(&soa, &parsed_labels).map_err(core_err_to_js)
}

/// Computes the Davies-Bouldin Index.
#[wasm_bindgen(js_name = daviesBouldinScore)]
pub fn js_davies_bouldin_score(layout: &JsLayout2d, labels: &JsValue) -> Result<f64, JsError> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let soa = layout.inner().to_layout2d();
    davies_bouldin_score(&soa, &parsed_labels).map_err(core_err_to_js)
}

/// Computes Pairwise Distance-Overlap Error (`PDOE`, `numBins = 100` default).
#[wasm_bindgen(js_name = pairwiseDistanceOverlapError)]
pub fn js_pairwise_distance_overlap_error(
    layout: &JsLayout2d,
    labels: &JsValue,
    num_bins: Option<f64>,
) -> Result<f64, JsError> {
    let defaults = ClusterBaselineConfig::default();
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let bins = opt_usize_with_default(num_bins, defaults.num_bins, "numBins")?;
    let soa = layout.inner().to_layout2d();
    pairwise_distance_overlap_error(&soa, &parsed_labels, bins).map_err(core_err_to_js)
}

/// Computes Sprawlter-NN composite overlap clutter measure from per-node `halfWidths` and `halfHeights`.
#[wasm_bindgen(js_name = sprawlterNn)]
pub fn js_sprawlter_nn(
    layout: &JsLayout2d,
    half_widths: &[f32],
    half_heights: &[f32],
) -> Result<f64, JsError> {
    let soa = layout.inner().to_layout2d();
    sprawlter_nn(&soa, half_widths, half_heights).map_err(core_err_to_js)
}

/// Computes Sprawlter-NN composite overlap clutter measure from uniform square `radii`.
#[wasm_bindgen(js_name = sprawlterNnRadii)]
pub fn js_sprawlter_nn_radii(layout: &JsLayout2d, radii: &[f32]) -> Result<f64, JsError> {
    let soa = layout.inner().to_layout2d();
    sprawlter_nn_radii(&soa, radii).map_err(core_err_to_js)
}

/// Computes all 5 Chen et al. (2020) NOR metrics including graph edge-length distortion (`elRsd`).
#[wasm_bindgen(js_name = norMetrics)]
pub fn js_nor_metrics(
    graph: &JsCsrGraph,
    orig_layout: &JsLayout2d,
    adjusted_layout: &JsLayout2d,
    half_widths: &[f32],
    half_heights: &[f32],
) -> Result<JsNorMetricsResult, JsError> {
    let orig_soa = orig_layout.inner().to_layout2d();
    let adj_soa = adjusted_layout.inner().to_layout2d();
    let inner = nor_metrics(
        graph.inner(),
        &orig_soa,
        &adj_soa,
        half_widths,
        half_heights,
    )
    .map_err(core_err_to_js)?;
    Ok(JsNorMetricsResult { inner })
}

/// Computes Chen et al. (2020) NOR metrics without a graph (`elRsd = 0.0`).
#[wasm_bindgen(js_name = norMetricsWithoutGraph)]
pub fn js_nor_metrics_without_graph(
    orig_layout: &JsLayout2d,
    adjusted_layout: &JsLayout2d,
    half_widths: &[f32],
    half_heights: &[f32],
) -> Result<JsNorMetricsResult, JsError> {
    let orig_soa = orig_layout.inner().to_layout2d();
    let adj_soa = adjusted_layout.inner().to_layout2d();
    let inner = nor_metrics_without_graph(&orig_soa, &adj_soa, half_widths, half_heights)
        .map_err(core_err_to_js)?;
    Ok(JsNorMetricsResult { inner })
}
