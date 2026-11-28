//! WebAssembly bindings for cluster-faithful layout refinement, squarified treemaps,
//! and Group-in-a-Box (`TR-GIB` / `FD-GIB`) solvers (`f32`).

use egraph::core::{CsrGraph, IterativeSolver};
use egraph::layout::{
    ClusterRefineConfig, ClusterRefineMethod, ClusterRefineSolver, GibBoxLayoutMode,
    GibIntraLayoutMode, GroupInABoxConfig, GroupInABoxSolver, refine_cluster_layout,
    squarified_treemap,
};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{
    core_err_to_js, ffi_err_to_js, invalid_input_err, parse_f64_sequence,
    parse_optional_f64_sequence, parse_u32_sequence, set_f64_prop,
};
use crate::js_options::{
    check_options_arg, get_prop, normalize_enum_token, opt_bool_prop, opt_f64_prop, opt_seed_prop,
    opt_string_prop, opt_usize_prop, solver_status_str,
};
use crate::layout_buffers::JsLayout2d;
use crate::solver_runner::{run_layout_in_place, run_steps_layout_in_place, step_layout_in_place};

fn parse_cluster_refine_method(raw: &str) -> Result<ClusterRefineMethod, JsValue> {
    match normalize_enum_token(raw)?.as_str() {
        "kmeans" | "clusterkmeans" => Ok(ClusterRefineMethod::Kmeans),
        "hac" | "clusterhac" => Ok(ClusterRefineMethod::Hac),
        _ => Err(invalid_input_err(format!(
            "unknown ClusterRefineMethod {raw:?}; expected 'kmeans' or 'hac'"
        ))),
    }
}

fn parse_cluster_refine_config(
    options: &JsValue,
    ctx: &str,
) -> Result<ClusterRefineConfig, JsValue> {
    let mut cfg = ClusterRefineConfig::default();
    if check_options_arg(options, ctx)? {
        if let Some(m_str) = opt_string_prop(options, "method")? {
            cfg.method = parse_cluster_refine_method(&m_str)?;
        }
        cfg.max_iterations = opt_usize_prop(options, "maxIterations", cfg.max_iterations)?;
        cfg.kmeans_n_init = opt_usize_prop(options, "kmeansNInit", cfg.kmeans_n_init)?;
        cfg.kmeans_max_iters =
            opt_usize_prop(options, "kmeansMaxIterations", cfg.kmeans_max_iters)?;
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
    /// Constructs a `ClusterRefineSolver` from `labels` without graph edge lengths.
    #[wasm_bindgen(constructor)]
    pub fn new(labels: &JsValue, options: &JsValue) -> Result<JsClusterRefineSolver, JsValue> {
        let parsed_labels = parse_u32_sequence(labels, "labels")?;
        let cfg = parse_cluster_refine_config(options, "ClusterRefineSolver")?;
        let inner = ClusterRefineSolver::<f32>::new(&parsed_labels, cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Constructs a `ClusterRefineSolver` with graph topology for `hac` mean edge-length scaling.
    #[wasm_bindgen(js_name = fromGraph)]
    pub fn from_graph(
        graph: &JsCsrGraph,
        labels: &JsValue,
        options: &JsValue,
    ) -> Result<JsClusterRefineSolver, JsValue> {
        let parsed_labels = parse_u32_sequence(labels, "labels")?;
        let cfg = parse_cluster_refine_config(options, "ClusterRefineSolver.fromGraph")?;
        let inner = ClusterRefineSolver::<f32>::from_graph(graph.inner(), &parsed_labels, cfg)
            .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one refinement round on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` refinement rounds on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsValue> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "ClusterRefineSolver::run_steps",
        )
    }

    /// Runs cluster refinement to completion on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
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

    /// Returns the maximum scheduled iterations (`maxIterations`).
    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> Option<usize> {
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
    labels: &JsValue,
    options: &JsValue,
) -> Result<usize, JsValue> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let cfg = parse_cluster_refine_config(options, "refineClusterLayout")?;
    let mut soa = layout.inner().to_layout2d();
    let moved = refine_cluster_layout(&mut soa, &parsed_labels, None::<&CsrGraph<f32>>, &cfg)
        .map_err(core_err_to_js)?;
    layout
        .inner_mut()
        .copy_from_layout2d(&soa)
        .map_err(ffi_err_to_js)?;
    Ok(moved)
}

/// Computes a Squarified Treemap partition returning flat `[xMin0, xMax0, yMin0, yMax0, ...]` (`Float64Array`).
#[wasm_bindgen(js_name = squarifiedTreemap)]
pub fn js_squarified_treemap(weights: &JsValue, options: &JsValue) -> Result<Vec<f64>, JsValue> {
    let w = parse_f64_sequence(weights, "weights")?;
    let mut x = 0.0;
    let mut y = 0.0;
    let mut width = 100.0;
    let mut height = 100.0;
    if check_options_arg(options, "squarifiedTreemap")? {
        x = opt_f64_prop(options, "x", x)?;
        y = opt_f64_prop(options, "y", y)?;
        width = opt_f64_prop(options, "width", width)?;
        height = opt_f64_prop(options, "height", height)?;
    }
    let rects = squarified_treemap(&w, x, y, width, height).map_err(core_err_to_js)?;
    let mut flat = Vec::with_capacity(rects.len() * 4);
    for r in rects {
        flat.extend_from_slice(&r);
    }
    Ok(flat)
}

fn parse_gib_box_mode(raw: &str) -> Result<GibBoxLayoutMode, JsValue> {
    match normalize_enum_token(raw)?.as_str() {
        "treemap" | "squarifiedtreemap" | "trgib" => Ok(GibBoxLayoutMode::SquarifiedTreemap),
        "forcedirected" | "fd" | "fdgib" => Ok(GibBoxLayoutMode::ForceDirected),
        _ => Err(invalid_input_err(format!(
            "unknown GibBoxLayoutMode {raw:?}; expected 'treemap' or 'forceDirected'"
        ))),
    }
}

fn parse_gib_intra_mode(raw: &str) -> Result<GibIntraLayoutMode, JsValue> {
    match normalize_enum_token(raw)?.as_str() {
        "fr" | "fruchtermanreingold" => Ok(GibIntraLayoutMode::Fr),
        "sgdfull" | "sgd" => Ok(GibIntraLayoutMode::SgdFull),
        "yifanhu" => Ok(GibIntraLayoutMode::YifanHu),
        _ => Err(invalid_input_err(format!(
            "unknown GibIntraLayoutMode {raw:?}; expected 'fr', 'sgdFull', or 'yifanHu'"
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
    /// Constructs a `GroupInABoxSolver` from `graph`, `labels`, and optional `GroupInABoxOptions`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        graph: &JsCsrGraph,
        labels: &JsValue,
        options: &JsValue,
    ) -> Result<JsGroupInABoxSolver, JsValue> {
        let parsed_labels = parse_u32_sequence(labels, "labels")?;
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
            cfg.fit_bounds = opt_bool_prop(options, "fitBounds", cfg.fit_bounds)?;
            cluster_weights = parse_optional_f64_sequence(
                &get_prop(options, "clusterWeights")?,
                "clusterWeights",
            )?;
            cfg.seed = opt_seed_prop(options)?;
        }
        let inner = match cluster_weights {
            Some(ref cw) => GroupInABoxSolver::<f32>::with_cluster_weights(
                graph.inner(),
                &parsed_labels,
                cw,
                cfg,
            ),
            None => GroupInABoxSolver::<f32>::new(graph.inner(), &parsed_labels, cfg),
        }
        .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Returns the computed cluster boxes ordered canonically by first appearance.
    pub fn boxes(&self) -> Result<js_sys::Array, JsValue> {
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
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` intra-box layout iterations on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsValue> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "GroupInABoxSolver::run_steps",
        )
    }

    /// Runs Group-in-a-Box intra-cluster optimization to completion on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
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
    pub fn total_iterations(&self) -> Option<usize> {
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
