//! WebAssembly bindings for cluster-faithful layout refinement, squarified treemaps,
//! and Group-in-a-Box (`TR-GIB` / `FD-GIB`) solvers (`f32`).

use egraph::core::IterativeSolver;
use egraph::layout::{
    ClusterRefineConfig, ClusterRefineMethod, ClusterRefineSolver, GibBoxLayoutMode,
    GibIntraLayoutMode, GroupInABoxConfig, GroupInABoxSolver, refine_cluster_layout,
    squarified_treemap,
};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{
    core_err_to_js, ffi_err_to_js, parse_f64_sequence, parse_optional_f64_sequence,
    parse_u32_sequence, set_f64_prop,
};
use crate::js_options::{
    check_options_arg, get_prop, normalize_enum_token, opt_f64_prop, opt_seed_prop,
    opt_string_prop, opt_usize_prop, solver_status_str,
};
use crate::layout_buffers::JsLayout2d;
use crate::solver_runner::{run_layout_in_place, run_steps_layout_in_place, step_layout_in_place};

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
