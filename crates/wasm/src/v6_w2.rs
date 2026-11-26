//! W2 v6 WebAssembly bindings for HDE, Ortmann Sparse Stress, Maxent-Stress,
//! LargeVis, DRGraph, Graph-CNE, GRIP, GEM, SEL, PFS', and canonical community-ordered
//! SFC layout baselines (`f32`-backed layouts with `f64` internal solver math; zero
//! `serde` / `getrandom` / threads).

use egraph::core::IterativeSolver;
use egraph::layout::{
    DrGraphConfig, GemConfig, GemSolver, GraphCneConfig, GraphCneObjective, GripConfig, GripSolver,
    LargeVisConfig, MaxentStressConfig, MaxentStressSolver, OrtmannPivotStrategy,
    OrtmannSparseStressConfig, PfsPrimeConfig, PfsPrimeOverlapRemoval, SelConfig, SelLayoutBackend,
    SelLayoutSolver, SelWeightMapping, SfcConfig, SfcCurveKind, SfcLayoutSolver, SgdSolver,
};
use egraph::linalg::{HdeConfig, hde, hde_nd, hde_nd_with_pivots, hde_with_pivots};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, parse_non_negative_u32, parse_u32_sequence};
use crate::js_options::{
    check_options_arg, get_prop, normalize_enum_token, opt_bool_prop, opt_f64_prop,
    opt_maybe_f64_prop, opt_maybe_usize_prop, opt_seed_prop, opt_string_prop, opt_usize_prop,
    solver_status_str,
};
use crate::layout_buffers::{JsLayout2d, JsLayoutNd};
use crate::v6_m3::{
    JsSgdSolver, run_layout_in_place, run_steps_layout_in_place, step_layout_in_place,
};

fn opt_maybe_u32_prop(obj: &JsValue, key: &str) -> Result<Option<u32>, JsError> {
    let val = get_prop(obj, key)?;
    if val.is_undefined() || val.is_null() {
        return Ok(None);
    }
    let num = val
        .as_f64()
        .ok_or_else(|| JsError::new(&format!("options.{key} must be an integer number")))?;
    parse_non_negative_u32(num, key).map(Some)
}

// ============================================================================
// W2-1a. Harel-Koren High-Dimensional Embedding (HDE)
// ============================================================================

fn parse_hde_config(
    options: &JsValue,
    is_nd: bool,
    ctx: &str,
) -> Result<(HdeConfig, Option<Vec<u32>>), JsError> {
    let mut cfg = HdeConfig::default();
    let mut pivots = None;
    if check_options_arg(options, ctx)? {
        cfg.dimensions = opt_usize_prop(options, "dimensions", cfg.dimensions)?;
        if is_nd {
            cfg.output_dims = opt_usize_prop(options, "outputDims", cfg.output_dims)?;
        }
        cfg.seed = opt_seed_prop(options)?;
        cfg.tol = opt_f64_prop(options, "tol", cfg.tol)?;
        cfg.max_sweeps = opt_usize_prop(options, "maxSweeps", cfg.max_sweeps)?;
        cfg.normalize = opt_bool_prop(options, "normalize", cfg.normalize)?;
        let piv_val = get_prop(options, "pivots")?;
        if !piv_val.is_undefined() && !piv_val.is_null() {
            pivots = Some(parse_u32_sequence(&piv_val, "pivots")?);
        }
    }
    Ok((cfg, pivots))
}

/// Computes a 2D Harel-Koren High-Dimensional Embedding (`HDE`) `Layout2d` (`f32`).
#[wasm_bindgen(js_name = hde)]
pub fn js_hde(graph: &JsCsrGraph, options: &JsValue) -> Result<JsLayout2d, JsError> {
    let (cfg, pivots) = parse_hde_config(options, false, "hde")?;
    let layout = match pivots {
        Some(ref pivs) => hde_with_pivots::<f32, f32>(graph.inner(), pivs, &cfg),
        None => hde::<f32, f32>(graph.inner(), &cfg),
    }
    .map_err(core_err_to_js)?;
    Ok(JsLayout2d::from_layout2d(&layout))
}

/// Computes an $N$-dimensional Harel-Koren High-Dimensional Embedding (`HDE`) `LayoutNd` (`f32`).
#[wasm_bindgen(js_name = hdeNd)]
pub fn js_hde_nd(graph: &JsCsrGraph, options: &JsValue) -> Result<JsLayoutNd, JsError> {
    let (cfg, pivots) = parse_hde_config(options, true, "hdeNd")?;
    let layout = match pivots {
        Some(ref pivs) => hde_nd_with_pivots::<f32, f32>(graph.inner(), pivs, &cfg),
        None => hde_nd::<f32, f32>(graph.inner(), &cfg),
    }
    .map_err(core_err_to_js)?;
    Ok(JsLayoutNd::from_inner(layout))
}

// ============================================================================
// W2-1b & W2-3. SgdSolver Constructors (Ortmann, LargeVis, DRGraph, Graph-CNE)
// ============================================================================

fn parse_ortmann_pivot_strategy(raw: &str) -> Result<OrtmannPivotStrategy, JsError> {
    match normalize_enum_token(raw).as_str() {
        "kmeanssp" => Ok(OrtmannPivotStrategy::KMeansSp),
        "maxminrandomsp" => Ok(OrtmannPivotStrategy::MaxMinRandomSp),
        "maxminsp" => Ok(OrtmannPivotStrategy::MaxMinSp),
        _ => Err(JsError::new(&format!(
            "unknown OrtmannPivotStrategy {raw:?}; expected 'kmeans_sp', 'maxmin_random_sp', or 'maxmin_sp'"
        ))),
    }
}

pub(crate) fn sgd_ortmann_sparse_stress(
    graph: &JsCsrGraph,
    options: &JsValue,
) -> Result<JsSgdSolver, JsError> {
    let mut cfg = OrtmannSparseStressConfig::default();
    if check_options_arg(options, "SgdSolver.ortmannSparseStress")? {
        cfg.num_pivots = opt_usize_prop(options, "numPivots", cfg.num_pivots)?;
        if let Some(strat_str) = opt_string_prop(options, "pivotStrategy")? {
            cfg.pivot_strategy = parse_ortmann_pivot_strategy(&strat_str)?;
        }
        cfg.iterations = opt_usize_prop(options, "iterations", cfg.iterations)?;
        cfg.eta_max = opt_maybe_f64_prop(options, "etaMax")?;
        cfg.eta_min = opt_maybe_f64_prop(options, "etaMin")?;
        cfg.eps = opt_f64_prop(options, "eps", cfg.eps)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    let inner =
        SgdSolver::<f32>::ortmann_sparse_stress(graph.inner(), cfg).map_err(core_err_to_js)?;
    Ok(JsSgdSolver::from_inner(inner))
}

pub(crate) fn sgd_largevis(graph: &JsCsrGraph, options: &JsValue) -> Result<JsSgdSolver, JsError> {
    let mut cfg = LargeVisConfig::default();
    if check_options_arg(options, "SgdSolver.largevis")? {
        cfg.iterations = opt_usize_prop(options, "iterations", cfg.iterations)?;
        cfg.negative_samples = opt_usize_prop(options, "negativeSamples", cfg.negative_samples)?;
        cfg.gamma = opt_f64_prop(options, "gamma", cfg.gamma)?;
        cfg.initial_lr = opt_f64_prop(options, "initialLr", cfg.initial_lr)?;
        cfg.min_lr = opt_f64_prop(options, "minLr", cfg.min_lr)?;
        cfg.degree_exponent = opt_f64_prop(options, "degreeExponent", cfg.degree_exponent)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    let inner = SgdSolver::<f32>::largevis(graph.inner(), cfg).map_err(core_err_to_js)?;
    Ok(JsSgdSolver::from_inner(inner))
}

pub(crate) fn sgd_drgraph(graph: &JsCsrGraph, options: &JsValue) -> Result<JsSgdSolver, JsError> {
    let mut cfg = DrGraphConfig::default();
    if check_options_arg(options, "SgdSolver.drgraph")? {
        cfg.k_order = opt_usize_prop(options, "kOrder", cfg.k_order)?;
        cfg.negative_samples = opt_usize_prop(options, "negativeSamples", cfg.negative_samples)?;
        cfg.b = opt_f64_prop(options, "b", cfg.b)?;
        cfg.gamma_coarse = opt_f64_prop(options, "gammaCoarse", cfg.gamma_coarse)?;
        cfg.gamma_fine = opt_f64_prop(options, "gammaFine", cfg.gamma_fine)?;
        cfg.coarsen_stop_ratio = opt_f64_prop(options, "coarsenStopRatio", cfg.coarsen_stop_ratio)?;
        cfg.iterations = opt_usize_prop(options, "iterations", cfg.iterations)?;
        cfg.initial_lr = opt_f64_prop(options, "initialLr", cfg.initial_lr)?;
        cfg.min_lr = opt_f64_prop(options, "minLr", cfg.min_lr)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    let inner = SgdSolver::<f32>::drgraph(graph.inner(), cfg).map_err(core_err_to_js)?;
    Ok(JsSgdSolver::from_inner(inner))
}

fn parse_graph_cne_objective(raw: &str) -> Result<GraphCneObjective, JsError> {
    match normalize_enum_token(raw).as_str() {
        "infonce" => Ok(GraphCneObjective::InfoNce),
        "negtsne" => Ok(GraphCneObjective::NegTSne),
        _ => Err(JsError::new(&format!(
            "unknown GraphCneObjective {raw:?}; expected 'infonce' or 'neg_tsne'"
        ))),
    }
}

pub(crate) fn sgd_graph_cne(graph: &JsCsrGraph, options: &JsValue) -> Result<JsSgdSolver, JsError> {
    let mut cfg = GraphCneConfig::default();
    if check_options_arg(options, "SgdSolver.graphCne")? {
        if let Some(obj_str) = opt_string_prop(options, "objective")? {
            cfg.objective = parse_graph_cne_objective(&obj_str)?;
        }
        cfg.negative_samples = opt_usize_prop(options, "negativeSamples", cfg.negative_samples)?;
        cfg.iterations = opt_usize_prop(options, "iterations", cfg.iterations)?;
        cfg.initial_lr = opt_f64_prop(options, "initialLr", cfg.initial_lr)?;
        cfg.min_lr = opt_f64_prop(options, "minLr", cfg.min_lr)?;
        cfg.repulsion_weight = opt_f64_prop(options, "repulsionWeight", cfg.repulsion_weight)?;
        cfg.row_normalize = opt_bool_prop(options, "rowNormalize", cfg.row_normalize)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    let inner = SgdSolver::<f32>::graph_cne(graph.inner(), cfg).map_err(core_err_to_js)?;
    Ok(JsSgdSolver::from_inner(inner))
}

// ============================================================================
// W2-2. MaxentStressSolver
// ============================================================================

/// Cooperative Gansner et al. (2013) Maxent-Stress layout solver (`f32`).
#[wasm_bindgen(js_name = MaxentStressSolver)]
#[derive(Debug, Clone)]
pub struct JsMaxentStressSolver {
    inner: MaxentStressSolver<f32>,
}

#[wasm_bindgen(js_class = MaxentStressSolver)]
impl JsMaxentStressSolver {
    /// Constructs a `MaxentStressSolver` from `graph` and optional `MaxentStressOptions`.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsMaxentStressSolver, JsError> {
        let mut cfg = MaxentStressConfig::default();
        if check_options_arg(options, "MaxentStressSolver")? {
            cfg.q = opt_maybe_f64_prop(options, "q")?;
            cfg.alpha_init = opt_f64_prop(options, "alphaInit", cfg.alpha_init)?;
            cfg.alpha_decay = opt_f64_prop(options, "alphaDecay", cfg.alpha_decay)?;
            cfg.alpha_min = opt_f64_prop(options, "alphaMin", cfg.alpha_min)?;
            cfg.stages = opt_usize_prop(options, "stages", cfg.stages)?;
            cfg.steps_per_stage = opt_usize_prop(options, "stepsPerStage", cfg.steps_per_stage)?;
            cfg.tol = opt_f64_prop(options, "tol", cfg.tol)?;
            cfg.cg_tol = opt_f64_prop(options, "cgTol", cfg.cg_tol)?;
            cfg.cg_max_iters = opt_usize_prop(options, "cgMaxIters", cfg.cg_max_iters)?;
            cfg.theta = opt_f64_prop(options, "theta", cfg.theta)?;
            cfg.use_barnes_hut = opt_bool_prop(options, "useBarnesHut", cfg.use_barnes_hut)?;
            cfg.eps_sq = opt_f64_prop(options, "epsSq", cfg.eps_sq)?;
            cfg.seed = opt_seed_prop(options)?;
        }
        let inner = MaxentStressSolver::<f32>::new(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one Maxent-Stress iteration on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` Maxent-Stress iterations on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "MaxentStressSolver::run_steps",
        )
    }

    /// Runs Maxent-Stress optimization to completion on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "MaxentStressSolver::run",
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

    /// Returns the scheduled iteration count (`stages * stepsPerStage`).
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

    /// Returns the current annealed entropy-repulsion weight $\alpha$.
    #[wasm_bindgen(getter, js_name = currentAlpha)]
    pub fn current_alpha(&self) -> f64 {
        self.inner.current_alpha()
    }

    /// Returns the resolved repulsion exponent $q$.
    #[wasm_bindgen(getter, js_name = effectiveQ)]
    pub fn effective_q(&self) -> f64 {
        self.inner.effective_q()
    }
}

// ============================================================================
// W2-4. GRIP, GEM, SEL, PFS', and Canonical SFC Layout Solvers
// ============================================================================

/// Cooperative Gajer & Kobourov (2002) GRIP multilevel force-directed layout solver (`f32`).
#[wasm_bindgen(js_name = GripSolver)]
#[derive(Debug, Clone)]
pub struct JsGripSolver {
    inner: GripSolver<f32>,
}

#[wasm_bindgen(js_class = GripSolver)]
impl JsGripSolver {
    /// Constructs a `GripSolver` from `graph` and optional `GripOptions`.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsGripSolver, JsError> {
        let mut cfg = GripConfig::default();
        if check_options_arg(options, "GripSolver")? {
            cfg.rounds_min = opt_usize_prop(options, "roundsMin", cfg.rounds_min)?;
            cfg.rounds_max = opt_usize_prop(options, "roundsMax", cfg.rounds_max)?;
            cfg.local_neighborhood_size = opt_usize_prop(
                options,
                "localNeighborhoodSize",
                cfg.local_neighborhood_size,
            )?;
            cfg.repulsion_scale = opt_f64_prop(options, "repulsionScale", cfg.repulsion_scale)?;
            cfg.edge_length = opt_f64_prop(options, "edgeLength", cfg.edge_length)?;
            cfg.temp_r = opt_f64_prop(options, "tempR", cfg.temp_r)?;
            cfg.temp_s = opt_f64_prop(options, "tempS", cfg.temp_s)?;
            cfg.seed = opt_seed_prop(options)?;
        }
        let inner = GripSolver::<f32>::new(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one GRIP refinement step on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` GRIP refinement steps on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "GripSolver::run_steps",
        )
    }

    /// Runs GRIP layout optimization to completion on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "GripSolver::run",
        )
    }

    /// Returns the solver lifecycle status string.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        solver_status_str(self.inner.status()).to_owned()
    }

    /// Returns the number of steps executed so far.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    /// Returns the total scheduled step count across filtration levels.
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

    /// Returns the maximum number of MIS filtration levels across connected components.
    #[wasm_bindgen(getter, js_name = numLevels)]
    pub fn num_levels(&self) -> usize {
        self.inner.num_levels()
    }
}

/// Cooperative Frick et al. (1995) GEM adaptive local-temperature layout solver (`f32`).
#[wasm_bindgen(js_name = GemSolver)]
#[derive(Debug, Clone)]
pub struct JsGemSolver {
    inner: GemSolver<f32>,
}

#[wasm_bindgen(js_class = GemSolver)]
impl JsGemSolver {
    /// Constructs a `GemSolver` from `graph` and optional `GemOptions`.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsGemSolver, JsError> {
        let mut cfg = GemConfig::default();
        if check_options_arg(options, "GemSolver")? {
            cfg.edge_length = opt_f64_prop(options, "edgeLength", cfg.edge_length)?;
            cfg.gravity = opt_f64_prop(options, "gravity", cfg.gravity)?;
            cfg.t_init = opt_f64_prop(options, "tInit", cfg.t_init)?;
            cfg.t_max = opt_f64_prop(options, "tMax", cfg.t_max)?;
            cfg.t_min = opt_f64_prop(options, "tMin", cfg.t_min)?;
            cfg.max_rounds = opt_maybe_usize_prop(options, "maxRounds")?;
            cfg.oscillation_angle =
                opt_f64_prop(options, "oscillationAngle", cfg.oscillation_angle)?;
            cfg.rotation_angle = opt_f64_prop(options, "rotationAngle", cfg.rotation_angle)?;
            cfg.oscillation_sensitivity = opt_f64_prop(
                options,
                "oscillationSensitivity",
                cfg.oscillation_sensitivity,
            )?;
            cfg.rotation_sensitivity = opt_maybe_f64_prop(options, "rotationSensitivity")?;
            cfg.perturbation = opt_f64_prop(options, "perturbation", cfg.perturbation)?;
            cfg.seed = opt_seed_prop(options)?;
        }
        let inner = GemSolver::<f32>::new(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one GEM round on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` GEM rounds on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "GemSolver::run_steps",
        )
    }

    /// Runs GEM layout optimization to completion on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "GemSolver::run",
        )
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

fn parse_sel_weight_mapping(raw: &str) -> Result<SelWeightMapping, JsError> {
    match normalize_enum_token(raw).as_str() {
        "sigmoid" => Ok(SelWeightMapping::Sigmoid),
        "linear" => Ok(SelWeightMapping::Linear),
        "tanh" => Ok(SelWeightMapping::Tanh),
        "customlogistic" => Ok(SelWeightMapping::CustomLogistic),
        "binary" => Ok(SelWeightMapping::Binary),
        _ => Err(JsError::new(&format!(
            "unknown SelWeightMapping {raw:?}; expected 'sigmoid', 'linear', 'tanh', 'custom_logistic', or 'binary'"
        ))),
    }
}

fn parse_sel_layout_backend(raw: &str) -> Result<SelLayoutBackend, JsError> {
    match normalize_enum_token(raw).as_str() {
        "fr" | "fruchtermanreingold" => Ok(SelLayoutBackend::Fr),
        "tfdp" => Ok(SelLayoutBackend::TFdp),
        _ => Err(JsError::new(&format!(
            "unknown SelLayoutBackend {raw:?}; expected 'fr' or 't_fdp'"
        ))),
    }
}

/// Cooperative Xue et al. (2025) Structural Entropy Layout (`SEL`) solver (`f32`).
#[wasm_bindgen(js_name = SelLayoutSolver)]
#[derive(Debug, Clone)]
pub struct JsSelLayoutSolver {
    inner: SelLayoutSolver<f32>,
}

#[wasm_bindgen(js_class = SelLayoutSolver)]
impl JsSelLayoutSolver {
    /// Constructs a `SelLayoutSolver` from `graph` and optional `SelOptions`.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsSelLayoutSolver, JsError> {
        let mut cfg = SelConfig::default();
        if check_options_arg(options, "SelLayoutSolver")? {
            cfg.tree_depth = opt_usize_prop(options, "treeDepth", cfg.tree_depth)?;
            cfg.pq_max_pops = opt_usize_prop(options, "pqMaxPops", cfg.pq_max_pops)?;
            if let Some(m_str) = opt_string_prop(options, "mapping")? {
                cfg.mapping = parse_sel_weight_mapping(&m_str)?;
            }
            if let Some(b_str) = opt_string_prop(options, "layoutBackend")? {
                cfg.layout_backend = parse_sel_layout_backend(&b_str)?;
            }
            cfg.iterations = opt_usize_prop(options, "iterations", cfg.iterations)?;
            cfg.seed = opt_seed_prop(options)?;
        }
        let inner = SelLayoutSolver::<f32>::new(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one SEL layout iteration on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` SEL layout iterations on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "SelLayoutSolver::run_steps",
        )
    }

    /// Runs SEL layout optimization to completion on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "SelLayoutSolver::run",
        )
    }

    /// Returns the SEL-modulated edge weights aligned with canonical undirected edges `(u < v)` (`Float64Array`).
    #[wasm_bindgen(js_name = modulatedWeights)]
    pub fn modulated_weights(&self) -> Vec<f64> {
        self.inner.modulated_weights().to_vec()
    }

    /// Returns the elbow-selected structural entropy threshold $\mathrm{SE}_{\mathrm{target}}$.
    #[wasm_bindgen(getter, js_name = seTarget)]
    pub fn se_target(&self) -> f64 {
        self.inner.se_target()
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

fn parse_pfs_prime_config(options: &JsValue, ctx: &str) -> Result<PfsPrimeConfig, JsError> {
    let mut cfg = PfsPrimeConfig::default();
    if check_options_arg(options, ctx)? {
        cfg.padding = opt_f64_prop(options, "padding", cfg.padding)?;
        cfg.seed = opt_seed_prop(options)?;
    }
    Ok(cfg)
}

/// Cooperative Hayashi et al. (1998) `PFS'` two-pass scan-line node overlap removal solver (`f32`).
#[wasm_bindgen(js_name = PfsPrimeOverlapRemoval)]
#[derive(Debug, Clone)]
pub struct JsPfsPrimeOverlapRemoval {
    inner: PfsPrimeOverlapRemoval<f32>,
}

#[wasm_bindgen(js_class = PfsPrimeOverlapRemoval)]
impl JsPfsPrimeOverlapRemoval {
    /// Constructs a `PfsPrimeOverlapRemoval` solver from per-node `halfWidths` and `halfHeights`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        half_widths: &[f32],
        half_heights: &[f32],
        options: &JsValue,
    ) -> Result<JsPfsPrimeOverlapRemoval, JsError> {
        let cfg = parse_pfs_prime_config(options, "PfsPrimeOverlapRemoval")?;
        let inner =
            PfsPrimeOverlapRemoval::<f32>::new(half_widths.to_vec(), half_heights.to_vec(), cfg)
                .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Constructs a `PfsPrimeOverlapRemoval` solver with uniform square `radii`.
    #[wasm_bindgen(js_name = fromRadii)]
    pub fn from_radii(
        radii: &[f32],
        options: &JsValue,
    ) -> Result<JsPfsPrimeOverlapRemoval, JsError> {
        let cfg = parse_pfs_prime_config(options, "PfsPrimeOverlapRemoval.fromRadii")?;
        let inner = PfsPrimeOverlapRemoval::<f32>::from_radii(radii.to_vec(), cfg)
            .map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes the single two-pass `PFS'` overlap removal step on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` `PFS'` passes on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "PfsPrimeOverlapRemoval::run_steps",
        )
    }

    /// Runs `PFS'` overlap removal to completion on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "PfsPrimeOverlapRemoval::run",
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

    /// Returns the scheduled pass count (`1`).
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

fn parse_sfc_curve_kind(raw: &str) -> Result<SfcCurveKind, JsError> {
    match normalize_enum_token(raw).as_str() {
        "hilbert" => Ok(SfcCurveKind::Hilbert),
        "morton" | "zorder" => Ok(SfcCurveKind::Morton),
        _ => Err(JsError::new(&format!(
            "unknown SfcCurveKind {raw:?}; expected 'hilbert' or 'morton'"
        ))),
    }
}

/// Canonical community-ordered Space-Filling Curve (`SFC`) layout solver (`f32`).
///
/// Note: `muelder2008_space_filling` (Muelder & Ma, IEEE TVCG 2008) is currently marked
/// `MISSING_PDF` in the project literature archive, so exact clustering, ordering, and
/// post-refinement parameters from that paper cannot be verified or claimed. This solver
/// exposes a canonical community-ordered Hilbert/Morton space-filling curve baseline using
/// `louvain_coarsen` and strictly injective 2D curve mapping (`totalIterations == 1`).
#[wasm_bindgen(js_name = SfcLayoutSolver)]
#[derive(Debug, Clone)]
pub struct JsSfcLayoutSolver {
    inner: SfcLayoutSolver<f32>,
}

#[wasm_bindgen(js_class = SfcLayoutSolver)]
impl JsSfcLayoutSolver {
    /// Constructs an `SfcLayoutSolver` from `graph` and optional `SfcOptions`.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsSfcLayoutSolver, JsError> {
        let mut cfg = SfcConfig::default();
        if check_options_arg(options, "SfcLayoutSolver")? {
            if let Some(c_str) = opt_string_prop(options, "curve")? {
                cfg.curve = parse_sfc_curve_kind(&c_str)?;
            }
            cfg.scale = opt_f64_prop(options, "scale", cfg.scale)?;
            cfg.resolution_order = opt_maybe_u32_prop(options, "resolutionOrder")?;
            cfg.seed = opt_seed_prop(options)?;
        }
        let inner = SfcLayoutSolver::<f32>::new(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes the single `SFC` coordinate assignment step on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` `SFC` steps on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "SfcLayoutSolver::run_steps",
        )
    }

    /// Runs `SFC` coordinate assignment to completion on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "SfcLayoutSolver::run",
        )
    }

    /// Returns the deterministic community-ordered vertex permutation (`Uint32Array`).
    #[wasm_bindgen(js_name = vertexOrder)]
    pub fn vertex_order(&self) -> Vec<u32> {
        self.inner.vertex_order().to_vec()
    }

    /// Returns the solver lifecycle status string.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        solver_status_str(self.inner.status()).to_owned()
    }

    /// Returns the number of steps executed so far.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    /// Returns the scheduled step count (`1`).
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
