//! Cooperative SoA `u32` `SgdSolver` (`f32`) bindings supporting `Layout2d`, `Layout3d`,
//! `LayoutNd`, and `ManifoldLayout2d`.

use egraph::core::{Layout2d, Layout3d, LayoutNd, ManifoldKind, ManifoldLayout2d};
use egraph::layout::{SgdScheduleConfig, SgdScheduleKind, SgdSolver, SparseBarrierStats};
use egraph::neighborhood::DogVaradhanStats;
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, invalid_input_err, unsupported_err};
use crate::js_options::{
    JsTargetKind, check_options_arg, normalize_enum_token, opt_f64_prop, opt_maybe_f64_prop,
    opt_seed_prop, opt_string_prop, opt_target_prop, opt_usize_prop, solver_status_str,
};
use crate::layout_buffers::{JsLayout2d, JsLayoutNd};
use crate::layout3d::JsLayout3d;
use crate::manifold::JsManifoldLayout2d;
use crate::solver_runner::{
    run_layout_in_place, run_layout3d_in_place, run_manifold_in_place, run_nd_in_place,
    run_steps_layout_in_place, run_steps_layout3d_in_place, run_steps_manifold_in_place,
    run_steps_nd_in_place, step_layout_in_place, step_layout3d_in_place, step_manifold_in_place,
    step_nd_in_place,
};

pub(crate) fn parse_sgd_schedule_kind(raw: &str) -> Result<SgdScheduleKind, JsValue> {
    match normalize_enum_token(raw)?.as_str() {
        "exponential" => Ok(SgdScheduleKind::Exponential),
        "linear" => Ok(SgdScheduleKind::Linear),
        "quadratic" => Ok(SgdScheduleKind::Quadratic),
        "reciprocal" => Ok(SgdScheduleKind::Reciprocal),
        "constant" => Ok(SgdScheduleKind::Constant),
        "squared" => Ok(SgdScheduleKind::Squared),
        _ => Err(invalid_input_err(format!(
            "unknown SgdScheduleKind {raw:?}; expected 'exponential', 'linear', 'quadratic', 'reciprocal', 'constant', or 'squared'"
        ))),
    }
}

pub(crate) fn parse_sgd_schedule_options(
    options: &JsValue,
    ctx: &str,
) -> Result<(SgdScheduleConfig, u64, JsTargetKind), JsValue> {
    let mut cfg = SgdScheduleConfig::default();
    let mut seed = crate::js_convert::DEFAULT_SEED;
    let mut target = JsTargetKind::Layout2d;
    if check_options_arg(options, ctx)? {
        cfg.iterations = opt_usize_prop(options, "iterations", cfg.iterations)?;
        cfg.eps = opt_f64_prop(options, "eps", cfg.eps)?;
        if let Some(k_str) = opt_string_prop(options, "kind")? {
            cfg.kind = parse_sgd_schedule_kind(&k_str)?;
        }
        cfg.step_scale = opt_f64_prop(options, "stepScale", cfg.step_scale)?;
        cfg.max_mu = opt_maybe_f64_prop(options, "maxMu")?;
        seed = opt_seed_prop(options)?;
        target = opt_target_prop(options)?;
    }
    Ok((cfg, seed, target))
}

pub(crate) fn target_kind_str(target: JsTargetKind) -> &'static str {
    match target {
        JsTargetKind::Layout2d => "layout2d",
        JsTargetKind::Layout3d => "layout3d",
        JsTargetKind::LayoutNd => "layoutNd",
        JsTargetKind::Manifold2d => "manifold2d",
    }
}

#[derive(Debug, Clone)]
pub(crate) enum SgdInner {
    Dim2(SgdSolver<Layout2d<f32>>),
    Dim3(SgdSolver<Layout3d<f32>>),
    DimN(SgdSolver<LayoutNd<f32>>),
    Manifold(SgdSolver<ManifoldLayout2d<f32>>),
}

impl SgdInner {
    pub(crate) fn target_kind(&self) -> JsTargetKind {
        match self {
            Self::Dim2(_) => JsTargetKind::Layout2d,
            Self::Dim3(_) => JsTargetKind::Layout3d,
            Self::DimN(_) => JsTargetKind::LayoutNd,
            Self::Manifold(_) => JsTargetKind::Manifold2d,
        }
    }

    pub(crate) fn num_nodes(&self) -> usize {
        match self {
            Self::Dim2(s) => s.num_nodes(),
            Self::Dim3(s) => s.num_nodes(),
            Self::DimN(s) => s.num_nodes(),
            Self::Manifold(s) => s.num_nodes(),
        }
    }

    pub(crate) fn num_springs(&self) -> usize {
        match self {
            Self::Dim2(s) => s.num_springs(),
            Self::Dim3(s) => s.num_springs(),
            Self::DimN(s) => s.num_springs(),
            Self::Manifold(s) => s.num_springs(),
        }
    }

    pub(crate) fn num_hinges(&self) -> usize {
        match self {
            Self::Dim2(s) => s.num_hinges(),
            Self::Dim3(s) => s.num_hinges(),
            Self::DimN(s) => s.num_hinges(),
            Self::Manifold(s) => s.num_hinges(),
        }
    }

    pub(crate) fn current_eta(&self) -> f64 {
        match self {
            Self::Dim2(s) => s.current_eta(),
            Self::Dim3(s) => s.current_eta(),
            Self::DimN(s) => s.current_eta(),
            Self::Manifold(s) => s.current_eta(),
        }
    }

    pub(crate) fn status(&self) -> egraph::core::SolverStatus {
        match self {
            Self::Dim2(s) => s.status(),
            Self::Dim3(s) => s.status(),
            Self::DimN(s) => s.status(),
            Self::Manifold(s) => s.status(),
        }
    }

    pub(crate) fn iteration(&self) -> usize {
        match self {
            Self::Dim2(s) => s.iteration(),
            Self::Dim3(s) => s.iteration(),
            Self::DimN(s) => s.iteration(),
            Self::Manifold(s) => s.iteration(),
        }
    }

    pub(crate) fn total_iterations(&self) -> Option<usize> {
        match self {
            Self::Dim2(s) => s.total_iterations(),
            Self::Dim3(s) => s.total_iterations(),
            Self::DimN(s) => s.total_iterations(),
            Self::Manifold(s) => s.total_iterations(),
        }
    }

    pub(crate) fn reset_schedule(
        &mut self,
        sched: SgdScheduleConfig,
        seed: u64,
    ) -> egraph::core::CoreResult<()> {
        match self {
            Self::Dim2(s) => s.reset_schedule(sched, seed),
            Self::Dim3(s) => s.reset_schedule(sched, seed),
            Self::DimN(s) => s.reset_schedule(sched, seed),
            Self::Manifold(s) => s.reset_schedule(sched, seed),
        }
    }
}

/// Cooperative SoA `u32` SGD solver supporting full APSP, explicit pairs, packed symmetric pairs,
/// random-pair sampling, Omega, streaming negatives, 3-tier sparse barrier, and Ortmann Sparse
/// Stress across `Layout2d`, `Layout3d`, `LayoutNd`, and `ManifoldLayout2d` (`f32`).
#[wasm_bindgen(js_name = SgdSolver)]
#[derive(Debug, Clone)]
pub struct JsSgdSolver {
    inner: SgdInner,
    barrier_stats: Option<SparseBarrierStats>,
    dog_stats: Option<DogVaradhanStats>,
    max_mu: Option<f64>,
    locked_manifold: Option<ManifoldKind>,
}

impl JsSgdSolver {
    pub(crate) fn from_inner(inner: SgdInner) -> Self {
        Self {
            inner,
            barrier_stats: None,
            dog_stats: None,
            max_mu: None,
            locked_manifold: None,
        }
    }

    pub(crate) fn from_schedule_and_inner(max_mu: Option<f64>, inner: SgdInner) -> Self {
        Self {
            inner,
            barrier_stats: None,
            dog_stats: None,
            max_mu,
            locked_manifold: None,
        }
    }

    pub(crate) fn from_barrier_parts(
        inner: SgdInner,
        barrier_stats: Option<SparseBarrierStats>,
        dog_stats: Option<DogVaradhanStats>,
        max_mu: Option<f64>,
    ) -> Self {
        Self {
            inner,
            barrier_stats,
            dog_stats,
            max_mu,
            locked_manifold: None,
        }
    }

    pub(crate) fn set_max_mu(&mut self, max_mu: Option<f64>) {
        self.max_mu = max_mu;
    }

    pub(crate) fn inner_mut(&mut self) -> &mut SgdInner {
        &mut self.inner
    }

    pub(crate) fn locked_manifold_mut(&mut self) -> &mut Option<ManifoldKind> {
        &mut self.locked_manifold
    }

    pub(crate) fn barrier_stats_inner(&self) -> Option<SparseBarrierStats> {
        self.barrier_stats.clone()
    }

    pub(crate) fn dog_stats_inner(&self) -> Option<DogVaradhanStats> {
        self.dog_stats.clone()
    }
}

#[wasm_bindgen(js_class = SgdSolver)]
impl JsSgdSolver {
    /// Constructs a full all-pairs shortest-path SGD solver from `graph`.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsCsrGraph, options: &JsValue) -> Result<JsSgdSolver, JsValue> {
        Self::full(graph, options)
    }

    /// Constructs a full all-pairs shortest-path SGD solver from `graph`.
    pub fn full(graph: &JsCsrGraph, options: &JsValue) -> Result<JsSgdSolver, JsValue> {
        let (sched, seed_u64, target) = parse_sgd_schedule_options(options, "SgdSolver.full")?;
        let max_mu = sched.max_mu;
        let inner = match target {
            JsTargetKind::Layout2d => SgdInner::Dim2(
                SgdSolver::<Layout2d<f32>>::full(graph.inner(), sched, seed_u64)
                    .map_err(core_err_to_js)?,
            ),
            JsTargetKind::Layout3d => SgdInner::Dim3(
                SgdSolver::<Layout3d<f32>>::full(graph.inner(), sched, seed_u64)
                    .map_err(core_err_to_js)?,
            ),
            JsTargetKind::LayoutNd => SgdInner::DimN(
                SgdSolver::<LayoutNd<f32>>::full(graph.inner(), sched, seed_u64)
                    .map_err(core_err_to_js)?,
            ),
            JsTargetKind::Manifold2d => SgdInner::Manifold(
                SgdSolver::<ManifoldLayout2d<f32>>::full(graph.inner(), sched, seed_u64)
                    .map_err(core_err_to_js)?,
            ),
        };
        Ok(Self::from_schedule_and_inner(max_mu, inner))
    }

    /// Executes one 2D Euclidean SGD iteration in place.
    #[wasm_bindgen(js_name = _step2d)]
    pub fn step_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            SgdInner::Dim2(s) => step_layout_in_place(s, layout),
            _ => Err(unsupported_err(
                "SgdSolver was not configured for target 'layout2d'",
            )),
        }
    }

    /// Executes up to `steps` 2D Euclidean SGD iterations in place.
    #[wasm_bindgen(js_name = _runSteps2d)]
    pub fn run_steps_2d(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            SgdInner::Dim2(s) => {
                run_steps_layout_in_place(s.num_nodes(), s, layout, steps, "SgdSolver::run_steps")
            }
            _ => Err(unsupported_err(
                "SgdSolver was not configured for target 'layout2d'",
            )),
        }
    }

    /// Runs 2D Euclidean SGD optimization to completion in place.
    #[wasm_bindgen(js_name = _run2d)]
    pub fn run_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            SgdInner::Dim2(s) => run_layout_in_place(s.num_nodes(), s, layout, "SgdSolver::run"),
            _ => Err(unsupported_err(
                "SgdSolver was not configured for target 'layout2d'",
            )),
        }
    }

    /// Executes one 3D Euclidean SGD iteration in place.
    #[wasm_bindgen(js_name = _step3d)]
    pub fn step_3d(&mut self, layout: &mut JsLayout3d) -> Result<String, JsValue> {
        match &mut self.inner {
            SgdInner::Dim3(s) => step_layout3d_in_place(s, layout),
            _ => Err(unsupported_err(
                "SgdSolver was not configured for target 'layout3d'",
            )),
        }
    }

    /// Executes up to `steps` 3D Euclidean SGD iterations in place.
    #[wasm_bindgen(js_name = _runSteps3d)]
    pub fn run_steps_3d(&mut self, layout: &mut JsLayout3d, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            SgdInner::Dim3(s) => run_steps_layout3d_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "SgdSolver::run_steps_3d",
            ),
            _ => Err(unsupported_err(
                "SgdSolver was not configured for target 'layout3d'",
            )),
        }
    }

    /// Runs 3D Euclidean SGD optimization to completion in place.
    #[wasm_bindgen(js_name = _run3d)]
    pub fn run_3d(&mut self, layout: &mut JsLayout3d) -> Result<String, JsValue> {
        match &mut self.inner {
            SgdInner::Dim3(s) => {
                run_layout3d_in_place(s.num_nodes(), s, layout, "SgdSolver::run_3d")
            }
            _ => Err(unsupported_err(
                "SgdSolver was not configured for target 'layout3d'",
            )),
        }
    }

    /// Executes one N-D Euclidean SGD iteration in place.
    #[wasm_bindgen(js_name = _stepNd)]
    pub fn step_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsValue> {
        match &mut self.inner {
            SgdInner::DimN(s) => step_nd_in_place(s, layout),
            _ => Err(unsupported_err(
                "SgdSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    /// Executes up to `steps` N-D Euclidean SGD iterations in place.
    #[wasm_bindgen(js_name = _runStepsNd)]
    pub fn run_steps_nd(&mut self, layout: &mut JsLayoutNd, steps: f64) -> Result<String, JsValue> {
        match &mut self.inner {
            SgdInner::DimN(s) => {
                run_steps_nd_in_place(s.num_nodes(), s, layout, steps, "SgdSolver::run_steps_nd")
            }
            _ => Err(unsupported_err(
                "SgdSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    /// Runs N-D Euclidean SGD optimization to completion in place.
    #[wasm_bindgen(js_name = _runNd)]
    pub fn run_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsValue> {
        match &mut self.inner {
            SgdInner::DimN(s) => run_nd_in_place(s.num_nodes(), s, layout, "SgdSolver::run_nd"),
            _ => Err(unsupported_err(
                "SgdSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    /// Executes one non-Euclidean manifold SGD iteration in place.
    #[wasm_bindgen(js_name = _stepManifold)]
    pub fn step_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            SgdInner::Manifold(s) => step_manifold_in_place(
                s,
                layout,
                &mut self.locked_manifold,
                s.num_nodes(),
                "SgdSolver::step_manifold",
            ),
            _ => Err(unsupported_err(
                "SgdSolver was not configured for target 'manifold2d'",
            )),
        }
    }

    /// Executes up to `steps` non-Euclidean manifold SGD iterations in place.
    #[wasm_bindgen(js_name = _runStepsManifold)]
    pub fn run_steps_manifold(
        &mut self,
        layout: &mut JsManifoldLayout2d,
        steps: f64,
    ) -> Result<String, JsValue> {
        match &mut self.inner {
            SgdInner::Manifold(s) => run_steps_manifold_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                &mut self.locked_manifold,
                "SgdSolver::run_steps_manifold",
            ),
            _ => Err(unsupported_err(
                "SgdSolver was not configured for target 'manifold2d'",
            )),
        }
    }

    /// Runs non-Euclidean manifold SGD optimization to completion in place.
    #[wasm_bindgen(js_name = _runManifold)]
    pub fn run_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsValue> {
        match &mut self.inner {
            SgdInner::Manifold(s) => run_manifold_in_place(
                s.num_nodes(),
                s,
                layout,
                &mut self.locked_manifold,
                "SgdSolver::run_manifold",
            ),
            _ => Err(unsupported_err(
                "SgdSolver was not configured for target 'manifold2d'",
            )),
        }
    }

    /// Returns the configured layout target kind (`'layout2d' | 'layout3d' | 'layoutNd' | 'manifold2d'`).
    #[wasm_bindgen(getter)]
    pub fn target(&self) -> String {
        target_kind_str(self.inner.target_kind()).to_owned()
    }

    /// Returns `true` when this solver targets `ManifoldLayout2d`.
    #[wasm_bindgen(getter, js_name = supportsManifold)]
    pub fn supports_manifold(&self) -> bool {
        matches!(self.inner, SgdInner::Manifold(_))
    }

    /// Returns the maximum step clamp multiplier `maxMu` (`undefined` when unbounded `<= 1.0`).
    #[wasm_bindgen(getter, js_name = maxMu)]
    pub fn max_mu(&self) -> Option<f64> {
        self.max_mu
    }

    /// Returns the number of two-sided spring pairs stored.
    #[wasm_bindgen(getter, js_name = numSprings)]
    pub fn num_springs(&self) -> usize {
        self.inner.num_springs()
    }

    /// Returns the number of one-sided hinge repulsion pairs stored.
    #[wasm_bindgen(getter, js_name = numHinges)]
    pub fn num_hinges(&self) -> usize {
        self.inner.num_hinges()
    }

    /// Returns the learning rate `eta` for the upcoming iteration (`0.0` once finished).
    #[wasm_bindgen(getter, js_name = currentEta)]
    pub fn current_eta(&self) -> f64 {
        self.inner.current_eta()
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

    /// Returns the scheduled maximum iteration count (`iterations`), if fixed.
    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> Option<usize> {
        self.inner.total_iterations()
    }

    /// Returns `true` when the solver has converged, completed, or failed.
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
