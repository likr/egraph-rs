//! Cooperative SoA `u32` `SgdSolver` (`f32`) core bindings and manifold execution methods.

use egraph::core::CoreError;
use egraph::layout::{SgdScheduleConfig, SgdScheduleKind, SgdSolver, SparseBarrierStats};
use egraph::neighborhood::DogVaradhanStats;
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, parse_non_negative_usize, parse_seed_u64};
use crate::js_options::{
    is_plain_js_object, normalize_enum_token, opt_f64_prop, opt_seed_prop, opt_string_prop,
    opt_usize_prop, reject_extra_args_with_options, solver_status_str,
};
use crate::layout_buffers::JsLayout2d;
use crate::manifold::JsManifoldLayout2d;
use crate::solver_runner::{run_layout_in_place, run_steps_layout_in_place, step_layout_in_place};

pub(crate) fn parse_sgd_schedule_kind(raw: &str) -> Result<SgdScheduleKind, JsError> {
    match normalize_enum_token(raw).as_str() {
        "exponential" => Ok(SgdScheduleKind::Exponential),
        "linear" => Ok(SgdScheduleKind::Linear),
        "quadratic" => Ok(SgdScheduleKind::Quadratic),
        "reciprocal" => Ok(SgdScheduleKind::Reciprocal),
        "constant" => Ok(SgdScheduleKind::Constant),
        "squared" => Ok(SgdScheduleKind::Squared),
        _ => Err(JsError::new(&format!(
            "unknown SgdScheduleKind {raw:?}; expected 'exponential', 'linear', 'quadratic', 'reciprocal', 'constant', or 'squared'"
        ))),
    }
}

pub(crate) fn parse_sgd_schedule_and_seed(
    t_max_or_options: &JsValue,
    eps: Option<f64>,
    kind: Option<String>,
    step_scale: Option<f64>,
    seed: Option<f64>,
) -> Result<(SgdScheduleConfig, u64), JsError> {
    let mut cfg = SgdScheduleConfig::default();
    if is_plain_js_object(t_max_or_options) {
        let has_extra = eps.is_some() || kind.is_some() || step_scale.is_some() || seed.is_some();
        reject_extra_args_with_options(has_extra)?;
        cfg.iterations = opt_usize_prop(t_max_or_options, "tMax", cfg.iterations)?;
        cfg.eps = opt_f64_prop(t_max_or_options, "eps", cfg.eps)?;
        if let Some(k_str) = opt_string_prop(t_max_or_options, "kind")? {
            cfg.kind = parse_sgd_schedule_kind(&k_str)?;
        }
        cfg.step_scale = opt_f64_prop(t_max_or_options, "stepScale", cfg.step_scale)?;
        let seed_u64 = opt_seed_prop(t_max_or_options)?;
        return Ok((cfg, seed_u64));
    }

    if !t_max_or_options.is_undefined() && !t_max_or_options.is_null() {
        let t_val = t_max_or_options.as_f64().ok_or_else(|| {
            JsError::new(
                "tMaxOrOptions must be a non-negative integer, options object, null, or undefined",
            )
        })?;
        cfg.iterations = parse_non_negative_usize(t_val, "tMax")?;
    }
    cfg.eps = eps.unwrap_or(cfg.eps);
    if let Some(k_str) = kind {
        cfg.kind = parse_sgd_schedule_kind(&k_str)?;
    }
    cfg.step_scale = step_scale.unwrap_or(cfg.step_scale);
    let seed_u64 = parse_seed_u64(seed)?;
    Ok((cfg, seed_u64))
}

fn default_inv_sq_weights_f32(dists: &[f32]) -> Vec<f32> {
    dists
        .iter()
        .map(|&d| {
            let df = f64::from(d);
            if df.is_finite() && df > 0.0 {
                (1.0 / (df * df)) as f32
            } else {
                0.0
            }
        })
        .collect()
}

/// Cooperative SoA `u32` SGD solver supporting 3-tier sparse barrier, streaming negatives,
/// Ortmann Sparse Stress, LargeVis, DRGraph, Graph-CNE, SNAP-tFDP, PaCMAP-Graph,
/// diagonal-preconditioned mini-batch stress SGD (`sgd2`), Euclidean `Layout2d` (`f32`),
/// and non-Euclidean `ManifoldLayout2d` (`f32`).
#[wasm_bindgen(js_name = SgdSolver)]
#[derive(Debug, Clone)]
pub struct JsSgdSolver {
    inner: SgdSolver<f32>,
    barrier_stats: Option<SparseBarrierStats>,
    dog_stats: Option<DogVaradhanStats>,
}

impl JsSgdSolver {
    pub(crate) fn from_inner(inner: SgdSolver<f32>) -> Self {
        Self {
            inner,
            barrier_stats: None,
            dog_stats: None,
        }
    }

    pub(crate) fn from_barrier_parts(
        inner: SgdSolver<f32>,
        barrier_stats: Option<SparseBarrierStats>,
        dog_stats: Option<DogVaradhanStats>,
    ) -> Self {
        Self {
            inner,
            barrier_stats,
            dog_stats,
        }
    }

    pub(crate) fn barrier_stats_inner(&self) -> Option<SparseBarrierStats> {
        self.barrier_stats
    }

    pub(crate) fn dog_stats_inner(&self) -> Option<DogVaradhanStats> {
        self.dog_stats
    }
}

#[wasm_bindgen(js_class = SgdSolver)]
impl JsSgdSolver {
    /// Constructs an `SgdSolver` from explicit flat SoA two-sided spring arrays.
    ///
    /// When `dji` is omitted, defaults to `dij`. When `wij`/`wji` are omitted, defaults to $1 / d^2$.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = fromPairs)]
    pub fn from_pairs(
        num_nodes: f64,
        src: &[u32],
        tgt: &[u32],
        dij: &[f32],
        dji: Option<Vec<f32>>,
        wij: Option<Vec<f32>>,
        wji: Option<Vec<f32>>,
        t_max_or_options: &JsValue,
        eps: Option<f64>,
        kind: Option<String>,
        step_scale: Option<f64>,
        seed: Option<f64>,
    ) -> Result<JsSgdSolver, JsError> {
        let n = parse_non_negative_usize(num_nodes, "numNodes")?;
        let (sched, seed_u64) =
            parse_sgd_schedule_and_seed(t_max_or_options, eps, kind, step_scale, seed)?;
        let dji_vec = dji.unwrap_or_else(|| dij.to_vec());
        let wij_was_none = wij.is_none();
        let wij_vec = wij.unwrap_or_else(|| default_inv_sq_weights_f32(dij));
        let wji_vec = wji.unwrap_or_else(|| {
            if wij_was_none {
                default_inv_sq_weights_f32(&dji_vec)
            } else {
                wij_vec.clone()
            }
        });
        let inner = SgdSolver::<f32>::from_pairs(
            n,
            src.to_vec(),
            tgt.to_vec(),
            dij.to_vec(),
            dji_vec,
            wij_vec,
            wji_vec,
            sched,
            seed_u64,
        )
        .map_err(core_err_to_js)?;
        Ok(Self::from_inner(inner))
    }

    /// Constructs an `SgdSolver` combining two-sided spring pairs and one-sided hinge pairs.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(js_name = fromPairsWithHinges)]
    pub fn from_pairs_with_hinges(
        num_nodes: f64,
        spring_src: &[u32],
        spring_tgt: &[u32],
        spring_dij: &[f32],
        spring_dji: &[f32],
        spring_wij: &[f32],
        spring_wji: &[f32],
        hinge_src: &[u32],
        hinge_tgt: &[u32],
        hinge_dij: &[f32],
        hinge_dji: &[f32],
        hinge_wij: &[f32],
        hinge_wji: &[f32],
        t_max_or_options: &JsValue,
        eps: Option<f64>,
        kind: Option<String>,
        step_scale: Option<f64>,
        seed: Option<f64>,
    ) -> Result<JsSgdSolver, JsError> {
        let n = parse_non_negative_usize(num_nodes, "numNodes")?;
        let (sched, seed_u64) =
            parse_sgd_schedule_and_seed(t_max_or_options, eps, kind, step_scale, seed)?;
        let inner = SgdSolver::<f32>::from_pairs_with_hinges(
            n,
            spring_src.to_vec(),
            spring_tgt.to_vec(),
            spring_dij.to_vec(),
            spring_dji.to_vec(),
            spring_wij.to_vec(),
            spring_wji.to_vec(),
            hinge_src.to_vec(),
            hinge_tgt.to_vec(),
            hinge_dij.to_vec(),
            hinge_dji.to_vec(),
            hinge_wij.to_vec(),
            hinge_wji.to_vec(),
            sched,
            seed_u64,
        )
        .map_err(core_err_to_js)?;
        Ok(Self::from_inner(inner))
    }

    /// Constructs a full all-pairs shortest-path SGD solver from `graph`.
    pub fn full(
        graph: &JsCsrGraph,
        t_max_or_options: &JsValue,
        eps: Option<f64>,
        kind: Option<String>,
        step_scale: Option<f64>,
        seed: Option<f64>,
    ) -> Result<JsSgdSolver, JsError> {
        let (sched, seed_u64) =
            parse_sgd_schedule_and_seed(t_max_or_options, eps, kind, step_scale, seed)?;
        let inner =
            SgdSolver::<f32>::full(graph.inner(), sched, seed_u64).map_err(core_err_to_js)?;
        Ok(Self::from_inner(inner))
    }

    /// Executes one Euclidean SGD iteration on `layout` in place.
    pub fn step(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(&mut self.inner, layout)
    }

    /// Executes up to `steps` Euclidean SGD iterations on `layout` in place.
    #[wasm_bindgen(js_name = runSteps)]
    pub fn run_steps(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        run_steps_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            steps,
            "SgdSolver::run_steps",
        )
    }

    /// Runs Euclidean SGD optimization to completion on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "SgdSolver::run",
        )
    }

    /// Executes one non-Euclidean SGD iteration on a `ManifoldLayout2d` in place.
    #[wasm_bindgen(js_name = stepManifold)]
    pub fn step_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsError> {
        let status = self
            .inner
            .step_manifold(layout.inner_mut())
            .map_err(core_err_to_js)?;
        Ok(solver_status_str(status).to_owned())
    }

    /// Executes up to `steps` non-Euclidean SGD iterations on a `ManifoldLayout2d` in place.
    #[wasm_bindgen(js_name = runStepsManifold)]
    pub fn run_steps_manifold(
        &mut self,
        layout: &mut JsManifoldLayout2d,
        steps: f64,
    ) -> Result<String, JsError> {
        let n_steps = parse_non_negative_usize(steps, "steps")?;
        if layout.len() != self.inner.num_nodes() {
            return Err(core_err_to_js(CoreError::dimension_mismatch(
                self.inner.num_nodes(),
                layout.len(),
                "SgdSolver::run_steps_manifold",
            )));
        }
        layout
            .inner()
            .as_layout()
            .validate_finite()
            .map_err(core_err_to_js)?;
        let status = self
            .inner
            .run_steps_manifold(layout.inner_mut(), n_steps)
            .map_err(core_err_to_js)?;
        Ok(solver_status_str(status).to_owned())
    }

    /// Runs non-Euclidean SGD optimization to completion on a `ManifoldLayout2d` in place.
    #[wasm_bindgen(js_name = runManifold)]
    pub fn run_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsError> {
        if layout.len() != self.inner.num_nodes() {
            return Err(core_err_to_js(CoreError::dimension_mismatch(
                self.inner.num_nodes(),
                layout.len(),
                "SgdSolver::run_manifold",
            )));
        }
        layout
            .inner()
            .as_layout()
            .validate_finite()
            .map_err(core_err_to_js)?;
        let status = self
            .inner
            .run_manifold(layout.inner_mut())
            .map_err(core_err_to_js)?;
        Ok(solver_status_str(status).to_owned())
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

    /// Returns the learning rate $\eta$ for the upcoming iteration (`0.0` once finished).
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

    /// Returns the scheduled maximum iteration count (`tMax`).
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
