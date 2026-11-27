//! Cooperative `SpatialIntruderRefiner` (`f32`) and refinement telemetry WebAssembly bindings.

use egraph::core::IterativeSolver;
use egraph::layout::{
    SpatialIntruderRefiner, SpatialRefinerConfig, SpatialRefinerRoundStats, SpatialRefinerStats,
};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{
    core_err_to_js, parse_non_negative_usize, parse_optional_usize, parse_seed_u64,
};
use crate::js_options::{
    is_plain_js_object, opt_bool_prop, opt_f64_prop, opt_seed_prop, opt_usize_prop,
    reject_extra_args_with_options, solver_status_str,
};
use crate::layout_buffers::JsLayout2d;
use crate::solver_runner::{run_layout_in_place, run_steps_layout_in_place, step_layout_in_place};

/// Per-round diagnostics emitted by `SpatialIntruderRefiner`.
#[wasm_bindgen(js_name = SpatialRefinerRoundStats)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JsSpatialRefinerRoundStats {
    inner: SpatialRefinerRoundStats,
}

#[wasm_bindgen(js_class = SpatialRefinerRoundStats)]
impl JsSpatialRefinerRoundStats {
    /// Returns the 1-indexed refinement round number.
    #[wasm_bindgen(getter)]
    pub fn round(&self) -> usize {
        self.inner.round
    }

    /// Returns the count of Tier 1 1-hop anchor pairs.
    #[wasm_bindgen(getter, js_name = nAnchors)]
    pub fn n_anchors(&self) -> usize {
        self.inner.n_anchors
    }

    /// Returns the count of active Tier 2 false-negative pairs.
    #[wasm_bindgen(getter, js_name = nFn)]
    pub fn n_fn(&self) -> usize {
        self.inner.n_fn
    }

    /// Returns the count of active Tier 3 spatial false-positive intruder pairs.
    #[wasm_bindgen(getter, js_name = nFp)]
    pub fn n_fp(&self) -> usize {
        self.inner.n_fp
    }

    /// Returns the count of active short-range separation hinge pairs.
    #[wasm_bindgen(getter, js_name = nSep)]
    pub fn n_sep(&self) -> usize {
        self.inner.n_sep
    }

    /// Returns the total active pair count passed to the inner SGD pass.
    #[wasm_bindgen(getter, js_name = nPairsTotal)]
    pub fn n_pairs_total(&self) -> usize {
        self.inner.n_pairs_total
    }

    /// Returns `true` when zero intruders/stragglers were found and SGD was skipped.
    #[wasm_bindgen(getter)]
    pub fn skipped(&self) -> bool {
        self.inner.skipped
    }
}

/// Cumulative diagnostics returned by `SpatialIntruderRefiner.stats()`.
#[wasm_bindgen(js_name = SpatialRefinerStats)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsSpatialRefinerStats {
    inner: SpatialRefinerStats,
}

#[wasm_bindgen(js_class = SpatialRefinerStats)]
impl JsSpatialRefinerStats {
    /// Returns the number of active refinement rounds executed.
    #[wasm_bindgen(getter, js_name = roundsExecuted)]
    pub fn rounds_executed(&self) -> usize {
        self.inner.rounds_executed
    }

    /// Returns cumulative Tier 1 anchor pairs across executed rounds.
    #[wasm_bindgen(getter, js_name = nAnchorsTotal)]
    pub fn n_anchors_total(&self) -> usize {
        self.inner.n_anchors_total
    }

    /// Returns cumulative Tier 2 false-negative pairs across executed rounds.
    #[wasm_bindgen(getter, js_name = nFnTotal)]
    pub fn n_fn_total(&self) -> usize {
        self.inner.n_fn_total
    }

    /// Returns cumulative Tier 3 false-positive intruder pairs across executed rounds.
    #[wasm_bindgen(getter, js_name = nFpTotal)]
    pub fn n_fp_total(&self) -> usize {
        self.inner.n_fp_total
    }

    /// Returns cumulative short-range separation hinge pairs across executed rounds.
    #[wasm_bindgen(getter, js_name = nSepTotal)]
    pub fn n_sep_total(&self) -> usize {
        self.inner.n_sep_total
    }

    /// Returns cumulative total active pairs across executed rounds.
    #[wasm_bindgen(getter, js_name = nPairsTotal)]
    pub fn n_pairs_total(&self) -> usize {
        self.inner.n_pairs_total
    }

    /// Returns the initial median 1-hop edge scale before refinement.
    #[wasm_bindgen(getter, js_name = initialScale)]
    pub fn initial_scale(&self) -> f64 {
        self.inner.initial_scale
    }

    /// Returns the final median 1-hop edge scale after refinement.
    #[wasm_bindgen(getter, js_name = finalScale)]
    pub fn final_scale(&self) -> f64 {
        self.inner.final_scale
    }

    /// Returns the relative distance floor applied during refinement.
    #[wasm_bindgen(getter, js_name = dFloor)]
    pub fn d_floor(&self) -> f64 {
        self.inner.d_floor
    }

    /// Returns the ratio of final layout std to initial layout std.
    #[wasm_bindgen(getter, js_name = scaleRatio)]
    pub fn scale_ratio(&self) -> f64 {
        self.inner.scale_ratio
    }

    /// Returns `true` when the hub-safe 2-hop fallback path was used.
    #[wasm_bindgen(getter, js_name = usedHubFallback)]
    pub fn used_hub_fallback(&self) -> bool {
        self.inner.used_hub_fallback
    }

    /// Returns `true` when Tier 3 FPs used one-sided hinge repulsion.
    #[wasm_bindgen(getter, js_name = hingeFp)]
    pub fn hinge_fp(&self) -> bool {
        self.inner.hinge_fp
    }

    /// Returns per-round diagnostic entries.
    #[wasm_bindgen(js_name = roundStats)]
    pub fn round_stats(&self) -> Vec<JsSpatialRefinerRoundStats> {
        self.inner
            .round_stats
            .iter()
            .copied()
            .map(|inner| JsSpatialRefinerRoundStats { inner })
            .collect()
    }
}

/// Cooperative $O(N \log N)$ `StaticKdTree` active-set spatial intruder refiner (`f32`).
#[wasm_bindgen(js_name = SpatialIntruderRefiner)]
#[derive(Debug, Clone)]
pub struct JsSpatialIntruderRefiner {
    inner: SpatialIntruderRefiner<f32>,
}

#[wasm_bindgen(js_class = SpatialIntruderRefiner)]
impl JsSpatialIntruderRefiner {
    /// Precomputes canonical 1-hop topology and hub-safe 2-hop structures for `graph`.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(constructor)]
    pub fn new(
        graph: &JsCsrGraph,
        rounds_or_options: &JsValue,
        k_cap: Option<f64>,
        margin_scale: Option<f64>,
        w_anchor: Option<f64>,
        w_fn: Option<f64>,
        w_fp: Option<f64>,
        fn_shrink: Option<f64>,
        round_decay: Option<f64>,
        t_max: Option<f64>,
        eps: Option<f64>,
        hinge_fp: Option<bool>,
        seed: Option<f64>,
        w_anchor2: Option<f64>,
        k_cap_decay: Option<f64>,
        k_cap_min: Option<f64>,
        min_sep_ratio: Option<f64>,
        w_sep: Option<f64>,
    ) -> Result<JsSpatialIntruderRefiner, JsError> {
        let mut cfg = SpatialRefinerConfig::default();
        if is_plain_js_object(rounds_or_options) {
            let has_extra = k_cap.is_some()
                || margin_scale.is_some()
                || w_anchor.is_some()
                || w_fn.is_some()
                || w_fp.is_some()
                || fn_shrink.is_some()
                || round_decay.is_some()
                || t_max.is_some()
                || eps.is_some()
                || hinge_fp.is_some()
                || seed.is_some()
                || w_anchor2.is_some()
                || k_cap_decay.is_some()
                || k_cap_min.is_some()
                || min_sep_ratio.is_some()
                || w_sep.is_some();
            reject_extra_args_with_options(has_extra)?;
            cfg.iterations = opt_usize_prop(rounds_or_options, "iterations", cfg.iterations)?;
            cfg.k_cap = opt_usize_prop(rounds_or_options, "kCap", cfg.k_cap)?;
            cfg.margin_scale = opt_f64_prop(rounds_or_options, "marginScale", cfg.margin_scale)?;
            cfg.w_anchor = opt_f64_prop(rounds_or_options, "wAnchor", cfg.w_anchor)?;
            cfg.w_anchor2 = opt_f64_prop(rounds_or_options, "wAnchor2", cfg.w_anchor2)?;
            cfg.w_fn = opt_f64_prop(rounds_or_options, "wFn", cfg.w_fn)?;
            cfg.w_fp = opt_f64_prop(rounds_or_options, "wFp", cfg.w_fp)?;
            cfg.fn_shrink = opt_f64_prop(rounds_or_options, "fnShrink", cfg.fn_shrink)?;
            cfg.round_decay = opt_f64_prop(rounds_or_options, "roundDecay", cfg.round_decay)?;
            cfg.k_cap_decay = opt_f64_prop(rounds_or_options, "kCapDecay", cfg.k_cap_decay)?;
            cfg.k_cap_min = opt_usize_prop(rounds_or_options, "kCapMin", cfg.k_cap_min)?;
            cfg.sgd_iterations =
                opt_usize_prop(rounds_or_options, "sgdIterations", cfg.sgd_iterations)?;
            cfg.eps = opt_f64_prop(rounds_or_options, "eps", cfg.eps)?;
            cfg.hinge_fp = opt_bool_prop(rounds_or_options, "hingeFp", cfg.hinge_fp)?;
            cfg.min_sep_ratio = opt_f64_prop(rounds_or_options, "minSepRatio", cfg.min_sep_ratio)?;
            cfg.w_sep = opt_f64_prop(rounds_or_options, "wSep", cfg.w_sep)?;
            cfg.seed = opt_seed_prop(rounds_or_options)?;
        } else {
            if !rounds_or_options.is_undefined() && !rounds_or_options.is_null() {
                let v = rounds_or_options.as_f64().ok_or_else(|| {
                    JsError::new("iterationsOrOptions must be a non-negative integer, options object, null, or undefined")
                })?;
                cfg.iterations = parse_non_negative_usize(v, "iterations")?;
            }
            cfg.k_cap = parse_optional_usize(k_cap, cfg.k_cap, "kCap")?;
            cfg.margin_scale = margin_scale.unwrap_or(cfg.margin_scale);
            cfg.w_anchor = w_anchor.unwrap_or(cfg.w_anchor);
            cfg.w_anchor2 = w_anchor2.unwrap_or(cfg.w_anchor2);
            cfg.w_fn = w_fn.unwrap_or(cfg.w_fn);
            cfg.w_fp = w_fp.unwrap_or(cfg.w_fp);
            cfg.fn_shrink = fn_shrink.unwrap_or(cfg.fn_shrink);
            cfg.round_decay = round_decay.unwrap_or(cfg.round_decay);
            cfg.k_cap_decay = k_cap_decay.unwrap_or(cfg.k_cap_decay);
            cfg.k_cap_min = parse_optional_usize(k_cap_min, cfg.k_cap_min, "kCapMin")?;
            cfg.sgd_iterations = parse_optional_usize(t_max, cfg.sgd_iterations, "sgdIterations")?;
            cfg.eps = eps.unwrap_or(cfg.eps);
            cfg.hinge_fp = hinge_fp.unwrap_or(cfg.hinge_fp);
            cfg.min_sep_ratio = min_sep_ratio.unwrap_or(cfg.min_sep_ratio);
            cfg.w_sep = w_sep.unwrap_or(cfg.w_sep);
            cfg.seed = parse_seed_u64(seed)?;
        }

        let inner =
            SpatialIntruderRefiner::<f32>::new(graph.inner(), cfg).map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    /// Executes one active-set refinement round on `layout` in place.
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
            "SpatialIntruderRefiner::run_steps",
        )
    }

    /// Runs refinement rounds to completion or early convergence on `layout` in place.
    pub fn run(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        run_layout_in_place(
            self.inner.num_nodes(),
            &mut self.inner,
            layout,
            "SpatialIntruderRefiner::run",
        )
    }

    /// Returns cumulative and per-round diagnostic statistics.
    pub fn stats(&self) -> JsSpatialRefinerStats {
        JsSpatialRefinerStats {
            inner: self.inner.stats().clone(),
        }
    }

    /// Returns the solver lifecycle status string.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> String {
        solver_status_str(self.inner.status()).to_owned()
    }

    /// Returns the number of refinement rounds completed so far.
    #[wasm_bindgen(getter)]
    pub fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    /// Returns the maximum scheduled refinement rounds.
    #[wasm_bindgen(getter, js_name = totalIterations)]
    pub fn total_iterations(&self) -> usize {
        self.inner.total_iterations()
    }

    /// Returns `true` when refinement has converged or completed.
    #[wasm_bindgen(getter, js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        self.inner.status().is_finished()
    }

    /// Returns the expected vertex count.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Returns the current dynamic active-set neighborhood cap `kCap`.
    #[wasm_bindgen(getter, js_name = currentKCap)]
    pub fn current_k_cap(&self) -> usize {
        self.inner.current_k_cap()
    }

    /// Updates the active-set neighborhood cap `kCap` for subsequent refinement rounds.
    #[wasm_bindgen(js_name = setKCap)]
    pub fn set_k_cap(&mut self, k_cap: f64) -> Result<(), JsError> {
        let cap = parse_non_negative_usize(k_cap, "kCap")?;
        self.inner.set_k_cap(cap).map_err(core_err_to_js)
    }
}
