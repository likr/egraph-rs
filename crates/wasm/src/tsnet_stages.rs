//! Custom stage definitions, row-normalized sampling options, and polymorphic steppers
//! for `TsNetSolver`.

use egraph::layout::{TsNetSolver, TsNetStage};
use egraph::neighborhood::TwoHopSamplingMode;
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{core_err_to_js, parse_non_negative_usize};
use crate::js_options::{
    get_prop, is_plain_js_object, opt_f64_prop, opt_maybe_usize_prop, opt_string_prop,
    opt_usize_prop,
};
use crate::layout_buffers::{JsLayout2d, JsLayoutNd};
use crate::manifold::JsManifoldLayout2d;
use crate::neighborhood_types::JsSparseProbabilities;
use crate::solver_runner::{
    run_layout_in_place, run_manifold_in_place, run_nd_in_place, run_steps_layout_in_place,
    run_steps_manifold_in_place, run_steps_nd_in_place, step_layout_in_place,
    step_manifold_in_place, step_nd_in_place,
};
use crate::tsnet::JsTsNetSolver;
use crate::tsnet_options::{parse_tsnet_config, parse_tsnet_sampling_mode};

/// Hyperparameters governing a single optimization stage of `TsNetSolver`.
#[wasm_bindgen(js_name = TsNetStage)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JsTsNetStage {
    inner: TsNetStage,
}

#[wasm_bindgen(js_class = TsNetStage)]
impl JsTsNetStage {
    /// Constructs a validated `TsNetStage`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        iterations: f64,
        exaggeration: f64,
        lambda_kl: f64,
        lambda_c: f64,
        lambda_r: f64,
    ) -> Result<JsTsNetStage, JsError> {
        let iters = parse_non_negative_usize(iterations, "iterations")?;
        let inner = TsNetStage {
            iterations: iters,
            exaggeration,
            lambda_kl,
            lambda_c,
            lambda_r,
        };
        inner.validate().map_err(core_err_to_js)?;
        Ok(Self { inner })
    }

    #[wasm_bindgen(getter)]
    pub fn iterations(&self) -> usize {
        self.inner.iterations
    }

    #[wasm_bindgen(getter)]
    pub fn exaggeration(&self) -> f64 {
        self.inner.exaggeration
    }

    #[wasm_bindgen(getter, js_name = lambdaKl)]
    pub fn lambda_kl(&self) -> f64 {
        self.inner.lambda_kl
    }

    #[wasm_bindgen(getter, js_name = lambdaC)]
    pub fn lambda_c(&self) -> f64 {
        self.inner.lambda_c
    }

    #[wasm_bindgen(getter, js_name = lambdaR)]
    pub fn lambda_r(&self) -> f64 {
        self.inner.lambda_r
    }
}

fn parse_stages_array(stages: &JsValue) -> Result<Vec<TsNetStage>, JsError> {
    if stages.is_null() || stages.is_undefined() || stages.is_string() || !stages.is_object() {
        return Err(JsError::new(
            "stages must be an array or iterable of TsNetStage objects",
        ));
    }
    let iter = js_sys::try_iter(stages)
        .map_err(|err| JsError::new(&format!("failed to iterate stages: {err:?}")))?
        .ok_or_else(|| JsError::new("stages must be an array or iterable of TsNetStage objects"))?;
    let mut out = Vec::new();
    for item_res in iter {
        let item =
            item_res.map_err(|err| JsError::new(&format!("failed to read stage item: {err:?}")))?;
        if item.is_null() || !item.is_object() {
            return Err(JsError::new(
                "each stage entry must be a TsNetStage or plain stage object",
            ));
        }
        let iters = opt_usize_prop(&item, "iterations", 0)?;
        let exagg = opt_f64_prop(&item, "exaggeration", 1.0)?;
        let lambda_kl = opt_f64_prop(&item, "lambdaKl", 1.0)?;
        let lambda_c = opt_f64_prop(&item, "lambdaC", 0.0)?;
        let lambda_r = opt_f64_prop(&item, "lambdaR", 0.0)?;
        let stage = TsNetStage {
            iterations: iters,
            exaggeration: exagg,
            lambda_kl,
            lambda_c,
            lambda_r,
        };
        stage.validate().map_err(core_err_to_js)?;
        out.push(stage);
    }
    Ok(out)
}

#[wasm_bindgen(js_class = TsNetSolver)]
impl JsTsNetSolver {
    /// Constructs a `TsNetSolver` from precomputed symmetric `SparseProbabilities` and explicit `stages`.
    #[wasm_bindgen(js_name = fromProbabilitiesWithStages)]
    pub fn from_probabilities_with_stages(
        probs: &JsSparseProbabilities,
        stages: &JsValue,
        options: &JsValue,
    ) -> Result<JsTsNetSolver, JsError> {
        let parsed_stages = parse_stages_array(stages)?;
        let cfg = parse_tsnet_config(
            options,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            &JsValue::UNDEFINED,
            None,
            None,
        )?;
        let inner = TsNetSolver::<f32>::from_probabilities_with_stages(
            probs.inner().clone(),
            cfg,
            parsed_stages,
        )
        .map_err(core_err_to_js)?;
        Ok(Self::from_inner(inner))
    }

    /// Constructs a `TsNetSolver` from `graph` using direct row-normalized `A1 + w2 * A2`
    /// affinity with configurable 2-hop sampling `mode` and optional `maxBridgeDegree`.
    #[wasm_bindgen(js_name = fromGraphRowNormalizedWithOptions)]
    pub fn from_graph_row_normalized_with_options(
        graph: &JsCsrGraph,
        options: &JsValue,
    ) -> Result<JsTsNetSolver, JsError> {
        let mut w2 = 0.5_f64;
        let mut max_2hop = None;
        let mut degree_beta = 0.0_f64;
        let mut mode = TwoHopSamplingMode::DegreeOrderedHubSuppressed;
        let mut max_bridge_degree = None;
        if is_plain_js_object(options) {
            w2 = opt_f64_prop(options, "w2", w2)?;
            max_2hop = opt_maybe_usize_prop(options, "max2hopPerNode")?;
            degree_beta = opt_f64_prop(options, "degreeBeta", degree_beta)?;
            if let Some(raw_mode) = opt_string_prop(options, "mode")? {
                mode = parse_tsnet_sampling_mode(&raw_mode)?;
            } else if let Some(raw_mode) = opt_string_prop(options, "samplingMode")? {
                mode = parse_tsnet_sampling_mode(&raw_mode)?;
            }
            max_bridge_degree = opt_maybe_usize_prop(options, "maxBridgeDegree")?;
        }
        let cfg = parse_tsnet_config(
            options,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            &JsValue::UNDEFINED,
            None,
            None,
        )?;
        let _ = &get_prop;
        let inner = TsNetSolver::<f32>::from_graph_row_normalized_with_options(
            graph.inner(),
            w2,
            max_2hop,
            degree_beta,
            mode,
            max_bridge_degree,
            cfg,
        )
        .map_err(core_err_to_js)?;
        Ok(Self::from_inner(inner))
    }

    /// Executes one optimization iteration on `Layout2d` in place.
    #[wasm_bindgen(js_name = _step2d)]
    pub fn step_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        step_layout_in_place(self.inner_mut(), layout)
    }

    /// Executes up to `steps` iterations on `Layout2d` in place.
    #[wasm_bindgen(js_name = _runSteps2d)]
    pub fn run_steps_2d(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsError> {
        let expected = self.inner().num_nodes();
        run_steps_layout_in_place(
            expected,
            self.inner_mut(),
            layout,
            steps,
            "TsNetSolver::run_steps",
        )
    }

    /// Runs optimization to completion on `Layout2d` in place.
    #[wasm_bindgen(js_name = _run2d)]
    pub fn run_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsError> {
        let expected = self.inner().num_nodes();
        run_layout_in_place(expected, self.inner_mut(), layout, "TsNetSolver::run")
    }

    /// Executes one optimization iteration on `LayoutNd` in place.
    #[wasm_bindgen(js_name = _stepNd)]
    pub fn step_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsError> {
        step_nd_in_place(self.inner_mut(), layout)
    }

    /// Executes up to `steps` iterations on `LayoutNd` in place.
    #[wasm_bindgen(js_name = _runStepsNd)]
    pub fn run_steps_nd(&mut self, layout: &mut JsLayoutNd, steps: f64) -> Result<String, JsError> {
        let expected = self.inner().num_nodes();
        run_steps_nd_in_place(
            expected,
            self.inner_mut(),
            layout,
            steps,
            "TsNetSolver::run_steps_nd",
        )
    }

    /// Runs optimization to completion on `LayoutNd` in place.
    #[wasm_bindgen(js_name = _runNd)]
    pub fn run_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsError> {
        let expected = self.inner().num_nodes();
        run_nd_in_place(expected, self.inner_mut(), layout, "TsNetSolver::run_nd")
    }

    /// Executes one optimization iteration on `ManifoldLayout2d` in place.
    #[wasm_bindgen(js_name = _stepManifold)]
    pub fn step_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsError> {
        step_manifold_in_place(self.inner_mut(), layout)
    }

    /// Executes up to `steps` iterations on `ManifoldLayout2d` in place.
    #[wasm_bindgen(js_name = _runStepsManifold)]
    pub fn run_steps_manifold(
        &mut self,
        layout: &mut JsManifoldLayout2d,
        steps: f64,
    ) -> Result<String, JsError> {
        let expected = self.inner().num_nodes();
        run_steps_manifold_in_place(
            expected,
            self.inner_mut(),
            layout,
            steps,
            "TsNetSolver::run_steps_manifold",
        )
    }

    /// Runs optimization to completion on `ManifoldLayout2d` in place.
    #[wasm_bindgen(js_name = _runManifold)]
    pub fn run_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsError> {
        let expected = self.inner().num_nodes();
        run_manifold_in_place(
            expected,
            self.inner_mut(),
            layout,
            "TsNetSolver::run_manifold",
        )
    }
}
