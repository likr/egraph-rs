//! Custom stage definitions, row-normalized sampling options, and polymorphic steppers
//! for `TsNetSolver`.

use egraph::layout::TsNetStage;
use egraph::neighborhood::TwoHopSamplingMode;
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{
    core_err_to_js, invalid_input_err, parse_non_negative_usize, unsupported_err,
};
use crate::js_options::{
    check_options_arg, is_plain_js_object, opt_f64_prop, opt_maybe_usize_prop, opt_usize_prop,
};
use crate::layout_buffers::{JsLayout2d, JsLayoutNd};
use crate::layout3d::JsLayout3d;
use crate::manifold::JsManifoldLayout2d;
use crate::neighborhood_types::JsSparseProbabilities;
use crate::solver_runner::{
    run_layout_in_place, run_layout3d_in_place, run_manifold_in_place, run_nd_in_place,
    run_steps_layout_in_place, run_steps_layout3d_in_place, run_steps_manifold_in_place,
    run_steps_nd_in_place, step_layout_in_place, step_layout3d_in_place, step_manifold_in_place,
    step_nd_in_place,
};
use crate::tsnet::{JsTsNetSolver, TsNetInner};
use crate::tsnet_options::{
    parse_sampling_mode_from_options, parse_stages_array, parse_tsnet_options,
};

/// Hyperparameters governing a single optimization stage of `TsNetSolver`.
#[wasm_bindgen(js_name = TsNetStage)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsTsNetStage {
    inner: TsNetStage,
}

#[wasm_bindgen(js_class = TsNetStage)]
impl JsTsNetStage {
    /// Constructs a validated `TsNetStage` from `(iterations, options?)` or `{ iterations, ... }`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        iterations_or_options: &JsValue,
        options: &JsValue,
    ) -> Result<JsTsNetStage, JsValue> {
        let stage = if is_plain_js_object(iterations_or_options) {
            if !options.is_undefined() && !options.is_null() {
                return Err(invalid_input_err(
                    "TsNetStage(options): unexpected second argument after options object",
                ));
            }
            let iters = opt_usize_prop(iterations_or_options, "iterations", 250)?;
            let exagg = opt_f64_prop(iterations_or_options, "exaggeration", 1.0)?;
            let l_kl = opt_f64_prop(iterations_or_options, "lambdaKl", 1.0)?;
            let l_c = opt_f64_prop(iterations_or_options, "lambdaC", 0.0)?;
            let l_r = opt_f64_prop(iterations_or_options, "lambdaR", 0.0)?;
            TsNetStage::new()
                .with_iterations(iters)
                .with_exaggeration(exagg)
                .with_lambda_kl(l_kl)
                .with_lambda_c(l_c)
                .with_lambda_r(l_r)
        } else {
            let iters_f = iterations_or_options.as_f64().ok_or_else(|| {
                invalid_input_err(
                    "TsNetStage: iterations must be a non-negative integer or options object",
                )
            })?;
            let iters = parse_non_negative_usize(iters_f, "iterations")?;
            let mut st = TsNetStage::new().with_iterations(iters);
            if check_options_arg(options, "TsNetStage")? {
                st = st
                    .with_exaggeration(opt_f64_prop(options, "exaggeration", 1.0)?)
                    .with_lambda_kl(opt_f64_prop(options, "lambdaKl", 1.0)?)
                    .with_lambda_c(opt_f64_prop(options, "lambdaC", 0.0)?)
                    .with_lambda_r(opt_f64_prop(options, "lambdaR", 0.0)?);
            }
            st
        };
        stage.validate().map_err(core_err_to_js)?;
        Ok(Self { inner: stage })
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

#[wasm_bindgen(js_class = TsNetSolver)]
impl JsTsNetSolver {
    /// Constructs a `TsNetSolver` from precomputed symmetric `SparseProbabilities` and explicit `stages`.
    #[wasm_bindgen(js_name = fromProbabilitiesWithStages)]
    pub fn from_probabilities_with_stages(
        probs: &JsSparseProbabilities,
        stages: &JsValue,
        options: &JsValue,
    ) -> Result<JsTsNetSolver, JsValue> {
        let parsed_stages = parse_stages_array(stages)?;
        let (mut cfg, _, backend) =
            parse_tsnet_options(options, "TsNetSolver.fromProbabilitiesWithStages")?;
        cfg = cfg.with_stages(parsed_stages);
        let inner = Self::build_probabilities(probs, cfg, backend)?;
        Ok(Self::from_parts(inner, None))
    }

    /// Constructs a `TsNetSolver` from `graph` using direct row-normalized `A1 + w2 * A2`
    /// affinity with configurable 2-hop sampling `mode` and optional `maxBridgeDegree`.
    #[wasm_bindgen(js_name = fromGraphRowNormalizedWithOptions)]
    pub fn from_graph_row_normalized_with_options(
        graph: &JsCsrGraph,
        options: &JsValue,
    ) -> Result<JsTsNetSolver, JsValue> {
        let mut w2 = 0.5_f64;
        let mut max_2hop = None;
        let mut degree_beta = 0.0_f64;
        let mut mode = TwoHopSamplingMode::DegreeOrderedHubSuppressed;
        let mut max_bridge_degree = None;
        if check_options_arg(options, "TsNetSolver.fromGraphRowNormalizedWithOptions")? {
            w2 = opt_f64_prop(options, "w2", w2)?;
            max_2hop = opt_maybe_usize_prop(options, "max2hopPerNode")?;
            degree_beta = opt_f64_prop(options, "degreeBeta", degree_beta)?;
            mode = parse_sampling_mode_from_options(options, mode)?;
            max_bridge_degree = opt_maybe_usize_prop(options, "maxBridgeDegree")?;
        }
        let (cfg, _, backend) =
            parse_tsnet_options(options, "TsNetSolver.fromGraphRowNormalizedWithOptions")?;
        let inner = Self::build_row_normalized(
            graph,
            w2,
            max_2hop,
            degree_beta,
            mode,
            max_bridge_degree,
            cfg,
            backend,
        )?;
        Ok(Self::from_parts(inner, Some(mode)))
    }

    #[wasm_bindgen(js_name = _step2d)]
    pub fn step_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        match self.inner_mut() {
            TsNetInner::Dim2(s) => step_layout_in_place(s, layout),
            _ => Err(unsupported_err(
                "TsNetSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runSteps2d)]
    pub fn run_steps_2d(&mut self, layout: &mut JsLayout2d, steps: f64) -> Result<String, JsValue> {
        match self.inner_mut() {
            TsNetInner::Dim2(s) => {
                run_steps_layout_in_place(s.num_nodes(), s, layout, steps, "TsNetSolver::run_steps")
            }
            _ => Err(unsupported_err(
                "TsNetSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _run2d)]
    pub fn run_2d(&mut self, layout: &mut JsLayout2d) -> Result<String, JsValue> {
        match self.inner_mut() {
            TsNetInner::Dim2(s) => {
                run_layout_in_place(s.num_nodes(), s, layout, "TsNetSolver::run")
            }
            _ => Err(unsupported_err(
                "TsNetSolver was not configured for target 'layout2d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _step3d)]
    pub fn step_3d(&mut self, layout: &mut JsLayout3d) -> Result<String, JsValue> {
        match self.inner_mut() {
            TsNetInner::Dim3(s) => step_layout3d_in_place(s, layout),
            _ => Err(unsupported_err(
                "TsNetSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runSteps3d)]
    pub fn run_steps_3d(&mut self, layout: &mut JsLayout3d, steps: f64) -> Result<String, JsValue> {
        match self.inner_mut() {
            TsNetInner::Dim3(s) => run_steps_layout3d_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                "TsNetSolver::run_steps_3d",
            ),
            _ => Err(unsupported_err(
                "TsNetSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _run3d)]
    pub fn run_3d(&mut self, layout: &mut JsLayout3d) -> Result<String, JsValue> {
        match self.inner_mut() {
            TsNetInner::Dim3(s) => {
                run_layout3d_in_place(s.num_nodes(), s, layout, "TsNetSolver::run_3d")
            }
            _ => Err(unsupported_err(
                "TsNetSolver was not configured for target 'layout3d'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _stepNd)]
    pub fn step_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsValue> {
        match self.inner_mut() {
            TsNetInner::DimN(s) => step_nd_in_place(s, layout),
            _ => Err(unsupported_err(
                "TsNetSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runStepsNd)]
    pub fn run_steps_nd(&mut self, layout: &mut JsLayoutNd, steps: f64) -> Result<String, JsValue> {
        match self.inner_mut() {
            TsNetInner::DimN(s) => {
                run_steps_nd_in_place(s.num_nodes(), s, layout, steps, "TsNetSolver::run_steps_nd")
            }
            _ => Err(unsupported_err(
                "TsNetSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _runNd)]
    pub fn run_nd(&mut self, layout: &mut JsLayoutNd) -> Result<String, JsValue> {
        match self.inner_mut() {
            TsNetInner::DimN(s) => run_nd_in_place(s.num_nodes(), s, layout, "TsNetSolver::run_nd"),
            _ => Err(unsupported_err(
                "TsNetSolver was not configured for target 'layoutNd'",
            )),
        }
    }

    #[wasm_bindgen(js_name = _stepManifold)]
    pub fn step_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsValue> {
        let mut locked = *self.locked_manifold_mut();
        let res = match self.inner_mut() {
            TsNetInner::Manifold(s) => {
                let n = s.num_nodes();
                step_manifold_in_place(s, layout, &mut locked, n, "TsNetSolver::step_manifold")
            }
            _ => Err(unsupported_err(
                "TsNetSolver was not configured for target 'manifold2d'",
            )),
        };
        *self.locked_manifold_mut() = locked;
        res
    }

    #[wasm_bindgen(js_name = _runStepsManifold)]
    pub fn run_steps_manifold(
        &mut self,
        layout: &mut JsManifoldLayout2d,
        steps: f64,
    ) -> Result<String, JsValue> {
        let mut locked = *self.locked_manifold_mut();
        let res = match self.inner_mut() {
            TsNetInner::Manifold(s) => run_steps_manifold_in_place(
                s.num_nodes(),
                s,
                layout,
                steps,
                &mut locked,
                "TsNetSolver::run_steps_manifold",
            ),
            _ => Err(unsupported_err(
                "TsNetSolver was not configured for target 'manifold2d'",
            )),
        };
        *self.locked_manifold_mut() = locked;
        res
    }

    #[wasm_bindgen(js_name = _runManifold)]
    pub fn run_manifold(&mut self, layout: &mut JsManifoldLayout2d) -> Result<String, JsValue> {
        let mut locked = *self.locked_manifold_mut();
        let res = match self.inner_mut() {
            TsNetInner::Manifold(s) => run_manifold_in_place(
                s.num_nodes(),
                s,
                layout,
                &mut locked,
                "TsNetSolver::run_manifold",
            ),
            _ => Err(unsupported_err(
                "TsNetSolver was not configured for target 'manifold2d'",
            )),
        };
        *self.locked_manifold_mut() = locked;
        res
    }
}
