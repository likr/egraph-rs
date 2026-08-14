//! tsNET layout algorithm bindings for WebAssembly.
//!
//! This module provides WebAssembly bindings for the tsNET graph layout algorithm and its builder.

use crate::{distance_matrix::JsDistanceMatrix, drawing::JsDrawingEuclidean2d};
use petgraph_layout_ts_net::{TsNet, TsNetBuilder};
use wasm_bindgen::prelude::*;

/// WebAssembly binding for constructing a `TsNet` algorithm instance via the Builder pattern.
#[wasm_bindgen(js_name = "TsNetBuilder")]
pub struct JsTsNetBuilder {
    builder: TsNetBuilder<f32>,
}

#[wasm_bindgen(js_class = "TsNetBuilder")]
impl JsTsNetBuilder {
    /// Creates a new `TsNetBuilder` with default hyperparameters.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            builder: TsNetBuilder::new(),
        }
    }

    /// Sets the target perplexity.
    pub fn perplexity(mut self, value: f32) -> Self {
        self.builder = self.builder.perplexity(value);
        self
    }

    /// Sets the number of iterations for Stage 1.
    #[wasm_bindgen(js_name = "iterationsStage1")]
    pub fn iterations_stage1(mut self, value: usize) -> Self {
        self.builder = self.builder.iterations_stage1(value);
        self
    }

    /// Sets the exaggeration factor for Stage 1.
    pub fn exaggeration(mut self, value: f32) -> Self {
        self.builder = self.builder.exaggeration(value);
        self
    }

    /// Sets the number of iterations for Stage 2.
    #[wasm_bindgen(js_name = "iterationsStage2")]
    pub fn iterations_stage2(mut self, value: usize) -> Self {
        self.builder = self.builder.iterations_stage2(value);
        self
    }

    /// Sets the compression parameter lambda_c for Stage 2.
    #[wasm_bindgen(js_name = "lambdaCStage2")]
    pub fn lambda_c_stage2(mut self, value: f32) -> Self {
        self.builder = self.builder.lambda_c_stage2(value);
        self
    }

    /// Sets the number of iterations for Stage 3.
    #[wasm_bindgen(js_name = "iterationsStage3")]
    pub fn iterations_stage3(mut self, value: usize) -> Self {
        self.builder = self.builder.iterations_stage3(value);
        self
    }

    /// Sets the compression parameter lambda_c for Stage 3.
    #[wasm_bindgen(js_name = "lambdaCStage3")]
    pub fn lambda_c_stage3(mut self, value: f32) -> Self {
        self.builder = self.builder.lambda_c_stage3(value);
        self
    }

    /// Sets the repulsion parameter lambda_r for Stage 3.
    #[wasm_bindgen(js_name = "lambdaRStage3")]
    pub fn lambda_r_stage3(mut self, value: f32) -> Self {
        self.builder = self.builder.lambda_r_stage3(value);
        self
    }

    /// Sets the learning rate.
    #[wasm_bindgen(js_name = "learningRate")]
    pub fn learning_rate(mut self, value: f32) -> Self {
        self.builder = self.builder.learning_rate(value);
        self
    }

    /// Sets the momentum parameter.
    pub fn momentum(mut self, value: f32) -> Self {
        self.builder = self.builder.momentum(value);
        self
    }

    /// Sets the distance power exponent.
    pub fn power(mut self, value: f32) -> Self {
        self.builder = self.builder.power(value);
        self
    }

    /// Sets the epsilon_r parameter.
    #[wasm_bindgen(js_name = "epsilonR")]
    pub fn epsilon_r(mut self, value: f32) -> Self {
        self.builder = self.builder.epsilon_r(value);
        self
    }

    /// Sets the maximum binary search iterations for finding node sigma_i.
    #[wasm_bindgen(js_name = "sigmaIters")]
    pub fn sigma_iters(mut self, value: usize) -> Self {
        self.builder = self.builder.sigma_iters(value);
        self
    }

    /// Sets the tolerance threshold for perplexity binary search convergence.
    #[wasm_bindgen(js_name = "sigmaTolerance")]
    pub fn sigma_tolerance(mut self, value: f32) -> Self {
        self.builder = self.builder.sigma_tolerance(value);
        self
    }

    /// Builds a configured `TsNet` layout instance.
    pub fn build(&self) -> Result<JsTsNet, JsValue> {
        let ts_net = self
            .builder
            .clone()
            .build()
            .map_err(|e| JsValue::from_str(&e))?;
        Ok(JsTsNet { ts_net })
    }
}

impl Default for JsTsNetBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// WebAssembly binding for the tsNET layout algorithm.
#[wasm_bindgen(js_name = "TsNet")]
pub struct JsTsNet {
    pub(crate) ts_net: TsNet<f32>,
}

#[wasm_bindgen(js_class = "TsNet")]
impl JsTsNet {
    /// Creates a new `TsNet` with default hyperparameters.
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsTsNet {
        JsTsNet {
            ts_net: TsNet::new(),
        }
    }

    /// Gets the target perplexity.
    #[wasm_bindgen(getter)]
    pub fn perplexity(&self) -> f32 {
        self.ts_net.perplexity
    }

    /// Sets the target perplexity.
    #[wasm_bindgen(setter)]
    pub fn set_perplexity(&mut self, value: f32) {
        self.ts_net.perplexity = value;
    }

    /// Gets the number of iterations for Stage 1.
    #[wasm_bindgen(getter, js_name = "iterationsStage1")]
    pub fn iterations_stage1(&self) -> usize {
        self.ts_net.iterations_stage1
    }

    /// Sets the number of iterations for Stage 1.
    #[wasm_bindgen(setter, js_name = "iterationsStage1")]
    pub fn set_iterations_stage1(&mut self, value: usize) {
        self.ts_net.iterations_stage1 = value;
    }

    /// Gets the exaggeration factor for Stage 1.
    #[wasm_bindgen(getter)]
    pub fn exaggeration(&self) -> f32 {
        self.ts_net.exaggeration
    }

    /// Sets the exaggeration factor for Stage 1.
    #[wasm_bindgen(setter)]
    pub fn set_exaggeration(&mut self, value: f32) {
        self.ts_net.exaggeration = value;
    }

    /// Gets the number of iterations for Stage 2.
    #[wasm_bindgen(getter, js_name = "iterationsStage2")]
    pub fn iterations_stage2(&self) -> usize {
        self.ts_net.iterations_stage2
    }

    /// Sets the number of iterations for Stage 2.
    #[wasm_bindgen(setter, js_name = "iterationsStage2")]
    pub fn set_iterations_stage2(&mut self, value: usize) {
        self.ts_net.iterations_stage2 = value;
    }

    /// Gets the lambda_c parameter for Stage 2.
    #[wasm_bindgen(getter, js_name = "lambdaCStage2")]
    pub fn lambda_c_stage2(&self) -> f32 {
        self.ts_net.lambda_c_stage2
    }

    /// Sets the lambda_c parameter for Stage 2.
    #[wasm_bindgen(setter, js_name = "lambdaCStage2")]
    pub fn set_lambda_c_stage2(&mut self, value: f32) {
        self.ts_net.lambda_c_stage2 = value;
    }

    /// Gets the number of iterations for Stage 3.
    #[wasm_bindgen(getter, js_name = "iterationsStage3")]
    pub fn iterations_stage3(&self) -> usize {
        self.ts_net.iterations_stage3
    }

    /// Sets the number of iterations for Stage 3.
    #[wasm_bindgen(setter, js_name = "iterationsStage3")]
    pub fn set_iterations_stage3(&mut self, value: usize) {
        self.ts_net.iterations_stage3 = value;
    }

    /// Gets the lambda_c parameter for Stage 3.
    #[wasm_bindgen(getter, js_name = "lambdaCStage3")]
    pub fn lambda_c_stage3(&self) -> f32 {
        self.ts_net.lambda_c_stage3
    }

    /// Sets the lambda_c parameter for Stage 3.
    #[wasm_bindgen(setter, js_name = "lambdaCStage3")]
    pub fn set_lambda_c_stage3(&mut self, value: f32) {
        self.ts_net.lambda_c_stage3 = value;
    }

    /// Gets the lambda_r parameter for Stage 3.
    #[wasm_bindgen(getter, js_name = "lambdaRStage3")]
    pub fn lambda_r_stage3(&self) -> f32 {
        self.ts_net.lambda_r_stage3
    }

    /// Sets the lambda_r parameter for Stage 3.
    #[wasm_bindgen(setter, js_name = "lambdaRStage3")]
    pub fn set_lambda_r_stage3(&mut self, value: f32) {
        self.ts_net.lambda_r_stage3 = value;
    }

    /// Gets the learning rate.
    #[wasm_bindgen(getter, js_name = "learningRate")]
    pub fn learning_rate(&self) -> f32 {
        self.ts_net.learning_rate
    }

    /// Sets the learning rate.
    #[wasm_bindgen(setter, js_name = "learningRate")]
    pub fn set_learning_rate(&mut self, value: f32) {
        self.ts_net.learning_rate = value;
    }

    /// Gets the momentum parameter.
    #[wasm_bindgen(getter)]
    pub fn momentum(&self) -> f32 {
        self.ts_net.momentum
    }

    /// Sets the momentum parameter.
    #[wasm_bindgen(setter)]
    pub fn set_momentum(&mut self, value: f32) {
        self.ts_net.momentum = value;
    }

    /// Gets the epsilon_d parameter (kept for backward compatibility).
    #[wasm_bindgen(getter, js_name = "epsilonD")]
    pub fn epsilon_d(&self) -> f32 {
        0.01
    }

    /// Sets the epsilon_d parameter.
    #[wasm_bindgen(setter, js_name = "epsilonD")]
    pub fn set_epsilon_d(&mut self, _value: f32) {}

    /// Gets the epsilon_r parameter.
    #[wasm_bindgen(getter, js_name = "epsilonR")]
    pub fn epsilon_r(&self) -> f32 {
        self.ts_net.epsilon_r
    }

    /// Sets the epsilon_r parameter.
    #[wasm_bindgen(setter, js_name = "epsilonR")]
    pub fn set_epsilon_r(&mut self, value: f32) {
        self.ts_net.epsilon_r = value;
    }

    /// Runs the tsNET layout algorithm on the drawing using the provided distance matrix.
    pub fn run(&self, drawing: &mut JsDrawingEuclidean2d, distance_matrix: &JsDistanceMatrix) {
        self.ts_net
            .run(drawing.drawing_mut(), &distance_matrix.inner);
    }
}

impl Default for JsTsNet {
    fn default() -> Self {
        Self::new()
    }
}
