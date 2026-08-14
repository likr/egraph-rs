//! tsNET and BH-tsNET layout algorithm bindings for WebAssembly.
//!
//! This module provides WebAssembly bindings for the tsNET and BH-tsNET graph layout algorithms and their builders.

use crate::{
    distance_matrix::JsDistanceMatrix, drawing::JsDrawingEuclidean2d, graph::JsGraph, rng::JsRng,
};
use petgraph_layout_ts_net::{BhTsNet, BhTsNetBuilder, TsNet, TsNetBuilder};
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

/// WebAssembly binding for constructing a `BhTsNet` algorithm instance via the Builder pattern.
#[wasm_bindgen(js_name = "BhTsNetBuilder")]
pub struct JsBhTsNetBuilder {
    builder: BhTsNetBuilder<f32>,
}

#[wasm_bindgen(js_class = "BhTsNetBuilder")]
impl JsBhTsNetBuilder {
    /// Creates a new `BhTsNetBuilder` with default hyperparameters.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            builder: BhTsNetBuilder::new(),
        }
    }

    /// Sets the target perplexity (default: 40.0).
    pub fn perplexity(mut self, value: f32) -> Self {
        self.builder = self.builder.perplexity(value);
        self
    }

    /// Sets the Barnes-Hut opening angle threshold theta (default: 0.5).
    pub fn theta(mut self, value: f32) -> Self {
        self.builder = self.builder.theta(value);
        self
    }

    /// Sets the number of nearest neighbors `k` for Partial BFS. Defaults to `3 * perplexity`.
    pub fn k(mut self, value: usize) -> Self {
        self.builder = self.builder.k(value);
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

    /// Builds a configured `BhTsNet` layout instance.
    pub fn build(&self) -> Result<JsBhTsNet, JsValue> {
        let bh_ts_net = self
            .builder
            .clone()
            .build()
            .map_err(|e| JsValue::from_str(&e))?;
        Ok(JsBhTsNet { bh_ts_net })
    }
}

impl Default for JsBhTsNetBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// WebAssembly binding for the BH-tsNET (Barnes-Hut tsNET) layout algorithm.
#[wasm_bindgen(js_name = "BhTsNet")]
pub struct JsBhTsNet {
    pub(crate) bh_ts_net: BhTsNet<f32>,
}

#[wasm_bindgen(js_class = "BhTsNet")]
impl JsBhTsNet {
    /// Creates a new `BhTsNet` with default hyperparameters.
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsBhTsNet {
        JsBhTsNet {
            bh_ts_net: BhTsNet::new(),
        }
    }

    /// Gets the target perplexity.
    #[wasm_bindgen(getter)]
    pub fn perplexity(&self) -> f32 {
        self.bh_ts_net.perplexity
    }

    /// Sets the target perplexity.
    #[wasm_bindgen(setter)]
    pub fn set_perplexity(&mut self, value: f32) {
        self.bh_ts_net.perplexity = value;
    }

    /// Gets the Barnes-Hut theta parameter.
    #[wasm_bindgen(getter)]
    pub fn theta(&self) -> f32 {
        self.bh_ts_net.theta
    }

    /// Sets the Barnes-Hut theta parameter.
    #[wasm_bindgen(setter)]
    pub fn set_theta(&mut self, value: f32) {
        self.bh_ts_net.theta = value;
    }

    /// Gets the number of nearest neighbors k.
    #[wasm_bindgen(getter)]
    pub fn k(&self) -> usize {
        self.bh_ts_net.k
    }

    /// Sets the number of nearest neighbors k.
    #[wasm_bindgen(setter)]
    pub fn set_k(&mut self, value: usize) {
        self.bh_ts_net.k = value;
    }

    /// Gets the number of iterations for Stage 1.
    #[wasm_bindgen(getter, js_name = "iterationsStage1")]
    pub fn iterations_stage1(&self) -> usize {
        self.bh_ts_net.iterations_stage1
    }

    /// Sets the number of iterations for Stage 1.
    #[wasm_bindgen(setter, js_name = "iterationsStage1")]
    pub fn set_iterations_stage1(&mut self, value: usize) {
        self.bh_ts_net.iterations_stage1 = value;
    }

    /// Gets the exaggeration factor for Stage 1.
    #[wasm_bindgen(getter)]
    pub fn exaggeration(&self) -> f32 {
        self.bh_ts_net.exaggeration
    }

    /// Sets the exaggeration factor for Stage 1.
    #[wasm_bindgen(setter)]
    pub fn set_exaggeration(&mut self, value: f32) {
        self.bh_ts_net.exaggeration = value;
    }

    /// Gets the number of iterations for Stage 2.
    #[wasm_bindgen(getter, js_name = "iterationsStage2")]
    pub fn iterations_stage2(&self) -> usize {
        self.bh_ts_net.iterations_stage2
    }

    /// Sets the number of iterations for Stage 2.
    #[wasm_bindgen(setter, js_name = "iterationsStage2")]
    pub fn set_iterations_stage2(&mut self, value: usize) {
        self.bh_ts_net.iterations_stage2 = value;
    }

    /// Gets the lambda_c parameter for Stage 2.
    #[wasm_bindgen(getter, js_name = "lambdaCStage2")]
    pub fn lambda_c_stage2(&self) -> f32 {
        self.bh_ts_net.lambda_c_stage2
    }

    /// Sets the lambda_c parameter for Stage 2.
    #[wasm_bindgen(setter, js_name = "lambdaCStage2")]
    pub fn set_lambda_c_stage2(&mut self, value: f32) {
        self.bh_ts_net.lambda_c_stage2 = value;
    }

    /// Gets the number of iterations for Stage 3.
    #[wasm_bindgen(getter, js_name = "iterationsStage3")]
    pub fn iterations_stage3(&self) -> usize {
        self.bh_ts_net.iterations_stage3
    }

    /// Sets the number of iterations for Stage 3.
    #[wasm_bindgen(setter, js_name = "iterationsStage3")]
    pub fn set_iterations_stage3(&mut self, value: usize) {
        self.bh_ts_net.iterations_stage3 = value;
    }

    /// Gets the lambda_c parameter for Stage 3.
    #[wasm_bindgen(getter, js_name = "lambdaCStage3")]
    pub fn lambda_c_stage3(&self) -> f32 {
        self.bh_ts_net.lambda_c_stage3
    }

    /// Sets the lambda_c parameter for Stage 3.
    #[wasm_bindgen(setter, js_name = "lambdaCStage3")]
    pub fn set_lambda_c_stage3(&mut self, value: f32) {
        self.bh_ts_net.lambda_c_stage3 = value;
    }

    /// Gets the lambda_r parameter for Stage 3.
    #[wasm_bindgen(getter, js_name = "lambdaRStage3")]
    pub fn lambda_r_stage3(&self) -> f32 {
        self.bh_ts_net.lambda_r_stage3
    }

    /// Sets the lambda_r parameter for Stage 3.
    #[wasm_bindgen(setter, js_name = "lambdaRStage3")]
    pub fn set_lambda_r_stage3(&mut self, value: f32) {
        self.bh_ts_net.lambda_r_stage3 = value;
    }

    /// Gets the learning rate.
    #[wasm_bindgen(getter, js_name = "learningRate")]
    pub fn learning_rate(&self) -> f32 {
        self.bh_ts_net.learning_rate
    }

    /// Sets the learning rate.
    #[wasm_bindgen(setter, js_name = "learningRate")]
    pub fn set_learning_rate(&mut self, value: f32) {
        self.bh_ts_net.learning_rate = value;
    }

    /// Gets the momentum parameter.
    #[wasm_bindgen(getter)]
    pub fn momentum(&self) -> f32 {
        self.bh_ts_net.momentum
    }

    /// Sets the momentum parameter.
    #[wasm_bindgen(setter)]
    pub fn set_momentum(&mut self, value: f32) {
        self.bh_ts_net.momentum = value;
    }

    /// Gets the epsilon_r parameter.
    #[wasm_bindgen(getter, js_name = "epsilonR")]
    pub fn epsilon_r(&self) -> f32 {
        self.bh_ts_net.epsilon_r
    }

    /// Sets the epsilon_r parameter.
    #[wasm_bindgen(setter, js_name = "epsilonR")]
    pub fn set_epsilon_r(&mut self, value: f32) {
        self.bh_ts_net.epsilon_r = value;
    }

    /// Runs the BH-tsNET layout algorithm on the drawing using the provided graph and RNG.
    pub fn run(&self, drawing: &mut JsDrawingEuclidean2d, graph: &JsGraph, rng: &mut JsRng) {
        self.bh_ts_net
            .run(drawing.drawing_mut(), graph.graph(), rng.get_mut());
    }
}

impl Default for JsBhTsNet {
    fn default() -> Self {
        Self::new()
    }
}

/// WebAssembly binding for constructing a `FitTsNet` algorithm instance via the Builder pattern.
#[wasm_bindgen(js_name = "FitTsNetBuilder")]
pub struct JsFitTsNetBuilder {
    builder: petgraph_layout_ts_net::FitTsNetBuilder<f32>,
}

#[wasm_bindgen(js_class = "FitTsNetBuilder")]
impl JsFitTsNetBuilder {
    /// Creates a new `FitTsNetBuilder` with default hyperparameters.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            builder: petgraph_layout_ts_net::FitTsNetBuilder::new(),
        }
    }

    /// Sets the number of intervals `I` for FFT interpolation (default: 25).
    pub fn intervals(mut self, value: usize) -> Self {
        self.builder = self.builder.intervals(value);
        self
    }

    /// Sets the number of interpolation points `P` per interval (default: 3).
    #[wasm_bindgen(js_name = "interpolationPoints")]
    pub fn interpolation_points(mut self, value: usize) -> Self {
        self.builder = self.builder.interpolation_points(value);
        self
    }

    /// Sets the target perplexity (default: 40.0).
    pub fn perplexity(mut self, value: f32) -> Self {
        self.builder = self.builder.perplexity(value);
        self
    }

    /// Sets the Barnes-Hut opening angle threshold theta (default: 0.5).
    pub fn theta(mut self, value: f32) -> Self {
        self.builder = self.builder.theta(value);
        self
    }

    /// Sets the number of nearest neighbors `k` for Partial BFS. Defaults to `3 * perplexity`.
    pub fn k(mut self, value: usize) -> Self {
        self.builder = self.builder.k(value);
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

    /// Builds a configured `FitTsNet` layout instance.
    pub fn build(&self) -> Result<JsFitTsNet, JsValue> {
        let fit_ts_net = self
            .builder
            .clone()
            .build()
            .map_err(|e| JsValue::from_str(&e))?;
        Ok(JsFitTsNet { fit_ts_net })
    }
}

impl Default for JsFitTsNetBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// WebAssembly binding for the FIt-tsNET (Fast Interpolation tsNET) layout algorithm.
#[wasm_bindgen(js_name = "FitTsNet")]
pub struct JsFitTsNet {
    pub(crate) fit_ts_net: petgraph_layout_ts_net::FitTsNet<f32>,
}

#[wasm_bindgen(js_class = "FitTsNet")]
impl JsFitTsNet {
    /// Creates a new `FitTsNet` with default hyperparameters.
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsFitTsNet {
        JsFitTsNet {
            fit_ts_net: petgraph_layout_ts_net::FitTsNet::new(),
        }
    }

    /// Gets the number of intervals.
    #[wasm_bindgen(getter)]
    pub fn intervals(&self) -> usize {
        self.fit_ts_net.intervals
    }

    /// Sets the number of intervals.
    #[wasm_bindgen(setter)]
    pub fn set_intervals(&mut self, value: usize) {
        self.fit_ts_net.intervals = value;
    }

    /// Gets the number of interpolation points.
    #[wasm_bindgen(getter, js_name = "interpolationPoints")]
    pub fn interpolation_points(&self) -> usize {
        self.fit_ts_net.interpolation_points
    }

    /// Sets the number of interpolation points.
    #[wasm_bindgen(setter, js_name = "interpolationPoints")]
    pub fn set_interpolation_points(&mut self, value: usize) {
        self.fit_ts_net.interpolation_points = value;
    }

    /// Gets the target perplexity.
    #[wasm_bindgen(getter)]
    pub fn perplexity(&self) -> f32 {
        self.fit_ts_net.perplexity
    }

    /// Sets the target perplexity.
    #[wasm_bindgen(setter)]
    pub fn set_perplexity(&mut self, value: f32) {
        self.fit_ts_net.perplexity = value;
    }

    /// Gets the Barnes-Hut theta parameter.
    #[wasm_bindgen(getter)]
    pub fn theta(&self) -> f32 {
        self.fit_ts_net.theta
    }

    /// Sets the Barnes-Hut theta parameter.
    #[wasm_bindgen(setter)]
    pub fn set_theta(&mut self, value: f32) {
        self.fit_ts_net.theta = value;
    }

    /// Gets the number of nearest neighbors k.
    #[wasm_bindgen(getter)]
    pub fn k(&self) -> usize {
        self.fit_ts_net.k
    }

    /// Sets the number of nearest neighbors k.
    #[wasm_bindgen(setter)]
    pub fn set_k(&mut self, value: usize) {
        self.fit_ts_net.k = value;
    }

    /// Gets the number of iterations for Stage 1.
    #[wasm_bindgen(getter, js_name = "iterationsStage1")]
    pub fn iterations_stage1(&self) -> usize {
        self.fit_ts_net.iterations_stage1
    }

    /// Sets the number of iterations for Stage 1.
    #[wasm_bindgen(setter, js_name = "iterationsStage1")]
    pub fn set_iterations_stage1(&mut self, value: usize) {
        self.fit_ts_net.iterations_stage1 = value;
    }

    /// Gets the exaggeration factor for Stage 1.
    #[wasm_bindgen(getter)]
    pub fn exaggeration(&self) -> f32 {
        self.fit_ts_net.exaggeration
    }

    /// Sets the exaggeration factor for Stage 1.
    #[wasm_bindgen(setter)]
    pub fn set_exaggeration(&mut self, value: f32) {
        self.fit_ts_net.exaggeration = value;
    }

    /// Gets the number of iterations for Stage 2.
    #[wasm_bindgen(getter, js_name = "iterationsStage2")]
    pub fn iterations_stage2(&self) -> usize {
        self.fit_ts_net.iterations_stage2
    }

    /// Sets the number of iterations for Stage 2.
    #[wasm_bindgen(setter, js_name = "iterationsStage2")]
    pub fn set_iterations_stage2(&mut self, value: usize) {
        self.fit_ts_net.iterations_stage2 = value;
    }

    /// Gets the lambda_c parameter for Stage 2.
    #[wasm_bindgen(getter, js_name = "lambdaCStage2")]
    pub fn lambda_c_stage2(&self) -> f32 {
        self.fit_ts_net.lambda_c_stage2
    }

    /// Sets the lambda_c parameter for Stage 2.
    #[wasm_bindgen(setter, js_name = "lambdaCStage2")]
    pub fn set_lambda_c_stage2(&mut self, value: f32) {
        self.fit_ts_net.lambda_c_stage2 = value;
    }

    /// Gets the number of iterations for Stage 3.
    #[wasm_bindgen(getter, js_name = "iterationsStage3")]
    pub fn iterations_stage3(&self) -> usize {
        self.fit_ts_net.iterations_stage3
    }

    /// Sets the number of iterations for Stage 3.
    #[wasm_bindgen(setter, js_name = "iterationsStage3")]
    pub fn set_iterations_stage3(&mut self, value: usize) {
        self.fit_ts_net.iterations_stage3 = value;
    }

    /// Gets the lambda_c parameter for Stage 3.
    #[wasm_bindgen(getter, js_name = "lambdaCStage3")]
    pub fn lambda_c_stage3(&self) -> f32 {
        self.fit_ts_net.lambda_c_stage3
    }

    /// Sets the lambda_c parameter for Stage 3.
    #[wasm_bindgen(setter, js_name = "lambdaCStage3")]
    pub fn set_lambda_c_stage3(&mut self, value: f32) {
        self.fit_ts_net.lambda_c_stage3 = value;
    }

    /// Gets the lambda_r parameter for Stage 3.
    #[wasm_bindgen(getter, js_name = "lambdaRStage3")]
    pub fn lambda_r_stage3(&self) -> f32 {
        self.fit_ts_net.lambda_r_stage3
    }

    /// Sets the lambda_r parameter for Stage 3.
    #[wasm_bindgen(setter, js_name = "lambdaRStage3")]
    pub fn set_lambda_r_stage3(&mut self, value: f32) {
        self.fit_ts_net.lambda_r_stage3 = value;
    }

    /// Gets the learning rate.
    #[wasm_bindgen(getter, js_name = "learningRate")]
    pub fn learning_rate(&self) -> f32 {
        self.fit_ts_net.learning_rate
    }

    /// Sets the learning rate.
    #[wasm_bindgen(setter, js_name = "learningRate")]
    pub fn set_learning_rate(&mut self, value: f32) {
        self.fit_ts_net.learning_rate = value;
    }

    /// Gets the momentum parameter.
    #[wasm_bindgen(getter)]
    pub fn momentum(&self) -> f32 {
        self.fit_ts_net.momentum
    }

    /// Sets the momentum parameter.
    #[wasm_bindgen(setter)]
    pub fn set_momentum(&mut self, value: f32) {
        self.fit_ts_net.momentum = value;
    }

    /// Gets the epsilon_r parameter.
    #[wasm_bindgen(getter, js_name = "epsilonR")]
    pub fn epsilon_r(&self) -> f32 {
        self.fit_ts_net.epsilon_r
    }

    /// Sets the epsilon_r parameter.
    #[wasm_bindgen(setter, js_name = "epsilonR")]
    pub fn set_epsilon_r(&mut self, value: f32) {
        self.fit_ts_net.epsilon_r = value;
    }

    /// Runs the FIt-tsNET layout algorithm on the drawing using the provided graph and RNG.
    pub fn run(&self, drawing: &mut JsDrawingEuclidean2d, graph: &JsGraph, rng: &mut JsRng) {
        self.fit_ts_net
            .run(drawing.drawing_mut(), graph.graph(), rng.get_mut());
    }
}

impl Default for JsFitTsNet {
    fn default() -> Self {
        Self::new()
    }
}
