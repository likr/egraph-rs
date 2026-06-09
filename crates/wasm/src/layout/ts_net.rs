//! tsNET layout algorithm bindings for WebAssembly.
//!
//! This module provides WebAssembly bindings for the tsNET graph layout algorithm.

use crate::{distance_matrix::JsDistanceMatrix, drawing::JsDrawingEuclidean2d};
use petgraph_layout_ts_net::TsNet;
use wasm_bindgen::prelude::*;

/// WebAssembly binding for the tsNET layout algorithm.
#[wasm_bindgen(js_name = "TsNet")]
pub struct JsTsNet {
    ts_net: TsNet<f32>,
}

#[wasm_bindgen(js_class = "TsNet")]
impl JsTsNet {
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

    /// Gets the epsilon_d parameter.
    #[wasm_bindgen(getter, js_name = "epsilonD")]
    pub fn epsilon_d(&self) -> f32 {
        self.ts_net.epsilon_d
    }

    /// Sets the epsilon_d parameter.
    #[wasm_bindgen(setter, js_name = "epsilonD")]
    pub fn set_epsilon_d(&mut self, value: f32) {
        self.ts_net.epsilon_d = value;
    }

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
