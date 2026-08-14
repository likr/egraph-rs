//! GUMAP (Graph UMAP) layout algorithm bindings for WebAssembly.
//!
//! This module provides WebAssembly bindings for the GUMAP graph layout algorithm and its builder.

use crate::{distance_matrix::JsDistanceMatrix, drawing::JsDrawingEuclidean2d, rng::JsRng};
use petgraph_layout_gumap::{GuMap, GuMapBuilder};
use wasm_bindgen::prelude::*;

/// WebAssembly binding for constructing a `GuMap` (Graph UMAP) algorithm instance via the Builder pattern.
#[wasm_bindgen(js_name = "GuMapBuilder")]
pub struct JsGuMapBuilder {
    builder: GuMapBuilder<f32>,
}

#[wasm_bindgen(js_class = "GuMapBuilder")]
impl JsGuMapBuilder {
    /// Creates a new `GuMapBuilder` with default hyperparameters.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            builder: GuMapBuilder::new(),
        }
    }

    /// Sets the number of nearest neighbors `k` (default: 15).
    pub fn k(mut self, value: usize) -> Self {
        self.builder = self.builder.k(value);
        self
    }

    /// Sets the minimum distance in low-dimensional space (default: 0.1).
    #[wasm_bindgen(js_name = "minDist")]
    pub fn min_dist(mut self, value: f32) -> Self {
        self.builder = self.builder.min_dist(value);
        self
    }

    /// Sets the scale of spread in low-dimensional space (default: 1.0).
    pub fn spread(mut self, value: f32) -> Self {
        self.builder = self.builder.spread(value);
        self
    }

    /// Sets the number of SGD optimization epochs (default: 300).
    #[wasm_bindgen(js_name = "nEpochs")]
    pub fn n_epochs(mut self, value: usize) -> Self {
        self.builder = self.builder.n_epochs(value);
        self
    }

    /// Sets the number of negative samples per attractive edge (default: 5).
    #[wasm_bindgen(js_name = "negativeSampleRate")]
    pub fn negative_sample_rate(mut self, value: usize) -> Self {
        self.builder = self.builder.negative_sample_rate(value);
        self
    }

    /// Sets the initial learning rate (default: 1.0).
    #[wasm_bindgen(js_name = "learningRate")]
    pub fn learning_rate(mut self, value: f32) -> Self {
        self.builder = self.builder.learning_rate(value);
        self
    }

    /// Explicitly sets the `a` parameter for low-dimensional probability $1 / (1 + a d^{2b})$.
    pub fn a(mut self, value: f32) -> Self {
        self.builder = self.builder.a(value);
        self
    }

    /// Explicitly sets the `b` parameter for low-dimensional probability $1 / (1 + a d^{2b})$.
    pub fn b(mut self, value: f32) -> Self {
        self.builder = self.builder.b(value);
        self
    }

    /// Sets the maximum binary search iterations for finding node sigma_i (default: 64).
    #[wasm_bindgen(js_name = "sigmaIters")]
    pub fn sigma_iters(mut self, value: usize) -> Self {
        self.builder = self.builder.sigma_iters(value);
        self
    }

    /// Sets the tolerance threshold for perplexity binary search convergence (default: 1e-5).
    #[wasm_bindgen(js_name = "sigmaTolerance")]
    pub fn sigma_tolerance(mut self, value: f32) -> Self {
        self.builder = self.builder.sigma_tolerance(value);
        self
    }

    /// Builds a configured `GuMap` layout instance.
    pub fn build(&self) -> Result<JsGuMap, JsValue> {
        let gumap = self
            .builder
            .clone()
            .build()
            .map_err(|e| JsValue::from_str(&e))?;
        Ok(JsGuMap { gumap })
    }
}

impl Default for JsGuMapBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// WebAssembly binding for the GUMAP (Graph UMAP) layout algorithm.
#[wasm_bindgen(js_name = "GuMap")]
pub struct JsGuMap {
    pub(crate) gumap: GuMap<f32>,
}

#[wasm_bindgen(js_class = "GuMap")]
impl JsGuMap {
    /// Creates a new `GuMap` with default hyperparameters.
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsGuMap {
        JsGuMap {
            gumap: GuMap::new(),
        }
    }

    /// Gets the number of nearest neighbors k.
    #[wasm_bindgen(getter)]
    pub fn k(&self) -> usize {
        self.gumap.k
    }

    /// Sets the number of nearest neighbors k.
    #[wasm_bindgen(setter)]
    pub fn set_k(&mut self, value: usize) {
        self.gumap.k = value;
    }

    /// Gets the minimum distance in low-dimensional space.
    #[wasm_bindgen(getter, js_name = "minDist")]
    pub fn min_dist(&self) -> f32 {
        self.gumap.min_dist
    }

    /// Sets the minimum distance in low-dimensional space.
    #[wasm_bindgen(setter, js_name = "minDist")]
    pub fn set_min_dist(&mut self, value: f32) {
        self.gumap.min_dist = value;
    }

    /// Gets the spread scale.
    #[wasm_bindgen(getter)]
    pub fn spread(&self) -> f32 {
        self.gumap.spread
    }

    /// Sets the spread scale.
    #[wasm_bindgen(setter)]
    pub fn set_spread(&mut self, value: f32) {
        self.gumap.spread = value;
    }

    /// Gets the number of SGD epochs.
    #[wasm_bindgen(getter, js_name = "nEpochs")]
    pub fn n_epochs(&self) -> usize {
        self.gumap.n_epochs
    }

    /// Sets the number of SGD epochs.
    #[wasm_bindgen(setter, js_name = "nEpochs")]
    pub fn set_n_epochs(&mut self, value: usize) {
        self.gumap.n_epochs = value;
    }

    /// Gets the negative sample rate.
    #[wasm_bindgen(getter, js_name = "negativeSampleRate")]
    pub fn negative_sample_rate(&self) -> usize {
        self.gumap.negative_sample_rate
    }

    /// Sets the negative sample rate.
    #[wasm_bindgen(setter, js_name = "negativeSampleRate")]
    pub fn set_negative_sample_rate(&mut self, value: usize) {
        self.gumap.negative_sample_rate = value;
    }

    /// Gets the initial learning rate.
    #[wasm_bindgen(getter, js_name = "learningRate")]
    pub fn learning_rate(&self) -> f32 {
        self.gumap.learning_rate
    }

    /// Sets the initial learning rate.
    #[wasm_bindgen(setter, js_name = "learningRate")]
    pub fn set_learning_rate(&mut self, value: f32) {
        self.gumap.learning_rate = value;
    }

    /// Gets the `a` parameter.
    #[wasm_bindgen(getter)]
    pub fn a(&self) -> f32 {
        self.gumap.a
    }

    /// Sets the `a` parameter.
    #[wasm_bindgen(setter)]
    pub fn set_a(&mut self, value: f32) {
        self.gumap.a = value;
    }

    /// Gets the `b` parameter.
    #[wasm_bindgen(getter)]
    pub fn b(&self) -> f32 {
        self.gumap.b
    }

    /// Sets the `b` parameter.
    #[wasm_bindgen(setter)]
    pub fn set_b(&mut self, value: f32) {
        self.gumap.b = value;
    }

    /// Runs the GUMAP layout algorithm on the drawing using the provided distance matrix and RNG.
    pub fn run(
        &self,
        drawing: &mut JsDrawingEuclidean2d,
        distance_matrix: &JsDistanceMatrix,
        rng: &mut JsRng,
    ) {
        self.gumap
            .run(drawing.drawing_mut(), &distance_matrix.inner, rng.get_mut());
    }
}

impl Default for JsGuMap {
    fn default() -> Self {
        Self::new()
    }
}
