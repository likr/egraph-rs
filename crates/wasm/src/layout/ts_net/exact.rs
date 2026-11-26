//! Exact tsNET layout algorithm binding for WebAssembly.

use crate::{distance_matrix::JsDistanceMatrix, drawing::JsDrawingEuclidean2d};
use petgraph_layout_ts_net::TsNet;
use wasm_bindgen::prelude::*;

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

    /// Creates a new `TsNet` with tsNET* hyperparameters (`initial_l_c = 0.1`).
    #[wasm_bindgen(js_name = "newStar")]
    pub fn new_star() -> JsTsNet {
        JsTsNet {
            ts_net: TsNet::new_star(),
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

    /// Gets the total number of epochs.
    #[wasm_bindgen(getter, js_name = "nEpochs")]
    pub fn n_epochs(&self) -> usize {
        self.ts_net.n_epochs
    }

    /// Sets the total number of epochs.
    #[wasm_bindgen(setter, js_name = "nEpochs")]
    pub fn set_n_epochs(&mut self, value: usize) {
        self.ts_net.n_epochs = value;
    }

    /// Gets the initial learning rate.
    #[wasm_bindgen(getter, js_name = "initialLr")]
    pub fn initial_lr(&self) -> f32 {
        self.ts_net.initial_lr
    }

    /// Sets the initial learning rate.
    #[wasm_bindgen(setter, js_name = "initialLr")]
    pub fn set_initial_lr(&mut self, value: f32) {
        self.ts_net.initial_lr = value;
    }

    /// Gets the final learning rate.
    #[wasm_bindgen(getter, js_name = "finalLr")]
    pub fn final_lr(&self) -> f32 {
        self.ts_net.final_lr
    }

    /// Sets the final learning rate.
    #[wasm_bindgen(setter, js_name = "finalLr")]
    pub fn set_final_lr(&mut self, value: f32) {
        self.ts_net.final_lr = value;
    }

    /// Gets the learning rate switch epoch.
    #[wasm_bindgen(getter, js_name = "lrSwitch")]
    pub fn lr_switch(&self) -> Option<usize> {
        self.ts_net.lr_switch
    }

    /// Sets the learning rate switch epoch.
    #[wasm_bindgen(setter, js_name = "lrSwitch")]
    pub fn set_lr_switch(&mut self, value: Option<usize>) {
        self.ts_net.lr_switch = value;
    }

    /// Gets the initial momentum.
    #[wasm_bindgen(getter, js_name = "initialMomentum")]
    pub fn initial_momentum(&self) -> f32 {
        self.ts_net.initial_momentum
    }

    /// Sets the initial momentum.
    #[wasm_bindgen(setter, js_name = "initialMomentum")]
    pub fn set_initial_momentum(&mut self, value: f32) {
        self.ts_net.initial_momentum = value;
    }

    /// Gets the final momentum.
    #[wasm_bindgen(getter, js_name = "finalMomentum")]
    pub fn final_momentum(&self) -> f32 {
        self.ts_net.final_momentum
    }

    /// Sets the final momentum.
    #[wasm_bindgen(setter, js_name = "finalMomentum")]
    pub fn set_final_momentum(&mut self, value: f32) {
        self.ts_net.final_momentum = value;
    }

    /// Gets the momentum switch epoch.
    #[wasm_bindgen(getter, js_name = "momentumSwitch")]
    pub fn momentum_switch(&self) -> Option<usize> {
        self.ts_net.momentum_switch
    }

    /// Sets the momentum switch epoch.
    #[wasm_bindgen(setter, js_name = "momentumSwitch")]
    pub fn set_momentum_switch(&mut self, value: Option<usize>) {
        self.ts_net.momentum_switch = value;
    }

    /// Gets the initial lambda_kl weight.
    #[wasm_bindgen(getter, js_name = "initialLKl")]
    pub fn initial_l_kl(&self) -> Option<f32> {
        self.ts_net.initial_l_kl
    }

    /// Sets the initial lambda_kl weight.
    #[wasm_bindgen(setter, js_name = "initialLKl")]
    pub fn set_initial_l_kl(&mut self, value: Option<f32>) {
        self.ts_net.initial_l_kl = value;
    }

    /// Gets the final lambda_kl weight.
    #[wasm_bindgen(getter, js_name = "finalLKl")]
    pub fn final_l_kl(&self) -> Option<f32> {
        self.ts_net.final_l_kl
    }

    /// Sets the final lambda_kl weight.
    #[wasm_bindgen(setter, js_name = "finalLKl")]
    pub fn set_final_l_kl(&mut self, value: Option<f32>) {
        self.ts_net.final_l_kl = value;
    }

    /// Gets the lambda_kl switch epoch.
    #[wasm_bindgen(getter, js_name = "lKlSwitch")]
    pub fn l_kl_switch(&self) -> Option<usize> {
        self.ts_net.l_kl_switch
    }

    /// Sets the lambda_kl switch epoch.
    #[wasm_bindgen(setter, js_name = "lKlSwitch")]
    pub fn set_l_kl_switch(&mut self, value: Option<usize>) {
        self.ts_net.l_kl_switch = value;
    }

    /// Gets the initial lambda_c weight.
    #[wasm_bindgen(getter, js_name = "initialLC")]
    pub fn initial_l_c(&self) -> Option<f32> {
        self.ts_net.initial_l_c
    }

    /// Sets the initial lambda_c weight.
    #[wasm_bindgen(setter, js_name = "initialLC")]
    pub fn set_initial_l_c(&mut self, value: Option<f32>) {
        self.ts_net.initial_l_c = value;
    }

    /// Gets the final lambda_c weight.
    #[wasm_bindgen(getter, js_name = "finalLC")]
    pub fn final_l_c(&self) -> Option<f32> {
        self.ts_net.final_l_c
    }

    /// Sets the final lambda_c weight.
    #[wasm_bindgen(setter, js_name = "finalLC")]
    pub fn set_final_l_c(&mut self, value: Option<f32>) {
        self.ts_net.final_l_c = value;
    }

    /// Gets the lambda_c switch epoch.
    #[wasm_bindgen(getter, js_name = "lCSwitch")]
    pub fn l_c_switch(&self) -> Option<usize> {
        self.ts_net.l_c_switch
    }

    /// Sets the lambda_c switch epoch.
    #[wasm_bindgen(setter, js_name = "lCSwitch")]
    pub fn set_l_c_switch(&mut self, value: Option<usize>) {
        self.ts_net.l_c_switch = value;
    }

    /// Gets the initial lambda_r weight.
    #[wasm_bindgen(getter, js_name = "initialLR")]
    pub fn initial_l_r(&self) -> Option<f32> {
        self.ts_net.initial_l_r
    }

    /// Sets the initial lambda_r weight.
    #[wasm_bindgen(setter, js_name = "initialLR")]
    pub fn set_initial_l_r(&mut self, value: Option<f32>) {
        self.ts_net.initial_l_r = value;
    }

    /// Gets the final lambda_r weight.
    #[wasm_bindgen(getter, js_name = "finalLR")]
    pub fn final_l_r(&self) -> Option<f32> {
        self.ts_net.final_l_r
    }

    /// Sets the final lambda_r weight.
    #[wasm_bindgen(setter, js_name = "finalLR")]
    pub fn set_final_l_r(&mut self, value: Option<f32>) {
        self.ts_net.final_l_r = value;
    }

    /// Gets the lambda_r switch epoch.
    #[wasm_bindgen(getter, js_name = "lRSwitch")]
    pub fn l_r_switch(&self) -> Option<usize> {
        self.ts_net.l_r_switch
    }

    /// Sets the lambda_r switch epoch.
    #[wasm_bindgen(setter, js_name = "lRSwitch")]
    pub fn set_l_r_switch(&mut self, value: Option<usize>) {
        self.ts_net.l_r_switch = value;
    }

    /// Gets the r_eps parameter.
    #[wasm_bindgen(getter, js_name = "rEps")]
    pub fn r_eps(&self) -> f32 {
        self.ts_net.r_eps
    }

    /// Sets the r_eps parameter.
    #[wasm_bindgen(setter, js_name = "rEps")]
    pub fn set_r_eps(&mut self, value: f32) {
        self.ts_net.r_eps = value;
    }

    /// Gets the sigma_iters parameter.
    #[wasm_bindgen(getter, js_name = "sigmaIters")]
    pub fn sigma_iters(&self) -> usize {
        self.ts_net.sigma_iters
    }

    /// Sets the sigma_iters parameter.
    #[wasm_bindgen(setter, js_name = "sigmaIters")]
    pub fn set_sigma_iters(&mut self, value: usize) {
        self.ts_net.sigma_iters = value;
    }

    /// Gets the autostop convergence tolerance.
    #[wasm_bindgen(getter)]
    pub fn autostop(&self) -> f32 {
        self.ts_net.autostop
    }

    /// Sets the autostop convergence tolerance.
    #[wasm_bindgen(setter)]
    pub fn set_autostop(&mut self, value: f32) {
        self.ts_net.autostop = value;
    }

    /// Gets the autostop convergence window size.
    #[wasm_bindgen(getter, js_name = "windowSize")]
    pub fn window_size(&self) -> usize {
        self.ts_net.window_size
    }

    /// Sets the autostop convergence window size.
    #[wasm_bindgen(setter, js_name = "windowSize")]
    pub fn set_window_size(&mut self, value: usize) {
        self.ts_net.window_size = value;
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
