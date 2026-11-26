//! Builder WebAssembly binding for the exact tsNET layout algorithm.

use super::exact::JsTsNet;
use petgraph_layout_ts_net::TsNetBuilder;
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

    /// Creates a new `TsNetBuilder` initialized with tsNET* hyperparameters (`initial_l_c = 0.1`).
    #[wasm_bindgen(js_name = "newStar")]
    pub fn new_star() -> Self {
        Self {
            builder: TsNetBuilder::new_star(),
        }
    }

    /// Sets the target perplexity.
    pub fn perplexity(mut self, value: f32) -> Self {
        self.builder = self.builder.perplexity(value);
        self
    }

    /// Sets the total number of epochs.
    #[wasm_bindgen(js_name = "nEpochs")]
    pub fn n_epochs(mut self, value: usize) -> Self {
        self.builder = self.builder.n_epochs(value);
        self
    }

    /// Sets the initial learning rate.
    #[wasm_bindgen(js_name = "initialLr")]
    pub fn initial_lr(mut self, value: f32) -> Self {
        self.builder = self.builder.initial_lr(value);
        self
    }

    /// Sets the final learning rate.
    #[wasm_bindgen(js_name = "finalLr")]
    pub fn final_lr(mut self, value: f32) -> Self {
        self.builder = self.builder.final_lr(value);
        self
    }

    /// Sets the epoch at which learning rate switches.
    #[wasm_bindgen(js_name = "lrSwitch")]
    pub fn lr_switch(mut self, value: Option<usize>) -> Self {
        self.builder = self.builder.lr_switch(value);
        self
    }

    /// Sets both the initial and final learning rate.
    #[wasm_bindgen(js_name = "learningRate")]
    pub fn learning_rate(mut self, value: f32) -> Self {
        self.builder = self.builder.learning_rate(value);
        self
    }

    /// Sets the initial momentum.
    #[wasm_bindgen(js_name = "initialMomentum")]
    pub fn initial_momentum(mut self, value: f32) -> Self {
        self.builder = self.builder.initial_momentum(value);
        self
    }

    /// Sets the final momentum.
    #[wasm_bindgen(js_name = "finalMomentum")]
    pub fn final_momentum(mut self, value: f32) -> Self {
        self.builder = self.builder.final_momentum(value);
        self
    }

    /// Sets the epoch at which momentum switches.
    #[wasm_bindgen(js_name = "momentumSwitch")]
    pub fn momentum_switch(mut self, value: Option<usize>) -> Self {
        self.builder = self.builder.momentum_switch(value);
        self
    }

    /// Sets both the initial and final momentum parameter.
    pub fn momentum(mut self, value: f32) -> Self {
        self.builder = self.builder.momentum(value);
        self
    }

    /// Sets the initial KL divergence weight.
    #[wasm_bindgen(js_name = "initialLKl")]
    pub fn initial_l_kl(mut self, value: Option<f32>) -> Self {
        self.builder = self.builder.initial_l_kl(value);
        self
    }

    /// Sets the final KL divergence weight.
    #[wasm_bindgen(js_name = "finalLKl")]
    pub fn final_l_kl(mut self, value: Option<f32>) -> Self {
        self.builder = self.builder.final_l_kl(value);
        self
    }

    /// Sets the epoch at which lambda_kl switches.
    #[wasm_bindgen(js_name = "lKlSwitch")]
    pub fn l_kl_switch(mut self, value: Option<usize>) -> Self {
        self.builder = self.builder.l_kl_switch(value);
        self
    }

    /// Sets the initial compression weight.
    #[wasm_bindgen(js_name = "initialLC")]
    pub fn initial_l_c(mut self, value: Option<f32>) -> Self {
        self.builder = self.builder.initial_l_c(value);
        self
    }

    /// Sets the final compression weight.
    #[wasm_bindgen(js_name = "finalLC")]
    pub fn final_l_c(mut self, value: Option<f32>) -> Self {
        self.builder = self.builder.final_l_c(value);
        self
    }

    /// Sets the epoch at which lambda_c switches.
    #[wasm_bindgen(js_name = "lCSwitch")]
    pub fn l_c_switch(mut self, value: Option<usize>) -> Self {
        self.builder = self.builder.l_c_switch(value);
        self
    }

    /// Sets the initial repulsion weight.
    #[wasm_bindgen(js_name = "initialLR")]
    pub fn initial_l_r(mut self, value: Option<f32>) -> Self {
        self.builder = self.builder.initial_l_r(value);
        self
    }

    /// Sets the final repulsion weight.
    #[wasm_bindgen(js_name = "finalLR")]
    pub fn final_l_r(mut self, value: Option<f32>) -> Self {
        self.builder = self.builder.final_l_r(value);
        self
    }

    /// Sets the epoch at which lambda_r switches.
    #[wasm_bindgen(js_name = "lRSwitch")]
    pub fn l_r_switch(mut self, value: Option<usize>) -> Self {
        self.builder = self.builder.l_r_switch(value);
        self
    }

    /// Sets the repulsion offset parameter `r_eps`.
    #[wasm_bindgen(js_name = "rEps")]
    pub fn r_eps(mut self, value: f32) -> Self {
        self.builder = self.builder.r_eps(value);
        self
    }

    /// Sets the maximum binary search iterations for finding node sigma_i.
    #[wasm_bindgen(js_name = "sigmaIters")]
    pub fn sigma_iters(mut self, value: usize) -> Self {
        self.builder = self.builder.sigma_iters(value);
        self
    }

    /// Sets the autostop convergence tolerance threshold.
    pub fn autostop(mut self, value: f32) -> Self {
        self.builder = self.builder.autostop(value);
        self
    }

    /// Sets the sliding window size for autostop convergence checking.
    #[wasm_bindgen(js_name = "windowSize")]
    pub fn window_size(mut self, value: usize) -> Self {
        self.builder = self.builder.window_size(value);
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
