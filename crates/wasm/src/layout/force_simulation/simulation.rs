use js_sys::Function;
use petgraph_layout_force_simulation::Simulation;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = Simulation)]
pub struct JsSimulation {
    simulation: Simulation,
}

#[wasm_bindgen(js_class = Simulation)]
impl JsSimulation {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsSimulation {
        JsSimulation {
            simulation: Simulation::new(),
        }
    }

    pub fn run(&mut self, f: &Function) {
        self.simulation.run(&mut |alpha| {
            f.call1(&JsValue::null(), &(alpha as f64).into()).ok();
        })
    }

    #[wasm_bindgen(js_name = runStep)]
    pub fn run_step(&mut self, n: usize, f: &Function) {
        self.simulation.run_step(n, &mut |alpha| {
            f.call1(&JsValue::null(), &(alpha as f64).into()).ok();
        })
    }

    #[wasm_bindgen(js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        self.simulation.is_finished()
    }

    pub fn reset(&mut self, alpha_start: f32) {
        self.simulation.reset(alpha_start);
    }

    #[wasm_bindgen(getter = alphaStart)]
    pub fn alpha(&mut self) -> f32 {
        self.simulation.alpha
    }

    #[wasm_bindgen(setter = alphaStart)]
    pub fn set_alpha(&mut self, value: f32) {
        self.simulation.alpha = value;
    }

    #[wasm_bindgen(getter = alphaMin)]
    pub fn alpha_min(&mut self) -> f32 {
        self.simulation.alpha_min
    }

    #[wasm_bindgen(setter = alphaMin)]
    pub fn set_alpha_min(&mut self, value: f32) {
        self.simulation.alpha_min = value;
    }

    #[wasm_bindgen(getter = alphaTarget)]
    pub fn alpha_target(&mut self) -> f32 {
        self.simulation.alpha_target
    }

    #[wasm_bindgen(setter = alphaTarget)]
    pub fn set_alpha_target(&mut self, value: f32) {
        self.simulation.alpha_target = value;
    }

    #[wasm_bindgen(getter = iterations)]
    pub fn iterations(&mut self) -> usize {
        self.simulation.iterations
    }

    #[wasm_bindgen(setter = iterations)]
    pub fn set_iterations(&mut self, value: usize) {
        self.simulation.iterations = value;
    }
}
