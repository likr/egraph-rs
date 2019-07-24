use crate::layout::force_directed::force::JsForce;
use egraph::layout::force_directed::{Simulation, SimulationBuilder};
use egraph_wasm_adapter::{JsGraph, JsGraphAdapter};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    pub type ForceObject;

    #[wasm_bindgen(method, structural)]
    fn force(this: &ForceObject) -> JsForce;
}

#[wasm_bindgen(js_name = Simulation)]
pub struct JsSimulation {
    simulation: Simulation,
}

impl JsSimulation {
    pub fn new(simulation: Simulation) -> JsSimulation {
        JsSimulation { simulation }
    }
}

#[wasm_bindgen(js_class = Simulation)]
impl JsSimulation {
    pub fn run(&mut self) {
        self.simulation.run();
    }

    pub fn step(&mut self) {
        self.simulation.step();
    }

    #[wasm_bindgen(js_name = stepN)]
    pub fn step_n(&mut self, n: usize) {
        self.simulation.step_n(n);
    }

    #[wasm_bindgen(js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        self.simulation.is_finished()
    }

    pub fn reset(&mut self, alpha_start: f32) {
        self.simulation.reset(alpha_start);
    }

    pub fn x(&self, u: usize) -> f32 {
        self.simulation.x(u)
    }

    pub fn y(&self, u: usize) -> f32 {
        self.simulation.y(u)
    }
}

#[wasm_bindgen(js_name = SimulationBuilder)]
pub struct JsSimulationBuilder {
    builder: SimulationBuilder<JsGraph, JsGraphAdapter>,
    forces: HashMap<usize, ForceObject>,
}

impl JsSimulationBuilder {
    pub fn builder(&self) -> &SimulationBuilder<JsGraph, JsGraphAdapter> {
        &self.builder
    }
}

#[wasm_bindgen(js_class = SimulationBuilder)]
impl JsSimulationBuilder {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsSimulationBuilder {
        JsSimulationBuilder {
            builder: SimulationBuilder::new(),
            forces: HashMap::new(),
        }
    }

    #[wasm_bindgen(js_name = defaultConnected)]
    pub fn default_connected() -> JsSimulationBuilder {
        JsSimulationBuilder {
            builder: SimulationBuilder::default_connected(),
            forces: HashMap::new(),
        }
    }

    #[wasm_bindgen(js_name = defaultNonConnected)]
    pub fn default_non_connected() -> JsSimulationBuilder {
        JsSimulationBuilder {
            builder: SimulationBuilder::default_non_connected(),
            forces: HashMap::new(),
        }
    }

    pub fn add(&mut self, force: ForceObject) -> usize {
        let index = self.builder.add(force.force().force());
        self.forces.insert(index, force);
        index
    }

    pub fn get(&self, index: usize) -> JsValue {
        self.forces[&index].clone()
    }

    pub fn remove(&mut self, index: usize) -> Option<ForceObject> {
        if let Some(_) = self.builder.remove(index) {
            self.forces.remove(&index)
        } else {
            None
        }
    }

    pub fn build(&mut self, graph: JsGraph) -> JsSimulation {
        let graph = JsGraphAdapter::new(graph);
        JsSimulation::new(self.builder.build(&graph))
    }

    #[wasm_bindgen(getter = alphaStart)]
    pub fn alpha_start(&mut self) -> f32 {
        self.builder.alpha_start
    }

    #[wasm_bindgen(setter = alphaStart)]
    pub fn set_alpha_start(&mut self, value: f32) {
        self.builder.alpha_start = value;
    }

    #[wasm_bindgen(getter = alphaMin)]
    pub fn alpha_min(&mut self) -> f32 {
        self.builder.alpha_min
    }

    #[wasm_bindgen(setter = alphaMin)]
    pub fn set_alpha_min(&mut self, value: f32) {
        self.builder.alpha_min = value;
    }

    #[wasm_bindgen(getter = alphaTarget)]
    pub fn alpha_target(&mut self) -> f32 {
        self.builder.alpha_target
    }

    #[wasm_bindgen(setter = alphaTarget)]
    pub fn set_alpha_target(&mut self, value: f32) {
        self.builder.alpha_target = value;
    }

    #[wasm_bindgen(getter = velocityDecay)]
    pub fn velocity_decay(&mut self) -> f32 {
        self.builder.velocity_decay
    }

    #[wasm_bindgen(setter = velocityDecay)]
    pub fn set_velocity_decay(&mut self, value: f32) {
        self.builder.velocity_decay = value;
    }

    #[wasm_bindgen(getter = iterations)]
    pub fn iterations(&mut self) -> usize {
        self.builder.iterations
    }

    #[wasm_bindgen(setter = iterations)]
    pub fn set_iterations(&mut self, value: usize) {
        self.builder.iterations = value;
    }
}
