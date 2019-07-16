use crate::layout::force_directed::force::JsForce;
use egraph::layout::force_directed::force::{LinkForce, ManyBodyForce, PositionForce};
use egraph::layout::force_directed::{Simulation, SimulationBuilder};
use egraph_wasm_adapter::{JsGraph, JsGraphAdapter};
use std::cell::RefCell;
use std::rc::Rc;
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
        }
    }

    #[wasm_bindgen(js_name = defaultSetting)]
    pub fn default_setting() -> JsSimulationBuilder {
        let mut builder = SimulationBuilder::new();
        let many_body_force = ManyBodyForce::new();
        builder.add(Rc::new(RefCell::new(many_body_force)));
        let link_force = LinkForce::new();
        builder.add(Rc::new(RefCell::new(link_force)));
        let mut position_force = PositionForce::new();
        position_force.x = Box::new(|_, _| Some(0.));
        position_force.y = Box::new(|_, _| Some(0.));
        builder.add(Rc::new(RefCell::new(position_force)));
        JsSimulationBuilder { builder }
    }

    pub fn add(&mut self, force: &ForceObject) {
        self.builder.add(force.force().force());
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
