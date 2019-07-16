use crate::layout::force_directed::force::JsForce;
use egraph::layout::force_directed::force::{LinkForce, ManyBodyForce, PositionForce};
use egraph::layout::force_directed::{initial_placement, Point, Simulation, SimulationContext};
use egraph_wasm_adapter::{JsGraph, JsGraphAdapter};
use js_sys::try_iter;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct NodeGeometry {
    indices: HashMap<usize, usize>,
    points: Vec<Point>,
}

#[wasm_bindgen]
impl NodeGeometry {
    #[wasm_bindgen(constructor)]
    pub fn new(graph: JsGraph) -> NodeGeometry {
        let indices = try_iter(&graph.nodes())
            .unwrap()
            .unwrap()
            .enumerate()
            .map(|(i, obj)| (obj.unwrap().as_f64().unwrap() as usize, i))
            .collect::<HashMap<_, _>>();
        let points = initial_placement(graph.node_count());
        NodeGeometry { indices, points }
    }

    pub fn x(&self, u: usize) -> f32 {
        self.points[self.indices[&u]].x
    }

    pub fn y(&self, u: usize) -> f32 {
        self.points[self.indices[&u]].y
    }
}

#[wasm_bindgen]
extern "C" {
    pub type ForceObject;

    #[wasm_bindgen(method, structural)]
    fn force(this: &ForceObject) -> JsForce;
}

#[wasm_bindgen(js_name = SimulationContext)]
pub struct JsSimulationContext {
    context: SimulationContext,
}

impl JsSimulationContext {
    pub fn new(context: SimulationContext) -> JsSimulationContext {
        JsSimulationContext { context }
    }
}

#[wasm_bindgen(js_class = SimulationContext)]
impl JsSimulationContext {
    pub fn start(&mut self, points: &mut NodeGeometry) {
        self.context.start(&mut points.points);
    }

    pub fn step(&mut self, points: &mut NodeGeometry) {
        self.context.step(&mut points.points);
    }

    #[wasm_bindgen(js_name = isFinished)]
    pub fn is_finished(&self) -> bool {
        self.context.is_finished()
    }
}

#[wasm_bindgen(js_name = Simulation)]
pub struct JsSimulation {
    simulation: Simulation<JsGraph, JsGraphAdapter>,
}

impl JsSimulation {
    pub fn simulation(&self) -> &Simulation<JsGraph, JsGraphAdapter> {
        &self.simulation
    }
}

#[wasm_bindgen(js_class = Simulation)]
impl JsSimulation {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsSimulation {
        JsSimulation {
            simulation: Simulation::new(),
        }
    }

    pub fn basic() -> JsSimulation {
        let mut simulation = Simulation::new();
        let many_body_force = ManyBodyForce::new();
        simulation.add(Rc::new(RefCell::new(many_body_force)));
        let link_force = LinkForce::new();
        simulation.add(Rc::new(RefCell::new(link_force)));
        let mut position_force = PositionForce::new();
        position_force.x = Box::new(|_, _| Some(0.));
        position_force.y = Box::new(|_, _| Some(0.));
        simulation.add(Rc::new(RefCell::new(position_force)));
        JsSimulation { simulation }
    }

    pub fn add(&mut self, force: &ForceObject) {
        self.simulation.add(force.force().force());
    }

    pub fn build(&mut self, graph: JsGraph) -> JsSimulationContext {
        let graph = JsGraphAdapter::new(graph);
        JsSimulationContext::new(self.simulation.build(&graph))
    }

    #[wasm_bindgen(getter = alphaStart)]
    pub fn alpha_start(&mut self) -> f32 {
        self.simulation.alpha_start
    }

    #[wasm_bindgen(setter = alphaStart)]
    pub fn set_alpha_start(&mut self, value: f32) {
        self.simulation.alpha_start = value;
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

    #[wasm_bindgen(getter = velocityDecay)]
    pub fn velocity_decay(&mut self) -> f32 {
        self.simulation.velocity_decay
    }

    #[wasm_bindgen(setter = velocityDecay)]
    pub fn set_velocity_decay(&mut self, value: f32) {
        self.simulation.velocity_decay = value;
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
