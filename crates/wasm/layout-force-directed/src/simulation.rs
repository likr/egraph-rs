use crate::force::Force;
use egraph::layout::force_directed::{initial_placement, Point, Simulation as EgSimulation};
use egraph::Graph;
use egraph_wasm_adapter::{JsGraph, JsGraphAdapter};
use wasm_bindgen::prelude::*;

#[derive(Serialize, Deserialize)]
pub struct PointGeometry {
    pub x: f32,
    pub y: f32,
}

#[derive(Serialize, Deserialize)]
pub struct NodeGeometry {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Serialize, Deserialize)]
pub struct LinkGeometry {
    pub bends: Vec<PointGeometry>,
}

#[derive(Serialize, Deserialize)]
pub struct GraphGeometry {
    pub nodes: Vec<NodeGeometry>,
    pub links: Vec<LinkGeometry>,
}

#[wasm_bindgen]
extern "C" {
    pub type JsForce;

    #[wasm_bindgen(method, structural)]
    fn force(this: &JsForce) -> Force;
}

#[wasm_bindgen]
pub struct Simulation {
    simulation: EgSimulation<JsGraph>,
}

impl Simulation {
    pub fn simulation(&self) -> &EgSimulation<JsGraph> {
        &self.simulation
    }
}

#[wasm_bindgen]
impl Simulation {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Simulation {
        Simulation {
            simulation: EgSimulation::new(),
        }
    }

    pub fn add(&mut self, force: &JsForce) {
        self.simulation.add(force.force().force());
    }

    pub fn start(&mut self, graph: JsGraph, initial_points: JsValue) -> JsValue {
        let graph = JsGraphAdapter::new(graph);
        let mut points = if initial_points.is_null() || initial_points.is_undefined() {
            initial_placement(graph.node_count())
        } else {
            initial_points
                .into_serde::<GraphGeometry>()
                .unwrap()
                .nodes
                .iter()
                .map(|p| Point::new(p.x, p.y))
                .collect::<Vec<_>>()
        };
        let mut context = self.simulation.build(&graph);
        context.start(&mut points);

        let result = GraphGeometry {
            nodes: graph
                .nodes()
                .map(|i| NodeGeometry {
                    x: points[i].x,
                    y: points[i].y,
                    width: 0.0,
                    height: 0.0,
                })
                .collect(),
            links: graph
                .edges()
                .map(|_| LinkGeometry { bends: Vec::new() })
                .collect(),
        };
        JsValue::from_serde(&result).unwrap()
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
