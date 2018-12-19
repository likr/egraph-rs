use super::super::super::graph::{Edge, EdgeType, Graph, IndexType, Node};
use super::force::force::Force;
use js_sys::{Array, Object, Reflect};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Simulation {
    simulation: egraph::layout::force_directed::Simulation<Node, Edge, EdgeType, IndexType>,
}

#[wasm_bindgen]
impl Simulation {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Simulation {
        Simulation {
            simulation: egraph::layout::force_directed::Simulation::new(),
        }
    }

    pub fn add(&mut self, force: &Force) {
        self.simulation.add(force.force());
    }

    pub fn start(&mut self, graph: &Graph) -> JsValue {
        let mut points =
            egraph::layout::force_directed::initial_placement(graph.graph().node_count());
        let mut context = self.simulation.build(&graph.graph());
        context.start(&mut points);
        let array = Array::new();
        for point in points.iter() {
            let obj = Object::new();
            Reflect::set(&obj, &"x".into(), &point.x.into())
                .ok()
                .unwrap();
            Reflect::set(&obj, &"y".into(), &point.y.into())
                .ok()
                .unwrap();
            array.push(&obj);
        }
        array.into()
    }
}
