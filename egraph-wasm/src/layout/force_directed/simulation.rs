use super::super::super::graph::{Edge, EdgeType, Graph, IndexType, Node};
use super::force::force::Force;
use js_sys::{Array, Object, Reflect};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Simulation {
    simulation: egraph::layout::force_directed::Simulation<Node, Edge, EdgeType, IndexType>,
}

impl Simulation {
    pub fn simulation(
        &self,
    ) -> &egraph::layout::force_directed::Simulation<Node, Edge, EdgeType, IndexType> {
        &self.simulation
    }
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

    pub fn start(&mut self, graph: &Graph, initial_points: JsValue) -> JsValue {
        let mut points = if initial_points.is_null() || initial_points.is_undefined() {
            egraph::layout::force_directed::initial_placement(graph.graph().node_count())
        } else {
            let a: Array = initial_points.into();
            a.values()
                .into_iter()
                .map(|p| {
                    let p = p.ok().unwrap();
                    if p.is_null() {
                        egraph::layout::force_directed::force::Point::new(0., 0.)
                    } else {
                        let x = Reflect::get(&p, &"x".into())
                            .ok()
                            .unwrap()
                            .as_f64()
                            .unwrap() as f32;
                        let y = Reflect::get(&p, &"y".into())
                            .ok()
                            .unwrap()
                            .as_f64()
                            .unwrap() as f32;
                        egraph::layout::force_directed::force::Point::new(x, y)
                    }
                })
                .collect::<Vec<_>>()
        };
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

    pub fn alpha_start(&mut self, value: JsValue) -> JsValue {
        if value.is_null() || value.is_undefined() {
            return self.simulation.alpha_start.into();
        }
        self.simulation.alpha_start = value.as_f64().unwrap() as f32;
        JsValue::undefined()
    }

    pub fn alpha_min(&mut self, value: JsValue) -> JsValue {
        if value.is_null() || value.is_undefined() {
            return self.simulation.alpha_min.into();
        }
        self.simulation.alpha_min = value.as_f64().unwrap() as f32;
        JsValue::undefined()
    }

    pub fn alpha_target(&mut self, value: JsValue) -> JsValue {
        if value.is_null() || value.is_undefined() {
            return self.simulation.alpha_target.into();
        }
        self.simulation.alpha_target = value.as_f64().unwrap() as f32;
        JsValue::undefined()
    }

    pub fn velocity_decay(&mut self, value: JsValue) -> JsValue {
        if value.is_null() || value.is_undefined() {
            return self.simulation.velocity_decay.into();
        }
        self.simulation.velocity_decay = value.as_f64().unwrap() as f32;
        JsValue::undefined()
    }

    pub fn iterations(&mut self, value: JsValue) -> JsValue {
        if value.is_null() || value.is_undefined() {
            return (self.simulation.iterations as f64).into();
        }
        self.simulation.iterations = value.as_f64().unwrap() as usize;
        JsValue::undefined()
    }
}
