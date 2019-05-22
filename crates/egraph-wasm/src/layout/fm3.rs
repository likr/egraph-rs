use super::super::graph::{Edge, EdgeType, Graph, IndexType, Node};
use super::force_directed::simulation::Simulation;
use js_sys::{Array, Object, Reflect};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct FM3 {
    fm3: egraph::layout::fm3::FM3<Node, Edge, EdgeType, IndexType>,
}

#[wasm_bindgen]
impl FM3 {
    #[wasm_bindgen(constructor)]
    pub fn new() -> FM3 {
        FM3 {
            fm3: egraph::layout::fm3::FM3::new(
                Box::new(|_, _| Object::new()),
                Box::new(|_, _, distance| {
                    let edge = Object::new();
                    Reflect::set(&edge, &"distance".into(), &distance.into())
                        .ok()
                        .unwrap();
                    edge
                }),
                Box::new(|graph, e| {
                    if Reflect::has(&graph[e], &"distance".into()).ok().unwrap() {
                        Reflect::get(&graph[e], &"distance".into())
                            .ok()
                            .unwrap()
                            .as_f64()
                            .unwrap() as f32
                    } else {
                        30.
                    }
                }),
            ),
        }
    }

    pub fn call(&self, graph: &Graph, simulation: &Simulation) -> JsValue {
        let points = self.fm3.call(&graph.graph(), simulation.simulation());
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

    #[wasm_bindgen(js_name = minSize)]
    pub fn min_size(&mut self, value: usize) {
        self.fm3.min_size = value;
    }

    #[wasm_bindgen(js_name = stepIteration)]
    pub fn step_iteration(&mut self, value: usize) {
        self.fm3.step_iteration = value;
    }

    #[wasm_bindgen(js_name = shrinkNode)]
    pub fn shrink_node(&mut self, f: &js_sys::Function) {
        let f = f.clone();
        self.fm3.shrink_node = Box::new(move |_, _| {
            let this = JsValue::NULL;
            f.call0(&this).ok().unwrap().into()
        });
    }

    #[wasm_bindgen(js_name = shrinkEdge)]
    pub fn shrink_edge(&mut self, f: &js_sys::Function) {
        let f = f.clone();
        self.fm3.shrink_edge = Box::new(move |_, _, _| {
            let this = JsValue::NULL;
            f.call0(&this).ok().unwrap().into()
        });
    }

    #[wasm_bindgen(js_name = linkDistance)]
    pub fn link_distance(&mut self, f: &js_sys::Function) {
        let f = f.clone();
        self.fm3.link_distance_accessor = Box::new(move |_, _| {
            let this = JsValue::NULL;
            f.call0(&this).ok().unwrap().as_f64().unwrap() as f32
        });
    }
}
