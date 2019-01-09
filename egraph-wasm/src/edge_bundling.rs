use crate::graph::Graph;
use js_sys::{Array, Object, Reflect};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct EdgeBundling {
    edge_bundling: egraph::layout::force_directed::edge_bundling::EdgeBundling,
}

#[wasm_bindgen]
impl EdgeBundling {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        EdgeBundling {
            edge_bundling: egraph::layout::force_directed::edge_bundling::EdgeBundling::new(),
        }
    }

    pub fn call(&self, graph: &Graph, point_array: Array) -> Array {
        let mut points = Vec::new();
        point_array.for_each(&mut |point, _, _| {
            let x = Reflect::get(&point, &"x".into())
                .ok()
                .unwrap()
                .as_f64()
                .unwrap() as f32;
            let y = Reflect::get(&point, &"y".into())
                .ok()
                .unwrap()
                .as_f64()
                .unwrap() as f32;
            points.push(egraph::layout::force_directed::force::Point::new(x, y));
        });
        let lines = self.edge_bundling.call(&graph.graph(), &points);
        let result = Array::new();
        for line in lines {
            let bends = Array::new();
            for point in line.points {
                let obj = Object::new();
                Reflect::set(&obj, &"x".into(), &point.x.into()).ok().unwrap();
                Reflect::set(&obj, &"y".into(), &point.y.into()).ok().unwrap();
                bends.push(&obj);
            }
            let obj = Object::new();
            Reflect::set(&obj, &"bends".into(), &bends).ok().unwrap();
            result.push(&obj);
        }
        result
    }
}
