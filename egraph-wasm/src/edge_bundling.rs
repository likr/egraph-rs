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
                Reflect::set(&obj, &"x".into(), &point.x.into())
                    .ok()
                    .unwrap();
                Reflect::set(&obj, &"y".into(), &point.y.into())
                    .ok()
                    .unwrap();
                bends.push(&obj);
            }
            let obj = Object::new();
            Reflect::set(&obj, &"bends".into(), &bends).ok().unwrap();
            result.push(&obj);
        }
        result
    }

    pub fn cycles(&mut self, value: JsValue) -> JsValue {
        if value.is_null() || value.is_undefined() {
            return (self.edge_bundling.cycles as f64).into();
        }
        self.edge_bundling.cycles = value.as_f64().unwrap() as usize;
        JsValue::undefined()
    }

    pub fn s0(&mut self, value: JsValue) -> JsValue {
        if value.is_null() || value.is_undefined() {
            return self.edge_bundling.s0.into();
        }
        self.edge_bundling.s0 = value.as_f64().unwrap() as f32;
        JsValue::undefined()
    }

    #[wasm_bindgen(js_name = sStep)]
    pub fn s_step(&mut self, value: JsValue) -> JsValue {
        if value.is_null() || value.is_undefined() {
            return self.edge_bundling.s_step.into();
        }
        self.edge_bundling.s_step = value.as_f64().unwrap() as f32;
        JsValue::undefined()
    }

    pub fn i0(&mut self, value: JsValue) -> JsValue {
        if value.is_null() || value.is_undefined() {
            return (self.edge_bundling.i0 as f64).into();
        }
        self.edge_bundling.i0 = value.as_f64().unwrap() as usize;
        JsValue::undefined()
    }

    #[wasm_bindgen(js_name = iStep)]
    pub fn i_step(&mut self, value: JsValue) -> JsValue {
        if value.is_null() || value.is_undefined() {
            return self.edge_bundling.i_step.into();
        }
        self.edge_bundling.i_step = value.as_f64().unwrap() as f32;
        JsValue::undefined()
    }

    #[wasm_bindgen(js_name = minimumEdgeCompatibility)]
    pub fn minimum_edge_compatibility(&mut self, value: JsValue) -> JsValue {
        if value.is_null() || value.is_undefined() {
            return self.edge_bundling.minimum_edge_compatibility.into();
        }
        self.edge_bundling.minimum_edge_compatibility = value.as_f64().unwrap() as f32;
        JsValue::undefined()
    }
}
