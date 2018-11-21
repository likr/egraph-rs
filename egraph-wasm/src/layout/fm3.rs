use super::super::graph::Graph;
use js_sys::{Array, Object, Reflect};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct FM3 {
    fm3: egraph::layout::fm3::FM3,
}

#[wasm_bindgen]
impl FM3 {
    #[wasm_bindgen(constructor)]
    pub fn new() -> FM3 {
        FM3 {
            fm3: egraph::layout::fm3::FM3::new(),
        }
    }

    pub fn call(&self, graph: &Graph) -> JsValue {
        let array = Array::new();
        let points = self.fm3.call(&graph.graph());
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
