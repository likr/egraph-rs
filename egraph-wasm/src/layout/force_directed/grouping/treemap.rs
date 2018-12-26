use super::super::super::super::graph::{Edge, EdgeType, Graph, IndexType, Node};
use egraph::layout::grouping::Grouping;
use js_sys::{Array, Object, Reflect};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct TreemapGrouping {
    grouping: egraph::layout::grouping::TreemapGrouping<Node, Edge, EdgeType, IndexType>,
}

#[wasm_bindgen]
impl TreemapGrouping {
    #[wasm_bindgen(constructor)]
    pub fn new() -> TreemapGrouping {
        TreemapGrouping {
            grouping: egraph::layout::grouping::TreemapGrouping::new(),
        }
    }

    pub fn call(&mut self, graph: &Graph, width: f64, height: f64) -> JsValue {
        let result = self
            .grouping
            .call(&graph.graph(), width as f32, height as f32);
        let array = Array::new();
        for (_, tile) in result.iter() {
            let obj = Object::new();
            Reflect::set(&obj, &"x".into(), &tile.x.into())
                .ok()
                .unwrap();
            Reflect::set(&obj, &"y".into(), &tile.y.into())
                .ok()
                .unwrap();
            array.push(&obj);
        }
        array.into()
    }

    pub fn group(&mut self, f: &js_sys::Function) {
        let f = f.clone();
        self.grouping.group = Box::new(move |_, a| {
            let this = JsValue::NULL;
            let index = JsValue::from_f64(a.index() as f64);
            f.call1(&this, &index).ok().unwrap().as_f64().unwrap() as usize
        });
    }

    pub fn size(&mut self, f: &js_sys::Function) {
        let f = f.clone();
        self.grouping.size = Box::new(move |_, a| {
            let this = JsValue::NULL;
            let index = JsValue::from_f64(a.index() as f64);
            f.call1(&this, &index).ok().unwrap().as_f64().unwrap() as f32
        });
    }
}