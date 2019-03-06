use super::super::super::super::graph::{Edge, EdgeType, Graph, IndexType, Node};
use egraph::layout::grouping::Grouping;
use js_sys::{Object, Reflect};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct ForceDirectedGrouping {
    grouping: egraph::layout::grouping::ForceDirectedGrouping<Node, Edge, EdgeType, IndexType>,
}

#[wasm_bindgen]
impl ForceDirectedGrouping {
    #[wasm_bindgen(constructor)]
    pub fn new() -> ForceDirectedGrouping {
        ForceDirectedGrouping {
            grouping: egraph::layout::grouping::ForceDirectedGrouping::new(),
        }
    }

    pub fn call(&mut self, graph: &Graph, width: f64, height: f64) -> JsValue {
        let tiles = self
            .grouping
            .call(&graph.graph(), width as f32, height as f32);
        let result = Object::new();
        for (&g, tile) in tiles.iter() {
            let obj = Object::new();
            Reflect::set(&obj, &"x".into(), &tile.x.into())
                .ok()
                .unwrap();
            Reflect::set(&obj, &"y".into(), &tile.y.into())
                .ok()
                .unwrap();
            Reflect::set(&obj, &"width".into(), &tile.width.into())
                .ok()
                .unwrap();
            Reflect::set(&obj, &"height".into(), &tile.height.into())
                .ok()
                .unwrap();
            Reflect::set(&result, &JsValue::from_f64(g as f64), &obj)
                .ok()
                .unwrap();
        }
        result.into()
    }

    pub fn group(&mut self, f: &js_sys::Function) {
        let f = f.clone();
        self.grouping.group = Box::new(move |_, a| {
            let this = JsValue::NULL;
            let index = JsValue::from_f64(a.index() as f64);
            f.call1(&this, &index).ok().unwrap().as_f64().unwrap() as usize
        });
    }

    #[wasm_bindgen(js_name = nodeSize)]
    pub fn node_size(&mut self, f: &js_sys::Function) {
        let f = f.clone();
        self.grouping.node_size = Box::new(move |_, a| {
            let this = JsValue::NULL;
            let index = JsValue::from_f64(a.index() as f64);
            f.call1(&this, &index).ok().unwrap().as_f64().unwrap() as f32
        });
    }

    #[wasm_bindgen(js_name = linkWeight)]
    pub fn link_weight(&mut self, f: &js_sys::Function) {
        let f = f.clone();
        self.grouping.link_weight = Box::new(move |_, e| {
            let this = JsValue::NULL;
            let index = JsValue::from_f64(e.index() as f64);
            f.call1(&this, &index).ok().unwrap().as_f64().unwrap() as f32
        });
    }

    #[wasm_bindgen(js_name = linkForceStrength)]
    pub fn link_force_strength(&mut self, f: &js_sys::Function) {
        let f = f.clone();
        self.grouping.link_force_strength = Box::new(move |_, e| {
            let this = JsValue::NULL;
            let index = JsValue::from_f64(e.index() as f64);
            f.call1(&this, &index).ok().unwrap().as_f64().unwrap() as f32
        });
    }

    #[wasm_bindgen(js_name = manyBodyForceStrength)]
    pub fn many_body_force_strength(&mut self, f: &js_sys::Function) {
        let f = f.clone();
        self.grouping.many_body_force_strength = Box::new(move |g, a| {
            let this = JsValue::NULL;
            let size = JsValue::from_f64(g[a].size as f64);
            f.call1(&this, &size).ok().unwrap().as_f64().unwrap() as f32
        });
    }

    #[wasm_bindgen(js_name = positionForceStrength)]
    pub fn position_force_strength(&mut self, f: &js_sys::Function) {
        let f = f.clone();
        self.grouping.position_force_strength = Box::new(move |g, a| {
            let this = JsValue::NULL;
            let size = JsValue::from_f64(g[a].size as f64);
            f.call1(&this, &size).ok().unwrap().as_f64().unwrap() as f32
        });
    }
}
