use super::GroupObject;
use egraph::grouping::treemap::TreemapGrouping;
use egraph::Graph;
use egraph_wasm_adapter::{JsGraph, JsGraphAdapter};
use js_sys::Function;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = TreemapGrouping)]
pub struct JsTreemapGrouping {
    grouping: TreemapGrouping<JsGraph, JsGraphAdapter>,
}

#[wasm_bindgen(js_class = TreemapGrouping)]
impl JsTreemapGrouping {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsTreemapGrouping {
        JsTreemapGrouping {
            grouping: TreemapGrouping::new(),
        }
    }

    pub fn call(&self, graph: JsGraph, width: f64, height: f64) -> JsValue {
        let graph = JsGraphAdapter::new(graph);
        let result = self
            .grouping
            .call(&graph, width as f32, height as f32)
            .iter()
            .map(|(&i, g)| {
                (
                    i,
                    GroupObject {
                        shape: "rect".into(),
                        x: g.x as f64,
                        y: g.y as f64,
                        width: g.width as f64,
                        height: g.height as f64,
                    },
                )
            })
            .collect::<HashMap<_, _>>();
        JsValue::from_serde(&result).unwrap()
    }

    #[wasm_bindgen(setter = group)]
    pub fn set_group(&mut self, f: &Function) {
        let f = f.clone();
        self.grouping.group = Box::new(move |graph, u| {
            let this = JsValue::NULL;
            let graph = graph.data();
            let u = JsValue::from_f64(u as f64);
            f.call2(&this, &graph, &u).ok().unwrap().as_f64().unwrap() as usize
        });
    }

    #[wasm_bindgen(setter = weight)]
    pub fn set_weight(&mut self, f: &Function) {
        let f = f.clone();
        self.grouping.weight = Box::new(move |graph, u| {
            let this = JsValue::NULL;
            let graph = graph.data();
            let u = JsValue::from_f64(u as f64);
            f.call2(&this, &graph, &u).ok().unwrap().as_f64().unwrap() as f32
        });
    }
}
