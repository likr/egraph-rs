use egraph::grouping::treemap::TreemapGrouping as EgTreemapGrouping;
use egraph_wasm_adapter::{JsGraph, JsGraphAdapter};
use js_sys::Function;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[derive(Serialize, Deserialize)]
pub struct Group {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

#[wasm_bindgen]
pub struct TreemapGrouping {
    grouping: EgTreemapGrouping<JsGraph>,
}

#[wasm_bindgen]
impl TreemapGrouping {
    #[wasm_bindgen(constructor)]
    pub fn new() -> TreemapGrouping {
        TreemapGrouping {
            grouping: EgTreemapGrouping::new(),
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
                    Group {
                        x: g.x as f64,
                        y: g.y as f64,
                        width: g.width as f64,
                        height: g.height as f64,
                    },
                )
            })
            .collect::<HashMap<usize, Group>>();
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

    #[wasm_bindgen(setter = size)]
    pub fn set_size(&mut self, f: &Function) {
        let f = f.clone();
        self.grouping.size = Box::new(move |graph, u| {
            let this = JsValue::NULL;
            let graph = graph.data();
            let u = JsValue::from_f64(u as f64);
            f.call2(&this, &graph, &u).ok().unwrap().as_f64().unwrap() as f32
        });
    }
}
