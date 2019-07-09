use egraph::algorithm::biclustering::quasi_biclique::QuasiBiclique;
use egraph::algorithm::biclustering::Biclustering;
use egraph_wasm_adapter::{JsGraph, JsGraphAdapter};
use std::collections::HashSet;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = QuasiBiclique)]
pub struct JsQuasiBiclique {
    biclustering: QuasiBiclique,
}

#[wasm_bindgen(js_class = QuasiBiclique)]
impl JsQuasiBiclique {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsQuasiBiclique {
        JsQuasiBiclique {
            biclustering: QuasiBiclique::new(),
        }
    }

    pub fn call(
        &self,
        graph: JsGraph,
        source: JsValue,
        target: JsValue,
    ) -> Result<JsValue, JsValue> {
        let graph = JsGraphAdapter::new(graph);
        let source = js_sys::try_iter(&source)?
            .ok_or("source is not iterable")?
            .map(|value| value.ok().unwrap().as_f64().unwrap() as usize)
            .collect::<HashSet<_>>();
        let target = js_sys::try_iter(&target)?
            .ok_or("target is not iterable")?
            .map(|value| value.ok().unwrap().as_f64().unwrap() as usize)
            .collect::<HashSet<_>>();
        let result = self.biclustering.call(&graph, &source, &target);
        Ok(JsValue::from_serde(&result).unwrap())
    }
}
