use egraph::algorithm::biclustering::mu_quasi_bicliques;
use egraph_wasm_adapter::{JsGraph, JsGraphAdapter};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = muQuasiBicliques)]
pub fn js_mu_quasi_bicliques(
    graph: JsGraph,
    source: JsValue,
    target: JsValue,
    mu: f64,
) -> Result<JsValue, JsValue> {
    let graph = JsGraphAdapter::new(graph);
    let source = js_sys::try_iter(&source)?
        .ok_or("source is not iterable")?
        .map(|value| value.ok().unwrap().as_f64().unwrap() as usize)
        .collect::<Vec<_>>();
    let target = js_sys::try_iter(&target)?
        .ok_or("target is not iterable")?
        .map(|value| value.ok().unwrap().as_f64().unwrap() as usize)
        .collect::<Vec<_>>();
    let result = mu_quasi_bicliques(&graph, &source, &target, mu);
    Ok(JsValue::from_serde(&result).unwrap())
}
