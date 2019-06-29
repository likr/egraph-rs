pub mod force_directed;
pub mod treemap;

use egraph::grouping::{aggregate_edges, aggregate_nodes};
use egraph::Graph;
use egraph_wasm_adapter::{JsGraph, JsGraphAdapter};
use js_sys::Function;
use wasm_bindgen::prelude::*;

#[derive(Serialize, Deserialize)]
pub struct GroupObject {
    #[serde(rename = "type")]
    pub shape: String,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

#[derive(Serialize, Deserialize)]
pub struct GroupNodeObject {
    pub id: usize,
    pub weight: f64,
}

#[derive(Serialize, Deserialize)]
pub struct GroupLinkObject {
    pub source: usize,
    pub target: usize,
    pub weight: f64,
}

#[wasm_bindgen(js_name = aggregateNodes)]
pub fn js_aggregate_nodes(graph: JsGraph, grouping: Function, weight: Function) -> JsValue {
    let graph = JsGraphAdapter::new(graph);
    let grouping: Box<dyn Fn(&JsGraphAdapter, usize) -> usize> = Box::new(move |graph, u| {
        let this = JsValue::NULL;
        let graph = graph.data();
        let u = JsValue::from_f64(u as f64);
        grouping
            .call2(&this, &graph, &u)
            .ok()
            .unwrap()
            .as_f64()
            .unwrap() as usize
    });
    let weight: Box<dyn Fn(&JsGraphAdapter, usize) -> f32> = Box::new(move |graph, u| {
        let this = JsValue::NULL;
        let graph = graph.data();
        let u = JsValue::from_f64(u as f64);
        weight
            .call2(&this, &graph, &u)
            .ok()
            .unwrap()
            .as_f64()
            .unwrap() as f32
    });
    let result = aggregate_nodes(&graph, &grouping, &weight)
        .iter()
        .map(|g| GroupNodeObject {
            id: g.id,
            weight: g.weight as f64,
        })
        .collect::<Vec<_>>();
    JsValue::from_serde(&result).unwrap()
}

#[wasm_bindgen(js_name = aggregateEdges)]
pub fn js_aggregate_edges(graph: JsGraph, grouping: Function, weight: Function) -> JsValue {
    let graph = JsGraphAdapter::new(graph);
    let grouping: Box<dyn Fn(&JsGraphAdapter, usize) -> usize> = Box::new(move |graph, u| {
        let this = JsValue::NULL;
        let graph = graph.data();
        let u = JsValue::from_f64(u as f64);
        grouping
            .call2(&this, &graph, &u)
            .ok()
            .unwrap()
            .as_f64()
            .unwrap() as usize
    });
    let weight: Box<dyn Fn(&JsGraphAdapter, usize, usize) -> f32> = Box::new(move |graph, u, v| {
        let this = JsValue::NULL;
        let graph = graph.data();
        let u = JsValue::from_f64(u as f64);
        let v = JsValue::from_f64(v as f64);
        weight
            .call3(&this, &graph, &u, &v)
            .ok()
            .unwrap()
            .as_f64()
            .unwrap() as f32
    });
    let result = aggregate_edges(&graph, &grouping, &weight)
        .iter()
        .map(|g| GroupLinkObject {
            source: g.source,
            target: g.target,
            weight: g.weight as f64,
        })
        .collect::<Vec<_>>();
    JsValue::from_serde(&result).unwrap()
}
