use crate::graph::JsGraph;
use js_sys::Function;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn fm3(
    graph: &JsGraph,
    min_size: usize,
    step_iteration: usize,
    _shrink_node: Function,
    _shrink_edge: Function,
    _link_distance_accessor: Function,
) -> JsValue {
    let coordinates = petgraph_layout_fm3::fm3(
        graph.graph(),
        min_size,
        step_iteration,
        &mut |_, _| JsValue::null(),
        &mut |_, _| JsValue::null(),
        &mut |_, _| 30.,
    )
    .into_iter()
    .map(|(u, xy)| (u.index(), xy))
    .collect::<HashMap<_, _>>();
    JsValue::from_serde(&coordinates).unwrap()
}
