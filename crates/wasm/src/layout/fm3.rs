use crate::graph::JsGraph;
use crate::layout::force_simulation::coordinates::JsCoordinates;
use js_sys::Function;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn fm3(
    graph: &JsGraph,
    min_size: usize,
    step_iteration: usize,
    _shrink_node: Function,
    _shrink_edge: Function,
    _link_distance_accessor: Function,
) -> JsCoordinates {
    JsCoordinates::new(petgraph_layout_fm3::fm3(
        graph.graph(),
        min_size,
        step_iteration,
        &mut |_, _| JsValue::null(),
        &mut |_, _| JsValue::null(),
        &mut |_, _| 30.,
    ))
}
