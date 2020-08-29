pub mod force;
pub mod simulation;

use crate::graph::JsGraph;
use crate::layout::force_simulation::force::JsForce;
use js_sys::{Array, Object, Reflect};
use std::collections::HashMap;
use wasm_bindgen::convert::IntoWasmAbi;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = initialPlacement)]
pub fn initial_placement(graph: &JsGraph) -> JsValue {
    let coordinates = petgraph_layout_force_simulation::initial_placement(graph.graph())
        .iter()
        .map(|(u, &(x, y))| (u.index(), (x, y)))
        .collect::<HashMap<usize, (f32, f32)>>();
    JsValue::from_serde(&coordinates).unwrap()
}

#[wasm_bindgen(js_name = forceConnected)]
pub fn force_connected(graph: &JsGraph) -> Array {
    let forces = petgraph_layout_force_simulation::force_connected(graph.graph());
    forces
        .into_iter()
        .map(|f| {
            let js_force = JsForce::with_box(f);
            let ptr = js_force.into_abi();
            let obj = Object::new();
            Reflect::set(&obj, &"ptr".into(), &JsValue::from_f64(ptr as f64)).ok();
            obj
        })
        .collect::<Array>()
}

#[wasm_bindgen(js_name = forceNonconnected)]
pub fn force_nonconnected(graph: &JsGraph) -> Array {
    let forces = petgraph_layout_force_simulation::force_nonconnected(graph.graph());
    forces
        .into_iter()
        .map(|f| {
            let js_force = JsForce::with_box(f);
            let ptr = js_force.into_abi();
            let obj = Object::new();
            Reflect::set(&obj, &"ptr".into(), &JsValue::from_f64(ptr as f64)).ok();
            obj
        })
        .collect::<Array>()
}
