pub mod force;
pub mod grouping;

use crate::graph::JsGraph;
use crate::layout::force_simulation::force::JsForce;
use js_sys::{Array, Function, Object, Reflect};
use wasm_bindgen::convert::IntoWasmAbi;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = forceGrouped)]
pub fn force_grouped(graph: &JsGraph, group_accessor: &Function) -> Array {
    let forces = petgraph_layout_grouped_force::force_grouped(graph.graph(), |_, u| {
        group_accessor
            .call1(&JsValue::null(), &JsValue::from_f64(u.index() as f64))
            .unwrap()
            .as_f64()
            .unwrap() as usize
    });
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
