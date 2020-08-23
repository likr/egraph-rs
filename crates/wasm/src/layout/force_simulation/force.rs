use crate::graph::JsGraph;
use petgraph_layout_force_simulation::force::{CenterForce, LinkForce, ManyBodyForce};
use petgraph_layout_force_simulation::Force;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = Force)]
pub struct JsForce {
    force: Box<dyn Force>,
}

impl JsForce {
    pub fn new<F: Force + 'static>(force: F) -> JsForce {
        JsForce {
            force: Box::new(force),
        }
    }

    pub fn with_box(force: Box<dyn Force>) -> JsForce {
        JsForce { force }
    }
}

impl AsRef<dyn Force> for JsForce {
    fn as_ref(&self) -> &(dyn Force + 'static) {
        self.force.as_ref()
    }
}

#[wasm_bindgen(js_name = CenterForce)]
pub struct JsCenterForce {}

#[wasm_bindgen(js_class = CenterForce)]
impl JsCenterForce {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsForce {
        JsForce::new(CenterForce::new())
    }
}

#[wasm_bindgen(js_name = LinkForce)]
pub struct JsLinkForce {}

#[wasm_bindgen(js_class = LinkForce)]
impl JsLinkForce {
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsGraph) -> JsForce {
        JsForce::new(LinkForce::new(graph.graph()))
    }
}

#[wasm_bindgen(js_name = ManyBodyForce)]
pub struct JsManyBodyForce {}

#[wasm_bindgen(js_class = ManyBodyForce)]
impl JsManyBodyForce {
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsGraph) -> JsForce {
        JsForce::new(ManyBodyForce::new(graph.graph()))
    }
}
