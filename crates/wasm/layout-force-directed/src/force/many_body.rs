use super::JsForce;
use egraph::layout::force_directed::force::ManyBodyForce;
use egraph::Graph;
use egraph_wasm_adapter::{JsGraph, JsGraphAdapter};
use js_sys::Function;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = ManyBodyForce)]
pub struct JsManyBodyForce {
    force: Rc<RefCell<ManyBodyForce<JsGraph, JsGraphAdapter>>>,
}

#[wasm_bindgen(js_class = ManyBodyForce)]
impl JsManyBodyForce {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsManyBodyForce {
        JsManyBodyForce {
            force: Rc::new(RefCell::new(ManyBodyForce::new())),
        }
    }

    pub fn force(&self) -> JsForce {
        JsForce::new(self.force.clone())
    }

    #[wasm_bindgen(setter = strength)]
    pub fn set_strength(&self, f: &Function) {
        let f = f.clone();
        self.force.borrow_mut().strength = Box::new(move |graph, u| {
            let this = JsValue::NULL;
            let graph = graph.data();
            let u = JsValue::from_f64(u as f64);
            f.call2(&this, &graph, &u).ok().unwrap().as_f64().unwrap() as f32
        });
    }
}
