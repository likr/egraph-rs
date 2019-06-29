use super::JsForce;
use egraph::layout::force_directed::force::CollideForce;
use egraph::Graph;
use egraph_wasm_adapter::{JsGraph, JsGraphAdapter};
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = CollideForce)]
pub struct JsCollideForce {
    force: Rc<RefCell<CollideForce<JsGraph, JsGraphAdapter>>>,
}

#[wasm_bindgen(js_class = CollideForce)]
impl JsCollideForce {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsCollideForce {
        JsCollideForce {
            force: Rc::new(RefCell::new(CollideForce::new())),
        }
    }

    pub fn force(&self) -> JsForce {
        JsForce::new(self.force.clone())
    }

    #[wasm_bindgen(setter = radius)]
    pub fn radius(&mut self, f: &js_sys::Function) {
        let f = f.clone();
        self.force.borrow_mut().radius = Box::new(move |graph, u| {
            let this = JsValue::NULL;
            let graph = graph.data();
            let u = JsValue::from_f64(u as f64);
            f.call2(&this, &graph, &u).ok().unwrap().as_f64().unwrap() as f32
        });
    }

    #[wasm_bindgen(getter = strength)]
    pub fn strength(&mut self) -> f32 {
        self.force.borrow_mut().strength
    }

    #[wasm_bindgen(setter = strength)]
    pub fn set_strength(&mut self, value: f32) {
        self.force.borrow_mut().strength = value;
    }
}
