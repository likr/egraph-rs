use super::Force;
use egraph::layout::force_directed::force::GroupLinkForce as EgGroupLinkForce;
use egraph_wasm_adapter::JsGraph;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct GroupLinkForce {
    force: Rc<RefCell<EgGroupLinkForce<JsGraph>>>,
}

#[wasm_bindgen]
impl GroupLinkForce {
    #[wasm_bindgen(constructor)]
    pub fn new() -> GroupLinkForce {
        GroupLinkForce {
            force: Rc::new(RefCell::new(EgGroupLinkForce::new())),
        }
    }

    pub fn force(&self) -> Force {
        Force::new(self.force.clone())
    }

    pub fn intra_group(&self, value: f64) {
        self.force.borrow_mut().intra_group = value as f32;
    }

    pub fn inter_group(&self, value: f64) {
        self.force.borrow_mut().inter_group = value as f32;
    }

    pub fn distance(&self, f: &js_sys::Function) {
        let f = f.clone();
        self.force.borrow_mut().distance = Box::new(move |_, u, v| {
            let this = JsValue::NULL;
            let u = JsValue::from_f64(u as f64);
            let v = JsValue::from_f64(v as f64);
            f.call2(&this, &u, &v).ok().unwrap().as_f64().unwrap() as f32
        });
    }

    pub fn group(&self, f: &js_sys::Function) {
        let f = f.clone();
        self.force.borrow_mut().group = Box::new(move |_, u| {
            let this = JsValue::NULL;
            let u = JsValue::from_f64(u as f64);
            f.call1(&this, &u).ok().unwrap().as_f64().unwrap() as usize
        });
    }
}
