use super::Force;
use egraph::layout::force_directed::force::LinkForce as EgLinkForce;
use egraph_wasm_adapter::JsGraph;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct LinkForce {
    force: Rc<RefCell<EgLinkForce<JsGraph>>>,
}

#[wasm_bindgen]
impl LinkForce {
    #[wasm_bindgen(constructor)]
    pub fn new() -> LinkForce {
        LinkForce {
            force: Rc::new(RefCell::new(EgLinkForce::new())),
        }
    }

    pub fn force(&self) -> Force {
        Force::new(self.force.clone())
    }

    pub fn strength(&self, f: &js_sys::Function) {
        let f = f.clone();
        self.force.borrow_mut().strength = Box::new(move |_, u, v| {
            let this = JsValue::NULL;
            let u = JsValue::from_f64(u as f64);
            let v = JsValue::from_f64(v as f64);
            f.call2(&this, &u, &v).ok().unwrap().as_f64().unwrap() as f32
        });
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
}
