use super::Force;
use egraph::layout::force_directed::force::CollideForce as EgCollideForce;
use egraph_wasm_adapter::JsGraph;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct CollideForce {
    force: Rc<RefCell<EgCollideForce<JsGraph>>>,
}

#[wasm_bindgen]
impl CollideForce {
    #[wasm_bindgen(constructor)]
    pub fn new() -> CollideForce {
        CollideForce {
            force: Rc::new(RefCell::new(EgCollideForce::new())),
        }
    }

    pub fn force(&self) -> Force {
        Force::new(self.force.clone())
    }

    pub fn radius(&mut self, f: &js_sys::Function) {
        let f = f.clone();
        self.force.borrow_mut().radius = Box::new(move |_, a| {
            let this = JsValue::NULL;
            let index = JsValue::from_f64(a as f64);
            f.call1(&this, &index).ok().unwrap().as_f64().unwrap() as f32
        });
    }

    pub fn strength(&mut self, value: f64) {
        self.force.borrow_mut().strength = value as f32;
    }
}
