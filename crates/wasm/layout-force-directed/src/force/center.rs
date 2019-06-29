use super::JsForce;
use egraph::layout::force_directed::force::CenterForce;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = CenterForce)]
pub struct JsCenterForce {
    force: Rc<RefCell<CenterForce>>,
}

#[wasm_bindgen(js_class = CenterForce)]
impl JsCenterForce {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsCenterForce {
        JsCenterForce {
            force: Rc::new(RefCell::new(CenterForce::new())),
        }
    }

    pub fn force(&self) -> JsForce {
        JsForce::new(self.force.clone())
    }
}
