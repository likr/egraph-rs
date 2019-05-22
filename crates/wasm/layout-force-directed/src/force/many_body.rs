use super::Force;
use egraph::layout::force_directed::force::ManyBodyForce as EgManyBodyForce;
use egraph_wasm_adapter::JsGraph;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct ManyBodyForce {
    force: Rc<RefCell<EgManyBodyForce<JsGraph>>>,
}

#[wasm_bindgen]
impl ManyBodyForce {
    #[wasm_bindgen(constructor)]
    pub fn new() -> ManyBodyForce {
        ManyBodyForce {
            force: Rc::new(RefCell::new(EgManyBodyForce::new())),
        }
    }

    pub fn force(&self) -> Force {
        Force::new(self.force.clone())
    }

    pub fn strength(&self, f: &js_sys::Function) {
        let f = f.clone();
        self.force.borrow_mut().strength = Box::new(move |_, a| {
            let this = JsValue::NULL;
            let index = JsValue::from_f64(a as f64);
            f.call1(&this, &index).ok().unwrap().as_f64().unwrap() as f32
        });
    }
}
