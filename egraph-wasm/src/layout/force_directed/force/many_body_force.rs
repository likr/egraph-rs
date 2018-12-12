use super::force::Force;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct ManyBodyForce {
    force: Rc<RefCell<egraph::layout::force_directed::force::ManyBodyForce>>,
}

#[wasm_bindgen]
impl ManyBodyForce {
    #[wasm_bindgen(constructor)]
    pub fn new() -> ManyBodyForce {
        ManyBodyForce {
            force: Rc::new(RefCell::new(
                egraph::layout::force_directed::force::ManyBodyForce::new(),
            )),
        }
    }

    pub fn force(&self) -> Force {
        Force::new(self.force.clone())
    }

    #[wasm_bindgen(js_name = getStrength)]
    pub fn get_strength(&self) -> f64 {
        self.force.borrow().strength as f64
    }

    #[wasm_bindgen(js_name = setStrength)]
    pub fn set_strength(&self, value: f64) {
        self.force.borrow_mut().strength = value as f32;
    }
}
