use super::Force;
use egraph::layout::force_directed::force::CenterForce as EgCenterForce;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct CenterForce {
    force: Rc<RefCell<EgCenterForce>>,
}

#[wasm_bindgen]
impl CenterForce {
    #[wasm_bindgen(constructor)]
    pub fn new() -> CenterForce {
        CenterForce {
            force: Rc::new(RefCell::new(EgCenterForce::new())),
        }
    }

    pub fn force(&self) -> Force {
        Force::new(self.force.clone())
    }
}
