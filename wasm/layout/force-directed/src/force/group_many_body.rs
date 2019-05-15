use super::Force;
use egraph_layout_force_directed::force::GroupManyBodyForce as EgGroupManyBodyForce;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct GroupManyBodyForce {
    force: Rc<RefCell<EgGroupManyBodyForce>>,
}

#[wasm_bindgen]
impl GroupManyBodyForce {
    #[wasm_bindgen(constructor)]
    pub fn new() -> GroupManyBodyForce {
        GroupManyBodyForce {
            force: Rc::new(RefCell::new(EgGroupManyBodyForce::new())),
        }
    }

    pub fn force(&self) -> Force {
        Force::new(self.force.clone())
    }

    pub fn strength(&self, f: &js_sys::Function) {
        let f = f.clone();
        self.force.borrow_mut().strength = Box::new(move |_, u| {
            let this = JsValue::NULL;
            let index = JsValue::from_f64(u as f64);
            f.call1(&this, &index).ok().unwrap().as_f64().unwrap() as f32
        });
    }

    pub fn group(&self, f: &js_sys::Function) {
        let f = f.clone();
        self.force.borrow_mut().group = Box::new(move |_, u| {
            let this = JsValue::NULL;
            let index = JsValue::from_f64(u as f64);
            f.call1(&this, &index).ok().unwrap().as_f64().unwrap() as usize
        });
    }
}
