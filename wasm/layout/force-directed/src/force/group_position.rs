use super::Force;
use egraph_layout_force_directed::force::GroupPositionForce as EgGroupPositionForce;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct GroupPositionForce {
    force: Rc<RefCell<EgGroupPositionForce>>,
}

#[wasm_bindgen]
impl GroupPositionForce {
    #[wasm_bindgen(constructor)]
    pub fn new() -> GroupPositionForce {
        GroupPositionForce {
            force: Rc::new(RefCell::new(EgGroupPositionForce::new())),
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

    pub fn group(&self, f: &js_sys::Function) {
        let f = f.clone();
        self.force.borrow_mut().group = Box::new(move |_, a| {
            let this = JsValue::NULL;
            let index = JsValue::from_f64(a as f64);
            f.call1(&this, &index).ok().unwrap().as_f64().unwrap() as usize
        });
    }

    #[wasm_bindgen(js_name = groupX)]
    pub fn group_x(&self, f: &js_sys::Function) {
        let f = f.clone();
        self.force.borrow_mut().group_x = Box::new(move |a| {
            let this = JsValue::NULL;
            let index = JsValue::from_f64(a as f64);
            f.call1(&this, &index).ok().unwrap().as_f64().unwrap() as f32
        });
    }

    #[wasm_bindgen(js_name = groupY)]
    pub fn group_y(&self, f: &js_sys::Function) {
        let f = f.clone();
        self.force.borrow_mut().group_y = Box::new(move |a| {
            let this = JsValue::NULL;
            let index = JsValue::from_f64(a as f64);
            f.call1(&this, &index).ok().unwrap().as_f64().unwrap() as f32
        });
    }
}