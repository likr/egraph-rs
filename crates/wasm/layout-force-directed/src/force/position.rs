use super::Force;
use egraph::layout::force_directed::force::PositionForce as EgPositionForce;
use egraph_wasm_adapter::JsGraph;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct PositionForce {
    force: Rc<RefCell<EgPositionForce<JsGraph>>>,
}

#[wasm_bindgen]
impl PositionForce {
    #[wasm_bindgen(constructor)]
    pub fn new() -> PositionForce {
        PositionForce {
            force: Rc::new(RefCell::new(EgPositionForce::new())),
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

    pub fn x(&self, f: &js_sys::Function) {
        let f = f.clone();
        self.force.borrow_mut().x = Box::new(move |_, a| {
            let this = JsValue::NULL;
            let index = JsValue::from_f64(a as f64);
            let result = f.call1(&this, &index).ok().unwrap();
            if result.is_null() || result.is_undefined() {
                None
            } else {
                Some(result.as_f64().unwrap() as f32)
            }
        });
    }

    pub fn y(&self, f: &js_sys::Function) {
        let f = f.clone();
        self.force.borrow_mut().y = Box::new(move |_, a| {
            let this = JsValue::NULL;
            let index = JsValue::from_f64(a as f64);
            let result = f.call1(&this, &index).ok().unwrap();
            if result.is_null() || result.is_undefined() {
                None
            } else {
                Some(result.as_f64().unwrap() as f32)
            }
        });
    }
}
