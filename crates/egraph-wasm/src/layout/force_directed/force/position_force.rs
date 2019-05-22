use super::super::super::super::graph::{Edge, EdgeType, IndexType, Node};
use super::force::Force;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct PositionForce {
    force: Rc<
        RefCell<
            egraph::layout::force_directed::force::PositionForce<Node, Edge, EdgeType, IndexType>,
        >,
    >,
}

#[wasm_bindgen]
impl PositionForce {
    #[wasm_bindgen(constructor)]
    pub fn new() -> PositionForce {
        PositionForce {
            force: Rc::new(RefCell::new(
                egraph::layout::force_directed::force::PositionForce::new(),
            )),
        }
    }

    pub fn force(&self) -> Force {
        Force::new(self.force.clone())
    }

    pub fn strength(&self, f: &js_sys::Function) {
        let f = f.clone();
        self.force.borrow_mut().strength = Box::new(move |_, a| {
            let this = JsValue::NULL;
            let index = JsValue::from_f64(a.index() as f64);
            f.call1(&this, &index).ok().unwrap().as_f64().unwrap() as f32
        });
    }

    pub fn x(&self, f: &js_sys::Function) {
        let f = f.clone();
        self.force.borrow_mut().x = Box::new(move |_, a| {
            let this = JsValue::NULL;
            let index = JsValue::from_f64(a.index() as f64);
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
            let index = JsValue::from_f64(a.index() as f64);
            let result = f.call1(&this, &index).ok().unwrap();
            if result.is_null() || result.is_undefined() {
                None
            } else {
                Some(result.as_f64().unwrap() as f32)
            }
        });
    }
}
