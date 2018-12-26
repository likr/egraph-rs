use super::super::super::super::graph::{Edge, EdgeType, IndexType, Node};
use super::force::Force;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct GroupPositionForce {
    force: Rc<
        RefCell<
            egraph::layout::force_directed::force::GroupPositionForce<
                Node,
                Edge,
                EdgeType,
                IndexType,
            >,
        >,
    >,
}

#[wasm_bindgen]
impl GroupPositionForce {
    #[wasm_bindgen(constructor)]
    pub fn new() -> GroupPositionForce {
        GroupPositionForce {
            force: Rc::new(RefCell::new(
                egraph::layout::force_directed::force::GroupPositionForce::new(),
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

    pub fn group(&self, f: &js_sys::Function) {
        let f = f.clone();
        self.force.borrow_mut().group = Box::new(move |_, a| {
            let this = JsValue::NULL;
            let index = JsValue::from_f64(a.index() as f64);
            f.call1(&this, &index).ok().unwrap().as_f64().unwrap() as usize
        });
    }

    pub fn group_x(&self, f: &js_sys::Function) {
        let f = f.clone();
        self.force.borrow_mut().group_x = Box::new(move |a| {
            let this = JsValue::NULL;
            let index = JsValue::from_f64(a as f64);
            f.call1(&this, &index).ok().unwrap().as_f64().unwrap() as f32
        });
    }

    pub fn group_y(&self, f: &js_sys::Function) {
        let f = f.clone();
        self.force.borrow_mut().group_y = Box::new(move |a| {
            let this = JsValue::NULL;
            let index = JsValue::from_f64(a as f64);
            f.call1(&this, &index).ok().unwrap().as_f64().unwrap() as f32
        });
    }
}
