use super::super::super::super::graph::{Edge, EdgeType, IndexType, Node};
use super::force::Force;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct GroupLinkForce {
    force: Rc<
        RefCell<
            egraph::layout::force_directed::force::GroupLinkForce<Node, Edge, EdgeType, IndexType>,
        >,
    >,
}

#[wasm_bindgen]
impl GroupLinkForce {
    #[wasm_bindgen(constructor)]
    pub fn new() -> GroupLinkForce {
        GroupLinkForce {
            force: Rc::new(RefCell::new(
                egraph::layout::force_directed::force::GroupLinkForce::new(),
            )),
        }
    }

    pub fn force(&self) -> Force {
        Force::new(self.force.clone())
    }

    pub fn intra_group(&self, value: f64) {
        self.force.borrow_mut().intra_group = value as f32;
    }

    pub fn inter_group(&self, value: f64) {
        self.force.borrow_mut().inter_group = value as f32;
    }

    pub fn distance(&self, f: &js_sys::Function) {
        let f = f.clone();
        self.force.borrow_mut().distance = Box::new(move |_, a| {
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
}
