use super::super::super::super::graph::{Edge, EdgeType, IndexType, Node};
use super::force::Force;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct LinkForce {
    force: Rc<
        RefCell<egraph::layout::force_directed::force::LinkForce<Node, Edge, EdgeType, IndexType>>,
    >,
}

#[wasm_bindgen]
impl LinkForce {
    #[wasm_bindgen(constructor)]
    pub fn new() -> LinkForce {
        LinkForce {
            force: Rc::new(RefCell::new(
                egraph::layout::force_directed::force::LinkForce::new(),
            )),
        }
    }

    pub fn force(&self) -> Force {
        Force::new(self.force.clone())
    }

    pub fn strength(&self, f: &js_sys::Function) {
        let f = f.clone();
        self.force.borrow_mut().strength = Box::new(move |_, e| {
            let this = JsValue::NULL;
            let index = JsValue::from_f64(e.index() as f64);
            f.call1(&this, &index).ok().unwrap().as_f64().unwrap() as f32
        });
    }

    pub fn distance(&self, f: &js_sys::Function) {
        let f = f.clone();
        self.force.borrow_mut().distance = Box::new(move |_, e| {
            let this = JsValue::NULL;
            let index = JsValue::from_f64(e.index() as f64);
            f.call1(&this, &index).ok().unwrap().as_f64().unwrap() as f32
        });
    }
}
