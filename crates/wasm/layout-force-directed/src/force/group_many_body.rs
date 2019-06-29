use super::JsForce;
use egraph::layout::force_directed::force::GroupManyBodyForce;
use egraph::Graph;
use egraph_wasm_adapter::{JsGraph, JsGraphAdapter};
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = GroupManyBodyForce)]
pub struct JsGroupManyBodyForce {
    force: Rc<RefCell<GroupManyBodyForce<JsGraph, JsGraphAdapter>>>,
}

#[wasm_bindgen(js_class = GroupManyBodyForce)]
impl JsGroupManyBodyForce {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsGroupManyBodyForce {
        JsGroupManyBodyForce {
            force: Rc::new(RefCell::new(GroupManyBodyForce::new())),
        }
    }

    pub fn force(&self) -> JsForce {
        JsForce::new(self.force.clone())
    }

    #[wasm_bindgen(setter = group)]
    pub fn set_group(&self, f: &js_sys::Function) {
        let f = f.clone();
        self.force.borrow_mut().group = Box::new(move |graph, u| {
            let this = JsValue::NULL;
            let graph = graph.data();
            let u = JsValue::from_f64(u as f64);
            f.call2(&this, &graph, &u).ok().unwrap().as_f64().unwrap() as usize
        });
    }

    #[wasm_bindgen(setter = strength)]
    pub fn set_strength(&self, f: &js_sys::Function) {
        let f = f.clone();
        self.force.borrow_mut().strength = Box::new(move |graph, u| {
            let this = JsValue::NULL;
            let graph = graph.data();
            let u = JsValue::from_f64(u as f64);
            f.call2(&this, &graph, &u).ok().unwrap().as_f64().unwrap() as f32
        });
    }
}
