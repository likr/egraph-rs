use super::JsForce;
use egraph::layout::force_directed::force::GroupCenterForce;
use egraph::Graph;
use egraph_wasm_adapter::{JsGraph, JsGraphAdapter};
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = GroupCenterForce)]
pub struct JsGroupCenterForce {
    force: Rc<RefCell<GroupCenterForce<JsGraph, JsGraphAdapter>>>,
}

#[wasm_bindgen(js_class = GroupCenterForce)]
impl JsGroupCenterForce {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsGroupCenterForce {
        JsGroupCenterForce {
            force: Rc::new(RefCell::new(GroupCenterForce::new())),
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

    #[wasm_bindgen(setter = groupX)]
    pub fn set_group_x(&self, f: &js_sys::Function) {
        let f = f.clone();
        self.force.borrow_mut().group_x = Box::new(move |index| {
            let this = JsValue::NULL;
            let index = JsValue::from_f64(index as f64);
            f.call1(&this, &index).ok().unwrap().as_f64().unwrap() as f32
        });
    }

    #[wasm_bindgen(setter = groupY)]
    pub fn set_group_y(&self, f: &js_sys::Function) {
        let f = f.clone();
        self.force.borrow_mut().group_y = Box::new(move |index| {
            let this = JsValue::NULL;
            let index = JsValue::from_f64(index as f64);
            f.call1(&this, &index).ok().unwrap().as_f64().unwrap() as f32
        });
    }
}