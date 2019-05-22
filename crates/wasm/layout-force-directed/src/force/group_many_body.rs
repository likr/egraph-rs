use super::Force;
use egraph::layout::force_directed::force::GroupManyBodyForce as EgGroupManyBodyForce;
use egraph_wasm_adapter::JsGraph;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct GroupManyBodyForce {
    force: Rc<RefCell<EgGroupManyBodyForce<JsGraph>>>,
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
