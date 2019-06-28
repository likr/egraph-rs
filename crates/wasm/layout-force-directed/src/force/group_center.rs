use super::Force;
use egraph::layout::force_directed::force::GroupCenterForce as EgGroupCenterForce;
use egraph_wasm_adapter::JsGraph;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct GroupCenterForce {
    force: Rc<RefCell<EgGroupCenterForce<JsGraph>>>,
}

#[wasm_bindgen]
impl GroupCenterForce {
    #[wasm_bindgen(constructor)]
    pub fn new() -> GroupCenterForce {
        GroupCenterForce {
            force: Rc::new(RefCell::new(EgGroupCenterForce::new())),
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