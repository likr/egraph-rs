use super::Force;
use egraph::layout::force_directed::force::GroupLinkForce as EgGroupLinkForce;
use egraph_wasm_adapter::JsGraph;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct GroupLinkForce {
    force: Rc<RefCell<EgGroupLinkForce<JsGraph>>>,
}

#[wasm_bindgen]
impl GroupLinkForce {
    #[wasm_bindgen(constructor)]
    pub fn new() -> GroupLinkForce {
        GroupLinkForce {
            force: Rc::new(RefCell::new(EgGroupLinkForce::new())),
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

    #[wasm_bindgen(getter = intraGroup)]
    pub fn intra_group(&self) -> f32 {
        self.force.borrow_mut().intra_group
    }

    #[wasm_bindgen(setter = intraGroup)]
    pub fn set_intra_group(&self, value: f32) {
        self.force.borrow_mut().intra_group = value;
    }

    #[wasm_bindgen(getter = interGroup)]
    pub fn inter_group(&self) -> f32 {
        self.force.borrow_mut().inter_group
    }

    #[wasm_bindgen(setter = interGroup)]
    pub fn set_inter_group(&self, value: f32) {
        self.force.borrow_mut().inter_group = value;
    }

    #[wasm_bindgen(setter = distance)]
    pub fn set_distance(&self, f: &js_sys::Function) {
        let f = f.clone();
        self.force.borrow_mut().distance = Box::new(move |graph, u, v| {
            let this = JsValue::NULL;
            let graph = graph.data();
            let u = JsValue::from_f64(u as f64);
            let v = JsValue::from_f64(v as f64);
            f.call3(&this, &graph, &u, &v)
                .ok()
                .unwrap()
                .as_f64()
                .unwrap() as f32
        });
    }
}
