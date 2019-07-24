use super::JsForce;
use egraph::layout::force_directed::force::RadialForce;
use egraph::Graph;
use egraph_wasm_adapter::{JsGraph, JsGraphAdapter};
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = RadialForce)]
pub struct JsRadialForce {
    force: Rc<RefCell<RadialForce<JsGraph, JsGraphAdapter>>>,
}

#[wasm_bindgen(js_class = RadialForce)]
impl JsRadialForce {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsRadialForce {
        JsRadialForce {
            force: Rc::new(RefCell::new(RadialForce::new())),
        }
    }

    pub fn force(&self) -> JsForce {
        JsForce::new(self.force.clone())
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

    #[wasm_bindgen(setter = radius)]
    pub fn set_radius(&self, f: &js_sys::Function) {
        let f = f.clone();
        self.force.borrow_mut().radius = Box::new(move |graph, u| {
            let this = JsValue::NULL;
            let graph = graph.data();
            let u = JsValue::from_f64(u as f64);
            let result = f.call2(&this, &graph, &u).ok().unwrap();
            if result.is_null() || result.is_undefined() {
                None
            } else {
                Some(result.as_f64().unwrap() as f32)
            }
        });
    }

    #[wasm_bindgen(setter = x)]
    pub fn set_x(&self, x: f32) {
        self.force.borrow_mut().x = x;
    }

    #[wasm_bindgen(setter = y)]
    pub fn set_y(&self, y: f32) {
        self.force.borrow_mut().y = y;
    }
}