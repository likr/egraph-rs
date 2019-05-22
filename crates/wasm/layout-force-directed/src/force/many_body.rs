use super::Force;
use egraph::layout::force_directed::force::ManyBodyForce as EgManyBodyForce;
use egraph_wasm_adapter::JsGraph;
use js_sys::Function;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct ManyBodyForce {
    force: Rc<RefCell<EgManyBodyForce<JsGraph>>>,
}

#[wasm_bindgen]
impl ManyBodyForce {
    #[wasm_bindgen(constructor)]
    pub fn new() -> ManyBodyForce {
        ManyBodyForce {
            force: Rc::new(RefCell::new(EgManyBodyForce::new())),
        }
    }

    pub fn force(&self) -> Force {
        Force::new(self.force.clone())
    }

    #[wasm_bindgen(setter = strength)]
    pub fn set_strength(&self, f: &Function) {
        let f = f.clone();
        self.force.borrow_mut().strength = Box::new(move |graph, u| {
            let this = JsValue::NULL;
            let graph = graph.data();
            let u = JsValue::from_f64(u as f64);
            f.call2(&this, &graph, &u).ok().unwrap().as_f64().unwrap() as f32
        });
    }
}
