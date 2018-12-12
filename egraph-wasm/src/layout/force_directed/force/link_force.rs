use super::super::super::super::graph::Graph;
use super::force::Force;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct LinkForce {
    force: Rc<RefCell<egraph::layout::force_directed::force::LinkForce>>,
}

#[wasm_bindgen]
impl LinkForce {
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &Graph) -> LinkForce {
        LinkForce {
            force: Rc::new(RefCell::new(
                egraph::layout::force_directed::force::LinkForce::new(&graph.graph()),
            )),
        }
    }

    pub fn force(&self) -> Force {
        Force::new(self.force.clone())
    }

    #[wasm_bindgen(js_name = getStrength)]
    pub fn get_strength(&self) -> f64 {
        self.force.borrow().strength as f64
    }

    #[wasm_bindgen(js_name = setStrength)]
    pub fn set_strength(&self, value: f64) {
        self.force.borrow_mut().strength = value as f32;
    }
}
