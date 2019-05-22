pub mod center;
pub mod collide;
pub mod group_center;
pub mod group_link;
pub mod group_many_body;
pub mod group_position;
pub mod link;
pub mod many_body;
pub mod position;

use egraph::layout::force_directed::force::Force as EgForce;
use egraph_wasm_adapter::JsGraph;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Force {
    force: Rc<RefCell<EgForce<JsGraph>>>,
}

impl Force {
    pub fn new(force: Rc<RefCell<EgForce<JsGraph>>>) -> Force {
        Force { force: force }
    }

    pub fn force(&self) -> Rc<RefCell<EgForce<JsGraph>>> {
        self.force.clone()
    }
}
