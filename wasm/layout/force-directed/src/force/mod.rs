pub mod center;
pub mod collide;
pub mod group_center;
pub mod group_link;
pub mod group_many_body;
pub mod group_position;
pub mod link;
pub mod many_body;
pub mod position;

use egraph_layout_force_directed::force::Force as EgForce;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Force {
    force: Rc<RefCell<EgForce>>,
}

impl Force {
    pub fn new(force: Rc<RefCell<EgForce>>) -> Force {
        Force { force: force }
    }

    pub fn force(&self) -> Rc<RefCell<EgForce>> {
        self.force.clone()
    }
}
