pub mod center;
pub mod collide;
pub mod group_center;
pub mod group_link;
pub mod group_many_body;
pub mod group_position;
pub mod link;
pub mod many_body;
pub mod position;
pub mod radial;

use egraph::layout::force_directed::force::Force;
use egraph_wasm_adapter::{JsGraph, JsGraphAdapter};
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = Force)]
pub struct JsForce {
    force: Rc<RefCell<dyn Force<JsGraph, JsGraphAdapter>>>,
}

impl JsForce {
    pub fn new(force: Rc<RefCell<dyn Force<JsGraph, JsGraphAdapter>>>) -> JsForce {
        JsForce { force: force }
    }

    pub fn force(&self) -> Rc<RefCell<dyn Force<JsGraph, JsGraphAdapter>>> {
        self.force.clone()
    }
}
