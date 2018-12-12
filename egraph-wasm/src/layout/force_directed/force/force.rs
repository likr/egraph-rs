use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Force {
    force: Rc<RefCell<egraph::layout::force_directed::Force>>,
}

impl Force {
    pub fn new(force: Rc<RefCell<egraph::layout::force_directed::Force>>) -> Force {
        Force { force: force }
    }

    pub fn force(&self) -> Rc<RefCell<egraph::layout::force_directed::Force>> {
        self.force.clone()
    }
}
