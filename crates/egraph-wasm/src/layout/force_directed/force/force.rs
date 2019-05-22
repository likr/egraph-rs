use super::super::super::super::graph::{Edge, EdgeType, IndexType, Node};
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Force {
    force: Rc<RefCell<egraph::layout::force_directed::Force<Node, Edge, EdgeType, IndexType>>>,
}

impl Force {
    pub fn new(
        force: Rc<RefCell<egraph::layout::force_directed::Force<Node, Edge, EdgeType, IndexType>>>,
    ) -> Force {
        Force { force: force }
    }

    pub fn force(
        &self,
    ) -> Rc<RefCell<egraph::layout::force_directed::Force<Node, Edge, EdgeType, IndexType>>> {
        self.force.clone()
    }
}
