use crate::graph::JsGraph;
use crate::layout::force_simulation::coordinates::JsCoordinates;
use petgraph_layout_force_atlas2::ForceAtlas2Force;
use petgraph_layout_force_simulation::{Force, ForceToNode};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = ForceAtlas2Force)]
pub struct JsForceAtlas2Force {
    force: ForceAtlas2Force,
}

#[wasm_bindgen(js_class=ForceAtlas2Force)]
impl JsForceAtlas2Force {
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsGraph) -> JsForceAtlas2Force {
        JsForceAtlas2Force {
            force: ForceAtlas2Force::new(graph.graph()),
        }
    }

    pub fn apply(&self, coordinates: &mut JsCoordinates, alpha: f32) {
        self.force.apply(coordinates.points_mut(), alpha);
    }

    #[wasm_bindgen(js_name = applyToNode)]
    pub fn apply_to_node(&self, u: usize, coordinates: &mut JsCoordinates, alpha: f32) {
        self.force.apply_to_node(u, coordinates.points_mut(), alpha)
    }
}
