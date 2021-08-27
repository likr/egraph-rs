use crate::graph::JsGraph;
use crate::layout::force_simulation::coordinates::JsCoordinates;
use petgraph_layout_force_simulation::{Force, ForceToNode};
use petgraph_layout_fruchterman_reingold::FruchtermanReingoldForce;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = FruchtermanReingoldForce)]
pub struct JsFruchtermanReingoldForce {
    force: FruchtermanReingoldForce,
}

#[wasm_bindgen(js_class=FruchtermanReingoldForce)]
impl JsFruchtermanReingoldForce {
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsGraph, k: f32, min_distance: f32) -> JsFruchtermanReingoldForce {
        JsFruchtermanReingoldForce {
            force: FruchtermanReingoldForce::new(graph.graph(), k, min_distance),
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
