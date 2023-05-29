use crate::{drawing::JsDrawing, graph::JsGraph};
use petgraph_layout_force_atlas2::ForceAtlas2;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = ForceAtlas2)]
pub struct JsForceAtlas2 {
    force_atlas2: ForceAtlas2<f32>,
}

#[wasm_bindgen(js_class=ForceAtlas2)]
impl JsForceAtlas2 {
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsGraph) -> JsForceAtlas2 {
        JsForceAtlas2 {
            force_atlas2: ForceAtlas2::new(graph.graph()),
        }
    }

    pub fn apply(&self, drawing: &mut JsDrawing, alpha: f32) {
        self.force_atlas2.apply(drawing.drawing_mut(), alpha);
    }

    #[wasm_bindgen(js_name = applyToNode)]
    pub fn apply_to_node(&self, u: usize, drawing: &mut JsDrawing, alpha: f32) {
        self.force_atlas2
            .apply_to_node(u, drawing.drawing_mut(), alpha)
    }
}
