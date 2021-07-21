use crate::layout::force_simulation::coordinates::JsCoordinates;
use petgraph_layout_force_simulation::force::{center, update_position};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = center)]
pub fn js_center(coordinates: &mut JsCoordinates) {
    center(coordinates.points_mut());
}

#[wasm_bindgen(js_name = updatePosition)]
pub fn js_update_position(coordinates: &mut JsCoordinates, velocity_decay: f32) {
    update_position(coordinates.points_mut(), velocity_decay);
}
