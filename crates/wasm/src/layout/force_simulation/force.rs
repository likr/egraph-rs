use crate::layout::force_simulation::coordinates::JsCoordinates;
use petgraph_layout_force_simulation::force::{center, clamp_region, update_position};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = center)]
pub fn js_center(coordinates: &mut JsCoordinates) {
    center(coordinates.points_mut());
}

#[wasm_bindgen(js_name = updatePosition)]
pub fn js_update_position(coordinates: &mut JsCoordinates, velocity_decay: f32) {
    update_position(coordinates.points_mut(), velocity_decay);
}

#[wasm_bindgen(js_name = clampRegion)]
pub fn js_clamp_region(coordinates: &mut JsCoordinates, x0: f32, y0: f32, x1: f32, y1: f32) {
    clamp_region(coordinates.points_mut(), x0, y0, x1, y1);
}
