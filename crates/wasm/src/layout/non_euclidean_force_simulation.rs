use crate::layout::force_simulation::coordinates::JsCoordinates;
use js_sys::Function;
use petgraph_layout_non_euclidean_force_simulation::apply_in_hyperbolic_space;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = applyInHyperbolicSpace)]
pub fn js_apply_in_hyperbolic_space(
    coordinates: &mut JsCoordinates,
    buffer: &mut JsCoordinates,
    velocity_decay: f32,
    f: Function,
) {
    apply_in_hyperbolic_space(
        coordinates.points_mut(),
        buffer.points_mut(),
        velocity_decay,
        &mut |u, _| {
            f.call1(&JsValue::null(), &(u as f32).into()).ok();
        },
    );
}
