use crate::layout::force_simulation::coordinates::JsCoordinates;
use js_sys::{Function, Reflect};
use petgraph_layout_force_simulation::force::update_with;
use wasm_bindgen::convert::*;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = updateWith)]
pub fn js_update_with(
    coordinates_obj: &JsValue,
    alpha: f32,
    velocity_decay: f32,
    f: &Function,
) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    let ptr = Reflect::get(&coordinates_obj, &"ptr".into())?
        .as_f64()
        .ok_or_else(|| "xxx")? as u32;
    let mut coordinates = unsafe { JsCoordinates::ref_mut_from_abi(ptr) };
    let mut points = coordinates
        .points_mut()
        .iter()
        .map(|&p| p)
        .collect::<Vec<_>>();
    std::mem::drop(coordinates);
    update_with(&mut points, alpha, velocity_decay, &mut |points, alpha| {
        points[0].x = 100.;
        f.call1(&JsValue::null(), &(alpha as f64).into()).ok();
    });
    Ok(())
}
