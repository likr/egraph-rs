use crate::layout::force_simulation::coordinates::JsCoordinates;
use petgraph_layout_non_euclidean_force_simulation::{HyperbolicSpace, Map, SphericalSpace};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name= HyperbolicSpace)]
pub struct JsHyperbolicSpace {}

#[wasm_bindgen(js_class = HyperbolicSpace)]
impl JsHyperbolicSpace {
    #[wasm_bindgen(js_name = toTangentSpace)]
    pub fn to_tangent_space(a: JsValue, b: JsValue) -> Result<JsValue, JsValue> {
        let x: (f32, f32) = a
            .into_serde()
            .map_err(|e| JsValue::from(format!("{}", e)))?;
        let y: (f32, f32) = b
            .into_serde()
            .map_err(|e| JsValue::from(format!("{}", e)))?;
        Ok(JsValue::from_serde(&HyperbolicSpace::to_tangent_space(x, y)).unwrap())
    }

    #[wasm_bindgen(js_name = fromTangentSpace)]
    pub fn from_tangent_space(a: JsValue, b: JsValue) -> Result<JsValue, JsValue> {
        let x: (f32, f32) = a
            .into_serde()
            .map_err(|e| JsValue::from(format!("{}", e)))?;
        let y: (f32, f32) = b
            .into_serde()
            .map_err(|e| JsValue::from(format!("{}", e)))?;
        Ok(JsValue::from_serde(&HyperbolicSpace::from_tangent_space(x, y)).unwrap())
    }

    #[wasm_bindgen(js_name = mapToTangentSpace)]
    pub fn map_to_tangent_space(
        i: usize,
        riemann_space: &JsCoordinates,
        tangent_space: &mut JsCoordinates,
    ) {
        HyperbolicSpace::map_to_tangent_space(
            i,
            riemann_space.points(),
            tangent_space.points_mut(),
        );
    }

    #[wasm_bindgen(js_name = updatePosition)]
    pub fn update_position(
        i: usize,
        riemann_space: &mut JsCoordinates,
        tangent_space: &JsCoordinates,
        velocity_decay: f32,
    ) {
        HyperbolicSpace::update_position(
            i,
            riemann_space.points_mut(),
            tangent_space.points(),
            velocity_decay,
        );
    }
}

#[wasm_bindgen(js_name= SphericalSpace)]
pub struct JsSphericalSpace {}

#[wasm_bindgen(js_class = SphericalSpace)]
impl JsSphericalSpace {
    #[wasm_bindgen(js_name = toTangentSpace)]
    pub fn to_tangent_space(a: JsValue, b: JsValue) -> Result<JsValue, JsValue> {
        let x: (f32, f32) = a
            .into_serde()
            .map_err(|e| JsValue::from(format!("{}", e)))?;
        let y: (f32, f32) = b
            .into_serde()
            .map_err(|e| JsValue::from(format!("{}", e)))?;
        Ok(JsValue::from_serde(&SphericalSpace::to_tangent_space(x, y)).unwrap())
    }

    #[wasm_bindgen(js_name = fromTangentSpace)]
    pub fn from_tangent_space(a: JsValue, b: JsValue) -> Result<JsValue, JsValue> {
        let x: (f32, f32) = a
            .into_serde()
            .map_err(|e| JsValue::from(format!("{}", e)))?;
        let y: (f32, f32) = b
            .into_serde()
            .map_err(|e| JsValue::from(format!("{}", e)))?;
        Ok(JsValue::from_serde(&SphericalSpace::from_tangent_space(x, y)).unwrap())
    }

    #[wasm_bindgen(js_name = mapToTangentSpace)]
    pub fn map_to_tangent_space(
        i: usize,
        riemann_space: &JsCoordinates,
        tangent_space: &mut JsCoordinates,
    ) {
        SphericalSpace::map_to_tangent_space(i, riemann_space.points(), tangent_space.points_mut());
    }

    #[wasm_bindgen(js_name = updatePosition)]
    pub fn update_position(
        i: usize,
        riemann_space: &mut JsCoordinates,
        tangent_space: &JsCoordinates,
        velocity_decay: f32,
    ) {
        SphericalSpace::update_position(
            i,
            riemann_space.points_mut(),
            tangent_space.points(),
            velocity_decay,
        );
    }
}
