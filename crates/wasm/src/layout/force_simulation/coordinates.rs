use crate::graph::{IndexType, JsGraph};
use petgraph::graph::{node_index, NodeIndex};
use petgraph_layout_force_simulation::{Coordinates, Point};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = Coordinates)]
pub struct JsCoordinates {
    coordinates: Coordinates<IndexType>,
}

impl JsCoordinates {
    pub fn new(coordinates: Coordinates<IndexType>) -> JsCoordinates {
        JsCoordinates { coordinates }
    }
    pub fn indices(&self) -> &[NodeIndex<IndexType>] {
        &self.coordinates.indices
    }

    pub fn indices_mut(&mut self) -> &mut [NodeIndex<IndexType>] {
        &mut self.coordinates.indices
    }

    pub fn points(&self) -> &[Point] {
        &self.coordinates.points
    }

    pub fn points_mut(&mut self) -> &mut [Point] {
        &mut self.coordinates.points
    }

    pub fn coordinates(&self) -> &Coordinates<IndexType> {
        &self.coordinates
    }

    pub fn coordinates_mut(&mut self) -> &mut Coordinates<IndexType> {
        &mut self.coordinates
    }

    pub fn position(&self, u: usize) -> Option<(f32, f32)> {
        let u = node_index(u);
        self.coordinates.position(u)
    }

    pub fn set_position(&mut self, u: usize, p: (f32, f32)) {
        let u = node_index(u);
        self.coordinates.set_position(u, p);
    }
}

#[wasm_bindgen(js_class = Coordinates)]
impl JsCoordinates {
    pub fn x(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        self.coordinates.x(u)
    }

    pub fn y(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        self.coordinates.y(u)
    }

    #[wasm_bindgen(js_name = setX)]
    pub fn set_x(&mut self, u: usize, x: f32) {
        let u = node_index(u);
        self.coordinates.set_x(u, x);
    }

    #[wasm_bindgen(js_name = setY)]
    pub fn set_y(&mut self, u: usize, y: f32) {
        let u = node_index(u);
        self.coordinates.set_y(u, y);
    }

    pub fn vx(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        self.coordinates.vx(u)
    }

    pub fn vy(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        self.coordinates.vy(u)
    }

    pub fn len(&self) -> usize {
        self.coordinates.len()
    }

    #[wasm_bindgen(js_name = toJSON)]
    pub fn to_json(&self) -> JsValue {
        let result = self
            .coordinates
            .iter()
            .map(|(u, p)| (u.index(), (p.x, p.y)))
            .collect::<HashMap<usize, (f32, f32)>>();
        JsValue::from_serde(&result).unwrap()
    }

    pub fn centralize(&mut self) {
        self.coordinates.centralize();
    }

    #[wasm_bindgen(js_name = updatePosition)]
    pub fn update_position(&mut self, velocity_decay: f32) {
        self.coordinates.update_position(velocity_decay);
    }

    #[wasm_bindgen(js_name = clampRegion)]
    pub fn clamp_region(&mut self, x0: f32, y0: f32, x1: f32, y1: f32) {
        self.coordinates.clamp_region(x0, y0, x1, y1);
    }

    #[wasm_bindgen(js_name = initialPlacement)]
    pub fn initial_placement(graph: &JsGraph) -> JsCoordinates {
        JsCoordinates::new(Coordinates::initial_placement(graph.graph()))
    }
}
