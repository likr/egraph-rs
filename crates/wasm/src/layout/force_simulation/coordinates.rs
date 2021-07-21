use crate::graph::{IndexType, JsGraph};
use petgraph::graph::{node_index, NodeIndex};
use petgraph_layout_force_simulation::{initial_placement, Coordinates, Point};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = Coordinates)]
pub struct JsCoordinates {
    coordinates: Coordinates<IndexType>,
}

impl JsCoordinates {
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
}

#[wasm_bindgen(js_name = initialPlacement)]
pub fn js_initial_placement(graph: &JsGraph) -> JsCoordinates {
    let coordinates = initial_placement(graph.graph());
    JsCoordinates { coordinates }
}
