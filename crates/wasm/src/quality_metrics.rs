use crate::{graph::JsGraph, layout::force_simulation::coordinates::JsCoordinates};
use petgraph_algorithm_shortest_path::warshall_floyd;
use petgraph_quality_metrics::{crossing_number, neighborhood_preservation, stress};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = stress)]
pub fn js_stress(graph: &JsGraph, coordinates: &JsCoordinates) -> f32 {
    let distance = warshall_floyd(graph.graph(), &mut |_| 1.0);
    stress(coordinates.coordinates(), &distance)
}

#[wasm_bindgen(js_name = crossingNumber)]
pub fn js_crossing_number(graph: &JsGraph, coordinates: &JsCoordinates) -> f32 {
    crossing_number(graph.graph(), coordinates.coordinates())
}

#[wasm_bindgen(js_name = neighborhoodPreservation)]
pub fn js_neighborhood_preservation(graph: &JsGraph, coordinates: &JsCoordinates) -> f32 {
    neighborhood_preservation(graph.graph(), coordinates.coordinates())
}
