use crate::{graph::JsGraph, layout::force_simulation::coordinates::JsCoordinates};
use petgraph_algorithm_shortest_path::warshall_floyd;
use petgraph_quality_metrics::{number_of_crossings, shape_quality, stress};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = numberOfCrossings)]
pub fn js_number_of_crossings(graph: &JsGraph, coordinates: &JsCoordinates) -> usize {
    number_of_crossings(graph.graph(), coordinates.coordinates())
}

#[wasm_bindgen(js_name = shapeQuality)]
pub fn js_shape_quality(graph: &JsGraph, coordinates: &JsCoordinates) -> f32 {
    shape_quality(graph.graph(), coordinates.coordinates())
}

#[wasm_bindgen(js_name = stress)]
pub fn js_stress(graph: &JsGraph, coordinates: &JsCoordinates) -> f32 {
    let distance = warshall_floyd(graph.graph(), &mut |_| 1.0);
    stress(coordinates.coordinates(), &distance)
}
