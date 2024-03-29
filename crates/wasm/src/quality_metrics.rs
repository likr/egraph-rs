use crate::{drawing::JsDrawing, graph::JsGraph};
use petgraph_algorithm_shortest_path::warshall_floyd;
use petgraph_quality_metrics::{crossing_number, neighborhood_preservation, stress};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = stress)]
pub fn js_stress(graph: &JsGraph, drawing: &JsDrawing) -> f32 {
    let distance = warshall_floyd(graph.graph(), &mut |_| 1.0);
    stress(drawing.drawing(), &distance)
}

#[wasm_bindgen(js_name = crossingNumber)]
pub fn js_crossing_number(graph: &JsGraph, drawing: &JsDrawing) -> f32 {
    crossing_number(graph.graph(), drawing.drawing())
}

#[wasm_bindgen(js_name = neighborhoodPreservation)]
pub fn js_neighborhood_preservation(graph: &JsGraph, drawing: &JsDrawing) -> f32 {
    neighborhood_preservation(graph.graph(), drawing.drawing())
}
