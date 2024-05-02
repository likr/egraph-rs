use crate::{
    drawing::{JsDrawingEuclidean2d, JsDrawingTorus2d},
    graph::JsGraph,
};
use petgraph_algorithm_shortest_path::warshall_floyd;
use petgraph_quality_metrics::{
    crossing_edges, crossing_edges_torus, crossing_number_with_crossing_edges,
    neighborhood_preservation, stress,
};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = stress)]
pub fn js_stress(graph: &JsGraph, drawing: &JsDrawingEuclidean2d) -> f32 {
    let distance = warshall_floyd(graph.graph(), &mut |_| 1.0);
    stress(drawing.drawing(), &distance)
}

#[wasm_bindgen(js_name = crossingNumber)]
pub fn js_crossing_number(graph: &JsGraph, drawing: &JsDrawingEuclidean2d) -> f32 {
    let crossings = crossing_edges(graph.graph(), drawing.drawing());
    crossing_number_with_crossing_edges(&crossings)
}

#[wasm_bindgen(js_name = crossingNumberWithDrawingTorus2d)]
pub fn js_crossing_number_with_drawing_torus_2d(
    graph: &JsGraph,
    drawing: &JsDrawingTorus2d,
) -> f32 {
    let crossings = crossing_edges_torus(graph.graph(), drawing.drawing());
    crossing_number_with_crossing_edges(&crossings)
}

#[wasm_bindgen(js_name = neighborhoodPreservation)]
pub fn js_neighborhood_preservation(graph: &JsGraph, drawing: &JsDrawingEuclidean2d) -> f32 {
    neighborhood_preservation(graph.graph(), drawing.drawing())
}
