//! Graph drawing quality metrics for WebAssembly.
//!
//! This module provides WebAssembly bindings for various metrics that evaluate
//! the quality of graph layouts. These metrics quantify different aesthetic
//! criteria such as edge crossings, distance preservation, and neighborhood
//! preservation, allowing for objective comparison between different layout algorithms.

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

/// Calculates the stress metric for a graph drawing.
///
/// Stress measures how well a drawing preserves the graph-theoretic distances
/// between nodes. Lower stress values indicate better distance preservation.
/// The metric is calculated as the sum of squared differences between graph-theoretic
/// distances and geometric distances, weighted by the inverse square of the
/// graph-theoretic distances.
#[wasm_bindgen(js_name = stress)]
pub fn js_stress(graph: &JsGraph, drawing: &JsDrawingEuclidean2d) -> f32 {
    let distance = warshall_floyd(graph.graph(), &mut |_| 1.0);
    stress(drawing.drawing(), &distance)
}

/// Calculates the crossing number of a graph drawing in Euclidean 2D space.
///
/// The crossing number counts how many edge pairs cross each other in the drawing.
/// Lower values indicate clearer drawings with fewer visual intersections.
#[wasm_bindgen(js_name = crossingNumber)]
pub fn js_crossing_number(graph: &JsGraph, drawing: &JsDrawingEuclidean2d) -> f32 {
    let crossings = crossing_edges(graph.graph(), drawing.drawing());
    crossing_number_with_crossing_edges(&crossings)
}

/// Calculates the crossing number of a graph drawing in torus 2D space.
///
/// This is similar to the regular crossing number, but accounts for the
/// different geometry of a torus where the surface wraps around in both dimensions.
#[wasm_bindgen(js_name = crossingNumberWithDrawingTorus2d)]
pub fn js_crossing_number_with_drawing_torus_2d(
    graph: &JsGraph,
    drawing: &JsDrawingTorus2d,
) -> f32 {
    let crossings = crossing_edges_torus(graph.graph(), drawing.drawing());
    crossing_number_with_crossing_edges(&crossings)
}

/// Calculates how well a drawing preserves node neighborhoods.
///
/// This metric measures how well the k-nearest neighbors in the drawing
/// correspond to the actual graph neighbors. A value closer to 1.0 indicates
/// better neighborhood preservation.
#[wasm_bindgen(js_name = neighborhoodPreservation)]
pub fn js_neighborhood_preservation(graph: &JsGraph, drawing: &JsDrawingEuclidean2d) -> f32 {
    neighborhood_preservation(graph.graph(), drawing.drawing())
}
