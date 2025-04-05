//! Edge bundling algorithms for WebAssembly.
//!
//! This module provides WebAssembly bindings for edge bundling algorithms,
//! which improve the readability of dense graphs by routing related edges
//! along similar paths to reduce visual clutter.
//!
//! The main algorithm provided is Force-Directed Edge Bundling (FDEB),
//! which uses forces between edge segments to bundle similar edges together.

use crate::{drawing::JsDrawingEuclidean2d, graph::JsGraph};
use petgraph_edge_bundling_fdeb::{fdeb, EdgeBundlingOptions};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

/// Applies the Force-Directed Edge Bundling (FDEB) algorithm to a graph drawing.
///
/// The FDEB algorithm reduces visual clutter in dense graphs by bundling similar edges
/// together, making the graph structure more apparent. The algorithm works by
/// subdividing edges into segments and applying forces between these segments to
/// bundle related edges together.
///
/// @param {Graph} graph - The graph whose edges to bundle
/// @param {DrawingEuclidean2d} drawing - The drawing containing node positions
/// @returns {Object} A map from edge indices to arrays of line segments, where each line segment
///                  is an array of points representing the bundled edge path
#[wasm_bindgen(js_name = fdeb)]
pub fn js_fdeb(graph: &JsGraph, drawing: JsDrawingEuclidean2d) -> JsValue {
    let options = EdgeBundlingOptions::<f32>::new();
    let bends = fdeb(graph.graph(), drawing.drawing(), &options)
        .into_iter()
        .map(|(e, lines)| (e.index(), lines))
        .collect::<HashMap<_, _>>();
    serde_wasm_bindgen::to_value(&bends).unwrap()
}
