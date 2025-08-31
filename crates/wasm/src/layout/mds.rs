//! Multidimensional Scaling (MDS) layout algorithms for WebAssembly.
//!
//! This module provides WebAssembly bindings for MDS layout algorithms,
//! which are distance-based approaches for visualizing similarity data.
//! MDS algorithms attempt to place nodes in a lower-dimensional space
//! such that the distances between nodes reflect their similarity or
//! dissimilarity in the graph.
//!
//! Two main variants are provided:
//! * Classical MDS - Uses eigendecomposition on the full distance matrix
//! * Pivot MDS - A faster approximation using a subset of nodes as pivots

use crate::{
    drawing::{JsDrawingEuclidean, JsDrawingEuclidean2d},
    graph::JsGraph,
};
use js_sys::{Array, Function};
use petgraph::{graph::node_index, stable_graph::NodeIndex, visit::EdgeRef};
use petgraph_layout_mds::{ClassicalMds, PivotMds};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

/// WebAssembly binding for Classical Multidimensional Scaling.
///
/// Classical MDS is a dimension reduction technique that uses eigendecomposition
/// of the distance matrix to project points into a lower-dimensional space while
/// preserving distances as well as possible. This implementation is accurate but
/// computationally expensive for large graphs as it requires O(n²) memory and O(n³)
/// computation time.
#[wasm_bindgen(js_name = "ClassicalMds")]
pub struct JsClassicalMds {
    mds: ClassicalMds<NodeIndex, f32>,
}

#[wasm_bindgen(js_class = "ClassicalMds")]
impl JsClassicalMds {
    /// Creates a new Classical MDS layout instance for the given graph.
    ///
    /// Takes a graph and a length function that determines desired edge lengths.
    /// The length function should take an edge index and return its desired length.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsGraph, length: &Function) -> JsClassicalMds {
        let mut length_map = HashMap::new();
        for e in graph.graph().edge_indices() {
            let c = length
                .call1(&JsValue::null(), &JsValue::from_f64(e.index() as f64))
                .unwrap()
                .as_f64()
                .unwrap() as f32;
            length_map.insert(e, c);
        }
        JsClassicalMds {
            mds: ClassicalMds::new(graph.graph(), |e| length_map[&e.id()]),
        }
    }

    /// Executes the Classical MDS algorithm to generate a 2D layout.
    ///
    /// This method computes a layout that aims to preserve graph-theoretic distances
    /// in a 2D Euclidean space.
    #[wasm_bindgen(js_name = "run2d")]
    pub fn run_2d(&self) -> JsDrawingEuclidean2d {
        JsDrawingEuclidean2d::new(self.mds.run_2d())
    }

    /// Executes the Classical MDS algorithm to generate an n-dimensional layout.
    ///
    /// This method allows generating layouts in arbitrary dimensions, which can be
    /// useful for advanced visualization techniques or further dimensionality reduction.
    ///
    /// The parameter specifies the number of dimensions for the layout.
    pub fn run(&self, d: usize) -> JsDrawingEuclidean {
        JsDrawingEuclidean::new(self.mds.run(d))
    }
}

/// WebAssembly binding for Pivot Multidimensional Scaling.
///
/// Pivot MDS is an efficient approximation of Classical MDS that uses a subset
/// of nodes as pivots to reduce computational complexity. This makes it suitable
/// for large graphs where Classical MDS would be too slow or memory-intensive.
/// The quality of the layout depends on the number and choice of pivot nodes.
#[wasm_bindgen(js_name = "PivotMds")]
pub struct JsPivotMds {
    mds: PivotMds<NodeIndex, f32>,
}

#[wasm_bindgen(js_class = "PivotMds")]
impl JsPivotMds {
    /// Creates a new Pivot MDS layout instance for the given graph.
    ///
    /// Takes a graph, a length function, and an array of node indices to use as pivots.
    /// The length function should take an edge index and return its desired length.
    /// The pivot nodes are used to approximate distances in the graph.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsGraph, length: &Function, sources: &Array) -> JsPivotMds {
        let sources = sources
            .iter()
            .map(|item| node_index(item.as_f64().unwrap() as usize))
            .collect::<Vec<_>>();
        let mut length_map = HashMap::new();
        for e in graph.graph().edge_indices() {
            let c = length
                .call1(&JsValue::null(), &JsValue::from_f64(e.index() as f64))
                .unwrap()
                .as_f64()
                .unwrap() as f32;
            length_map.insert(e, c);
        }
        JsPivotMds {
            mds: PivotMds::new(graph.graph(), |e| length_map[&e.id()], &sources),
        }
    }

    /// Executes the Pivot MDS algorithm to generate a 2D layout.
    ///
    /// This method computes an efficient layout approximation using the pivot nodes.
    /// The resulting layout preserves distances between nodes and pivots, which often
    /// produces good overall distance preservation.
    #[wasm_bindgen(js_name = "run2d")]
    pub fn run_2d(&self) -> JsDrawingEuclidean2d {
        JsDrawingEuclidean2d::new(self.mds.run_2d())
    }
}
