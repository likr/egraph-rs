//! 2D Euclidean drawing functionality for WebAssembly.
//!
//! This module provides a WebAssembly binding for representing graph drawings
//! in 2D Euclidean space, allowing nodes to be positioned with x,y coordinates.

use crate::graph::{IndexType, JsGraph};
use js_sys::Array;
use petgraph::graph::{node_index, NodeIndex};
use petgraph_drawing::{Drawing, DrawingEuclidean2d};
use wasm_bindgen::prelude::*;

type NodeId = NodeIndex<IndexType>;

/// WebAssembly binding for 2D Euclidean graph drawings.
///
/// This struct provides a JavaScript interface for creating and manipulating
/// graph drawings in a 2D Euclidean space, where nodes have x,y coordinates.
#[wasm_bindgen(js_name = DrawingEuclidean2d)]
pub struct JsDrawingEuclidean2d {
    drawing: DrawingEuclidean2d<NodeId, f32>,
}

impl JsDrawingEuclidean2d {
    pub fn new(drawing: DrawingEuclidean2d<NodeId, f32>) -> Self {
        Self { drawing }
    }

    pub fn drawing(&self) -> &DrawingEuclidean2d<NodeId, f32> {
        &self.drawing
    }

    pub fn drawing_mut(&mut self) -> &mut DrawingEuclidean2d<NodeId, f32> {
        &mut self.drawing
    }
}

#[wasm_bindgen(js_class = DrawingEuclidean2d)]
impl JsDrawingEuclidean2d {
    /// Gets the x-coordinate of the node at the given index.
    ///
    /// Returns None if the node is not present in the drawing.
    pub fn x(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        self.drawing.x(u)
    }

    /// Gets the y-coordinate of the node at the given index.
    ///
    /// Returns None if the node is not present in the drawing.
    pub fn y(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        self.drawing.y(u)
    }

    /// Sets the x-coordinate of the node at the given index.
    #[wasm_bindgen(js_name = setX)]
    pub fn set_x(&mut self, u: usize, x: f32) {
        let u = node_index(u);
        self.drawing.set_x(u, x);
    }

    /// Sets the y-coordinate of the node at the given index.
    #[wasm_bindgen(js_name = setY)]
    pub fn set_y(&mut self, u: usize, y: f32) {
        let u = node_index(u);
        self.drawing.set_y(u, y);
    }

    /// Returns the number of nodes in the drawing.
    pub fn len(&self) -> usize {
        self.drawing.len()
    }

    /// Returns whether the drawing is empty (has no nodes).
    pub fn is_empty(&self) -> bool {
        self.drawing.is_empty()
    }

    /// Centralizes the drawing by moving all nodes so their center of mass is at the origin.
    ///
    /// This method adjusts all node coordinates so that the average position is (0,0).
    pub fn centralize(&mut self) {
        self.drawing.centralize();
    }

    /// Restricts node positions to be within the given rectangular region.
    ///
    /// Nodes outside the specified region will be moved to the closest point inside the region.
    #[wasm_bindgen(js_name = clampRegion)]
    pub fn clamp_region(&mut self, x0: f32, y0: f32, x1: f32, y1: f32) {
        self.drawing.clamp_region(x0, y0, x1, y1);
    }

    /// Returns line segments representing an edge between two nodes.
    ///
    /// For Euclidean 2D drawings, this typically returns a single straight line segment
    /// between the two node positions.
    #[wasm_bindgen(js_name = edgeSegments)]
    pub fn edge_segments(&self, u: usize, v: usize) -> Option<Box<[JsValue]>> {
        self.drawing
            .edge_segments(node_index(u), node_index(v))
            .map(|segments| {
                segments
                    .iter()
                    .map(|&(p, q)| {
                        let js_p = Array::new();
                        js_p.push(&JsValue::from_f64(p.0 as f64));
                        js_p.push(&JsValue::from_f64(p.1 as f64));
                        let js_q = Array::new();
                        js_q.push(&JsValue::from_f64(q.0 as f64));
                        js_q.push(&JsValue::from_f64(q.1 as f64));
                        let js_segment = Array::new();
                        js_segment.push(&js_p);
                        js_segment.push(&js_q);
                        js_segment.into()
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
            })
    }

    /// Creates a new drawing with an initial random placement of nodes from the given graph.
    ///
    /// Nodes are initially placed at random positions within a unit square.
    #[wasm_bindgen(js_name = initialPlacement)]
    pub fn initial_placement(graph: &JsGraph) -> Self {
        Self::new(DrawingEuclidean2d::initial_placement(graph.graph()))
    }
}
