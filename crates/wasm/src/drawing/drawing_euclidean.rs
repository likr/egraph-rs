//! N-dimensional Euclidean drawing functionality for WebAssembly.
//!
//! This module provides a WebAssembly binding for representing graph drawings
//! in n-dimensional Euclidean space. Unlike the 2D-specific implementations,
//! this drawing type supports arbitrary dimensions, making it suitable for
//! embedding graphs in higher-dimensional spaces.

use crate::graph::IndexType;
use petgraph::graph::{node_index, NodeIndex};
use petgraph_drawing::{Drawing, DrawingEuclidean};
use wasm_bindgen::prelude::*;

type NodeId = NodeIndex<IndexType>;

/// WebAssembly binding for n-dimensional Euclidean graph drawings.
///
/// This struct provides a JavaScript interface for creating and manipulating
/// graph drawings in n-dimensional Euclidean space. It allows for positioning
/// nodes in arbitrary dimensions, which can be useful for specialized layout
/// algorithms or for visualizing high-dimensional data.
#[wasm_bindgen(js_name = DrawingEuclidean)]
pub struct JsDrawingEuclidean {
    drawing: DrawingEuclidean<NodeId, f32>,
}

impl JsDrawingEuclidean {
    pub fn new(drawing: DrawingEuclidean<NodeId, f32>) -> Self {
        Self { drawing }
    }

    pub fn drawing(&self) -> &DrawingEuclidean<NodeId, f32> {
        &self.drawing
    }

    pub fn drawing_mut(&mut self) -> &mut DrawingEuclidean<NodeId, f32> {
        &mut self.drawing
    }
}

#[wasm_bindgen(js_class = DrawingEuclidean)]
impl JsDrawingEuclidean {
    /// Gets the coordinate of the node at the given index in the specified dimension.
    ///
    /// Returns None if the node is not present in the drawing or if the dimension
    /// is out of bounds.
    ///
    /// @param {number} u - The node index
    /// @param {number} d - The dimension index (0 for x, 1 for y, 2 for z, etc.)
    /// @returns {number|null} The coordinate if the node exists, null otherwise
    pub fn get(&self, u: usize, d: usize) -> Option<f32> {
        let u = node_index(u);
        self.drawing.get(u, d)
    }

    /// Sets the coordinate of the node at the given index in the specified dimension.
    ///
    /// @param {number} u - The node index
    /// @param {number} d - The dimension index (0 for x, 1 for y, 2 for z, etc.)
    /// @param {number} value - The new coordinate value
    pub fn set(&mut self, u: usize, d: usize, value: f32) {
        let u = node_index(u);
        self.drawing.set(u, d, value);
    }

    /// Returns the number of nodes in the drawing.
    ///
    /// @returns {number} The number of nodes with coordinates
    pub fn len(&self) -> usize {
        self.drawing.len()
    }

    /// Returns whether the drawing is empty (has no nodes).
    ///
    /// @returns {boolean} True if the drawing has no nodes, false otherwise
    pub fn is_empty(&self) -> bool {
        self.drawing.is_empty()
    }
}
