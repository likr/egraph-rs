//! 2D spherical drawing functionality for WebAssembly.
//!
//! This module provides a WebAssembly binding for representing graph drawings
//! on the surface of a sphere, which can be useful for visualizing global
//! networks, planetary data, or when attempting to reduce edge crossings
//! compared to planar layouts.

use crate::graph::{IndexType, JsGraph};
use petgraph::graph::{node_index, NodeIndex};
use petgraph_drawing::{Drawing, DrawingSpherical2d};
use wasm_bindgen::prelude::*;

type NodeId = NodeIndex<IndexType>;

/// WebAssembly binding for 2D spherical graph drawings.
///
/// This struct provides a JavaScript interface for creating and manipulating
/// graph drawings on the surface of a sphere, commonly represented using
/// longitude and latitude coordinates. Spherical drawings are useful for
/// global visualization contexts and can minimize certain aesthetic issues
/// that occur with planar layouts.
#[wasm_bindgen(js_name = DrawingSpherical2d)]
pub struct JsDrawingSpherical2d {
    drawing: DrawingSpherical2d<NodeId, f32>,
}

impl JsDrawingSpherical2d {
    pub fn new(drawing: DrawingSpherical2d<NodeId, f32>) -> Self {
        Self { drawing }
    }

    pub fn drawing(&self) -> &DrawingSpherical2d<NodeId, f32> {
        &self.drawing
    }

    pub fn drawing_mut(&mut self) -> &mut DrawingSpherical2d<NodeId, f32> {
        &mut self.drawing
    }
}

#[wasm_bindgen(js_class = DrawingSpherical2d)]
impl JsDrawingSpherical2d {
    /// Gets the longitude coordinate of the node at the given index.
    ///
    /// Returns None if the node is not present in the drawing.
    ///
    /// @param {number} u - The node index
    /// @returns {number|null} The longitude coordinate if the node exists, null otherwise
    pub fn lon(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        self.drawing.lon(u)
    }

    /// Gets the latitude coordinate of the node at the given index.
    ///
    /// Returns None if the node is not present in the drawing.
    ///
    /// @param {number} u - The node index
    /// @returns {number|null} The latitude coordinate if the node exists, null otherwise
    pub fn lat(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        self.drawing.lat(u)
    }

    /// Sets the longitude coordinate of the node at the given index.
    ///
    /// @param {number} u - The node index
    /// @param {number} value - The new longitude coordinate in radians
    #[wasm_bindgen(js_name = setX)]
    pub fn set_lon(&mut self, u: usize, value: f32) {
        let u = node_index(u);
        self.drawing.set_lon(u, value);
    }

    /// Sets the latitude coordinate of the node at the given index.
    ///
    /// @param {number} u - The node index
    /// @param {number} value - The new latitude coordinate in radians
    #[wasm_bindgen(js_name = setY)]
    pub fn set_lat(&mut self, u: usize, value: f32) {
        let u = node_index(u);
        self.drawing.set_lat(u, value);
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

    /// Creates a new drawing with an initial random placement of nodes from the given graph.
    ///
    /// Nodes are initially placed around the sphere in a way that distributes them evenly.
    ///
    /// @param {Graph} graph - The graph whose nodes to position
    /// @returns {DrawingSpherical2d} A new drawing with initial positions for all nodes in the graph
    #[wasm_bindgen(js_name = initialPlacement)]
    pub fn initial_placement(graph: &JsGraph) -> Self {
        Self::new(DrawingSpherical2d::initial_placement(graph.graph()))
    }
}
