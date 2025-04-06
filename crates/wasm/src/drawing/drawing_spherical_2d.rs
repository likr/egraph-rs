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
    pub fn lon(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        self.drawing.lon(u)
    }

    /// Gets the latitude coordinate of the node at the given index.
    ///
    /// Returns None if the node is not present in the drawing.
    pub fn lat(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        self.drawing.lat(u)
    }

    /// Sets the longitude coordinate of the node at the given index.
    #[wasm_bindgen(js_name = setLon)]
    pub fn set_lon(&mut self, u: usize, value: f32) {
        let u = node_index(u);
        self.drawing.set_lon(u, value);
    }

    /// Sets the latitude coordinate of the node at the given index.
    #[wasm_bindgen(js_name = setLat)]
    pub fn set_lat(&mut self, u: usize, value: f32) {
        let u = node_index(u);
        self.drawing.set_lat(u, value);
    }

    /// Returns the number of nodes in the drawing.
    pub fn len(&self) -> usize {
        self.drawing.len()
    }

    /// Returns whether the drawing is empty (has no nodes).
    pub fn is_empty(&self) -> bool {
        self.drawing.is_empty()
    }

    /// Creates a new drawing with an initial random placement of nodes from the given graph.
    ///
    /// Nodes are initially placed around the sphere in a way that distributes them evenly.
    #[wasm_bindgen(js_name = initialPlacement)]
    pub fn initial_placement(graph: &JsGraph) -> Self {
        Self::new(DrawingSpherical2d::initial_placement(graph.graph()))
    }
}
