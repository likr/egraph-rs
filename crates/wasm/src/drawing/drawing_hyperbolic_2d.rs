//! 2D hyperbolic drawing functionality for WebAssembly.
//!
//! This module provides a WebAssembly binding for representing graph drawings
//! in 2D hyperbolic space, which can be useful for visualizing large graphs
//! with hierarchical structure due to the exponential growth of area with distance.

use crate::graph::{IndexType, JsGraph};
use petgraph::graph::{node_index, NodeIndex};
use petgraph_drawing::{Drawing, DrawingHyperbolic2d};
use wasm_bindgen::prelude::*;

type NodeId = NodeIndex<IndexType>;

/// WebAssembly binding for 2D hyperbolic graph drawings.
///
/// This struct provides a JavaScript interface for creating and manipulating
/// graph drawings in a 2D hyperbolic space, commonly visualized using the
/// Poincaré disk model. Hyperbolic space has the property that the area
/// grows exponentially with the radius, making it useful for visualizing
/// large graphs with hierarchical structure.
#[wasm_bindgen(js_name = DrawingHyperbolic2d)]
pub struct JsDrawingHyperbolic2d {
    drawing: DrawingHyperbolic2d<NodeId, f32>,
}

impl JsDrawingHyperbolic2d {
    pub fn new(drawing: DrawingHyperbolic2d<NodeId, f32>) -> Self {
        Self { drawing }
    }

    pub fn drawing(&self) -> &DrawingHyperbolic2d<NodeId, f32> {
        &self.drawing
    }

    pub fn drawing_mut(&mut self) -> &mut DrawingHyperbolic2d<NodeId, f32> {
        &mut self.drawing
    }
}

#[wasm_bindgen(js_class = DrawingHyperbolic2d)]
impl JsDrawingHyperbolic2d {
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
    pub fn set_x(&mut self, u: usize, value: f32) {
        let u = node_index(u);
        self.drawing.set_x(u, value);
    }

    /// Sets the y-coordinate of the node at the given index.
    #[wasm_bindgen(js_name = setY)]
    pub fn set_y(&mut self, u: usize, value: f32) {
        let u = node_index(u);
        self.drawing.set_y(u, value);
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
    /// Nodes are initially placed in a way that respects hyperbolic space properties.
    #[wasm_bindgen(js_name = initialPlacement)]
    pub fn initial_placement(graph: &JsGraph) -> Self {
        Self::new(DrawingHyperbolic2d::initial_placement(graph.graph()))
    }
}
