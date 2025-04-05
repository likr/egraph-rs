//! 2D torus drawing functionality for WebAssembly.
//!
//! This module provides a WebAssembly binding for representing graph drawings
//! on the surface of a torus (donut shape). Torus drawings have the advantage
//! of eliminating edge effects at boundaries, as the surface wraps around in
//! both dimensions.

use crate::graph::{IndexType, JsGraph};
use js_sys::Array;
use petgraph::graph::{node_index, NodeIndex};
use petgraph_drawing::{Drawing, DrawingTorus2d};
use wasm_bindgen::prelude::*;

type NodeId = NodeIndex<IndexType>;

/// WebAssembly binding for 2D torus graph drawings.
///
/// This struct provides a JavaScript interface for creating and manipulating
/// graph drawings on the surface of a torus. A torus is a donut-shaped surface
/// that wraps around in both dimensions, providing a continuous layout space
/// without boundaries. This can be beneficial for certain types of networks
/// where boundary effects are undesirable.
#[wasm_bindgen(js_name = DrawingTorus2d)]
pub struct JsDrawingTorus2d {
    drawing: DrawingTorus2d<NodeId, f32>,
}

impl JsDrawingTorus2d {
    pub fn new(drawing: DrawingTorus2d<NodeId, f32>) -> Self {
        Self { drawing }
    }

    pub fn drawing(&self) -> &DrawingTorus2d<NodeId, f32> {
        &self.drawing
    }

    pub fn drawing_mut(&mut self) -> &mut DrawingTorus2d<NodeId, f32> {
        &mut self.drawing
    }
}

#[wasm_bindgen(js_class = DrawingTorus2d)]
impl JsDrawingTorus2d {
    /// Gets the x-coordinate of the node at the given index.
    ///
    /// Returns None if the node is not present in the drawing.
    /// The coordinate is a value in the range [0, 1] representing a position
    /// on the torus surface.
    ///
    /// @param {number} u - The node index
    /// @returns {number|null} The x-coordinate if the node exists, null otherwise
    pub fn x(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        self.drawing.x(u)
    }

    /// Gets the y-coordinate of the node at the given index.
    ///
    /// Returns None if the node is not present in the drawing.
    /// The coordinate is a value in the range [0, 1] representing a position
    /// on the torus surface.
    ///
    /// @param {number} u - The node index
    /// @returns {number|null} The y-coordinate if the node exists, null otherwise
    pub fn y(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        self.drawing.y(u)
    }

    /// Sets the x-coordinate of the node at the given index.
    ///
    /// The coordinate should be a value in the range [0, 1]. Values outside
    /// this range will wrap around due to the torus geometry.
    ///
    /// @param {number} u - The node index
    /// @param {number} x - The new x-coordinate
    #[wasm_bindgen(js_name = setX)]
    pub fn set_x(&mut self, u: usize, x: f32) {
        let u = node_index(u);
        self.drawing.set_x(u, x);
    }

    /// Sets the y-coordinate of the node at the given index.
    ///
    /// The coordinate should be a value in the range [0, 1]. Values outside
    /// this range will wrap around due to the torus geometry.
    ///
    /// @param {number} u - The node index
    /// @param {number} y - The new y-coordinate
    #[wasm_bindgen(js_name = setY)]
    pub fn set_y(&mut self, u: usize, y: f32) {
        let u = node_index(u);
        self.drawing.set_y(u, y);
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

    /// Returns line segments representing an edge between two nodes.
    ///
    /// For torus drawings, an edge may be split into multiple line segments
    /// due to the wrapping around nature of the torus. This method returns all
    /// segments needed to properly draw the edge.
    ///
    /// @param {number} u - The source node index
    /// @param {number} v - The target node index
    /// @returns {Array|null} An array of line segments, each containing two points [[x1,y1], [x2,y2]], or null if either node is not in the drawing
    #[wasm_bindgen(js_name = edgeSegments)]
    pub fn edge_segments(&self, u: usize, v: usize) -> Option<Box<[JsValue]>> {
        self.drawing
            .edge_segments(node_index(u), node_index(v))
            .map(|segments| {
                segments
                    .iter()
                    .map(|&(p, q)| {
                        let js_p = Array::new();
                        js_p.push(&JsValue::from_f64(p.0 .0 as f64));
                        js_p.push(&JsValue::from_f64(p.1 .0 as f64));
                        let js_q = Array::new();
                        js_q.push(&JsValue::from_f64(q.0 .0 as f64));
                        js_q.push(&JsValue::from_f64(q.1 .0 as f64));
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
    /// Nodes are initially placed at random positions within the torus surface.
    ///
    /// @param {Graph} graph - The graph whose nodes to position
    /// @returns {DrawingTorus2d} A new drawing with initial positions for all nodes in the graph
    #[wasm_bindgen(js_name = initialPlacement)]
    pub fn initial_placement(graph: &JsGraph) -> Self {
        Self::new(DrawingTorus2d::initial_placement(graph.graph()))
    }
}
