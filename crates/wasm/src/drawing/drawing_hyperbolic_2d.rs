use crate::graph::{IndexType, JsGraph};
use petgraph::graph::{node_index, NodeIndex};
use petgraph_drawing::{Drawing, DrawingHyperbolic2d};
use wasm_bindgen::prelude::*;

type NodeId = NodeIndex<IndexType>;

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
    pub fn x(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        self.drawing.x(u)
    }

    pub fn y(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        self.drawing.y(u)
    }

    #[wasm_bindgen(js_name = setX)]
    pub fn set_x(&mut self, u: usize, value: f32) {
        let u = node_index(u);
        self.drawing.set_x(u, value);
    }

    #[wasm_bindgen(js_name = setY)]
    pub fn set_y(&mut self, u: usize, value: f32) {
        let u = node_index(u);
        self.drawing.set_y(u, value);
    }

    pub fn len(&self) -> usize {
        self.drawing.len()
    }

    #[wasm_bindgen(js_name = initialPlacement)]
    pub fn initial_placement(graph: &JsGraph) -> Self {
        Self::new(DrawingHyperbolic2d::initial_placement(graph.graph()))
    }
}
