use crate::graph::IndexType;
use petgraph::graph::{node_index, NodeIndex};
use petgraph_drawing::{Drawing, DrawingEuclidean};
use wasm_bindgen::prelude::*;

type NodeId = NodeIndex<IndexType>;

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
    pub fn get(&self, u: usize, d: usize) -> Option<f32> {
        let u = node_index(u);
        self.drawing.get(u, d)
    }

    pub fn set(&mut self, u: usize, d: usize, value: f32) {
        let u = node_index(u);
        self.drawing.set(u, d, value);
    }

    pub fn len(&self) -> usize {
        self.drawing.len()
    }

    pub fn is_empty(&self) -> bool {
        self.drawing.is_empty()
    }
}
