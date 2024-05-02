use crate::graph::{IndexType, JsGraph};
use petgraph::graph::{node_index, NodeIndex};
use petgraph_drawing::{Drawing, DrawingSpherical2d};
use wasm_bindgen::prelude::*;

type NodeId = NodeIndex<IndexType>;

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
    pub fn lon(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        self.drawing.lon(u)
    }

    pub fn lat(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        self.drawing.lat(u)
    }

    #[wasm_bindgen(js_name = setX)]
    pub fn set_lon(&mut self, u: usize, value: f32) {
        let u = node_index(u);
        self.drawing.set_lon(u, value);
    }

    #[wasm_bindgen(js_name = setY)]
    pub fn set_lat(&mut self, u: usize, value: f32) {
        let u = node_index(u);
        self.drawing.set_lat(u, value);
    }

    pub fn len(&self) -> usize {
        self.drawing.len()
    }

    #[wasm_bindgen(js_name = initialPlacement)]
    pub fn initial_placement(graph: &JsGraph) -> Self {
        Self::new(DrawingSpherical2d::initial_placement(graph.graph()))
    }
}
