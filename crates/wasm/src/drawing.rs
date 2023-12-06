use crate::graph::{IndexType, JsGraph};
use petgraph::graph::{node_index, NodeIndex};
use petgraph_drawing::{Drawing2D, Tuple2D};
use wasm_bindgen::prelude::*;

type NodeId = NodeIndex<IndexType>;
type DrawingImpl = Drawing2D<NodeId, f32>;

#[wasm_bindgen(js_name = Drawing)]
pub struct JsDrawing {
    drawing: DrawingImpl,
}

impl JsDrawing {
    pub fn new(drawing: DrawingImpl) -> Self {
        Self { drawing }
    }
    pub fn indices(&self) -> &[NodeId] {
        &self.drawing.indices
    }

    pub fn indices_mut(&mut self) -> &mut [NodeId] {
        &mut self.drawing.indices
    }

    pub fn coordinates(&self) -> &[Tuple2D<f32>] {
        &self.drawing.coordinates
    }

    pub fn coordinates_mut(&mut self) -> &mut [Tuple2D<f32>] {
        &mut self.drawing.coordinates
    }

    pub fn drawing(&self) -> &DrawingImpl {
        &self.drawing
    }

    pub fn drawing_mut(&mut self) -> &mut DrawingImpl {
        &mut self.drawing
    }

    pub fn position(&self, u: usize) -> Option<&Tuple2D<f32>> {
        let u = node_index(u);
        self.drawing.position(u)
    }

    pub fn set_position(&mut self, u: usize, p: Tuple2D<f32>) {
        let u = node_index(u);
        self.drawing.position_mut(u).map(|q| *q = p);
    }
}

#[wasm_bindgen(js_class = Drawing)]
impl JsDrawing {
    pub fn x(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        self.drawing.x(u)
    }

    pub fn y(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        self.drawing.y(u)
    }

    #[wasm_bindgen(js_name = setX)]
    pub fn set_x(&mut self, u: usize, x: f32) {
        let u = node_index(u);
        self.drawing.set_x(u, x);
    }

    #[wasm_bindgen(js_name = setY)]
    pub fn set_y(&mut self, u: usize, y: f32) {
        let u = node_index(u);
        self.drawing.set_y(u, y);
    }

    pub fn len(&self) -> usize {
        self.drawing.len()
    }

    pub fn centralize(&mut self) {
        self.drawing.centralize();
    }

    #[wasm_bindgen(js_name = clampRegion)]
    pub fn clamp_region(&mut self, x0: f32, y0: f32, x1: f32, y1: f32) {
        self.drawing.clamp_region(x0, y0, x1, y1);
    }

    #[wasm_bindgen(js_name = initialPlacement)]
    pub fn initial_placement(graph: &JsGraph) -> Self {
        Self::new(Drawing2D::initial_placement(graph.graph()))
    }
}
