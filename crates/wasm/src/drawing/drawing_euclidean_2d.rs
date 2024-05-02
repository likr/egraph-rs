use crate::graph::{IndexType, JsGraph};
use js_sys::Array;
use petgraph::graph::{node_index, NodeIndex};
use petgraph_drawing::{Drawing, DrawingEuclidean2d};
use wasm_bindgen::prelude::*;

type NodeId = NodeIndex<IndexType>;

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

    #[wasm_bindgen(js_name = initialPlacement)]
    pub fn initial_placement(graph: &JsGraph) -> Self {
        Self::new(DrawingEuclidean2d::initial_placement(graph.graph()))
    }
}
