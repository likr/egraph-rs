use crate::graph::{IndexType, JsGraph};
use js_sys::Array;
use petgraph::graph::{node_index, NodeIndex};
use petgraph_drawing::{Drawing2D, DrawingTorus};
use wasm_bindgen::prelude::*;

type NodeId = NodeIndex<IndexType>;
pub enum DrawingType {
    Drawing2D(Drawing2D<NodeId, f32>),
    DrawingTorus(DrawingTorus<NodeId, f32>),
}

#[wasm_bindgen(js_name = Drawing)]
pub struct JsDrawing {
    drawing: DrawingType,
}

impl JsDrawing {
    pub fn new_drawing_2d(drawing: Drawing2D<NodeId, f32>) -> Self {
        Self {
            drawing: DrawingType::Drawing2D(drawing),
        }
    }

    pub fn new_drawing_torus(drawing: DrawingTorus<NodeId, f32>) -> Self {
        Self {
            drawing: DrawingType::DrawingTorus(drawing),
        }
    }

    pub fn drawing(&self) -> &DrawingType {
        &self.drawing
    }

    pub fn drawing_mut(&mut self) -> &mut DrawingType {
        &mut self.drawing
    }
}

#[wasm_bindgen(js_class = Drawing)]
impl JsDrawing {
    pub fn x(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        match self.drawing() {
            DrawingType::Drawing2D(drawing) => drawing.x(u),
            DrawingType::DrawingTorus(drawing) => drawing.x(u),
        }
    }

    pub fn y(&self, u: usize) -> Option<f32> {
        let u = node_index(u);
        match self.drawing() {
            DrawingType::Drawing2D(drawing) => drawing.y(u),
            DrawingType::DrawingTorus(drawing) => drawing.y(u),
        }
    }

    #[wasm_bindgen(js_name = setX)]
    pub fn set_x(&mut self, u: usize, x: f32) {
        let u = node_index(u);
        match self.drawing_mut() {
            DrawingType::Drawing2D(drawing) => drawing.set_x(u, x),
            DrawingType::DrawingTorus(drawing) => drawing.set_x(u, x),
        };
    }

    #[wasm_bindgen(js_name = setY)]
    pub fn set_y(&mut self, u: usize, y: f32) {
        let u = node_index(u);
        match self.drawing_mut() {
            DrawingType::Drawing2D(drawing) => drawing.set_y(u, y),
            DrawingType::DrawingTorus(drawing) => drawing.set_y(u, y),
        };
    }

    pub fn len(&self) -> usize {
        match self.drawing() {
            DrawingType::Drawing2D(drawing) => drawing.len(),
            DrawingType::DrawingTorus(drawing) => drawing.len(),
        }
    }

    pub fn centralize(&mut self) {
        match self.drawing_mut() {
            DrawingType::Drawing2D(drawing) => drawing.centralize(),
            _ => unimplemented!(),
        };
    }

    #[wasm_bindgen(js_name = clampRegion)]
    pub fn clamp_region(&mut self, x0: f32, y0: f32, x1: f32, y1: f32) {
        match self.drawing_mut() {
            DrawingType::Drawing2D(drawing) => drawing.clamp_region(x0, y0, x1, y1),
            _ => unimplemented!(),
        };
    }

    #[wasm_bindgen(js_name = edgeSegments)]
    pub fn edge_segments(&self, u: usize, v: usize) -> Option<Box<[JsValue]>> {
        match self.drawing() {
            DrawingType::Drawing2D(drawing) => drawing
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
                }),
            DrawingType::DrawingTorus(drawing) => drawing
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
                }),
        }
    }
}

#[wasm_bindgen(js_name = Drawing2D)]
pub struct JsDrawing2D;

#[wasm_bindgen(js_class = Drawing2D)]
impl JsDrawing2D {
    #[wasm_bindgen(js_name = initialPlacement)]
    pub fn initial_placement(graph: &JsGraph) -> JsDrawing {
        JsDrawing::new_drawing_2d(Drawing2D::initial_placement(graph.graph()))
    }
}

#[wasm_bindgen(js_name = DrawingTorus)]
pub struct JsDrawingTorus;

#[wasm_bindgen(js_class = DrawingTorus)]
impl JsDrawingTorus {
    #[wasm_bindgen(js_name = initialPlacement)]
    pub fn initial_placement(graph: &JsGraph) -> JsDrawing {
        JsDrawing::new_drawing_torus(DrawingTorus::initial_placement(graph.graph()))
    }
}
