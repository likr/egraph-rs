use crate::graph::{IndexType, JsGraph};
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

    // pub fn indices(&self) -> &[NodeId] {
    //     match self.drawing {
    //         DrawingType::Drawing2D(drawing) => &drawing.indices,
    //         DrawingType::DrawingTorus(drawing) => &drawing.indices,
    //     }
    // }

    // pub fn indices_mut(&mut self) -> &mut [NodeId] {
    //     match self.drawing {
    //         DrawingType::Drawing2D(mut drawing) => &mut drawing.indices,
    //         DrawingType::DrawingTorus(mut drawing) => &mut drawing.indices,
    //     }
    // }

    // pub fn coordinates(&self) -> &[Tuple2D<f32>] {
    //     match self.drawing {
    //         DrawingType::Drawing2D(mut drawing) => &drawing.coordinates,
    //         DrawingType::DrawingTorus(mut drawing) => &drawing.coordinates,
    //     }
    // }

    // pub fn coordinates_mut(&mut self) -> &mut [Tuple2D<f32>] {
    //     &mut self.drawing.coordinates
    // }

    pub fn drawing(&self) -> &DrawingType {
        &self.drawing
    }

    pub fn drawing_mut(&mut self) -> &mut DrawingType {
        &mut self.drawing
    }

    // pub fn position(&self, u: usize) -> Option<&Tuple2D<f32>> {
    //     let u = node_index(u);
    //     self.drawing.position(u)
    // }

    // pub fn set_position(&mut self, u: usize, p: Tuple2D<f32>) {
    //     let u = node_index(u);
    //     self.drawing.position_mut(u).map(|q| *q = p);
    // }
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
