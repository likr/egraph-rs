use crate::{drawing::JsDrawing, graph::JsGraph, rng::JsRng};
use js_sys::Function;
use petgraph::visit::EdgeRef;
use petgraph_layout_sgd::{FullSgd, Sgd, SgdScheduler, SparseSgd};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = "SgdScheduler")]
pub struct JsSgdScheduler {
    scheduler: SgdScheduler,
}

#[wasm_bindgen(js_class = "SgdScheduler")]
impl JsSgdScheduler {
    pub fn run(&mut self, f: &Function) {
        self.scheduler.run(&mut |eta| {
            f.call1(&JsValue::null(), &(eta as f64).into()).ok();
        })
    }

    pub fn step(&mut self, f: &Function) {
        self.scheduler.step(&mut |eta| {
            f.call1(&JsValue::null(), &(eta as f64).into()).ok();
        })
    }

    #[wasm_bindgen(js_name = "isFinished")]
    pub fn is_finished(&self) -> bool {
        self.scheduler.is_finished()
    }
}

#[wasm_bindgen(js_name = "FullSgd")]
pub struct JsFullSgd {
    sgd: FullSgd,
}

#[wasm_bindgen(js_class = "FullSgd")]
impl JsFullSgd {
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsGraph, length: &Function) -> JsFullSgd {
        let mut length_map = HashMap::new();
        for e in graph.graph().edge_indices() {
            let c = length
                .call1(&JsValue::null(), &JsValue::from_f64(e.index() as f64))
                .unwrap()
                .as_f64()
                .unwrap() as f32;
            length_map.insert(e, c);
        }
        JsFullSgd {
            sgd: FullSgd::new(graph.graph(), |e| length_map[&e.id()]),
        }
    }

    pub fn shuffle(&mut self, rng: &mut JsRng) {
        self.sgd.shuffle(rng.get_mut());
    }

    pub fn apply(&self, drawing: &mut JsDrawing, eta: f32) {
        self.sgd.apply(drawing.drawing_mut(), eta);
    }

    pub fn scheduler(&self, t_max: usize, epsilon: f32) -> JsSgdScheduler {
        JsSgdScheduler {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }
}

#[wasm_bindgen(js_name = "SparseSgd")]
pub struct JsSparseSgd {
    sgd: SparseSgd,
}

#[wasm_bindgen(js_class = "SparseSgd")]
impl JsSparseSgd {
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &JsGraph, length: &Function, h: usize, rng: &mut JsRng) -> JsSparseSgd {
        let mut length_map = HashMap::new();
        for e in graph.graph().edge_indices() {
            let c = length
                .call1(&JsValue::null(), &JsValue::from_f64(e.index() as f64))
                .unwrap()
                .as_f64()
                .unwrap() as f32;
            length_map.insert(e, c);
        }
        JsSparseSgd {
            sgd: SparseSgd::new_with_rng(graph.graph(), |e| length_map[&e.id()], h, rng.get_mut()),
        }
    }

    pub fn shuffle(&mut self, rng: &mut JsRng) {
        self.sgd.shuffle(rng.get_mut());
    }

    pub fn apply(&self, drawing: &mut JsDrawing, eta: f32) {
        self.sgd.apply(drawing.drawing_mut(), eta);
    }

    pub fn scheduler(&self, t_max: usize, epsilon: f32) -> JsSgdScheduler {
        JsSgdScheduler {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }
}
