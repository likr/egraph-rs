use crate::graph::JsGraph;
use crate::layout::force_simulation::coordinates::JsCoordinates;
use js_sys::Function;
use petgraph::visit::EdgeRef;
use petgraph_layout_sgd::{FullSgd, Sgd, SgdScheduler, SparseSgd};
use rand::prelude::*;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = "SgdScheduler")]
pub struct JsSgdScheduler {
    scheduler: SgdScheduler,
}

#[wasm_bindgen(js_class = "SgdScheduler")]
impl JsSgdScheduler {
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
            sgd: FullSgd::new(graph.graph(), &mut |e| length_map[&e.id()]),
        }
    }

    pub fn shuffle(&mut self) {
        let mut rng = thread_rng();
        self.sgd.shuffle(&mut rng);
    }

    pub fn apply(&self, coordinates: &mut JsCoordinates, eta: f32) {
        self.sgd.apply(coordinates.coordinates_mut(), eta);
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
    pub fn new(graph: &JsGraph, length: &Function, h: usize) -> JsSparseSgd {
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
            sgd: SparseSgd::new(graph.graph(), &mut |e| length_map[&e.id()], h),
        }
    }

    pub fn shuffle(&mut self) {
        let mut rng = thread_rng();
        self.sgd.shuffle(&mut rng);
    }

    pub fn apply(&self, coordinates: &mut JsCoordinates, eta: f32) {
        self.sgd.apply(coordinates.coordinates_mut(), eta);
    }

    pub fn scheduler(&self, t_max: usize, epsilon: f32) -> JsSgdScheduler {
        JsSgdScheduler {
            scheduler: self.sgd.scheduler(t_max, epsilon),
        }
    }
}
