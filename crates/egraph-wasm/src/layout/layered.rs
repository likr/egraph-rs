use super::super::graph::Graph;
use super::convert::convert_to_object;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct SugiyamaLayout {
    layouter: egraph_layered::SugiyamaLayout<usize>,
}

#[wasm_bindgen]
impl SugiyamaLayout {
    #[wasm_bindgen(constructor)]
    pub fn new() -> SugiyamaLayout {
        SugiyamaLayout {
            layouter: egraph_layered::SugiyamaLayout::new(),
        }
    }

    pub fn call(&self, graph: &Graph) -> JsValue {
        let layout = self.layouter.call(&graph.graph());
        convert_to_object(layout)
    }

    pub fn set_min_width_ranking(&mut self, width_limit: usize) {
        let mut ranking = egraph_layered::ranking::MinWidthRanking::new();
        ranking.width_limit = width_limit;
        self.layouter.ranking_module = Box::new(ranking);
    }
}
