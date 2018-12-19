use super::super::super::super::graph::{Edge, EdgeType, Graph, IndexType, Node};
use egraph::layout::grouping::Grouping;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct TreemapGrouping {
    grouping: egraph::layout::grouping::TreemapGrouping<Node, Edge, EdgeType, IndexType>,
    group: Box<
        Fn(
            &petgraph::Graph<Node, Edge, EdgeType, IndexType>,
            petgraph::graph::NodeIndex<IndexType>,
        ) -> usize,
    >,
}

#[wasm_bindgen]
impl TreemapGrouping {
    #[wasm_bindgen(constructor)]
    pub fn new() -> TreemapGrouping {
        TreemapGrouping {
            grouping: egraph::layout::grouping::TreemapGrouping::new(),
            group: Box::new(|_, _| 0),
        }
    }

    pub fn call(&mut self, graph: &Graph, width: f64, height: f64) {
        let result = self
            .grouping
            .call(&graph.graph(), width as f32, height as f32);
    }
}
