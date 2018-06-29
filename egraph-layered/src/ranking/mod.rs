use std::collections::HashMap;
use petgraph::Graph;
use petgraph::graph::NodeIndex;
use super::graph::{Node, Edge};

mod longest_path;

pub trait RankingModule {
    fn call(&self, graph: &Graph<Node, Edge>) -> HashMap<NodeIndex, usize>;
}

pub struct LongetPathRanking {}

impl LongetPathRanking {
    pub fn new() -> LongetPathRanking {
        LongetPathRanking {}
    }
}

impl RankingModule for LongetPathRanking {
    fn call(&self, graph: &Graph<Node, Edge>) -> HashMap<NodeIndex, usize> {
        longest_path::longest_path(&graph)
    }
}
