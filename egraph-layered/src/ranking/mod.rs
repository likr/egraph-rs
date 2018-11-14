use petgraph::graph::NodeIndex;
use petgraph::Graph;
use std::collections::HashMap;

mod longest_path;
pub mod min_width;
mod ranking;

pub use self::ranking::RankingModule;

pub struct LongetPathRanking {}

impl LongetPathRanking {
    pub fn new() -> LongetPathRanking {
        LongetPathRanking {}
    }
}

impl<N, E> RankingModule<N, E> for LongetPathRanking {
    fn call(&self, graph: &Graph<N, E>) -> HashMap<NodeIndex, usize> {
        longest_path::longest_path(&graph)
    }
}
