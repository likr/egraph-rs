use super::RankingModule;
use crate::algorithm::ranking::longest_path;
use crate::{Graph, NodeIndex};
use std::collections::HashMap;

pub struct LongetPathRanking {}

impl LongetPathRanking {
    pub fn new() -> LongetPathRanking {
        LongetPathRanking {}
    }
}

impl<D, G: Graph<D>> RankingModule<D, G> for LongetPathRanking {
    fn call(&self, graph: &G) -> HashMap<NodeIndex, usize> {
        longest_path(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egraph_petgraph_adapter::PetgraphWrapper;
    use petgraph::Graph;

    #[test]
    fn test_longest_path() {
        let mut graph = Graph::<_, _>::new();
        let a = graph.add_node("a");
        let b = graph.add_node("b");
        let c = graph.add_node("c");
        let d = graph.add_node("d");
        let e = graph.add_node("e");
        graph.add_edge(a, b, ());
        graph.add_edge(b, c, ());
        graph.add_edge(d, c, ());
        graph.add_edge(d, e, ());
        let graph = PetgraphWrapper::new(graph);
        let layers = longest_path(&graph);
        assert_eq!(layers[&a.index()], 0);
        assert_eq!(layers[&b.index()], 1);
        assert_eq!(layers[&c.index()], 2);
        assert_eq!(layers[&d.index()], 0);
        assert_eq!(layers[&e.index()], 1);
    }
}
