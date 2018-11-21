use super::ranking::RankingModule;
use petgraph::graph::{IndexType, NodeIndex};
use petgraph::{Directed, EdgeDirection, Graph};
use std::collections::HashMap;

fn dfs<N, E, Ix: IndexType>(
    graph: &Graph<N, E, Directed, Ix>,
    layers: &mut HashMap<NodeIndex<Ix>, usize>,
    u: NodeIndex<Ix>,
    depth: usize,
) {
    for v in graph.neighbors(u) {
        if layers.contains_key(&v) {
            let layer = layers.get_mut(&v).unwrap();
            if *layer <= depth {
                *layer = depth + 1
            }
        } else {
            layers.insert(v, depth + 1);
        }
        dfs(graph, layers, v, depth + 1);
    }
}

pub fn longest_path<N, E, Ix: IndexType>(
    graph: &Graph<N, E, Directed, Ix>,
) -> HashMap<NodeIndex<Ix>, usize> {
    let mut result = HashMap::new();
    for u in graph.externals(EdgeDirection::Incoming) {
        result.insert(u, 0);
        dfs(graph, &mut result, u, 0);
    }
    result
}

pub struct LongetPathRanking {}

impl LongetPathRanking {
    pub fn new() -> LongetPathRanking {
        LongetPathRanking {}
    }
}

impl<N, E, Ix: IndexType> RankingModule<N, E, Ix> for LongetPathRanking {
    fn call(&self, graph: &Graph<N, E, Directed, Ix>) -> HashMap<NodeIndex<Ix>, usize> {
        longest_path(&graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::Graph;

    #[test]
    fn test_longest_path() {
        let mut graph = Graph::<&str, &str>::new();
        let a = graph.add_node("a");
        let b = graph.add_node("b");
        let c = graph.add_node("c");
        let d = graph.add_node("d");
        let e = graph.add_node("e");
        graph.add_edge(a, b, "");
        graph.add_edge(b, c, "");
        graph.add_edge(d, c, "");
        graph.add_edge(d, e, "");
        let layers = longest_path(&graph);
        assert_eq!(*layers.get(&a).unwrap(), 0);
        assert_eq!(*layers.get(&b).unwrap(), 1);
        assert_eq!(*layers.get(&c).unwrap(), 2);
        assert_eq!(*layers.get(&d).unwrap(), 0);
        assert_eq!(*layers.get(&e).unwrap(), 1);
    }
}
