use fixedbitset::FixedBitSet;
use petgraph::graph::{IndexType, NodeIndex};
use petgraph::visit::{VisitMap, Visitable};
use petgraph::{Directed, Graph};
use std::collections::HashSet;

fn dfs<N, E, Ix: IndexType>(
    graph: &Graph<N, E, Directed, Ix>,
    map: &mut FixedBitSet,
    ancestors: &mut HashSet<NodeIndex<Ix>>,
    result: &mut Vec<(NodeIndex<Ix>, NodeIndex<Ix>)>,
    u: NodeIndex<Ix>,
) {
    if map.is_visited(&u) {
        return;
    }
    map.visit(u);
    ancestors.insert(u);
    for v in graph.neighbors(u) {
        if ancestors.contains(&v) {
            result.push((u, v));
        } else {
            dfs(graph, map, ancestors, result, v)
        }
    }
    ancestors.remove(&u);
}

pub fn cycle_edges<N, E, Ix: IndexType>(
    graph: &Graph<N, E, Directed, Ix>,
) -> Vec<(NodeIndex<Ix>, NodeIndex<Ix>)> {
    let mut map = graph.visit_map();
    let mut ancestors = HashSet::new();
    let mut result = vec![];
    for u in graph.node_indices() {
        dfs(&graph, &mut map, &mut ancestors, &mut result, u)
    }
    result
}

pub fn remove_cycle<N, E, Ix: IndexType>(graph: &mut Graph<N, E, Directed, Ix>) {
    for (u, v) in cycle_edges(graph) {
        let index = graph.find_edge(u, v).unwrap();
        let weight = graph.remove_edge(index).unwrap();
        graph.add_edge(v, u, weight);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::Graph;

    #[test]
    fn it_works() {
        let mut graph = Graph::<&str, &str>::new();
        let a = graph.add_node("a");
        let b = graph.add_node("b");
        let c = graph.add_node("c");
        graph.add_edge(a, b, "");
        graph.add_edge(b, c, "");
        graph.add_edge(c, a, "");
        assert_eq!(cycle_edges(&graph), vec![(c, a)]);
    }

    #[test]
    fn test_remove_cycle() {
        let mut graph = Graph::<&str, &str>::new();
        let a = graph.add_node("a");
        let b = graph.add_node("b");
        let c = graph.add_node("c");
        graph.add_edge(a, b, "");
        graph.add_edge(b, c, "");
        graph.add_edge(c, a, "");
        remove_cycle(&mut graph);
        assert!(graph.find_edge(a, b).is_some());
        assert!(graph.find_edge(a, c).is_some());
        assert!(graph.find_edge(b, c).is_some());
        assert_eq!(graph.edge_count(), 3);
    }
}
