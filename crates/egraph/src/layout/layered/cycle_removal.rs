use crate::{Graph, NodeIndex};
use std::collections::HashSet;

fn dfs<D, G: Graph<D>>(
    graph: &G,
    visited: &mut HashSet<NodeIndex>,
    ancestors: &mut HashSet<NodeIndex>,
    result: &mut Vec<(NodeIndex, NodeIndex)>,
    u: NodeIndex,
) {
    if visited.contains(&u) {
        return;
    }
    visited.insert(u);
    ancestors.insert(u);
    for v in graph.out_nodes(u) {
        if ancestors.contains(&v) {
            result.push((u, v));
        } else {
            dfs(graph, visited, ancestors, result, v)
        }
    }
    ancestors.remove(&u);
}

pub fn cycle_edges<D, G: Graph<D>>(graph: &G) -> Vec<(NodeIndex, NodeIndex)> {
    let mut visited = HashSet::new();
    let mut ancestors = HashSet::new();
    let mut result = vec![];
    for u in graph.nodes() {
        dfs(graph, &mut visited, &mut ancestors, &mut result, u)
    }
    result
}

// pub fn remove_cycle<D, G: Graph<D>>(graph: &mut G) {
//     for (u, v) in cycle_edges(graph) {
//         let index = graph.find_edge(u, v).unwrap();
//         let weight = graph.remove_edge(index).unwrap();
//         graph.add_edge(v, u, weight);
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use egraph_petgraph_adapter::PetgraphWrapper;
    use petgraph::Graph;

    #[test]
    fn it_works() {
        let mut graph = Graph::new();
        let a = graph.add_node("a");
        let b = graph.add_node("b");
        let c = graph.add_node("c");
        graph.add_edge(a, b, ());
        graph.add_edge(b, c, ());
        graph.add_edge(c, a, ());
        let graph = PetgraphWrapper::new(graph);
        assert_eq!(cycle_edges(&graph), vec![(c.index(), a.index())]);
    }
}
