use petgraph::graph::{Graph, IndexType, NodeIndex};
use petgraph::unionfind::UnionFind;
use petgraph::EdgeType;
use std::collections::HashMap;

pub fn connected_components<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
) -> HashMap<NodeIndex<Ix>, usize> {
    let mut components = UnionFind::new(graph.node_count());
    let indices = graph
        .node_indices()
        .enumerate()
        .map(|(i, u)| (u, i))
        .collect::<HashMap<NodeIndex<Ix>, usize>>();
    for e in graph.edge_indices() {
        let (u, v) = graph.edge_endpoints(e).unwrap();
        components.union(indices[&u], indices[&v]);
    }
    let mut result = HashMap::new();
    for u in graph.node_indices() {
        result.insert(u, components.find(indices[&u]));
    }
    result
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_connected_components() {
        let mut graph = Graph::new_undirected();
        let u1 = graph.add_node(());
        let u2 = graph.add_node(());
        let u3 = graph.add_node(());
        let u4 = graph.add_node(());
        let u5 = graph.add_node(());
        graph.add_edge(u1, u2, ());
        graph.add_edge(u1, u3, ());
        graph.add_edge(u2, u3, ());
        graph.add_edge(u4, u5, ());
        let components = connected_components(&graph);
        assert_eq!(components[&u1], components[&u2]);
        assert_eq!(components[&u1], components[&u3]);
        assert_ne!(components[&u3], components[&u4]);
        assert_eq!(components[&u4], components[&u5]);
    }
}
