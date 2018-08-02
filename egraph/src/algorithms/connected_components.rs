extern crate petgraph;

use std::collections::VecDeque;
use petgraph::{Graph, EdgeType};
use petgraph::graph::IndexType;

pub fn connected_components<N, E, Ty: EdgeType, Ix: IndexType>(graph: &Graph<N, E, Ty, Ix>) -> Vec<usize> {
    let mut components = graph.node_indices()
        .map(|u| u.index())
        .collect::<Vec<_>>();
    let mut visited = graph.node_indices()
        .map(|_| false)
        .collect::<Vec<_>>();
    let mut queue = VecDeque::new();
    for u in graph.node_indices() {
        if visited[u.index()] {
            continue;
        }
        queue.push_back(u);
        let c = components[u.index()];
        while queue.len() > 0 {
            let v = queue.pop_front().unwrap();
            if visited[v.index()] {
                continue;
            }
            visited[v.index()] = true;
            components[v.index()] = c;
            for w in graph.neighbors_undirected(v) {
                queue.push_back(w);
            }
        }
    }
    components
}

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
    assert_eq!(components[0], 0);
    assert_eq!(components[1], 0);
    assert_eq!(components[2], 0);
    assert_eq!(components[3], 3);
    assert_eq!(components[4], 3);
}
