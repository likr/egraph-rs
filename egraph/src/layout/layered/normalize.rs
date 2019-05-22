use super::graph::{Edge, Node};
use petgraph::graph::{IndexType, NodeIndex};
use petgraph::prelude::*;
use std::collections::HashMap;

pub fn normalize<Ix: IndexType>(
    graph: &mut Graph<Node<Ix>, Edge, Directed, Ix>,
    layers_map: &mut HashMap<NodeIndex<Ix>, usize>,
) {
    for e in graph.edge_indices() {
        let edge = graph[e].clone();
        let (u, v) = graph.edge_endpoints(e).unwrap();
        let h_u = *layers_map.get(&u).unwrap();
        let h_v = *layers_map.get(&v).unwrap();
        let length = h_v - h_u;
        if length == 1 {
            continue;
        }
        let mut w0 = u;
        for i in h_u + 1..h_v {
            let w1 = graph.add_node(Node::new_dummy(e));
            layers_map.insert(w1, i);
            graph.add_edge(w0, w1, Edge::new_split(&edge));
            w0 = w1;
        }
        graph.add_edge(w0, v, Edge::new_split(&edge));
        graph.remove_edge(e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::Graph;

    #[test]
    fn it_works() {
        let mut graph = Graph::new();
        let u1 = graph.add_node(Node::new());
        let u2 = graph.add_node(Node::new());
        let u3 = graph.add_node(Node::new());
        graph.add_edge(u1, u2, Edge::new());
        graph.add_edge(u1, u3, Edge::new_reversed());
        graph.add_edge(u2, u3, Edge::new());

        let mut layers_map = HashMap::new();
        layers_map.insert(u1, 0);
        layers_map.insert(u2, 1);
        layers_map.insert(u3, 2);

        normalize(&mut graph, &mut layers_map);
        assert_eq!(graph.node_count(), 4);
        assert_eq!(graph.edge_count(), 4);
    }
}
