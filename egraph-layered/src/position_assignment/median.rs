use super::super::graph::{Edge, Node};
use petgraph::graph::NodeIndex;
use petgraph::{Directed, EdgeDirection, Graph};
use std::iter::FromIterator;

pub fn median(
    graph: &Graph<Node, Edge, Directed>,
    u: NodeIndex,
    direction: EdgeDirection,
) -> Option<(NodeIndex, NodeIndex)> {
    let mut vertices = Vec::from_iter(graph.neighbors_directed(u, direction));
    if vertices.len() == 0 {
        None
    } else {
        vertices.sort_by_key(|v| graph[*v].order);
        let n = vertices.len();
        if n % 2 == 0 {
            Some((vertices[n / 2 - 1], vertices[n / 2]))
        } else {
            Some((vertices[n / 2], vertices[n / 2]))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::graph::{Edge, Node};
    use super::*;
    use petgraph::{EdgeDirection, Graph};

    #[test]
    fn test_median_outgoing() {
        let mut graph = Graph::new();
        let u1 = graph.add_node(Node::new());
        let v1 = graph.add_node(Node::new());
        let v2 = graph.add_node(Node::new());
        let v3 = graph.add_node(Node::new());
        let v4 = graph.add_node(Node::new());
        graph.add_edge(u1, v1, Edge::new());
        graph.add_edge(u1, v2, Edge::new());
        graph.add_edge(u1, v3, Edge::new());
        graph.add_edge(u1, v4, Edge::new());
        graph[v1].order = 1;
        graph[v2].order = 2;
        graph[v3].order = 3;
        graph[v4].order = 4;
        let (left, right) = median(&graph, u1, EdgeDirection::Outgoing).unwrap();
        assert_eq!(left, v2);
        assert_eq!(right, v3);
    }

    #[test]
    fn test_median_incoming() {
        let mut graph = Graph::new();
        let u1 = graph.add_node(Node::new());
        let u2 = graph.add_node(Node::new());
        let u3 = graph.add_node(Node::new());
        let u4 = graph.add_node(Node::new());
        let v1 = graph.add_node(Node::new());
        graph.add_edge(u1, v1, Edge::new());
        graph.add_edge(u2, v1, Edge::new());
        graph.add_edge(u3, v1, Edge::new());
        graph.add_edge(u4, v1, Edge::new());
        graph[u1].order = 1;
        graph[u2].order = 2;
        graph[u3].order = 3;
        graph[u4].order = 4;
        let (left, right) = median(&graph, v1, EdgeDirection::Incoming).unwrap();
        assert_eq!(left, u2);
        assert_eq!(right, u3);
    }
}
