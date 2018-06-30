use super::super::graph::{Edge, Node};
use petgraph::graph::NodeIndex;
use petgraph::{Directed, Graph};

fn segment(
    graph: &Graph<Node, Edge, Directed>,
    h1: &Vec<NodeIndex>,
) -> (Vec<(NodeIndex, NodeIndex)>, Vec<(NodeIndex, NodeIndex)>) {
    let mut inner = vec![];
    let mut outer = vec![];
    for u in h1 {
        for v in graph.neighbors(*u) {
            if graph[*u].dummy && graph[v].dummy {
                inner.push((*u, v));
            } else {
                outer.push((*u, v));
            }
        }
    }
    (inner, outer)
}

pub fn mark_conflicts(graph: &mut Graph<Node, Edge, Directed>, layers: &Vec<Vec<NodeIndex>>) {
    for i in 1..(layers.len() - 1) {
        let h1 = layers.get(i).unwrap();
        let (inner, outer) = segment(graph, &h1);
        for (u1, v1) in inner {
            for &(u2, v2) in &outer {
                let ou1 = graph[u1].order;
                let ou2 = graph[u2].order;
                let ov1 = graph[v1].order;
                let ov2 = graph[v2].order;
                if (ou1 < ou2 && ov1 > ov2) || (ou1 > ou2 && ov1 < ov2) {
                    let index = graph.find_edge(u2, v2).unwrap();
                    graph[index].conflict = true;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::graph::{Edge, Node};
    use super::*;
    use petgraph::Graph;

    #[test]
    fn test_mark_conflicts() {
        let mut graph = Graph::new();
        let a1 = graph.add_node(Node {
            width: 10,
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let a2 = graph.add_node(Node {
            width: 10,
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let b1 = graph.add_node(Node {
            width: 10,
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let b2 = graph.add_node(Node {
            width: 10,
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let b3 = graph.add_node(Node {
            width: 10,
            order: 2,
            dummy: true,
            ..Node::new()
        });
        let b4 = graph.add_node(Node {
            width: 10,
            order: 3,
            dummy: false,
            ..Node::new()
        });
        let b5 = graph.add_node(Node {
            width: 10,
            order: 4,
            dummy: true,
            ..Node::new()
        });
        let b6 = graph.add_node(Node {
            width: 10,
            order: 5,
            dummy: true,
            ..Node::new()
        });
        let b7 = graph.add_node(Node {
            width: 10,
            order: 6,
            dummy: false,
            ..Node::new()
        });
        let b8 = graph.add_node(Node {
            width: 10,
            order: 7,
            dummy: false,
            ..Node::new()
        });
        let c1 = graph.add_node(Node {
            width: 10,
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let c2 = graph.add_node(Node {
            width: 10,
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let c3 = graph.add_node(Node {
            width: 10,
            order: 2,
            dummy: true,
            ..Node::new()
        });
        let c4 = graph.add_node(Node {
            width: 10,
            order: 3,
            dummy: true,
            ..Node::new()
        });
        let c5 = graph.add_node(Node {
            width: 10,
            order: 4,
            dummy: true,
            ..Node::new()
        });
        let c6 = graph.add_node(Node {
            width: 10,
            order: 5,
            dummy: false,
            ..Node::new()
        });
        let d1 = graph.add_node(Node {
            width: 10,
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let d2 = graph.add_node(Node {
            width: 10,
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let d3 = graph.add_node(Node {
            width: 10,
            order: 2,
            dummy: true,
            ..Node::new()
        });
        let d4 = graph.add_node(Node {
            width: 10,
            order: 3,
            dummy: true,
            ..Node::new()
        });
        let d5 = graph.add_node(Node {
            width: 10,
            order: 4,
            dummy: true,
            ..Node::new()
        });
        let d6 = graph.add_node(Node {
            width: 10,
            order: 5,
            dummy: false,
            ..Node::new()
        });
        let d7 = graph.add_node(Node {
            width: 10,
            order: 6,
            dummy: true,
            ..Node::new()
        });
        let e1 = graph.add_node(Node {
            width: 10,
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let e2 = graph.add_node(Node {
            width: 10,
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let e3 = graph.add_node(Node {
            width: 10,
            order: 2,
            dummy: false,
            ..Node::new()
        });
        let a1b1 = graph.add_edge(a1, b1, Edge::new());
        let a1b6 = graph.add_edge(a1, b6, Edge::new());
        let a1b8 = graph.add_edge(a1, b8, Edge::new());
        let a2b3 = graph.add_edge(a2, b3, Edge::new());
        let a2b5 = graph.add_edge(a2, b5, Edge::new());
        let b2c2 = graph.add_edge(b2, c2, Edge::new());
        let b3c2 = graph.add_edge(b3, c2, Edge::new());
        let b4c2 = graph.add_edge(b4, c2, Edge::new());
        let b5c3 = graph.add_edge(b5, c3, Edge::new());
        let b6c4 = graph.add_edge(b6, c4, Edge::new());
        let b7c2 = graph.add_edge(b7, c2, Edge::new());
        let b7c6 = graph.add_edge(b7, c6, Edge::new());
        let b8c2 = graph.add_edge(b8, c2, Edge::new());
        let b8c5 = graph.add_edge(b8, c5, Edge::new());
        let c1d1 = graph.add_edge(c1, d1, Edge::new());
        let c1d2 = graph.add_edge(c1, d2, Edge::new());
        let c1d6 = graph.add_edge(c1, d6, Edge::new());
        let c3d4 = graph.add_edge(c3, d4, Edge::new());
        let c4d5 = graph.add_edge(c4, d5, Edge::new());
        let c5d6 = graph.add_edge(c5, d6, Edge::new());
        let c6d3 = graph.add_edge(c6, d3, Edge::new());
        let c6d7 = graph.add_edge(c6, d7, Edge::new());
        let d1e1 = graph.add_edge(d1, e1, Edge::new());
        let d1e2 = graph.add_edge(d1, e2, Edge::new());
        let d2e2 = graph.add_edge(d2, e2, Edge::new());
        let d3e1 = graph.add_edge(d3, e1, Edge::new());
        let d4e3 = graph.add_edge(d4, e3, Edge::new());
        let d5e3 = graph.add_edge(d5, e3, Edge::new());
        let d6e3 = graph.add_edge(d6, e3, Edge::new());
        let d7e3 = graph.add_edge(d7, e3, Edge::new());
        let layers = vec![
            vec![a1, a2],
            vec![b1, b2, b3, b4, b5, b6, b7, b8],
            vec![c1, c2, c3, c4, c5, c6],
            vec![d1, d2, d3, d4, d5, d6, d7],
            vec![e1, e2, e3],
        ];
        mark_conflicts(&mut graph, &layers);
        assert!(!graph[a1b1].conflict);
        assert!(!graph[a1b6].conflict);
        assert!(!graph[a1b8].conflict);
        assert!(!graph[a2b3].conflict);
        assert!(!graph[a2b5].conflict);
        assert!(!graph[b2c2].conflict);
        assert!(!graph[b3c2].conflict);
        assert!(!graph[b4c2].conflict);
        assert!(!graph[b5c3].conflict);
        assert!(!graph[b6c4].conflict);
        assert!(graph[b7c2].conflict);
        assert!(!graph[b7c6].conflict);
        assert!(graph[b8c2].conflict);
        assert!(!graph[b8c5].conflict);
        assert!(!graph[c1d1].conflict);
        assert!(!graph[c1d2].conflict);
        assert!(graph[c1d6].conflict);
        assert!(!graph[c3d4].conflict);
        assert!(!graph[c4d5].conflict);
        assert!(!graph[c5d6].conflict);
        assert!(graph[c6d3].conflict);
        assert!(!graph[c6d7].conflict);
        assert!(!graph[d1e1].conflict);
        assert!(!graph[d1e2].conflict);
        assert!(!graph[d2e2].conflict);
        assert!(!graph[d3e1].conflict);
        assert!(!graph[d4e3].conflict);
        assert!(!graph[d5e3].conflict);
        assert!(!graph[d6e3].conflict);
        assert!(!graph[d7e3].conflict);
    }
}
