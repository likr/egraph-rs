use petgraph::{Graph, Directed, EdgeDirection};
use petgraph::graph::NodeIndex;
use super::super::graph::{Node, Edge};
use super::median::median;

fn iter_layer<'a, I: Iterator<Item = &'a NodeIndex>>(
    graph: &mut Graph<Node, Edge, Directed>,
    rtol: bool,
    btot: bool,
    layer: I,
) {
    let mut r = if rtol {
        i32::max_value()
    } else {
        i32::min_value()
    };
    let edge_direction = if btot {
        EdgeDirection::Outgoing
    } else {
        EdgeDirection::Incoming
    };
    for &v in layer {
        match median(graph, v, edge_direction) {
            Some((left, right)) => {
                let medians = if left == right {
                    vec![left]
                } else {
                    if rtol {
                        vec![right, left]
                    } else {
                        vec![left, right]
                    }
                };
                for u in medians {
                    let edge = if btot {
                        graph.find_edge(v, u).unwrap()
                    } else {
                        graph.find_edge(u, v).unwrap()
                    };
                    if graph[v].align.unwrap() == v && !graph[edge].conflict {
                        let u_order = graph[u].order as i32;
                        if (rtol && r > u_order) || (!rtol && r < u_order) {
                            graph[v].align = graph[u].root;
                            graph[v].root = graph[u].root;
                            graph[u].align = Some(v);
                            r = u_order;
                            break;
                        }
                    }
                }
            }
            None => {}
        };
    }
}

fn iter_layers<'a, I: Iterator<Item = &'a Vec<NodeIndex>>>(
    graph: &mut Graph<Node, Edge, Directed>,
    rtol: bool,
    btot: bool,
    layers: I,
) {
    for layer in layers.skip(1) {
        if rtol {
            iter_layer(graph, rtol, btot, layer.iter().rev());
        } else {
            iter_layer(graph, rtol, btot, layer.iter());
        };
    }
}

pub fn vertical_alignment(
    graph: &mut Graph<Node, Edge, Directed>,
    layers: &Vec<Vec<NodeIndex>>,
    rtol: bool,
    btot: bool,
) {
    for u in graph.node_indices() {
        graph[u].root = Some(u);
        graph[u].align = Some(u);
    }
    if btot {
        iter_layers(graph, rtol, btot, layers.iter().rev());
    } else {
        iter_layers(graph, rtol, btot, layers.iter());
    };
}

#[cfg(test)]
mod tests {
    use petgraph::Graph;
    use super::*;
    use super::super::super::graph::{Node, Edge};

    #[test]
    fn test_vertical_alignment_lt() {
        let mut graph = Graph::new();
        let a1 = graph.add_node(Node {
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let a2 = graph.add_node(Node {
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let b1 = graph.add_node(Node {
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let b2 = graph.add_node(Node {
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let b3 = graph.add_node(Node {
            order: 2,
            dummy: true,
            ..Node::new()
        });
        let b4 = graph.add_node(Node {
            order: 3,
            dummy: false,
            ..Node::new()
        });
        let b5 = graph.add_node(Node {
            order: 4,
            dummy: true,
            ..Node::new()
        });
        let b6 = graph.add_node(Node {
            order: 5,
            dummy: true,
            ..Node::new()
        });
        let b7 = graph.add_node(Node {
            order: 6,
            dummy: false,
            ..Node::new()
        });
        let b8 = graph.add_node(Node {
            order: 7,
            dummy: false,
            ..Node::new()
        });
        let c1 = graph.add_node(Node {
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let c2 = graph.add_node(Node {
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let c3 = graph.add_node(Node {
            order: 2,
            dummy: true,
            ..Node::new()
        });
        let c4 = graph.add_node(Node {
            order: 3,
            dummy: true,
            ..Node::new()
        });
        let c5 = graph.add_node(Node {
            order: 4,
            dummy: true,
            ..Node::new()
        });
        let c6 = graph.add_node(Node {
            order: 5,
            dummy: false,
            ..Node::new()
        });
        let d1 = graph.add_node(Node {
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let d2 = graph.add_node(Node {
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let d3 = graph.add_node(Node {
            order: 2,
            dummy: true,
            ..Node::new()
        });
        let d4 = graph.add_node(Node {
            order: 3,
            dummy: true,
            ..Node::new()
        });
        let d5 = graph.add_node(Node {
            order: 4,
            dummy: true,
            ..Node::new()
        });
        let d6 = graph.add_node(Node {
            order: 5,
            dummy: false,
            ..Node::new()
        });
        let d7 = graph.add_node(Node {
            order: 6,
            dummy: true,
            ..Node::new()
        });
        let e1 = graph.add_node(Node {
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let e2 = graph.add_node(Node {
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let e3 = graph.add_node(Node {
            order: 2,
            dummy: false,
            ..Node::new()
        });
        graph.add_edge(
            a1,
            b1,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            a1,
            b6,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            a1,
            b8,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            a2,
            b3,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            a2,
            b5,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            b2,
            c2,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            b3,
            c2,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            b4,
            c2,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            b5,
            c3,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            b6,
            c4,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            b7,
            c2,
            Edge {
                conflict: true,
                ..Edge::new()
            },
        );
        graph.add_edge(
            b7,
            c6,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            b8,
            c2,
            Edge {
                conflict: true,
                ..Edge::new()
            },
        );
        graph.add_edge(
            b8,
            c5,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            c1,
            d1,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            c1,
            d2,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            c1,
            d6,
            Edge {
                conflict: true,
                ..Edge::new()
            },
        );
        graph.add_edge(
            c3,
            d4,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            c4,
            d5,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            c5,
            d6,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            c6,
            d3,
            Edge {
                conflict: true,
                ..Edge::new()
            },
        );
        graph.add_edge(
            c6,
            d7,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            d1,
            e1,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            d1,
            e2,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            d2,
            e2,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            d3,
            e1,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            d4,
            e3,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            d5,
            e3,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            d6,
            e3,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            d7,
            e3,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        let layers = vec![
            vec![a1, a2],
            vec![b1, b2, b3, b4, b5, b6, b7, b8],
            vec![c1, c2, c3, c4, c5, c6],
            vec![d1, d2, d3, d4, d5, d6, d7],
            vec![e1, e2, e3],
        ];
        vertical_alignment(&mut graph, &layers, false, false);
        assert_eq!(graph[a1].root.unwrap(), a1);
        assert_eq!(graph[a1].align.unwrap(), b1);
        assert_eq!(graph[a2].root.unwrap(), a2);
        assert_eq!(graph[a2].align.unwrap(), b3);
        assert_eq!(graph[b1].root.unwrap(), a1);
        assert_eq!(graph[b1].align.unwrap(), a1);
        assert_eq!(graph[b2].root.unwrap(), b2);
        assert_eq!(graph[b2].align.unwrap(), b2);
        assert_eq!(graph[b3].root.unwrap(), a2);
        assert_eq!(graph[b3].align.unwrap(), a2);
        assert_eq!(graph[b4].root.unwrap(), b4);
        assert_eq!(graph[b4].align.unwrap(), c2);
        assert_eq!(graph[b5].root.unwrap(), b5);
        assert_eq!(graph[b5].align.unwrap(), c3);
        assert_eq!(graph[b6].root.unwrap(), b6);
        assert_eq!(graph[b6].align.unwrap(), c4);
        assert_eq!(graph[b7].root.unwrap(), b7);
        assert_eq!(graph[b7].align.unwrap(), b7);
        assert_eq!(graph[b8].root.unwrap(), b8);
        assert_eq!(graph[b8].align.unwrap(), c5);
        assert_eq!(graph[c1].root.unwrap(), c1);
        assert_eq!(graph[c1].align.unwrap(), d1);
        assert_eq!(graph[c2].root.unwrap(), b4);
        assert_eq!(graph[c2].align.unwrap(), b4);
        assert_eq!(graph[c3].root.unwrap(), b5);
        assert_eq!(graph[c3].align.unwrap(), d4);
        assert_eq!(graph[c4].root.unwrap(), b6);
        assert_eq!(graph[c4].align.unwrap(), d5);
        assert_eq!(graph[c5].root.unwrap(), b8);
        assert_eq!(graph[c5].align.unwrap(), d6);
        assert_eq!(graph[c6].root.unwrap(), c6);
        assert_eq!(graph[c6].align.unwrap(), d7);
        assert_eq!(graph[d1].root.unwrap(), c1);
        assert_eq!(graph[d1].align.unwrap(), e1);
        assert_eq!(graph[d2].root.unwrap(), d2);
        assert_eq!(graph[d2].align.unwrap(), e2);
        assert_eq!(graph[d3].root.unwrap(), d3);
        assert_eq!(graph[d3].align.unwrap(), d3);
        assert_eq!(graph[d4].root.unwrap(), b5);
        assert_eq!(graph[d4].align.unwrap(), b5);
        assert_eq!(graph[d5].root.unwrap(), b6);
        assert_eq!(graph[d5].align.unwrap(), e3);
        assert_eq!(graph[d6].root.unwrap(), b8);
        assert_eq!(graph[d6].align.unwrap(), b8);
        assert_eq!(graph[d7].root.unwrap(), c6);
        assert_eq!(graph[d7].align.unwrap(), c6);
        assert_eq!(graph[e1].root.unwrap(), c1);
        assert_eq!(graph[e1].align.unwrap(), c1);
        assert_eq!(graph[e2].root.unwrap(), d2);
        assert_eq!(graph[e2].align.unwrap(), d2);
        assert_eq!(graph[e3].root.unwrap(), b6);
        assert_eq!(graph[e3].align.unwrap(), b6);
    }

    #[test]
    fn test_vertical_alignment_lb() {
        let mut graph = Graph::new();
        let a1 = graph.add_node(Node {
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let a2 = graph.add_node(Node {
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let b1 = graph.add_node(Node {
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let b2 = graph.add_node(Node {
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let b3 = graph.add_node(Node {
            order: 2,
            dummy: true,
            ..Node::new()
        });
        let b4 = graph.add_node(Node {
            order: 3,
            dummy: false,
            ..Node::new()
        });
        let b5 = graph.add_node(Node {
            order: 4,
            dummy: true,
            ..Node::new()
        });
        let b6 = graph.add_node(Node {
            order: 5,
            dummy: true,
            ..Node::new()
        });
        let b7 = graph.add_node(Node {
            order: 6,
            dummy: false,
            ..Node::new()
        });
        let b8 = graph.add_node(Node {
            order: 7,
            dummy: false,
            ..Node::new()
        });
        let c1 = graph.add_node(Node {
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let c2 = graph.add_node(Node {
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let c3 = graph.add_node(Node {
            order: 2,
            dummy: true,
            ..Node::new()
        });
        let c4 = graph.add_node(Node {
            order: 3,
            dummy: true,
            ..Node::new()
        });
        let c5 = graph.add_node(Node {
            order: 4,
            dummy: true,
            ..Node::new()
        });
        let c6 = graph.add_node(Node {
            order: 5,
            dummy: false,
            ..Node::new()
        });
        let d1 = graph.add_node(Node {
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let d2 = graph.add_node(Node {
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let d3 = graph.add_node(Node {
            order: 2,
            dummy: true,
            ..Node::new()
        });
        let d4 = graph.add_node(Node {
            order: 3,
            dummy: true,
            ..Node::new()
        });
        let d5 = graph.add_node(Node {
            order: 4,
            dummy: true,
            ..Node::new()
        });
        let d6 = graph.add_node(Node {
            order: 5,
            dummy: false,
            ..Node::new()
        });
        let d7 = graph.add_node(Node {
            order: 6,
            dummy: true,
            ..Node::new()
        });
        let e1 = graph.add_node(Node {
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let e2 = graph.add_node(Node {
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let e3 = graph.add_node(Node {
            order: 2,
            dummy: false,
            ..Node::new()
        });
        graph.add_edge(
            a1,
            b1,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            a1,
            b6,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            a1,
            b8,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            a2,
            b3,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            a2,
            b5,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            b2,
            c2,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            b3,
            c2,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            b4,
            c2,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            b5,
            c3,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            b6,
            c4,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            b7,
            c2,
            Edge {
                conflict: true,
                ..Edge::new()
            },
        );
        graph.add_edge(
            b7,
            c6,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            b8,
            c2,
            Edge {
                conflict: true,
                ..Edge::new()
            },
        );
        graph.add_edge(
            b8,
            c5,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            c1,
            d1,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            c1,
            d2,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            c1,
            d6,
            Edge {
                conflict: true,
                ..Edge::new()
            },
        );
        graph.add_edge(
            c3,
            d4,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            c4,
            d5,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            c5,
            d6,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            c6,
            d3,
            Edge {
                conflict: true,
                ..Edge::new()
            },
        );
        graph.add_edge(
            c6,
            d7,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            d1,
            e1,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            d1,
            e2,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            d2,
            e2,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            d3,
            e1,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            d4,
            e3,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            d5,
            e3,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            d6,
            e3,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        graph.add_edge(
            d7,
            e3,
            Edge {
                conflict: false,
                ..Edge::new()
            },
        );
        let layers = vec![
            vec![a1, a2],
            vec![b1, b2, b3, b4, b5, b6, b7, b8],
            vec![c1, c2, c3, c4, c5, c6],
            vec![d1, d2, d3, d4, d5, d6, d7],
            vec![e1, e2, e3],
        ];
        vertical_alignment(&mut graph, &layers, false, true);
        assert_eq!(graph[a1].root.unwrap(), d5);
        assert_eq!(graph[a1].align.unwrap(), d5);
        assert_eq!(graph[a2].root.unwrap(), a2);
        assert_eq!(graph[a2].align.unwrap(), a2);
        assert_eq!(graph[b1].root.unwrap(), b1);
        assert_eq!(graph[b1].align.unwrap(), b1);
        assert_eq!(graph[b2].root.unwrap(), c2);
        assert_eq!(graph[b2].align.unwrap(), c2);
        assert_eq!(graph[b3].root.unwrap(), b3);
        assert_eq!(graph[b3].align.unwrap(), b3);
        assert_eq!(graph[b4].root.unwrap(), b4);
        assert_eq!(graph[b4].align.unwrap(), b4);
        assert_eq!(graph[b5].root.unwrap(), e3);
        assert_eq!(graph[b5].align.unwrap(), e3);
        assert_eq!(graph[b6].root.unwrap(), d5);
        assert_eq!(graph[b6].align.unwrap(), a1);
        assert_eq!(graph[b7].root.unwrap(), d7);
        assert_eq!(graph[b7].align.unwrap(), d7);
        assert_eq!(graph[b8].root.unwrap(), b8);
        assert_eq!(graph[b8].align.unwrap(), b8);
        assert_eq!(graph[c1].root.unwrap(), e2);
        assert_eq!(graph[c1].align.unwrap(), e2);
        assert_eq!(graph[c2].root.unwrap(), c2);
        assert_eq!(graph[c2].align.unwrap(), b2);
        assert_eq!(graph[c3].root.unwrap(), e3);
        assert_eq!(graph[c3].align.unwrap(), b5);
        assert_eq!(graph[c4].root.unwrap(), d5);
        assert_eq!(graph[c4].align.unwrap(), b6);
        assert_eq!(graph[c5].root.unwrap(), d6);
        assert_eq!(graph[c5].align.unwrap(), d6);
        assert_eq!(graph[c6].root.unwrap(), d7);
        assert_eq!(graph[c6].align.unwrap(), b7);
        assert_eq!(graph[d1].root.unwrap(), e1);
        assert_eq!(graph[d1].align.unwrap(), e1);
        assert_eq!(graph[d2].root.unwrap(), e2);
        assert_eq!(graph[d2].align.unwrap(), c1);
        assert_eq!(graph[d3].root.unwrap(), d3);
        assert_eq!(graph[d3].align.unwrap(), d3);
        assert_eq!(graph[d4].root.unwrap(), e3);
        assert_eq!(graph[d4].align.unwrap(), c3);
        assert_eq!(graph[d5].root.unwrap(), d5);
        assert_eq!(graph[d5].align.unwrap(), c4);
        assert_eq!(graph[d6].root.unwrap(), d6);
        assert_eq!(graph[d6].align.unwrap(), c5);
        assert_eq!(graph[d7].root.unwrap(), d7);
        assert_eq!(graph[d7].align.unwrap(), c6);
        assert_eq!(graph[e1].root.unwrap(), e1);
        assert_eq!(graph[e1].align.unwrap(), d1);
        assert_eq!(graph[e2].root.unwrap(), e2);
        assert_eq!(graph[e2].align.unwrap(), d2);
        assert_eq!(graph[e3].root.unwrap(), e3);
        assert_eq!(graph[e3].align.unwrap(), d4);
    }
}
