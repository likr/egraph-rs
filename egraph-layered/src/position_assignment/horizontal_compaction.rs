use super::super::graph::{Edge, Node};
use petgraph::graph::IndexType;
use petgraph::prelude::*;

fn place_block<Ix: IndexType>(
    graph: &mut Graph<Node<Ix>, Edge, Directed, Ix>,
    layers: &Vec<Vec<NodeIndex<Ix>>>,
    v: NodeIndex<Ix>,
    rtol: bool,
) {
    if graph[v].x > 0 {
        return;
    }
    let mut w = v;
    loop {
        let w_layer = graph[w].layer;
        let w_order = graph[w].order;

        if (rtol && w_order < layers[w_layer].len() - 1) || (!rtol && w_order > 0) {
            let p = if rtol {
                layers[w_layer][w_order + 1]
            } else {
                layers[w_layer][w_order - 1]
            };
            let u = graph[p].root.unwrap();
            place_block(graph, layers, u, rtol);
            if graph[v].sink.unwrap() == v {
                graph[v].sink = graph[u].sink;
            }
            if graph[v].sink == graph[u].sink {
                let p_width = graph[p].width as i32;
                let w_width = graph[w].width as i32;
                let new_x = graph[u].x + (p_width + w_width) / 2;
                if new_x > graph[v].x {
                    graph[v].x = new_x;
                }
            } else {
                let p_width = graph[p].width as i32;
                let w_width = graph[w].width as i32;
                let new_shift = graph[v].x - graph[u].x - (p_width + w_width) / 2;
                let u_sink = graph[u].sink.unwrap();
                if new_shift < graph[u_sink].shift {
                    graph[u_sink].shift = new_shift;
                }
            }
        }
        w = graph[w].align.unwrap();
        if w == v {
            break;
        }
    }
}

pub fn horizontal_compaction<Ix: IndexType>(
    graph: &mut Graph<Node<Ix>, Edge, Directed, Ix>,
    layers: &Vec<Vec<NodeIndex<Ix>>>,
    rtol: bool,
) {
    for u in graph.node_indices() {
        graph[u].sink = Some(u);
        graph[u].shift = i32::max_value();
        graph[u].x = 0;
    }
    for u in graph.node_indices() {
        if graph[u].root.unwrap() == u {
            place_block(graph, layers, u, rtol);
        }
    }
    for u in graph.node_indices() {
        graph[u].x = graph[graph[u].root.unwrap()].x;
    }
    for u in graph.node_indices() {
        let shift = graph[graph[graph[u].root.unwrap()].sink.unwrap()].shift;
        if shift < i32::max_value() {
            graph[u].x += shift;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::graph::{Edge, Node};
    use super::super::vertical_alignment::vertical_alignment;
    use super::*;
    use petgraph::Graph;

    #[test]
    fn test_vertical_alignment_lt() {
        let mut graph = Graph::new();
        let a1 = graph.add_node(Node {
            width: 10,
            layer: 0,
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let a2 = graph.add_node(Node {
            width: 10,
            layer: 0,
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let b1 = graph.add_node(Node {
            width: 10,
            layer: 1,
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let b2 = graph.add_node(Node {
            width: 10,
            layer: 1,
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let b3 = graph.add_node(Node {
            width: 10,
            layer: 1,
            order: 2,
            dummy: true,
            ..Node::new()
        });
        let b4 = graph.add_node(Node {
            width: 10,
            layer: 1,
            order: 3,
            dummy: false,
            ..Node::new()
        });
        let b5 = graph.add_node(Node {
            width: 10,
            layer: 1,
            order: 4,
            dummy: true,
            ..Node::new()
        });
        let b6 = graph.add_node(Node {
            width: 10,
            layer: 1,
            order: 5,
            dummy: true,
            ..Node::new()
        });
        let b7 = graph.add_node(Node {
            width: 10,
            layer: 1,
            order: 6,
            dummy: false,
            ..Node::new()
        });
        let b8 = graph.add_node(Node {
            width: 10,
            layer: 1,
            order: 7,
            dummy: false,
            ..Node::new()
        });
        let c1 = graph.add_node(Node {
            width: 10,
            layer: 2,
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let c2 = graph.add_node(Node {
            width: 10,
            layer: 2,
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let c3 = graph.add_node(Node {
            width: 10,
            layer: 2,
            order: 2,
            dummy: true,
            ..Node::new()
        });
        let c4 = graph.add_node(Node {
            width: 10,
            layer: 2,
            order: 3,
            dummy: true,
            ..Node::new()
        });
        let c5 = graph.add_node(Node {
            width: 10,
            layer: 2,
            order: 4,
            dummy: true,
            ..Node::new()
        });
        let c6 = graph.add_node(Node {
            width: 10,
            layer: 2,
            order: 5,
            dummy: false,
            ..Node::new()
        });
        let d1 = graph.add_node(Node {
            width: 10,
            layer: 3,
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let d2 = graph.add_node(Node {
            width: 10,
            layer: 3,
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let d3 = graph.add_node(Node {
            width: 10,
            layer: 3,
            order: 2,
            dummy: true,
            ..Node::new()
        });
        let d4 = graph.add_node(Node {
            width: 10,
            layer: 3,
            order: 3,
            dummy: true,
            ..Node::new()
        });
        let d5 = graph.add_node(Node {
            width: 10,
            layer: 3,
            order: 4,
            dummy: true,
            ..Node::new()
        });
        let d6 = graph.add_node(Node {
            width: 10,
            layer: 3,
            order: 5,
            dummy: false,
            ..Node::new()
        });
        let d7 = graph.add_node(Node {
            width: 10,
            layer: 3,
            order: 6,
            dummy: true,
            ..Node::new()
        });
        let e1 = graph.add_node(Node {
            width: 10,
            layer: 4,
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let e2 = graph.add_node(Node {
            width: 10,
            layer: 4,
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let e3 = graph.add_node(Node {
            width: 10,
            layer: 4,
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
        graph[a1].align = Some(b1);
        graph[a2].align = Some(b3);
        graph[b1].align = Some(a1);
        graph[b2].align = Some(b2);
        graph[b3].align = Some(a2);
        graph[b4].align = Some(c2);
        graph[b5].align = Some(c3);
        graph[b6].align = Some(c4);
        graph[b7].align = Some(b7);
        graph[b8].align = Some(c5);
        graph[c1].align = Some(d1);
        graph[c2].align = Some(b4);
        graph[c3].align = Some(d4);
        graph[c4].align = Some(d5);
        graph[c5].align = Some(d6);
        graph[c6].align = Some(d7);
        graph[d1].align = Some(e1);
        graph[d2].align = Some(e2);
        graph[d3].align = Some(d3);
        graph[d4].align = Some(b5);
        graph[d5].align = Some(e3);
        graph[d6].align = Some(b8);
        graph[d7].align = Some(c6);
        graph[e1].align = Some(c1);
        graph[e2].align = Some(d2);
        graph[e3].align = Some(b6);
        graph[a1].root = Some(a1);
        graph[a2].root = Some(a2);
        graph[b1].root = Some(a1);
        graph[b2].root = Some(b2);
        graph[b3].root = Some(a2);
        graph[b4].root = Some(b4);
        graph[b5].root = Some(b5);
        graph[b6].root = Some(b6);
        graph[b7].root = Some(b7);
        graph[b8].root = Some(b8);
        graph[c1].root = Some(c1);
        graph[c2].root = Some(b4);
        graph[c3].root = Some(b5);
        graph[c4].root = Some(b6);
        graph[c5].root = Some(b8);
        graph[c6].root = Some(c6);
        graph[d1].root = Some(c1);
        graph[d2].root = Some(d2);
        graph[d3].root = Some(d3);
        graph[d4].root = Some(b5);
        graph[d5].root = Some(b6);
        graph[d6].root = Some(b8);
        graph[d7].root = Some(c6);
        graph[e1].root = Some(c1);
        graph[e2].root = Some(d2);
        graph[e3].root = Some(b6);
        let layers = vec![
            vec![a1, a2],
            vec![b1, b2, b3, b4, b5, b6, b7, b8],
            vec![c1, c2, c3, c4, c5, c6],
            vec![d1, d2, d3, d4, d5, d6, d7],
            vec![e1, e2, e3],
        ];
        horizontal_compaction(&mut graph, &layers, false);
        assert_eq!(graph[a1].x, 0);
        assert_eq!(graph[a2].x, 20);
        assert_eq!(graph[b1].x, 0);
        assert_eq!(graph[b2].x, 10);
        assert_eq!(graph[b3].x, 20);
        assert_eq!(graph[b4].x, 30);
        assert_eq!(graph[b5].x, 40);
        assert_eq!(graph[b6].x, 50);
        assert_eq!(graph[b7].x, 60);
        assert_eq!(graph[b8].x, 70);
        assert_eq!(graph[c1].x, 10);
        assert_eq!(graph[c2].x, 30);
        assert_eq!(graph[c3].x, 40);
        assert_eq!(graph[c4].x, 50);
        assert_eq!(graph[c5].x, 70);
        assert_eq!(graph[c6].x, 80);
        assert_eq!(graph[d1].x, 10);
        assert_eq!(graph[d2].x, 20);
        assert_eq!(graph[d3].x, 30);
        assert_eq!(graph[d4].x, 40);
        assert_eq!(graph[d5].x, 50);
        assert_eq!(graph[d6].x, 70);
        assert_eq!(graph[d7].x, 80);
        assert_eq!(graph[e1].x, 10);
        assert_eq!(graph[e2].x, 20);
        assert_eq!(graph[e3].x, 50);
    }

    #[test]
    fn test_vertical_alignment_rt() {
        let mut graph = Graph::new();
        let a1 = graph.add_node(Node {
            width: 10,
            layer: 0,
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let a2 = graph.add_node(Node {
            width: 10,
            layer: 0,
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let b1 = graph.add_node(Node {
            width: 10,
            layer: 1,
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let b2 = graph.add_node(Node {
            width: 10,
            layer: 1,
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let b3 = graph.add_node(Node {
            width: 10,
            layer: 1,
            order: 2,
            dummy: true,
            ..Node::new()
        });
        let b4 = graph.add_node(Node {
            width: 10,
            layer: 1,
            order: 3,
            dummy: false,
            ..Node::new()
        });
        let b5 = graph.add_node(Node {
            width: 10,
            layer: 1,
            order: 4,
            dummy: true,
            ..Node::new()
        });
        let b6 = graph.add_node(Node {
            width: 10,
            layer: 1,
            order: 5,
            dummy: true,
            ..Node::new()
        });
        let b7 = graph.add_node(Node {
            width: 10,
            layer: 1,
            order: 6,
            dummy: false,
            ..Node::new()
        });
        let b8 = graph.add_node(Node {
            width: 10,
            layer: 1,
            order: 7,
            dummy: false,
            ..Node::new()
        });
        let c1 = graph.add_node(Node {
            width: 10,
            layer: 2,
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let c2 = graph.add_node(Node {
            width: 10,
            layer: 2,
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let c3 = graph.add_node(Node {
            width: 10,
            layer: 2,
            order: 2,
            dummy: true,
            ..Node::new()
        });
        let c4 = graph.add_node(Node {
            width: 10,
            layer: 2,
            order: 3,
            dummy: true,
            ..Node::new()
        });
        let c5 = graph.add_node(Node {
            width: 10,
            layer: 2,
            order: 4,
            dummy: true,
            ..Node::new()
        });
        let c6 = graph.add_node(Node {
            width: 10,
            layer: 2,
            order: 5,
            dummy: false,
            ..Node::new()
        });
        let d1 = graph.add_node(Node {
            width: 10,
            layer: 3,
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let d2 = graph.add_node(Node {
            width: 10,
            layer: 3,
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let d3 = graph.add_node(Node {
            width: 10,
            layer: 3,
            order: 2,
            dummy: true,
            ..Node::new()
        });
        let d4 = graph.add_node(Node {
            width: 10,
            layer: 3,
            order: 3,
            dummy: true,
            ..Node::new()
        });
        let d5 = graph.add_node(Node {
            width: 10,
            layer: 3,
            order: 4,
            dummy: true,
            ..Node::new()
        });
        let d6 = graph.add_node(Node {
            width: 10,
            layer: 3,
            order: 5,
            dummy: false,
            ..Node::new()
        });
        let d7 = graph.add_node(Node {
            width: 10,
            layer: 3,
            order: 6,
            dummy: true,
            ..Node::new()
        });
        let e1 = graph.add_node(Node {
            width: 10,
            layer: 4,
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let e2 = graph.add_node(Node {
            width: 10,
            layer: 4,
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let e3 = graph.add_node(Node {
            width: 10,
            layer: 4,
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
        vertical_alignment(&mut graph, &layers, true, false);
        horizontal_compaction(&mut graph, &layers, true);
        assert_eq!(graph[a1].x, 10);
        assert_eq!(graph[a2].x, 0);
        assert_eq!(graph[b1].x, 90);
        assert_eq!(graph[b2].x, 80);
        assert_eq!(graph[b3].x, 70);
        assert_eq!(graph[b4].x, 60);
        assert_eq!(graph[b5].x, 50);
        assert_eq!(graph[b6].x, 40);
        assert_eq!(graph[b7].x, 20);
        assert_eq!(graph[b8].x, 10);
        assert_eq!(graph[c1].x, 70);
        assert_eq!(graph[c2].x, 60);
        assert_eq!(graph[c3].x, 50);
        assert_eq!(graph[c4].x, 40);
        assert_eq!(graph[c5].x, 30);
        assert_eq!(graph[c6].x, 20);
        assert_eq!(graph[d1].x, 80);
        assert_eq!(graph[d2].x, 70);
        assert_eq!(graph[d3].x, 60);
        assert_eq!(graph[d4].x, 50);
        assert_eq!(graph[d5].x, 40);
        assert_eq!(graph[d6].x, 30);
        assert_eq!(graph[d7].x, 20);
        assert_eq!(graph[e1].x, 80);
        assert_eq!(graph[e2].x, 70);
        assert_eq!(graph[e3].x, 30);
    }

    #[test]
    fn test_vertical_alignment_lb() {
        let mut graph = Graph::new();
        let a1 = graph.add_node(Node {
            width: 10,
            layer: 0,
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let a2 = graph.add_node(Node {
            width: 10,
            layer: 0,
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let b1 = graph.add_node(Node {
            width: 10,
            layer: 1,
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let b2 = graph.add_node(Node {
            width: 10,
            layer: 1,
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let b3 = graph.add_node(Node {
            width: 10,
            layer: 1,
            order: 2,
            dummy: true,
            ..Node::new()
        });
        let b4 = graph.add_node(Node {
            width: 10,
            layer: 1,
            order: 3,
            dummy: false,
            ..Node::new()
        });
        let b5 = graph.add_node(Node {
            width: 10,
            layer: 1,
            order: 4,
            dummy: true,
            ..Node::new()
        });
        let b6 = graph.add_node(Node {
            width: 10,
            layer: 1,
            order: 5,
            dummy: true,
            ..Node::new()
        });
        let b7 = graph.add_node(Node {
            width: 10,
            layer: 1,
            order: 6,
            dummy: false,
            ..Node::new()
        });
        let b8 = graph.add_node(Node {
            width: 10,
            layer: 1,
            order: 7,
            dummy: false,
            ..Node::new()
        });
        let c1 = graph.add_node(Node {
            width: 10,
            layer: 2,
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let c2 = graph.add_node(Node {
            width: 10,
            layer: 2,
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let c3 = graph.add_node(Node {
            width: 10,
            layer: 2,
            order: 2,
            dummy: true,
            ..Node::new()
        });
        let c4 = graph.add_node(Node {
            width: 10,
            layer: 2,
            order: 3,
            dummy: true,
            ..Node::new()
        });
        let c5 = graph.add_node(Node {
            width: 10,
            layer: 2,
            order: 4,
            dummy: true,
            ..Node::new()
        });
        let c6 = graph.add_node(Node {
            width: 10,
            layer: 2,
            order: 5,
            dummy: false,
            ..Node::new()
        });
        let d1 = graph.add_node(Node {
            width: 10,
            layer: 3,
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let d2 = graph.add_node(Node {
            width: 10,
            layer: 3,
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let d3 = graph.add_node(Node {
            width: 10,
            layer: 3,
            order: 2,
            dummy: true,
            ..Node::new()
        });
        let d4 = graph.add_node(Node {
            width: 10,
            layer: 3,
            order: 3,
            dummy: true,
            ..Node::new()
        });
        let d5 = graph.add_node(Node {
            width: 10,
            layer: 3,
            order: 4,
            dummy: true,
            ..Node::new()
        });
        let d6 = graph.add_node(Node {
            width: 10,
            layer: 3,
            order: 5,
            dummy: false,
            ..Node::new()
        });
        let d7 = graph.add_node(Node {
            width: 10,
            layer: 3,
            order: 6,
            dummy: true,
            ..Node::new()
        });
        let e1 = graph.add_node(Node {
            width: 10,
            layer: 4,
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let e2 = graph.add_node(Node {
            width: 10,
            layer: 4,
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let e3 = graph.add_node(Node {
            width: 10,
            layer: 4,
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
        horizontal_compaction(&mut graph, &layers, false);
        assert_eq!(graph[a1].x, 60);
        assert_eq!(graph[a2].x, 70);
        assert_eq!(graph[b1].x, 10);
        assert_eq!(graph[b2].x, 20);
        assert_eq!(graph[b3].x, 30);
        assert_eq!(graph[b4].x, 40);
        assert_eq!(graph[b5].x, 50);
        assert_eq!(graph[b6].x, 60);
        assert_eq!(graph[b7].x, 80);
        assert_eq!(graph[b8].x, 90);
        assert_eq!(graph[c1].x, 10);
        assert_eq!(graph[c2].x, 20);
        assert_eq!(graph[c3].x, 50);
        assert_eq!(graph[c4].x, 60);
        assert_eq!(graph[c5].x, 70);
        assert_eq!(graph[c6].x, 80);
        assert_eq!(graph[d1].x, 0);
        assert_eq!(graph[d2].x, 10);
        assert_eq!(graph[d3].x, 20);
        assert_eq!(graph[d4].x, 50);
        assert_eq!(graph[d5].x, 60);
        assert_eq!(graph[d6].x, 70);
        assert_eq!(graph[d7].x, 80);
        assert_eq!(graph[e1].x, 0);
        assert_eq!(graph[e2].x, 10);
        assert_eq!(graph[e3].x, 50);
    }

    #[test]
    fn test_vertical_alignment_rb() {
        let mut graph = Graph::new();
        let a1 = graph.add_node(Node {
            width: 10,
            layer: 0,
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let a2 = graph.add_node(Node {
            width: 10,
            layer: 0,
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let b1 = graph.add_node(Node {
            width: 10,
            layer: 1,
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let b2 = graph.add_node(Node {
            width: 10,
            layer: 1,
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let b3 = graph.add_node(Node {
            width: 10,
            layer: 1,
            order: 2,
            dummy: true,
            ..Node::new()
        });
        let b4 = graph.add_node(Node {
            width: 10,
            layer: 1,
            order: 3,
            dummy: false,
            ..Node::new()
        });
        let b5 = graph.add_node(Node {
            width: 10,
            layer: 1,
            order: 4,
            dummy: true,
            ..Node::new()
        });
        let b6 = graph.add_node(Node {
            width: 10,
            layer: 1,
            order: 5,
            dummy: true,
            ..Node::new()
        });
        let b7 = graph.add_node(Node {
            width: 10,
            layer: 1,
            order: 6,
            dummy: false,
            ..Node::new()
        });
        let b8 = graph.add_node(Node {
            width: 10,
            layer: 1,
            order: 7,
            dummy: false,
            ..Node::new()
        });
        let c1 = graph.add_node(Node {
            width: 10,
            layer: 2,
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let c2 = graph.add_node(Node {
            width: 10,
            layer: 2,
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let c3 = graph.add_node(Node {
            width: 10,
            layer: 2,
            order: 2,
            dummy: true,
            ..Node::new()
        });
        let c4 = graph.add_node(Node {
            width: 10,
            layer: 2,
            order: 3,
            dummy: true,
            ..Node::new()
        });
        let c5 = graph.add_node(Node {
            width: 10,
            layer: 2,
            order: 4,
            dummy: true,
            ..Node::new()
        });
        let c6 = graph.add_node(Node {
            width: 10,
            layer: 2,
            order: 5,
            dummy: false,
            ..Node::new()
        });
        let d1 = graph.add_node(Node {
            width: 10,
            layer: 3,
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let d2 = graph.add_node(Node {
            width: 10,
            layer: 3,
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let d3 = graph.add_node(Node {
            width: 10,
            layer: 3,
            order: 2,
            dummy: true,
            ..Node::new()
        });
        let d4 = graph.add_node(Node {
            width: 10,
            layer: 3,
            order: 3,
            dummy: true,
            ..Node::new()
        });
        let d5 = graph.add_node(Node {
            width: 10,
            layer: 3,
            order: 4,
            dummy: true,
            ..Node::new()
        });
        let d6 = graph.add_node(Node {
            width: 10,
            layer: 3,
            order: 5,
            dummy: false,
            ..Node::new()
        });
        let d7 = graph.add_node(Node {
            width: 10,
            layer: 3,
            order: 6,
            dummy: true,
            ..Node::new()
        });
        let e1 = graph.add_node(Node {
            width: 10,
            layer: 4,
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let e2 = graph.add_node(Node {
            width: 10,
            layer: 4,
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let e3 = graph.add_node(Node {
            width: 10,
            layer: 4,
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
        vertical_alignment(&mut graph, &layers, true, true);
        horizontal_compaction(&mut graph, &layers, true);
        assert_eq!(graph[a1].x, 50);
        assert_eq!(graph[a2].x, 40);
        assert_eq!(graph[b1].x, 80);
        assert_eq!(graph[b2].x, 70);
        assert_eq!(graph[b3].x, 60);
        assert_eq!(graph[b4].x, 50);
        assert_eq!(graph[b5].x, 40);
        assert_eq!(graph[b6].x, 30);
        assert_eq!(graph[b7].x, 20);
        assert_eq!(graph[b8].x, 10);
        assert_eq!(graph[c1].x, 60);
        assert_eq!(graph[c2].x, 50);
        assert_eq!(graph[c3].x, 40);
        assert_eq!(graph[c4].x, 30);
        assert_eq!(graph[c5].x, 10);
        assert_eq!(graph[c6].x, 0);
        assert_eq!(graph[d1].x, 70);
        assert_eq!(graph[d2].x, 60);
        assert_eq!(graph[d3].x, 50);
        assert_eq!(graph[d4].x, 40);
        assert_eq!(graph[d5].x, 30);
        assert_eq!(graph[d6].x, 10);
        assert_eq!(graph[d7].x, 0);
        assert_eq!(graph[e1].x, 50);
        assert_eq!(graph[e2].x, 10);
        assert_eq!(graph[e3].x, 0);
    }
}
