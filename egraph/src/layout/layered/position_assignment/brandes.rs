use super::super::graph::{Edge, Node};
use super::horizontal_compaction::horizontal_compaction;
use super::mark_conflicts::mark_conflicts;
use super::vertical_alignment::vertical_alignment;
use petgraph::graph::IndexType;
use petgraph::prelude::*;
use std::iter::FromIterator;

fn set_y<Ix: IndexType>(
    graph: &mut Graph<Node<Ix>, Edge, Directed, Ix>,
    layers: &Vec<Vec<NodeIndex<Ix>>>,
) {
    let mut y_offset = 0;
    for layer in layers {
        let max_height = layer.iter().map(|u| graph[*u].height).max().unwrap() as i32;
        y_offset += max_height / 2;
        for &u in layer {
            graph[u].y = y_offset;
        }
        y_offset += max_height / 2
    }
}

fn normalize<Ix: IndexType>(graph: &mut Graph<Node<Ix>, Edge, Directed, Ix>) {
    let x_min = graph
        .node_indices()
        .map(|u| graph[u].x - graph[u].orig_width as i32 / 2)
        .min()
        .unwrap();
    let y_min = graph
        .node_indices()
        .map(|u| graph[u].y - graph[u].orig_height as i32 / 2)
        .min()
        .unwrap();
    for u in graph.node_indices() {
        graph[u].x -= x_min;
        graph[u].y -= y_min;
    }
}

pub fn brandes<Ix: IndexType>(
    graph: &mut Graph<Node<Ix>, Edge, Directed, Ix>,
    layers: &Vec<Vec<NodeIndex<Ix>>>,
) {
    mark_conflicts(graph, layers);
    let directions = vec![(false, false), (true, false), (false, true), (true, true)];
    let mut xs = Vec::from_iter(graph.node_indices().map(|_| [0; 4]));
    let mut left = [0; 4];
    let mut right = [0; 4];
    for (i, &direction) in directions.iter().enumerate() {
        let (rtol, btot) = direction;
        vertical_alignment(graph, layers, rtol, btot);
        horizontal_compaction(graph, layers, rtol);
        if rtol {
            for u in graph.node_indices() {
                graph[u].x = -graph[u].x;
            }
        }
        left[i] = graph.node_indices().map(|u| graph[u].x).min().unwrap();
        right[i] = graph.node_indices().map(|u| graph[u].x).max().unwrap();
        for (j, u) in graph.node_indices().enumerate() {
            xs[j][i] = graph[u].x;
        }
    }
    let min_width_index = (0..4).min_by_key(|&i| right[i] - left[i]).unwrap();
    for (i, &direction) in directions.iter().enumerate() {
        let (rtol, _) = direction;
        if rtol {
            for j in 0..graph.node_count() {
                xs[j][i] += right[min_width_index] - right[i];
            }
        } else {
            for j in 0..graph.node_count() {
                xs[j][i] += left[min_width_index] - left[i];
            }
        }
    }

    for (i, u) in graph.node_indices().enumerate() {
        xs[i].sort();
        graph[u].x = (xs[i][1] + xs[i][2]) / 2;
    }

    set_y(graph, layers);
    normalize(graph);
}

#[cfg(test)]
mod tests {
    use super::super::super::graph::{Edge, Node};
    use super::*;
    use petgraph::Graph;

    #[test]
    fn test_brandes() {
        let mut graph = Graph::new();
        let a1 = graph.add_node(Node {
            width: 10,
            height: 10,
            orig_width: 10,
            orig_height: 10,
            layer: 0,
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let a2 = graph.add_node(Node {
            width: 10,
            height: 10,
            orig_width: 10,
            orig_height: 10,
            layer: 0,
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let b1 = graph.add_node(Node {
            width: 10,
            height: 10,
            orig_width: 10,
            orig_height: 10,
            layer: 1,
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let b2 = graph.add_node(Node {
            width: 10,
            height: 10,
            orig_width: 10,
            orig_height: 10,
            layer: 1,
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let b3 = graph.add_node(Node {
            width: 10,
            height: 10,
            orig_width: 10,
            orig_height: 10,
            layer: 1,
            order: 2,
            dummy: true,
            ..Node::new()
        });
        let b4 = graph.add_node(Node {
            width: 10,
            height: 10,
            orig_width: 10,
            orig_height: 10,
            layer: 1,
            order: 3,
            dummy: false,
            ..Node::new()
        });
        let b5 = graph.add_node(Node {
            width: 10,
            height: 10,
            orig_width: 10,
            orig_height: 10,
            layer: 1,
            order: 4,
            dummy: true,
            ..Node::new()
        });
        let b6 = graph.add_node(Node {
            width: 10,
            height: 10,
            orig_width: 10,
            orig_height: 10,
            layer: 1,
            order: 5,
            dummy: true,
            ..Node::new()
        });
        let b7 = graph.add_node(Node {
            width: 10,
            height: 10,
            orig_width: 10,
            orig_height: 10,
            layer: 1,
            order: 6,
            dummy: false,
            ..Node::new()
        });
        let b8 = graph.add_node(Node {
            width: 10,
            height: 10,
            orig_width: 10,
            orig_height: 10,
            layer: 1,
            order: 7,
            dummy: false,
            ..Node::new()
        });
        let c1 = graph.add_node(Node {
            width: 10,
            height: 10,
            orig_width: 10,
            orig_height: 10,
            layer: 2,
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let c2 = graph.add_node(Node {
            width: 10,
            height: 10,
            orig_width: 10,
            orig_height: 10,
            layer: 2,
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let c3 = graph.add_node(Node {
            width: 10,
            height: 10,
            orig_width: 10,
            orig_height: 10,
            layer: 2,
            order: 2,
            dummy: true,
            ..Node::new()
        });
        let c4 = graph.add_node(Node {
            width: 10,
            height: 10,
            orig_width: 10,
            orig_height: 10,
            layer: 2,
            order: 3,
            dummy: true,
            ..Node::new()
        });
        let c5 = graph.add_node(Node {
            width: 10,
            height: 10,
            orig_width: 10,
            orig_height: 10,
            layer: 2,
            order: 4,
            dummy: true,
            ..Node::new()
        });
        let c6 = graph.add_node(Node {
            width: 10,
            height: 10,
            orig_width: 10,
            orig_height: 10,
            layer: 2,
            order: 5,
            dummy: false,
            ..Node::new()
        });
        let d1 = graph.add_node(Node {
            width: 10,
            height: 10,
            orig_width: 10,
            orig_height: 10,
            layer: 3,
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let d2 = graph.add_node(Node {
            width: 10,
            height: 10,
            orig_width: 10,
            orig_height: 10,
            layer: 3,
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let d3 = graph.add_node(Node {
            width: 10,
            height: 10,
            orig_width: 10,
            orig_height: 10,
            layer: 3,
            order: 2,
            dummy: true,
            ..Node::new()
        });
        let d4 = graph.add_node(Node {
            width: 10,
            height: 10,
            orig_width: 10,
            orig_height: 10,
            layer: 3,
            order: 3,
            dummy: true,
            ..Node::new()
        });
        let d5 = graph.add_node(Node {
            width: 10,
            height: 10,
            orig_width: 10,
            orig_height: 10,
            layer: 3,
            order: 4,
            dummy: true,
            ..Node::new()
        });
        let d6 = graph.add_node(Node {
            width: 10,
            height: 10,
            orig_width: 10,
            orig_height: 10,
            layer: 3,
            order: 5,
            dummy: false,
            ..Node::new()
        });
        let d7 = graph.add_node(Node {
            width: 10,
            height: 10,
            orig_width: 10,
            orig_height: 10,
            layer: 3,
            order: 6,
            dummy: true,
            ..Node::new()
        });
        let e1 = graph.add_node(Node {
            width: 10,
            height: 10,
            orig_width: 10,
            orig_height: 10,
            layer: 4,
            order: 0,
            dummy: false,
            ..Node::new()
        });
        let e2 = graph.add_node(Node {
            width: 10,
            height: 10,
            orig_width: 10,
            orig_height: 10,
            layer: 4,
            order: 1,
            dummy: false,
            ..Node::new()
        });
        let e3 = graph.add_node(Node {
            width: 10,
            height: 10,
            orig_width: 10,
            orig_height: 10,
            layer: 4,
            order: 2,
            dummy: false,
            ..Node::new()
        });
        graph.add_edge(a1, b1, Edge::new());
        graph.add_edge(a1, b6, Edge::new());
        graph.add_edge(a1, b8, Edge::new());
        graph.add_edge(a2, b3, Edge::new());
        graph.add_edge(a2, b5, Edge::new());
        graph.add_edge(b2, c2, Edge::new());
        graph.add_edge(b3, c2, Edge::new());
        graph.add_edge(b4, c2, Edge::new());
        graph.add_edge(b5, c3, Edge::new());
        graph.add_edge(b6, c4, Edge::new());
        graph.add_edge(b7, c2, Edge::new());
        graph.add_edge(b7, c6, Edge::new());
        graph.add_edge(b8, c2, Edge::new());
        graph.add_edge(b8, c5, Edge::new());
        graph.add_edge(c1, d1, Edge::new());
        graph.add_edge(c1, d2, Edge::new());
        graph.add_edge(c1, d6, Edge::new());
        graph.add_edge(c3, d4, Edge::new());
        graph.add_edge(c4, d5, Edge::new());
        graph.add_edge(c5, d6, Edge::new());
        graph.add_edge(c6, d3, Edge::new());
        graph.add_edge(c6, d7, Edge::new());
        graph.add_edge(d1, e1, Edge::new());
        graph.add_edge(d1, e2, Edge::new());
        graph.add_edge(d2, e2, Edge::new());
        graph.add_edge(d3, e1, Edge::new());
        graph.add_edge(d4, e3, Edge::new());
        graph.add_edge(d5, e3, Edge::new());
        graph.add_edge(d6, e3, Edge::new());
        graph.add_edge(d7, e3, Edge::new());
        let layers = vec![
            vec![a1, a2],
            vec![b1, b2, b3, b4, b5, b6, b7, b8],
            vec![c1, c2, c3, c4, c5, c6],
            vec![d1, d2, d3, d4, d5, d6, d7],
            vec![e1, e2, e3],
        ];
        brandes(&mut graph, &layers);
        assert_eq!(graph[a1].x, 50);
        assert_eq!(graph[a2].x, 60);
        assert_eq!(graph[b1].x, 5);
        assert_eq!(graph[b2].x, 15);
        assert_eq!(graph[b3].x, 25);
        assert_eq!(graph[b4].x, 35);
        assert_eq!(graph[b5].x, 45);
        assert_eq!(graph[b6].x, 55);
        assert_eq!(graph[b7].x, 65);
        assert_eq!(graph[b8].x, 75);
        assert_eq!(graph[c1].x, 15);
        assert_eq!(graph[c2].x, 30);
        assert_eq!(graph[c3].x, 45);
        assert_eq!(graph[c4].x, 55);
        assert_eq!(graph[c5].x, 75);
        assert_eq!(graph[c6].x, 85);
        assert_eq!(graph[d1].x, 10);
        assert_eq!(graph[d2].x, 20);
        assert_eq!(graph[d3].x, 30);
        assert_eq!(graph[d4].x, 45);
        assert_eq!(graph[d5].x, 55);
        assert_eq!(graph[d6].x, 75);
        assert_eq!(graph[d7].x, 85);
        assert_eq!(graph[e1].x, 10);
        assert_eq!(graph[e2].x, 20);
        assert_eq!(graph[e3].x, 55);
        assert_eq!(graph[a1].y, 5);
        assert_eq!(graph[a2].y, 5);
        assert_eq!(graph[b1].y, 15);
        assert_eq!(graph[b2].y, 15);
        assert_eq!(graph[b3].y, 15);
        assert_eq!(graph[b4].y, 15);
        assert_eq!(graph[b5].y, 15);
        assert_eq!(graph[b6].y, 15);
        assert_eq!(graph[b7].y, 15);
        assert_eq!(graph[b8].y, 15);
        assert_eq!(graph[c1].y, 25);
        assert_eq!(graph[c2].y, 25);
        assert_eq!(graph[c3].y, 25);
        assert_eq!(graph[c4].y, 25);
        assert_eq!(graph[c5].y, 25);
        assert_eq!(graph[c6].y, 25);
        assert_eq!(graph[d1].y, 35);
        assert_eq!(graph[d2].y, 35);
        assert_eq!(graph[d3].y, 35);
        assert_eq!(graph[d4].y, 35);
        assert_eq!(graph[d5].y, 35);
        assert_eq!(graph[d6].y, 35);
        assert_eq!(graph[d7].y, 35);
        assert_eq!(graph[e1].y, 45);
        assert_eq!(graph[e2].y, 45);
        assert_eq!(graph[e3].y, 45);
    }
}
