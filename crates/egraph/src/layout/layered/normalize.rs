use crate::{Graph, NodeIndex};
use std::collections::HashMap;

pub enum NodeType {
    Original {
        id: NodeIndex,
    },
    Dummy {
        source: NodeIndex,
        target: NodeIndex,
    },
}

pub struct NodeAlignment {
    pub index: usize,
    pub layer: usize,
    pub order: usize,
    pub width: f32,
    pub node_type: NodeType,
}

pub fn normalize<D, G: Graph<D>>(
    graph: &G,
    layers_map: &HashMap<NodeIndex, usize>,
    sizes: &HashMap<NodeIndex, (f32, f32)>,
    horizontal_margin: f32,
) -> Vec<NodeAlignment> {
    let mut nodes = Vec::new();
    for u in graph.nodes() {
        let u_layer = layers_map[&u];
        for v in graph.out_nodes(u) {
            let v_layer = layers_map[&v];
            let length = v_layer - u_layer;
            if length == 1 {
                continue;
            }
            for layer in u_layer + 1..v_layer {
                nodes.push(NodeAlignment {
                    index: nodes.len(),
                    layer: layer,
                    order: 0,
                    width: horizontal_margin,
                    node_type: NodeType::Dummy {
                        source: u,
                        target: v,
                    },
                });
            }
        }
        let (width, _) = sizes[&u];
        nodes.push(NodeAlignment {
            index: nodes.len(),
            layer: u_layer,
            order: 0,
            width: width + horizontal_margin,
            node_type: NodeType::Original { id: u },
        });
    }
    nodes
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
