use petgraph::prelude::*;
use petgraph_drawing::DrawingEuclidean2d;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, BufWriter},
};

#[derive(Clone, Serialize, Deserialize)]
struct NodeData<N> {
    id: usize,
    x: Option<f32>,
    y: Option<f32>,
    data: Option<N>,
}

#[derive(Clone, Serialize, Deserialize)]
struct LinkData<E> {
    source: usize,
    target: usize,
    data: Option<E>,
}

#[derive(Clone, Serialize, Deserialize)]
struct GraphData<N, E> {
    nodes: Vec<NodeData<N>>,
    links: Vec<LinkData<E>>,
}

pub fn read_graph<N: Clone + DeserializeOwned, E: Clone + DeserializeOwned>(
    input_path: &str,
) -> (
    Graph<Option<N>, Option<E>, Undirected>,
    DrawingEuclidean2d<NodeIndex, f32>,
) {
    let file = File::open(input_path).unwrap();
    let reader = BufReader::new(file);
    let input_graph: GraphData<N, E> = serde_json::from_reader(reader).unwrap();

    let mut graph = Graph::new_undirected();
    let mut node_ids = HashMap::new();
    for node in input_graph.nodes.iter() {
        node_ids.insert(node.id, graph.add_node(node.data.clone()));
    }
    for link in input_graph.links.iter() {
        graph.add_edge(
            node_ids[&link.source],
            node_ids[&link.target],
            link.data.clone(),
        );
    }
    let mut drawing = DrawingEuclidean2d::initial_placement(&graph);
    for node in input_graph.nodes.iter() {
        let u = node_ids[&node.id];
        if let Some(x) = node.x {
            drawing.set_x(u, x);
        }
        if let Some(y) = node.y {
            drawing.set_y(u, y);
        }
    }
    (graph, drawing)
}

pub fn write_graph<N: Clone + Serialize, E: Clone + Serialize>(
    graph: &Graph<Option<N>, Option<E>, Undirected>,
    drawing: &DrawingEuclidean2d<NodeIndex, f32>,
    output_path: &str,
) {
    let output = GraphData {
        nodes: graph
            .node_indices()
            .map(|u| NodeData {
                id: u.index(),
                x: Some(drawing.x(u).unwrap()),
                y: Some(drawing.y(u).unwrap()),
                data: graph[u].clone(),
            })
            .collect::<Vec<_>>(),
        links: graph
            .edge_indices()
            .map(|e| {
                let (source, target) = graph.edge_endpoints(e).unwrap();
                LinkData {
                    source: source.index(),
                    target: target.index(),
                    data: graph[e].clone(),
                }
            })
            .collect::<Vec<_>>(),
    };

    let file = File::create(output_path).unwrap();
    let writer = BufWriter::new(file);
    serde_json::to_writer(writer, &output).unwrap();
}
