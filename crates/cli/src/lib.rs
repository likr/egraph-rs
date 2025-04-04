//! This crate provides utility functions for reading and writing graph data
//! in JSON format, specifically designed for use within the `egraph-rs`
//! command-line tools. It handles serialization and deserialization
//! between `petgraph` graph structures (with optional node/edge data)
//! and a simple JSON representation that includes node positions.

use petgraph::prelude::*;
use petgraph_drawing::DrawingEuclidean2d;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, BufWriter},
};

/// Represents node data for serialization/deserialization.
#[derive(Clone, Serialize, Deserialize)]
struct NodeData<N> {
    id: usize,
    x: Option<f32>,
    y: Option<f32>,
    data: Option<N>,
}

/// Represents link (edge) data for serialization/deserialization.
#[derive(Clone, Serialize, Deserialize)]
struct LinkData<E> {
    source: usize,
    target: usize,
    data: Option<E>,
}

/// Represents the overall graph structure for serialization/deserialization.
#[derive(Clone, Serialize, Deserialize)]
struct GraphData<N, E> {
    nodes: Vec<NodeData<N>>,
    links: Vec<LinkData<E>>,
}

/// Type alias for an undirected graph using `petgraph::Graph`.
type UndirectedGraph<N, E> = Graph<Option<N>, Option<E>, Undirected>;
/// Type alias for a 2D Euclidean drawing using `petgraph_drawing::DrawingEuclidean2d`.
type Drawing2D = DrawingEuclidean2d<NodeIndex, f32>;

/// Reads a graph and its drawing information from a JSON file.
///
/// The JSON file is expected to follow the `GraphData` structure.
/// It deserializes node and edge data of types `N` and `E` respectively.
/// Initial node positions (if present in the JSON) are used to populate the `Drawing2D`.
///
/// # Arguments
///
/// * `input_path` - Path to the input JSON file.
///
/// # Returns
///
/// A tuple containing the `UndirectedGraph<N, E>` and its corresponding `Drawing2D`.
///
/// # Panics
///
/// Panics if the file cannot be opened or if JSON deserialization fails.
pub fn read_graph<N: Clone + DeserializeOwned, E: Clone + DeserializeOwned>(
    input_path: &str,
) -> (UndirectedGraph<N, E>, Drawing2D) {
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

/// Writes a graph and its drawing information to a JSON file.
///
/// Serializes the graph structure and node positions into the `GraphData` format.
/// Node and edge data types `N` and `E` must implement `Serialize`.
///
/// # Arguments
///
/// * `graph` - The `UndirectedGraph<N, E>` to write.
/// * `drawing` - The `Drawing2D` containing node positions.
/// * `output_path` - Path to the output JSON file.
///
/// # Panics
///
/// Panics if the file cannot be created or if JSON serialization fails.
pub fn write_graph<N: Clone + Serialize, E: Clone + Serialize>(
    graph: &UndirectedGraph<N, E>,
    drawing: &Drawing2D,
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
