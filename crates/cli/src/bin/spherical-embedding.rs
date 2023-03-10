use argparse::{ArgumentParser, Store};
use petgraph::prelude::*;
use petgraph_layout_force_simulation::{Coordinates, ForceToNode, Simulation};
use petgraph_layout_fruchterman_reingold::FruchtermanReingoldForce;
use petgraph_layout_non_euclidean_force_simulation::{Map, SphericalSpace};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    collections::HashMap,
    f32::consts::PI,
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
) -> (Graph<Option<N>, Option<E>, Undirected>, Coordinates<u32>) {
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
    let mut coordinates = Coordinates::initial_placement(&graph);
    for node in input_graph.nodes.iter() {
        let u = node_ids[&node.id];
        if let Some(x) = node.x {
            coordinates.set_x(u, x);
        }
        if let Some(y) = node.y {
            coordinates.set_x(u, y);
        }
    }
    (graph, coordinates)
}

pub fn write_graph<N: Clone + Serialize, E: Clone + Serialize>(
    graph: &Graph<Option<N>, Option<E>, Undirected>,
    coordinates: &Coordinates<u32>,
    output_path: &str,
) {
    let output = GraphData {
        nodes: graph
            .node_indices()
            .map(|u| NodeData {
                id: u.index(),
                x: Some(coordinates.x(u).unwrap()),
                y: Some(coordinates.y(u).unwrap()),
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

fn parse_args(input_path: &mut String, output_path: &mut String) {
    let mut parser = ArgumentParser::new();
    parser
        .refer(input_path)
        .add_argument("input", Store, "input file path")
        .required();
    parser
        .refer(output_path)
        .add_argument("output", Store, "output file path")
        .required();
    parser.parse_args_or_exit();
}

fn layout(graph: &Graph<Option<()>, Option<()>, Undirected>, coordinates: &mut Coordinates<u32>) {
    for (i, u) in graph.node_indices().enumerate() {
        coordinates.set_x(u, (2. * PI * i as f32) / graph.node_count() as f32);
        coordinates.set_y(u, i as f32 + 1.);
    }
    let mut tangent_space = Coordinates::initial_placement(&graph);
    let mut simulation = Simulation::new();
    let forces = [FruchtermanReingoldForce::new(&graph, 0.5, 0.01)];
    simulation.run(&mut |alpha| {
        for u in graph.node_indices() {
            SphericalSpace::map_to_tangent_space(
                u.index(),
                &mut coordinates.points,
                &mut tangent_space.points,
            );
            for force in forces.iter() {
                force.apply_to_node(u.index(), &mut tangent_space.points, alpha);
            }
            SphericalSpace::update_position(
                u.index(),
                &mut coordinates.points,
                &mut tangent_space.points,
                0.6,
            );
        }
    });
}
fn main() {
    let mut input_path = "".to_string();
    let mut output_path = "".to_string();
    parse_args(&mut input_path, &mut output_path);
    let (input_graph, mut coordinates) = read_graph(&input_path);
    layout(&input_graph, &mut coordinates);
    write_graph(&input_graph, &coordinates, &output_path);
}
