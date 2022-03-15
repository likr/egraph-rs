use argparse::{ArgumentParser, Store};
use petgraph::prelude::*;
use petgraph_layout_force_simulation::{Coordinates, ForceToNode, Simulation};
use petgraph_layout_fruchterman_reingold::FruchtermanReingoldForce;
use petgraph_layout_non_euclidean_force_simulation::{Map, SphericalSpace};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    f32::consts::PI,
    fs::File,
    io::{BufReader, BufWriter},
};

#[derive(Clone, Serialize, Deserialize)]
struct InputNode {
    id: usize,
}

#[derive(Clone, Serialize, Deserialize)]
struct InputLink {
    source: usize,
    target: usize,
}

#[derive(Clone, Serialize, Deserialize)]
struct InputGraph {
    nodes: Vec<InputNode>,
    links: Vec<InputLink>,
}

#[derive(Clone, Serialize, Deserialize)]
struct OutputNode {
    id: usize,
    x: f32,
    y: f32,
}

#[derive(Clone, Serialize, Deserialize)]
struct OutputLink {
    source: usize,
    target: usize,
}

#[derive(Clone, Serialize, Deserialize)]
struct OutputGraph {
    nodes: Vec<OutputNode>,
    links: Vec<OutputLink>,
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

fn read_graph(input_path: &str) -> Graph<InputNode, InputLink, Undirected> {
    let file = File::open(input_path).unwrap();
    let reader = BufReader::new(file);
    let input_graph: InputGraph = serde_json::from_reader(reader).unwrap();

    let mut graph = Graph::new_undirected();
    let mut node_ids = HashMap::new();
    for node in input_graph.nodes.iter() {
        node_ids.insert(node.id, graph.add_node(node.clone()));
    }
    for link in input_graph.links.iter() {
        graph.add_edge(node_ids[&link.source], node_ids[&link.target], link.clone());
    }
    graph
}

fn layout(graph: &Graph<InputNode, InputLink, Undirected>) -> Coordinates<u32> {
    let mut coordinates = Coordinates::initial_placement(&graph);
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
    coordinates
}

fn construct_output(
    graph: &Graph<InputNode, InputLink, Undirected>,
    coordinates: &Coordinates<u32>,
) -> OutputGraph {
    OutputGraph {
        nodes: graph
            .node_indices()
            .map(|u| OutputNode {
                id: graph[u].id,
                x: coordinates.x(u).unwrap(),
                y: coordinates.y(u).unwrap(),
            })
            .collect::<Vec<_>>(),
        links: graph
            .edge_indices()
            .map(|e| OutputLink {
                source: graph[e].source,
                target: graph[e].target,
            })
            .collect::<Vec<_>>(),
    }
}

fn write_result(output_path: &str, graph: &OutputGraph) {
    let file = File::create(output_path).unwrap();
    let writer = BufWriter::new(file);
    serde_json::to_writer(writer, graph).unwrap();
}

fn main() {
    let mut input_path = "".to_string();
    let mut output_path = "".to_string();
    parse_args(&mut input_path, &mut output_path);
    let input_graph = read_graph(&input_path);
    let coordinates = layout(&input_graph);
    let output_graph = construct_output(&input_graph, &coordinates);
    write_result(&output_path, &output_graph);
}
