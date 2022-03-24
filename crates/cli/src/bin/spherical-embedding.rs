use argparse::{ArgumentParser, Store};
use egraph_cli::{read_graph, write_graph};
use petgraph::prelude::*;
use petgraph_layout_force_simulation::{Coordinates, ForceToNode, Simulation};
use petgraph_layout_fruchterman_reingold::FruchtermanReingoldForce;
use petgraph_layout_non_euclidean_force_simulation::{Map, SphericalSpace};
use std::f32::consts::PI;

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
