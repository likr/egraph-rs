//! Applies the Stochastic Gradient Descent (SGD) layout algorithm to a graph.
//!
//! Reads a graph layout from an input JSON file, applies the SGD layout algorithm
//! to update node positions, and writes the resulting layout to an output JSON file.
//!
//! # Usage
//!
//! `cargo run --bin sgd -- <input.json> <output.json>`
//!
//! * `<input.json>`: Path to the input JSON file (GraphData format). Initial coordinates are used.
//! * `<output.json>`: Path to the output JSON file for the layout after SGD.

use argparse::{ArgumentParser, Store};
use egraph_cli::{read_graph, write_graph};
use petgraph::prelude::*;
use petgraph_drawing::DrawingEuclidean2d;
use petgraph_layout_sgd::{Scheduler, SchedulerExponential, Sgd, SparseSgd};
use rand::thread_rng;

/// Parses command-line arguments for input and output file paths.
///
/// # Arguments
///
/// * `input_path` - Mutable string to store the input file path.
/// * `output_path` - Mutable string to store the output file path.
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

/// Applies the Sparse Stochastic Gradient Descent (SGD) layout algorithm.
///
/// Modifies the provided `coordinates` in place using the `SparseSgd` implementation
/// with an exponential learning rate scheduler.
///
/// # Arguments
///
/// * `graph` - The graph to layout (node/edge data ignored).
/// * `coordinates` - Mutable `DrawingEuclidean2d` containing the initial and resulting node positions.
fn layout(
    graph: &Graph<Option<()>, Option<()>, Undirected>,
    coordinates: &mut DrawingEuclidean2d<NodeIndex, f32>,
) {
    let mut rng = thread_rng();
    let mut sgd = SparseSgd::new_with_rng(graph, |_| 30., 281, &mut rng);
    let mut scheduler = sgd.scheduler::<SchedulerExponential<f32>>(867, 0.1);
    scheduler.run(&mut |eta| {
        sgd.shuffle(&mut rng);
        sgd.apply(coordinates, eta);
    });
}

/// Main entry point: parses args, reads graph, applies layout, writes result.
fn main() {
    let mut input_path = "".to_string();
    let mut output_path = "".to_string();
    parse_args(&mut input_path, &mut output_path);
    let (input_graph, mut coordinates) = read_graph(&input_path);
    layout(&input_graph, &mut coordinates);
    write_graph(&input_graph, &coordinates, &output_path);
}
