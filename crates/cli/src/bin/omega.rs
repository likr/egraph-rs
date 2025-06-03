//! Applies the Omega SGD layout algorithm to a graph using spectral coordinates.
//!
//! Reads a graph layout from an input JSON file, applies the Omega layout algorithm
//! to update node positions, and writes the resulting layout to an output JSON file.
//!
//! The Omega algorithm differs from standard SGD by using spectral coordinates derived
//! from the graph Laplacian to construct node pairs more intelligently:
//! 1. Computes the smallest d non-zero eigenvalues and eigenvectors of the graph Laplacian
//! 2. Creates d-dimensional coordinates by dividing eigenvectors by sqrt of eigenvalues  
//! 3. Adds edge-based node pairs using Euclidean distances from coordinates
//! 4. Adds k random node pairs per node using Euclidean distances (avoiding duplicates)
//!
//! # Usage
//!
//! `cargo run --bin omega -- <input.json> <output.json>`
//!
//! * `<input.json>`: Path to the input JSON file (GraphData format). Initial coordinates are used.
//! * `<output.json>`: Path to the output JSON file for the layout after Omega SGD.
//!
//! # Algorithm Parameters
//!
//! * Spectral dimensions (d): 2 (for 2D layouts)
//! * Random pairs per node (k): 30 (balances quality and performance)
//! * Iterations: 1000
//! * Learning rate: Exponential scheduler from high initial rate to 0.1 final rate

use argparse::{ArgumentParser, Store};
use egraph_cli::{read_graph, write_pos};
use petgraph::prelude::*;
use petgraph_drawing::DrawingEuclidean2d;
use petgraph_layout_omega::Omega;
use petgraph_layout_sgd::{Scheduler, SchedulerExponential, Sgd};
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

/// Applies the Omega SGD layout algorithm using spectral coordinates.
///
/// Modifies the provided `coordinates` in place using the `Omega` implementation
/// with an exponential learning rate scheduler. The algorithm uses spectral coordinates
/// derived from the graph Laplacian to construct more meaningful node pairs for SGD.
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

    // Algorithm parameters:
    // d = 2: Number of spectral dimensions (for 2D layouts)
    // k = 30: Number of random pairs per node (same as SparseSgd)
    let d = 10;
    let k = 200;

    let mut omega = Omega::new(graph, |_| 30.0, d, k, &mut rng);

    // Use same iteration count and learning rate schedule as SGD
    // 1000 iterations with exponential decay to final eta of 0.1
    let mut scheduler = omega.scheduler::<SchedulerExponential<f32>>(1000, 0.1);

    scheduler.run(&mut |eta| {
        omega.shuffle(&mut rng);
        omega.apply(coordinates, eta);
    });
}

/// Main entry point: parses args, reads graph, applies Omega layout, writes result.
fn main() {
    let mut input_path = "".to_string();
    let mut output_path = "".to_string();
    parse_args(&mut input_path, &mut output_path);
    let (input_graph, mut coordinates) = read_graph(&input_path);
    layout(&input_graph, &mut coordinates);
    write_pos(&input_graph, &coordinates, &output_path);
}
