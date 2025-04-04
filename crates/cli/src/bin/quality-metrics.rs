//! Computes various quality metrics for a given graph drawing.
//!
//! Reads a graph layout from an input JSON file, calculates quality metrics,
//! and writes the results to an output JSON file.
//!
//! # Usage
//!
//! `cargo run --bin quality-metrics -- <input.json> <output.json>`
//!
//! * `<input.json>`: Path to the input JSON file (GraphData format).
//! * `<output.json>`: Path to the output JSON file for metrics.

use argparse::{ArgumentParser, Store};
use egraph_cli::read_graph;
use petgraph::prelude::*;
use petgraph_algorithm_shortest_path::warshall_floyd;
use petgraph_drawing::DrawingEuclidean2d;
use petgraph_quality_metrics::{quality_metrics, QualityMetric};
use std::{collections::HashMap, fs::File, io::BufWriter};

// Define type aliases for clarity and to address clippy warning
type MyGraph = Graph<Option<()>, Option<()>, Undirected>;
type MyDrawing = DrawingEuclidean2d<NodeIndex, f32>;

/// Parses command-line arguments for input and output file paths.
///
/// # Arguments
///
/// * `input_path` - Mutable string to store the input file path.
/// * `output_path` - Mutable string to store the output file path.
fn parse_args(input_path: &mut String, output_path: &mut String) {
    let mut parser = ArgumentParser::new();
    parser.set_description("Compute graph drawing quality metrics.");
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

/// Computes quality metrics for the given graph and drawing.
///
/// Calculates all-pairs shortest paths using Warshall-Floyd and then computes metrics.
///
/// # Arguments
///
/// * `graph` - The graph (node/edge data ignored).
/// * `drawing` - The 2D node coordinates.
///
/// # Returns
///
/// A vector of tuples `(QualityMetric, f32)` representing metric results.
fn compute_metrics(
    graph: &MyGraph,     // Use the alias here
    drawing: &MyDrawing, // Use the alias here
) -> Vec<(QualityMetric, f32)> {
    let distance = warshall_floyd(graph, &mut |_| 1.);
    quality_metrics(graph, drawing, &distance)
}

/// Writes the computed quality metrics to a JSON file.
///
/// The output is a JSON object mapping metric names to their values.
///
/// # Arguments
///
/// * `output` - Slice containing `(QualityMetric, f32)` results.
/// * `output_path` - Path to the output JSON file.
///
/// # Panics
///
/// Panics on file creation or serialization errors.
fn write_result(output: &[(QualityMetric, f32)], output_path: &str) {
    let file = File::create(output_path).unwrap();
    let writer = BufWriter::new(file);
    serde_json::to_writer(
        writer,
        &output
            .iter()
            .map(|&(q, v)| (q.name(), v))
            .collect::<HashMap<_, _>>(),
    )
    .unwrap();
}

/// Main entry point: parses args, reads graph, computes metrics, writes results.
fn main() {
    let mut input_path = "".to_string();
    let mut output_path = "".to_string();
    parse_args(&mut input_path, &mut output_path);
    // Use the type aliases
    let (graph, coordinates): (MyGraph, MyDrawing) = read_graph(&input_path);
    let quality_metrics = compute_metrics(&graph, &coordinates);
    write_result(&quality_metrics, &output_path);
}
