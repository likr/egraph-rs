use argparse::{ArgumentParser, Store};
use egraph_cli::read_graph;
use petgraph::prelude::*;
use petgraph_algorithm_shortest_path::warshall_floyd;
use petgraph_layout_force_simulation::Coordinates;
use petgraph_quality_metrics::{number_of_crossings, shape_quality, stress};
use serde::Serialize;
use std::{fs::File, io::BufWriter};

#[derive(Serialize)]
struct QualityMetrics {
    #[serde(rename = "numberOfCrossings")]
    number_of_crossings: usize,
    #[serde(rename = "shapeQuality")]
    shape_quality: f32,
    stress: f32,
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

fn compute_metrics(
    graph: &Graph<Option<()>, Option<()>, Undirected>,
    coordinates: &Coordinates<u32>,
) -> QualityMetrics {
    let distance = warshall_floyd(graph, &mut |_| 1.);
    QualityMetrics {
        number_of_crossings: number_of_crossings(graph, coordinates),
        shape_quality: shape_quality(graph, coordinates),
        stress: stress(coordinates, &distance),
    }
}

fn write_result(output: &QualityMetrics, output_path: &str) {
    let file = File::create(output_path).unwrap();
    let writer = BufWriter::new(file);
    serde_json::to_writer(writer, &output).unwrap();
}

fn main() {
    let mut input_path = "".to_string();
    let mut output_path = "".to_string();
    parse_args(&mut input_path, &mut output_path);
    let (graph, coordinates) = read_graph(&input_path);
    let quality_metrics = compute_metrics(&graph, &coordinates);
    write_result(&quality_metrics, &output_path);
}
