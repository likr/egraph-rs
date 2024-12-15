use argparse::{ArgumentParser, Store};
use egraph_cli::read_graph;
use petgraph::prelude::*;
use petgraph_algorithm_shortest_path::warshall_floyd;
use petgraph_drawing::DrawingEuclidean2d;
use petgraph_quality_metrics::{quality_metrics, QualityMetric};
use std::{collections::HashMap, fs::File, io::BufWriter};

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
    drawing: &DrawingEuclidean2d<NodeIndex, f32>,
) -> Vec<(QualityMetric, f32)> {
    let distance = warshall_floyd(graph, &mut |_| 1.);
    quality_metrics(graph, drawing, &distance)
}

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

fn main() {
    let mut input_path = "".to_string();
    let mut output_path = "".to_string();
    parse_args(&mut input_path, &mut output_path);
    let (graph, coordinates) = read_graph(&input_path);
    let quality_metrics = compute_metrics(&graph, &coordinates);
    write_result(&quality_metrics, &output_path);
}
