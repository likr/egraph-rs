use argparse::{ArgumentParser, Store};
use egraph_cli::{read_graph, write_graph};
use petgraph::prelude::*;
use petgraph_drawing::Drawing;
use petgraph_layout_sgd::{Sgd, SparseSgd};
use rand::thread_rng;

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

fn layout(
    graph: &Graph<Option<()>, Option<()>, Undirected>,
    coordinates: &mut Drawing<NodeIndex, f32>,
) {
    let mut rng = thread_rng();
    let mut sgd = SparseSgd::new_with_rng(graph, |_| 30., 281, &mut rng);
    let mut scheduler = sgd.scheduler(867, 0.1);
    scheduler.run(&mut |eta| {
        sgd.shuffle(&mut rng);
        sgd.apply(coordinates, eta);
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
