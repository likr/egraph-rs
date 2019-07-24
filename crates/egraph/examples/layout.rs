#[macro_use]
extern crate serde_derive;

use egraph::edge_bundling::force_directed::ForceDirectedEdgeBundling;
use egraph::layout::force_directed::SimulationBuilder;
use egraph_petgraph_adapter::PetgraphWrapper;
use petgraph::prelude::*;
use std::collections::HashMap;

#[derive(Serialize, Deserialize)]
struct NodeData {
    id: usize,
}

#[derive(Serialize, Deserialize)]
struct LinkData {
    source: usize,
    target: usize,
}

#[derive(Serialize, Deserialize)]
struct GraphData {
    nodes: Vec<NodeData>,
    links: Vec<LinkData>,
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let mut opts = getopts::Options::new();
    opts.reqopt("f", "file", "input filename", "FILENAME");
    let matches = match opts.parse(&args[1..]) {
        Ok(m) => m,
        Err(f) => panic!(f.to_string()),
    };
    let filename = matches.opt_str("f").unwrap();

    let path = std::path::Path::new(&filename);
    let file = std::fs::File::open(&path).unwrap();
    let data: GraphData = serde_json::from_reader(&file).unwrap();

    let mut graph = Graph::new();
    let mut indices = HashMap::new();
    for node in data.nodes {
        indices.insert(node.id, graph.add_node(node));
    }
    for link in data.links {
        graph.add_edge(indices[&link.source], indices[&link.target], link);
    }
    let graph = PetgraphWrapper::new(graph);

    eprintln!("start");
    let builder = SimulationBuilder::default();
    let mut simulation = builder.build(&graph);
    simulation.run();

    eprintln!("bundling edges");
    let edge_bundling = ForceDirectedEdgeBundling::new();
    let lines = edge_bundling.call(&graph, &simulation.points);

    eprintln!("writing result");
    let width = 800.;
    let height = 800.;
    let margin = 10.;
    println!(
        "<svg version=\"1.1\" width=\"{}\" height=\"{}\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">",
        width + margin * 2., height + margin * 2.,
    );
    println!(
        "<g transform=\"translate({},{})\">",
        width / 2. + margin,
        height / 2. + margin,
    );
    for line in lines.iter() {
        let d = line
            .points
            .iter()
            .map(|p| format!("{} {}", p.x, p.y))
            .collect::<Vec<_>>()
            .join(" L ");
        println!(
            "<path d=\"M {}\" fill=\"none\" stroke=\"#999\" opacity=\"0.3\" />",
            d
        );
    }
    for point in simulation.points.iter() {
        println!(
            "<circle cx=\"{}\" cy=\"{}\" r=\"5\" fill=\"green\" />",
            point.x, point.y
        );
    }
    println!("</g>\n</svg>");
}
