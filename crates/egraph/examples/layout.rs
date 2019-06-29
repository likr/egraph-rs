#[macro_use]
extern crate serde_derive;

use egraph::edge_bundling::force_directed::ForceDirectedEdgeBundling;
use egraph::layout::force_directed::force::{CenterForce, LinkForce, ManyBodyForce};
use egraph::layout::force_directed::{initial_placement, Simulation};
use egraph::Graph as EGraph;
use egraph_petgraph_adapter::PetgraphWrapper;
use petgraph::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

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

    let mut points = initial_placement(graph.node_count());

    eprintln!("start");
    let many_body_force = Rc::new(RefCell::new(ManyBodyForce::new()));
    let link_force = Rc::new(RefCell::new(LinkForce::new()));
    let center_force = Rc::new(RefCell::new(CenterForce::new()));
    let mut simulation = Simulation::new();
    simulation.add(many_body_force);
    simulation.add(link_force);
    simulation.add(center_force);
    let mut context = simulation.build(&graph);
    context.start(&mut points);

    eprintln!("bundling edges");
    let edge_bundling = ForceDirectedEdgeBundling::new();
    let lines = edge_bundling.call(&graph, &points);

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
    for point in points.iter() {
        println!(
            "<circle cx=\"{}\" cy=\"{}\" r=\"5\" fill=\"green\" />",
            point.x, point.y
        );
    }
    println!("</g>\n</svg>");
}
