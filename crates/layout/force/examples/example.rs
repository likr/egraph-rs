#[macro_use]
extern crate serde_derive;

use petgraph::prelude::*;
use petgraph_edge_bundling_fdeb::{fdeb, EdgeBundlingOptions};
use petgraph_layout_force::{force_connected, force_nonconnected};
use petgraph_layout_force_simulation::{apply_forces, initial_placement, Simulation};
use petgraph_layout_grouped_force::force_grouped;
use std::collections::HashMap;

#[derive(Serialize, Deserialize)]
struct NodeData {
    id: usize,
    group: Option<usize>,
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

enum LayoutMethod {
    Connected,
    Nonconnected,
    Grouped,
}

fn load_graph(filename: &String) -> UnGraph<NodeData, LinkData> {
    let path = std::path::Path::new(filename);
    let file = std::fs::File::open(&path).unwrap();
    let data: GraphData = serde_json::from_reader(&file).unwrap();

    let mut graph = Graph::new_undirected();
    let mut indices = HashMap::new();
    for node in data.nodes {
        indices.insert(node.id, graph.add_node(node));
    }
    for link in data.links {
        graph.add_edge(indices[&link.source], indices[&link.target], link);
    }
    graph
}

fn layout(
    graph: &UnGraph<NodeData, LinkData>,
    layout_method: &LayoutMethod,
) -> HashMap<NodeIndex, (f32, f32)> {
    let forces = match layout_method {
        LayoutMethod::Connected => force_connected(&graph),
        LayoutMethod::Nonconnected => force_nonconnected(&graph),
        LayoutMethod::Grouped => force_grouped(&graph, |graph, u| graph[u].group.unwrap()),
    };
    let mut coordinates = initial_placement(&graph);
    let mut simulation = Simulation::new();
    simulation.run(&mut |alpha| {
        apply_forces(&mut coordinates.points, &forces, alpha, 0.6);
    });
    coordinates
        .iter()
        .map(|(u, p)| (u, (p.x, p.y)))
        .collect::<HashMap<_, _>>()
}

fn edge_bundling(
    graph: &UnGraph<NodeData, LinkData>,
    coordinates: &HashMap<NodeIndex, (f32, f32)>,
) -> HashMap<EdgeIndex, Vec<(f32, f32)>> {
    let options = EdgeBundlingOptions::new();
    fdeb(graph, coordinates, &options)
}

fn print_svg(
    graph: &UnGraph<NodeData, LinkData>,
    coordinates: &HashMap<NodeIndex, (f32, f32)>,
    lines: &HashMap<EdgeIndex, Vec<(f32, f32)>>,
) {
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
    println!("<g>");
    for e in graph.edge_indices() {
        let d = lines[&e]
            .iter()
            .map(|(x, y)| format!("{} {}", x, y))
            .collect::<Vec<_>>()
            .join(" L ");
        println!(
            "<path d=\"M {}\" fill=\"none\" stroke=\"#999\" opacity=\"0.3\" />",
            d
        );
    }
    println!("</g>");
    println!("<g>");
    for u in graph.node_indices() {
        let (x, y) = coordinates[&u];
        println!(
            "<circle cx=\"{}\" cy=\"{}\" r=\"5\" fill=\"green\" />",
            x, y,
        );
    }
    println!("</g>");
    println!("</g>");
    println!("</svg>");
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let mut opts = getopts::Options::new();
    opts.reqopt("f", "file", "input filename", "FILENAME");
    opts.optopt(
        "l",
        "layout",
        "layout method",
        "connected, nonconnected, or grouped",
    );
    let matches = match opts.parse(&args[1..]) {
        Ok(m) => m,
        Err(f) => panic!("{}", f.to_string()),
    };
    let filename = matches.opt_str("f").unwrap();
    let layout_method = if let Some(layout) = matches.opt_str("l") {
        match &*layout {
            "connected" => LayoutMethod::Connected,
            "nonconnected" => LayoutMethod::Nonconnected,
            "grouped" => LayoutMethod::Grouped,
            _ => panic!("invalid layout method"),
        }
    } else {
        LayoutMethod::Connected
    };
    let graph = load_graph(&filename);
    let coordinates = layout(&graph, &layout_method);
    let lines = edge_bundling(&graph, &coordinates);
    print_svg(&graph, &coordinates, &lines);
}
