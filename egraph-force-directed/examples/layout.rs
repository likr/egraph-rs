#[macro_use]
extern crate serde_derive;

extern crate getopts;
extern crate serde;
extern crate serde_json;
extern crate fd_layout;

use fd_layout::force::{Point, Link};
use fd_layout::many_body_force::ManyBodyForce;
use fd_layout::link_force::LinkForce;
use fd_layout::center_force::CenterForce;
use fd_layout::simulation::Simulation;
use fd_layout::edge_bundling::edge_bundling;

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
    let graph: GraphData = serde_json::from_reader(&file).unwrap();

    let mut points = graph
        .nodes
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let r = (i as usize as f32).sqrt();
            let theta = std::f32::consts::PI * (3. - (5. as f32).sqrt()) * (i as usize as f32);
            let x = r * theta.cos();
            let y = r * theta.sin();
            Point::new(x, y)
        })
        .collect::<Vec<_>>();
    let links = graph
        .links
        .iter()
        .map(|link| Link::new(link.source, link.target))
        .collect::<Vec<_>>();

    eprintln!("start");
    let forces = {
        let mut forces = Vec::new();
        forces.push(Box::new(ManyBodyForce::new()));
        forces.push(Box::new(LinkForce::new(&links)));
        forces.push(Box::new(CenterForce::new()));
        forces
    };
    start_simulation(&mut points, &forces);
    eprintln!("bundling edges");
    let lines = edge_bundling(&points, &links);

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
        let d = line.points
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
            point.x,
            point.y
        );
    }
    println!("</g>\n</svg>");
}
