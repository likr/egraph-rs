#[macro_use]
extern crate serde_derive;

extern crate getopts;
extern crate serde;
extern crate serde_json;
extern crate fd_layout;

use fd_layout::force::{Point, Link, Force};
use fd_layout::link_force::LinkForce;
use fd_layout::group_force::{Group, GroupForce};
use fd_layout::simulation::start_simulation;
use fd_layout::edge_bundling::edge_bundling;

#[derive(Serialize, Deserialize)]
struct NodeData {
    group: usize,
}

#[derive(Serialize, Deserialize)]
struct LinkData {
    source: usize,
    target: usize,
    value: f32,
}

#[derive(Serialize, Deserialize)]
struct GroupData {
    x: f32,
    y: f32,
    r: f32,
}

#[derive(Serialize, Deserialize)]
struct GraphData {
    nodes: Vec<NodeData>,
    links: Vec<LinkData>,
    groups: Vec<GroupData>,
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
    let mut graph: GraphData = serde_json::from_reader(&file).unwrap();

    let scale = 30.;
    for group in graph.groups.iter_mut() {
        group.x *= scale;
        group.y *= scale;
        group.r *= scale;
    }

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
        .map(|link| {
            let source_group = graph.nodes[link.source].group;
            let target_group = graph.nodes[link.target].group;
            let (length, strength) = if source_group == target_group {
                (graph.groups[source_group].r, 0.1)
            } else {
                (1.0, 0.)
            };
            Link {
                source: link.source,
                target: link.target,
                length: length,
                strength: strength,
            }
        })
        .collect::<Vec<_>>();
    let groups = graph
        .groups
        .iter()
        .map(|group| Group::new(group.x, group.y))
        .collect::<Vec<_>>();
    let node_groups = graph
        .nodes
        .iter()
        .map(|node| node.group)
        .collect::<Vec<_>>();
    let group_colors = {
        let n = graph.groups.len();
        (0..n)
            .map(|i| format!("hsl({}, 100%, 50%)", 360 / n * i))
            .collect::<Vec<_>>()
    };

    eprintln!("start");
    let forces = {
        let mut forces: Vec<Box<Force>> = Vec::new();
        forces.push(Box::new(LinkForce::new(&links)));
        let mut group_force = GroupForce::new(groups, node_groups);
        group_force.strength = 0.08;
        forces.push(Box::new(group_force));
        forces
    };
    start_simulation(&mut points, &forces);
    eprintln!("bundling edges");
    let lines = edge_bundling(&points, &links);

    eprintln!("writing result");
    {
        let width = 800.;
        let height = 800.;
        let margin = 10.;
        println!(
            "<svg version=\"1.1\" width=\"{}\" height=\"{}\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">",
            width + margin * 2., height + margin * 2.,
            );
        let circle = &graph.groups[0];
        println!(
            "<g transform=\"translate({},{})scale({})translate({},{})\">",
            width / 2. + margin,
            height / 2. + margin,
            (width as f32).min(height as f32) / circle.r / 2.,
            -circle.x,
            -circle.y,
            );
    }
    for group in graph.groups {
        println!(
            "<circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"none\" stroke=\"black\" />",
            group.x,
            group.y,
            group.r
        );
    }
    for line in lines.iter() {
        let d = line.points
            .iter()
            .map(|p| format!("{} {}", p.x, p.y))
            .collect::<Vec<_>>()
            .join(" L ");
        println!(
            "<path d=\"M {}\" fill=\"none\" stroke=\"#888\" stroke-width=\"5\" opacity=\"0.2\" />",
            d
        );
    }
    for (point, node) in points.iter().zip(graph.nodes.iter()) {
        println!(
            "<circle cx=\"{}\" cy=\"{}\" r=\"10\" fill=\"{}\" />",
            point.x,
            point.y,
            group_colors[node.group],
        );
    }
    println!("</g>\n</svg>");
}
