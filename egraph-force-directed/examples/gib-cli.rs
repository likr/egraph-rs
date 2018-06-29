#[macro_use]
extern crate serde_derive;

extern crate fd_layout;
extern crate getopts;
extern crate serde;
extern crate serde_json;

use std::str::FromStr;
use fd_layout::force::{Force, Link, Point};
use fd_layout::link_force::LinkForce;
use fd_layout::group_force::{Group, GroupForce};
use fd_layout::simulation::start_simulation;

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
    dx: f32,
    dy: f32,
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
    opts.optopt("s", "scale", "scale", "SCALE");
    let matches = match opts.parse(&args[1..]) {
        Ok(m) => m,
        Err(f) => panic!(f.to_string()),
    };
    let filename = matches.opt_str("f").unwrap();
    let scale = match matches.opt_str("s") {
        Some(s) => f32::from_str(&s).ok().unwrap(),
        None => 5.,
    };

    let path = std::path::Path::new(&filename);
    let file = std::fs::File::open(&path).unwrap();
    let mut graph: GraphData = serde_json::from_reader(&file).unwrap();

    for group in graph.groups.iter_mut() {
        group.x += group.dx / 2.;
        group.y += group.dy / 2.;
        group.x *= scale;
        group.y *= scale;
        group.dx *= scale;
        group.dy *= scale;
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
                let dx = graph.groups[source_group].dx;
                let dy = graph.groups[source_group].dy;
                (dx.min(dy) * 0.8, 0.1)
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

    eprintln!("writing result");
    for point in points.iter() {
        println!("{}\t{}", point.x / scale, point.y / scale,);
    }
}
