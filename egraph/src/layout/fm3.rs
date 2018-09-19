extern crate rand;
extern crate petgraph;

use std::f32::consts::PI;
use petgraph::{Graph, EdgeType};
use petgraph::graph::{IndexType, NodeIndex};
use ::layout::force_directed::{initial_placement, initial_links};
use ::layout::force_directed::force::{Force, Point, CenterForce, LinkForce, ManyBodyForce};
use ::layout::force_directed::simulation::start_simulation;

fn shuffled_nodes<N, E, Ty: EdgeType, Ix: IndexType>(graph: &Graph<N, E, Ty, Ix>) -> Vec<NodeIndex<Ix>> {
    let mut nodes = graph.node_indices().collect::<Vec<_>>();
    for _ in 0..nodes.len() {
        let i = rand::random::<usize>() % nodes.len();
        let j = rand::random::<usize>() % nodes.len();
        nodes.swap(i, j);
    }
    nodes
}

pub struct Node {
    pub group: usize,
    pub parent: usize,
    pub node_type: Option<NodeType>,
}

impl Node {
    fn new() -> Node {
        Node {
            group: 0,
            parent: 0,
            node_type: None,
        }
    }
}

pub struct Edge {
    pub length: f64,
    pub count: usize,
}

impl Edge {
    fn new() -> Edge {
        Edge {
            length: 30.,
            count: 0,
        }
    }
}

#[derive(Debug)]
pub enum NodeType {
    SunNode,
    PlanetNode,
    MoonNode,
}

fn solar_system_partition(graph: &mut Graph<Node, Edge>) {
    let nodes = shuffled_nodes(graph);
    let mut visited = graph.node_indices()
        .map(|_| false)
        .collect::<Vec<_>>();
    let mut i = 0;
    for s in nodes {
        if visited[s.index()] {
            continue;
        }
        {
            let s_node = graph.node_weight_mut(s).unwrap();
            s_node.group = i;
            s_node.parent = s.index();
            s_node.node_type = Some(NodeType::SunNode);
        }
        visited[s.index()] = true;
        let mut s_neighbors = graph.neighbors_undirected(s).detach();
        while let Some(p) = s_neighbors.next_node(graph) {
            if visited[p.index()] {
                continue;
            }
            {
                let p_node = graph.node_weight_mut(p).unwrap();
                p_node.group = i;
                p_node.parent = s.index();
                p_node.node_type = Some(NodeType::PlanetNode);
            }
            visited[p.index()] = true;
            let mut p_neighbors = graph.neighbors_undirected(p).detach();
            while let Some(m) = p_neighbors.next_node(graph) {
                if !visited[m.index()] {
                    {
                        let m_node = graph.node_weight_mut(m).unwrap();
                        m_node.group = i;
                        m_node.parent = p.index();
                        m_node.node_type = Some(NodeType::MoonNode);
                    }
                    visited[m.index()] = true;
                }
            }
        }
        i += 1;
    }
}

fn edge_length(graph: &Graph<Node, Edge>, u: NodeIndex, v: NodeIndex) -> f64 {
    let (e, _) = graph.find_edge_undirected(u, v).unwrap();
    graph.edge_weight(e).unwrap().length
}

fn path_length(graph: &Graph<Node, Edge>, u: NodeIndex) -> f64 {
    match &graph[u].node_type {
        Some(NodeType::PlanetNode) => {
            let s = NodeIndex::new(graph.node_weight(u).unwrap().parent);
            edge_length(graph, u, s)
        },
        Some(NodeType::MoonNode) => {
            let p = NodeIndex::new(graph.node_weight(u).unwrap().parent);
            let s = NodeIndex::new(graph.node_weight(p).unwrap().parent);
            edge_length(graph, u, p) + edge_length(graph, p, s)
        },
        _ => 0.,
    }
}

fn collapse(graph: &Graph<Node, Edge>) -> Graph<Node, Edge> {
    let mut shrinked_graph : Graph<Node, Edge> = Graph::new();
    let num_groups = graph.raw_nodes().iter().map(|node| node.weight.group).max().unwrap() + 1;
    for _ in 0..num_groups {
        shrinked_graph.add_node(Node::new());
    }
    for e in graph.edge_indices() {
        let (u0, v0) = graph.edge_endpoints(e).unwrap();
        if graph[u0].group == graph[v0].group {
            continue;
        }
        let e_u_length = path_length(graph, u0);
        let e_v_length = path_length(graph, v0);
        let e_length = edge_length(graph, u0, v0);
        let p_length = e_u_length + e_length + e_v_length;
        let u1 = NodeIndex::new(graph.node_weight(u0).unwrap().group);
        let v1 = NodeIndex::new(graph.node_weight(v0).unwrap().group);
        match shrinked_graph.find_edge_undirected(u1, v1) {
            Some((e1, _)) => {
                let edge = shrinked_graph.edge_weight_mut(e1).unwrap();
                edge.length += p_length;
                edge.count += 1;
            }
            None => {
                let mut edge = Edge::new();
                edge.length = p_length;
                edge.count = 1;
                shrinked_graph.add_edge(u1, v1, edge);
            }
        }
    }
    for e in shrinked_graph.edge_indices() {
        let edge = shrinked_graph.edge_weight_mut(e).unwrap();
        if edge.count > 0 {
            edge.length /= edge.count as f64;
        }
    }
    shrinked_graph
}

fn expand(graph0: &Graph<Node, Edge>, graph1: &Graph<Node, Edge>, graph1_points: &Vec<Point>) -> Vec<Point> {
    let mut points = Vec::new();
    for u in graph0.node_indices() {
        let mut x = 0.;
        let mut y = 0.;
        let mut count = 0;
        let s1 = NodeIndex::new(graph0[u].group);
        let s1_x = graph1_points[s1.index()].x as f64;
        let s1_y = graph1_points[s1.index()].y as f64;
        for v in graph0.neighbors_undirected(u) {
            if graph0[u].group == graph0[v].group {
                continue;
            }
            let t1 = NodeIndex::new(graph0[v].group);
            let t1_x = graph1_points[t1.index()].x as f64;
            let t1_y = graph1_points[t1.index()].y as f64;
            let scale = path_length(graph0, u) / edge_length(graph1, s1, t1);
            x += (t1_x - s1_x) * scale + s1_x;
            y += (t1_y - s1_y) * scale + s1_y;
            count += 1;
        }
        if count > 0 {
            points.push(Point::new(x as f32 / count as f32, y as f32 / count as f32));
        } else {
            let theta = rand::random::<f32>() * 2. * PI;
            let r = path_length(graph0, u) as f32;
            let x = r * theta.cos() + s1_x as f32;
            let y = r * theta.sin() + s1_y as f32;
            points.push(Point::new(x, y));
        }
    }
    points
}

fn layout(graph: &Graph<Node, Edge>) -> Vec<Point> {
    let links = initial_links(graph);
    let mut forces : Vec<Box<Force>> = Vec::new();
    forces.push(Box::new(ManyBodyForce::new()));
    forces.push(Box::new(LinkForce::new_with_links(links)));
    forces.push(Box::new(CenterForce::new()));
    let mut points = initial_placement(graph.node_count());
    start_simulation(&mut points, &forces);
    points
}

fn layout_with_initial_placement(graph: &Graph<Node, Edge>, points: &mut Vec<Point>) {
    let links = initial_links(graph);
    let mut forces : Vec<Box<Force>> = Vec::new();
    forces.push(Box::new(ManyBodyForce::new()));
    forces.push(Box::new(LinkForce::new_with_links(links)));
    forces.push(Box::new(CenterForce::new()));
    start_simulation(points, &forces);
}

pub struct FM3 {
    pub min_size: usize,
}

impl FM3 {
    pub fn new() -> FM3 {
        FM3 {
            min_size: 100,
        }
    }

    pub fn call<N, E, Ty: EdgeType, Ix: IndexType>(&self, graph: &Graph<N, E, Ty, Ix>) -> Vec<Point> {
        let mut shrinked_graphs = Vec::new();
        let mut g0 = Graph::new();
        for _node in graph.node_indices() {
            g0.add_node(Node::new());
        }
        for edge in graph.raw_edges() {
            g0.add_edge(NodeIndex::new(edge.source().index()), NodeIndex::new(edge.target().index()), Edge::new());
        }
        while g0.node_count() > self.min_size {
            solar_system_partition(&mut g0);
            let g1 = collapse(&mut g0);
            shrinked_graphs.push(g0);
            g0 = g1;
        }
        let mut gk = g0;
        let mut g1_points = layout(&mut gk);
        while !shrinked_graphs.is_empty() {
            let g0 = shrinked_graphs.pop().unwrap();
            let mut g0_points = expand(&g0, &gk, &g1_points);
            layout_with_initial_placement(&g0, &mut g0_points);
            g1_points = g0_points;
            gk = g0;
        }
        g1_points
    }
}

#[test]
fn test_fm3() {
    let rows = 10;
    let cols = 10;
    let mut graph = Graph::new();
    let nodes = (0..rows * cols).map(|_| graph.add_node(())).collect::<Vec<_>>();
    for i in 0..rows {
        for j in 0..cols {
            if i != rows - 1 {
                graph.add_edge(nodes[i * cols + j], nodes[(i + 1) * cols + j], ());
            }
            if j != cols - 1 {
                graph.add_edge(nodes[i * cols + j], nodes[i * cols + j + 1], ());
            }
        }
    }
    let fm3 = FM3::new();
    let points = fm3.call(&graph);
    for point in points {
        println!("{:?}", point);
    }
}
