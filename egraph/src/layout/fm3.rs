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
    for _ in 0..1000 {
        let i = rand::random::<usize>() % nodes.len();
        let j = rand::random::<usize>() % nodes.len();
        nodes.swap(i, j);
    }
    nodes
}

pub struct Node {
    pub radius: f64,
    pub group: usize,
    pub parent: usize,
    pub node_type: Option<NodeType>,
}

impl Node {
    fn new() -> Node {
        Node {
            radius: 0.,
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
            length: 0.,
            count: 0,
        }
    }
}

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

fn collapse(graph: &mut Graph<Node, Edge>) -> Graph<Node, Edge> {
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
        let e_u_length = match &graph[u0].node_type {
            Some(NodeType::PlanetNode) => {
                let s0 = NodeIndex::new(graph.node_weight(u0).unwrap().parent);
                edge_length(graph, u0, s0)
            },
            Some(NodeType::MoonNode) => {
                let p0 = NodeIndex::new(graph.node_weight(u0).unwrap().parent);
                let s0 = NodeIndex::new(graph.node_weight(p0).unwrap().parent);
                edge_length(graph, u0, p0) + edge_length(graph, p0, s0)
            },
            _ => 0.,
        };
        let e_v_length = match &graph[v0].node_type {
            Some(NodeType::PlanetNode) => {
                let t0 = NodeIndex::new(graph.node_weight(v0).unwrap().parent);
                edge_length(graph, v0, t0)
            },
            Some(NodeType::MoonNode) => {
                let q0 = NodeIndex::new(graph.node_weight(v0).unwrap().parent);
                let t0 = NodeIndex::new(graph.node_weight(q0).unwrap().parent);
                edge_length(graph, v0, q0) + edge_length(graph, q0, t0)
            },
            _ => 0.,
        };
        graph[u0].radius = e_u_length;
        graph[v0].radius = e_v_length;
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
                shrinked_graph.add_edge(u1, v1, edge);
            }
        }
    }
    for e in shrinked_graph.edge_indices() {
        let edge = shrinked_graph.edge_weight_mut(e).unwrap();
        edge.length /= edge.count as f64;
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
            let scale = graph0[u].radius / edge_length(graph1, s1, t1);
            x += (t1_x - s1_x) * scale + s1_x;
            y += (t1_y - s1_y) * scale + s1_y;
            count += 1;
        }
        if count > 0 {
            points.push(Point::new(x as f32 / count as f32, y as f32 / count as f32));
        } else {
            let theta = rand::random::<f32>() * 2. * PI;
            let r = graph0[u].radius as f32;
            let x = r * theta.cos() + s1_x as f32;
            let y = r * theta.sin() + s1_y as f32;
            points.push(Point::new(x, y));
        }
    }
    points
}

fn layout(graph: &mut Graph<Node, Edge>) -> Vec<Point> {
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

pub fn fm3(graph: Graph<Node, Edge>) {
    let mut shrinked_graphs = Vec::new();
    let mut g0 = graph;
    while g0.node_count() > 100 {
        println!("{} {}", g0.node_count(), g0.edge_count());
        solar_system_partition(&mut g0);
        let g1 = collapse(&mut g0);
        shrinked_graphs.push(g0);
        g0 = g1;
    }
    let mut gk = g0;
    println!("{} {}", gk.node_count(), gk.edge_count());
    let mut g1_points = layout(&mut gk);
    for i in 1..shrinked_graphs.len() {
        let g0 = &shrinked_graphs[i - 1];
        let g1 = &shrinked_graphs[i];
        let mut g0_points = expand(&g0, &g1, &g1_points);
        layout_with_initial_placement(&g0, &mut g0_points);
        g1_points = g0_points;
    }

}

#[test]
fn test_fm3() {
    let rows = 100;
    let cols = 100;
    let mut graph = Graph::new();
    let nodes = (0..rows * cols).map(|_| graph.add_node(Node::new())).collect::<Vec<_>>();
    for i in 0..rows {
        for j in 0..cols {
            if i != rows - 1 {
                graph.add_edge(nodes[i * cols + j], nodes[(i + 1) * cols + j], Edge::new());
            }
            if j != cols - 1 {
                graph.add_edge(nodes[i * cols + j], nodes[i * cols + j + 1], Edge::new());
            }
        }
    }
    fm3(graph);
}
