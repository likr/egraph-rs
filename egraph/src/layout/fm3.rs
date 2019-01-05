use algorithms::connected_components;
use layout::force_directed::force::Point;
use layout::force_directed::initial_placement;
use layout::force_directed::simulation::Simulation;
use petgraph::graph::IndexType;
use petgraph::prelude::*;
use petgraph::EdgeType;
use rand::prelude::*;
use std::collections::{HashMap, HashSet};
use std::f32::consts::PI;

#[derive(Debug)]
pub enum NodeType {
    Unknown,
    SunNode,
    PlanetNode,
    MoonNode,
}

fn solar_system_partition<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
    rng: &mut StdRng,
) -> (Vec<usize>, Vec<usize>, Vec<NodeType>) {
    let nodes = {
        let mut nodes = graph.node_indices().collect::<Vec<_>>();
        rng.shuffle(&mut nodes);
        nodes
    };
    let mut groups = graph.node_indices().map(|_| 0).collect::<Vec<_>>();
    let mut parents = graph.node_indices().map(|_| 0).collect::<Vec<_>>();
    let mut node_types = graph
        .node_indices()
        .map(|_| NodeType::Unknown)
        .collect::<Vec<_>>();
    let mut visited = graph.node_indices().map(|_| false).collect::<Vec<_>>();
    let mut i = 0;
    for s in nodes {
        if visited[s.index()] {
            continue;
        }
        groups[s.index()] = i;
        parents[s.index()] = s.index();
        node_types[s.index()] = NodeType::SunNode;
        visited[s.index()] = true;
        let mut s_neighbors = graph.neighbors_undirected(s).detach();
        while let Some(p) = s_neighbors.next_node(graph) {
            if visited[p.index()] {
                continue;
            }
            groups[p.index()] = i;
            parents[p.index()] = s.index();
            node_types[p.index()] = NodeType::PlanetNode;
            visited[p.index()] = true;
            let mut p_neighbors = graph.neighbors_undirected(p).detach();
            while let Some(m) = p_neighbors.next_node(graph) {
                if !visited[m.index()] {
                    groups[m.index()] = i;
                    parents[m.index()] = p.index();
                    node_types[m.index()] = NodeType::MoonNode;
                    visited[m.index()] = true;
                }
            }
        }
        i += 1;
    }
    (groups, parents, node_types)
}

fn edge_length<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
    link_distance_accessor: &Box<Fn(&Graph<N, E, Ty, Ix>, EdgeIndex<Ix>) -> f32>,
    a: NodeIndex<Ix>,
    b: NodeIndex<Ix>,
) -> f32 {
    let (e, _) = graph.find_edge_undirected(a, b).unwrap();
    link_distance_accessor(graph, e)
}

fn path_length<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
    node_parents: &Vec<usize>,
    node_types: &Vec<NodeType>,
    link_distance_accessor: &Box<Fn(&Graph<N, E, Ty, Ix>, EdgeIndex<Ix>) -> f32>,
    u: NodeIndex<Ix>,
) -> f32 {
    match node_types[u.index()] {
        NodeType::PlanetNode => {
            let s = NodeIndex::new(node_parents[u.index()]);
            edge_length(graph, link_distance_accessor, u, s)
        }
        NodeType::MoonNode => {
            let p = NodeIndex::new(node_parents[u.index()]);
            let s = NodeIndex::new(node_parents[p.index()]);
            edge_length(graph, link_distance_accessor, u, p)
                + edge_length(graph, link_distance_accessor, p, s)
        }
        _ => 0.,
    }
}

fn collapse<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
    node_groups: &Vec<usize>,
    node_parents: &Vec<usize>,
    node_types: &Vec<NodeType>,
    shrink_node: &Box<Fn(&Graph<N, E, Ty, Ix>, &Vec<NodeIndex<Ix>>) -> N>,
    shrink_edge: &Box<Fn(&Graph<N, E, Ty, Ix>, &Vec<EdgeIndex<Ix>>, f32) -> E>,
    link_distance_accessor: &Box<Fn(&Graph<N, E, Ty, Ix>, EdgeIndex<Ix>) -> f32>,
) -> Graph<N, E, Ty, Ix> {
    let mut shrinked_graph = Graph::default();
    let num_groups = graph
        .node_indices()
        .map(|a| node_groups[a.index()])
        .max()
        .unwrap()
        + 1;
    for g in 0..num_groups {
        let node_indices = graph
            .node_indices()
            .filter(|&a| node_groups[a.index()] == g)
            .collect();
        shrinked_graph.add_node(shrink_node(graph, &node_indices));
    }

    let mut group_edge_indices = HashMap::new();
    for e in graph.edge_indices() {
        let (u0, v0) = graph.edge_endpoints(e).unwrap();
        let key = {
            let gu = node_groups[u0.index()];
            let gv = node_groups[v0.index()];
            if gu == gv {
                continue;
            }
            if gu > gv {
                (gv, gu)
            } else {
                (gu, gv)
            }
        };
        if !group_edge_indices.contains_key(&key) {
            group_edge_indices.insert(key, Vec::new());
        }
        group_edge_indices.get_mut(&key).unwrap().push(e);
    }

    for ((gu, gv), edge_indices) in group_edge_indices.iter() {
        let mut total_edge_length = 0.;
        for &e in edge_indices.iter() {
            let (u0, v0) = graph.edge_endpoints(e).unwrap();
            let e_u_length =
                path_length(graph, node_parents, node_types, link_distance_accessor, u0);
            let e_v_length =
                path_length(graph, node_parents, node_types, link_distance_accessor, v0);

            let e_length = edge_length(graph, link_distance_accessor, u0, v0);
            total_edge_length += e_u_length + e_length + e_v_length;
        }
        total_edge_length /= edge_indices.len() as f32;
        shrinked_graph.add_edge(
            NodeIndex::new(gu.index()),
            NodeIndex::new(gv.index()),
            shrink_edge(graph, edge_indices, total_edge_length),
        );
    }
    shrinked_graph
}

fn expand<N, E, Ty: EdgeType, Ix: IndexType>(
    graph0: &Graph<N, E, Ty, Ix>,
    graph1: &Graph<N, E, Ty, Ix>,
    graph1_points: &Vec<Point>,
    node_groups: &Vec<usize>,
    node_parents: &Vec<usize>,
    node_types: &Vec<NodeType>,
    link_distance_accessor: &Box<Fn(&Graph<N, E, Ty, Ix>, EdgeIndex<Ix>) -> f32>,
    rng: &mut StdRng,
) -> Vec<Point> {
    let mut points = Vec::new();
    for u in graph0.node_indices() {
        let mut x = 0.;
        let mut y = 0.;
        let mut count = 0;
        let s1 = NodeIndex::new(node_groups[u.index()]);
        let s1_x = graph1_points[s1.index()].x as f32;
        let s1_y = graph1_points[s1.index()].y as f32;
        for v in graph0.neighbors_undirected(u) {
            if node_groups[u.index()] == node_groups[v.index()] {
                continue;
            }
            let t1 = NodeIndex::new(node_groups[v.index()]);
            let t1_x = graph1_points[t1.index()].x as f32;
            let t1_y = graph1_points[t1.index()].y as f32;
            let scale = path_length(graph0, node_parents, node_types, link_distance_accessor, u)
                / edge_length(graph1, link_distance_accessor, s1, t1);
            x += (t1_x - s1_x) * scale + s1_x;
            y += (t1_y - s1_y) * scale + s1_y;
            count += 1;
        }
        if count > 0 {
            points.push(Point::new(x as f32 / count as f32, y as f32 / count as f32));
        } else {
            let theta = rng.gen::<f32>() * 2. * PI;
            let r = path_length(graph0, node_parents, node_types, link_distance_accessor, u) as f32;
            let x = r * theta.cos() + s1_x as f32;
            let y = r * theta.sin() + s1_y as f32;
            points.push(Point::new(x, y));
        }
    }
    points
}

fn layout<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
    simulation: &Simulation<N, E, Ty, Ix>,
    iteration: usize,
    alpha: &mut f32,
    decay: f32,
) -> Vec<Point> {
    let mut points = initial_placement(graph.node_count());
    layout_with_initial_placement(graph, &mut points, simulation, iteration, alpha, decay);
    points
}

fn layout_with_initial_placement<N, E, Ty: EdgeType, Ix: IndexType>(
    graph: &Graph<N, E, Ty, Ix>,
    points: &mut Vec<Point>,
    simulation: &Simulation<N, E, Ty, Ix>,
    iteration: usize,
    alpha: &mut f32,
    decay: f32,
) {
    let mut context = simulation.build(&graph);
    for _i in 0..iteration {
        context.step(points);
        *alpha += -(*alpha) * decay;
    }
}

pub struct FM3<N, E, Ty: EdgeType, Ix: IndexType> {
    pub simulation: Simulation<N, E, Ty, Ix>,
    pub min_size: usize,
    pub step_iteration: usize,
    pub shrink_node: Box<Fn(&Graph<N, E, Ty, Ix>, &Vec<NodeIndex<Ix>>) -> N>,
    pub shrink_edge: Box<Fn(&Graph<N, E, Ty, Ix>, &Vec<EdgeIndex<Ix>>, f32) -> E>,
    pub link_distance_accessor: Box<Fn(&Graph<N, E, Ty, Ix>, EdgeIndex<Ix>) -> f32>,
}

impl<N, E, Ty: EdgeType, Ix: IndexType> FM3<N, E, Ty, Ix> {
    pub fn new(
        simulation: Simulation<N, E, Ty, Ix>,
        shrink_node: Box<Fn(&Graph<N, E, Ty, Ix>, &Vec<NodeIndex<Ix>>) -> N>,
        shrink_edge: Box<Fn(&Graph<N, E, Ty, Ix>, &Vec<EdgeIndex<Ix>>, f32) -> E>,
        link_distance_accessor: Box<Fn(&Graph<N, E, Ty, Ix>, EdgeIndex<Ix>) -> f32>,
    ) -> FM3<N, E, Ty, Ix> {
        FM3 {
            simulation,
            min_size: 100,
            step_iteration: 100,
            shrink_node,
            shrink_edge,
            link_distance_accessor,
        }
    }

    pub fn call(&self, graph: &Graph<N, E, Ty, Ix>) -> Vec<Point> {
        let seed = [0; 32];
        let mut rng: StdRng = SeedableRng::from_seed(seed);

        let num_components = connected_components(graph)
            .iter()
            .collect::<HashSet<_>>()
            .len();
        let mut shrinked_graphs = Vec::new();
        let mut g0 = graph.map(
            |a, _| (self.shrink_node)(graph, &vec![a]),
            |e, _| (self.shrink_edge)(graph, &vec![e], (self.link_distance_accessor)(graph, e)),
        );

        while g0.node_count() > self.min_size + num_components - 1 {
            let (groups, parents, types) = solar_system_partition(&g0, &mut rng);
            let g1 = collapse(
                &g0,
                &groups,
                &parents,
                &types,
                &self.shrink_node,
                &self.shrink_edge,
                &self.link_distance_accessor,
            );
            shrinked_graphs.push((g0, groups, parents, types));
            g0 = g1;
        }

        let total_iteration = self.step_iteration * (shrinked_graphs.len() + 1);
        let alpha_min = 0.001;
        let mut alpha = 1.;
        let decay = 1. - (alpha_min as f32).powf(1. / total_iteration as f32);

        let mut gk = g0;
        let mut g1_points = layout(
            &mut gk,
            &self.simulation,
            self.step_iteration,
            &mut alpha,
            decay,
        );

        while !shrinked_graphs.is_empty() {
            let (g0, groups, parents, types) = shrinked_graphs.pop().unwrap();
            let mut g0_points = expand(
                &g0,
                &gk,
                &g1_points,
                &groups,
                &parents,
                &types,
                &self.link_distance_accessor,
                &mut rng,
            );
            layout_with_initial_placement(
                &g0,
                &mut g0_points,
                &self.simulation,
                self.step_iteration,
                &mut alpha,
                decay,
            );
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
    let nodes = (0..rows * cols)
        .map(|_| graph.add_node(()))
        .collect::<Vec<_>>();
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
