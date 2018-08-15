extern crate rand;
extern crate petgraph;

use petgraph::{Graph, EdgeType};
use petgraph::graph::{IndexType, NodeIndex};

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
    pub group: usize,
    pub parent_index: usize,
    pub node_type: Option<NodeType>,
}

pub struct Edge {
    pub length: f32,
    pub count: usize,
}

pub enum NodeType {
    Unknown,
    SunNode,
    PlanetNode,
    MoonNode,
    PMNode,
}

fn solar_system_partition<N, E, Ty: EdgeType, Ix: IndexType>(graph: &Graph<N, E, Ty, Ix>) -> Vec<usize> {
    let nodes = shuffled_nodes(graph);
    let mut groups = graph.node_indices()
        .map(|_| 0)
        .collect::<Vec<_>>();
    let mut node_types = graph.node_indices()
        .map(|_| NodeType::Unknown)
        .collect::<Vec<_>>();
    let mut visited = graph.node_indices()
        .map(|_| false)
        .collect::<Vec<_>>();
    let mut i = 0;
    for s in nodes {
        if visited[s.index()] {
            continue;
        }
        groups[s.index()] = i;
        node_types[s.index()] = NodeType::SunNode;
        visited[s.index()] = true;
        for p in graph.neighbors_undirected(s) {
            if visited[p.index()] {
                continue;
            }
            groups[p.index()] = i;
            node_types[p.index()] = NodeType::PlanetNode;
            visited[p.index()] = true;
            let mut has_moon_node = false;
            for m in graph.neighbors_undirected(p) {
                if !visited[m.index()] {
                    groups[m.index()] = i;
                    node_types[m.index()] = NodeType::MoonNode;
                    visited[m.index()] = true;
                    has_moon_node = true;
                }
            }
            if has_moon_node {
                node_types[p.index()] = NodeType::PMNode;
            }
        }
        i += 1;
    }
    groups
}

fn collaps<N, E, Ty: EdgeType, Ix: IndexType>(graph: &Graph<N, E, Ty, Ix>, groups: &Vec<usize>) -> Graph<Node, Edge> {
    let mut shrinked_graph : Graph<Node, Edge> = Graph::new();
    let num_groups = groups.iter().max().unwrap() + 1;
    for _ in 0..num_groups {
        shrinked_graph.add_node(Node {group: 0, parent_index: 0, node_type: None});
    }
    for e in graph.edge_indices() {
        let (u0, v0) = graph.edge_endpoints(e).unwrap();
        let u1 = NodeIndex::new(groups[u0.index()]);
        let v1 = NodeIndex::new(groups[v0.index()]);
        match shrinked_graph.find_edge_undirected(u1, v1) {
            Some((e1, _)) => {
                let edge = shrinked_graph.edge_weight_mut(e1).unwrap();
                edge.count += 1;
            }
            None => {
                shrinked_graph.add_edge(u1, v1, Edge {length: 1.0, count: 1});
            }
        }
    }
    for e in shrinked_graph.edge_indices() {
        let edge = shrinked_graph.edge_weight_mut(e).unwrap();
        edge.length /= edge.count as f32;
    }
    shrinked_graph
}

pub fn fm3<N, E, Ty: EdgeType, Ix: IndexType>(graph: &Graph<N, E, Ty, Ix>) {
    let groups = solar_system_partition(graph);
    let collapsed_graph = collaps(graph, &groups);
    println!("{} {}", collapsed_graph.node_count(), collapsed_graph.edge_count());
}

#[test]
fn test_fm3() {
    let rows = 10;
    let cols = 10;
    let mut graph = Graph::new_undirected();
    let nodes = (0..rows * cols).map(|_| graph.add_node(Node {group: 0, parent_index: 0, node_type: None})).collect::<Vec<_>>();
    for i in 0..rows {
        for j in 0..cols {
            if i != rows - 1 {
                graph.add_edge(nodes[i * cols + j], nodes[(i + 1) * cols + j], Edge {length: 1.0, count: 1});
            }
            if j != cols - 1 {
                graph.add_edge(nodes[i * cols + j], nodes[i * cols + j + 1], Edge {length: 1.0, count: 1});
            }
        }
    }
    fm3(&graph);
}
