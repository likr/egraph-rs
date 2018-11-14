use algorithms::biclustering::{Bicluster, Biclustering};
use petgraph::graph::{IndexType, NodeIndex};
use petgraph::{EdgeType, Graph};
use std::collections::{HashMap, HashSet};

pub fn edge_concentration<
    N,
    E,
    Ty: EdgeType,
    Ix: IndexType,
    NF: Fn(usize) -> N,
    EF: Fn(usize) -> E,
    DF: Fn(&Bicluster) -> N,
>(
    graph: &Graph<N, E, Ty, Ix>,
    biclusters: &Vec<Bicluster>,
    new_node: NF,
    new_edge: EF,
    dummy_node: DF,
) -> Graph<N, E, Ty, Ix> {
    let mut transformed = Graph::<N, E, Ty, Ix>::with_capacity(0, 0);
    for node in graph.node_indices() {
        transformed.add_node(new_node(node.index()));
    }
    for bicluster in biclusters {
        let d = transformed.add_node(dummy_node(&bicluster));
        for &u in bicluster.source.iter() {
            transformed.add_edge(NodeIndex::new(u), d, new_edge(u));
        }
        for &v in bicluster.target.iter() {
            transformed.add_edge(d, NodeIndex::new(v), new_edge(v));
        }
    }
    transformed
}

pub fn inter_group_edge_concentration<
    N,
    E,
    Ty: EdgeType,
    Ix: IndexType,
    BC: Biclustering,
    NF: Fn(usize) -> N,
    EF1: Fn(usize, usize) -> E,
    EF2: Fn(usize) -> E,
    DF: Fn(&Bicluster) -> N,
    DE: Fn(&Bicluster) -> E,
>(
    graph: &Graph<N, E, Ty, Ix>,
    groups: &Vec<usize>,
    biclustering: &BC,
    new_node: NF,
    intra_group_edge: EF1,
    inter_group_edge: EF2,
    dummy_node: DF,
    dummy_edge: DE,
) -> Graph<N, E, Ty, Ix> {
    let mut transformed = Graph::<N, E, Ty, Ix>::with_capacity(0, 0);
    for node in graph.node_indices() {
        transformed.add_node(new_node(node.index()));
    }
    for edge in graph.raw_edges() {
        let u = edge.source();
        let v = edge.target();
        if groups[u.index()] == groups[v.index()] {
            transformed.add_edge(u, v, intra_group_edge(u.index(), v.index()));
        }
    }

    let mut group_vertices = HashMap::new();
    for (i, &g) in groups.iter().enumerate() {
        if !group_vertices.contains_key(&g) {
            group_vertices.insert(g, HashSet::new());
        }
        let s = group_vertices.get_mut(&g).unwrap();
        s.insert(i);
    }

    let group_keys = group_vertices.keys().collect::<Vec<_>>();
    for i in 0..group_keys.len() {
        let g1 = group_vertices.get(group_keys[i]).unwrap();
        for j in (i + 1)..group_keys.len() {
            let g2 = group_vertices.get(group_keys[j]).unwrap();
            let biclusters = biclustering.call(&graph, g1, g2);
            for bicluster in biclusters {
                let d1 = transformed.add_node(dummy_node(&bicluster));
                for &u in bicluster.source.iter() {
                    transformed.add_edge(NodeIndex::new(u), d1, inter_group_edge(u));
                }
                let d2 = transformed.add_node(dummy_node(&bicluster));
                for &v in bicluster.target.iter() {
                    transformed.add_edge(d2, NodeIndex::new(v), inter_group_edge(v));
                }
                transformed.add_edge(d1, d2, dummy_edge(&bicluster));
            }
        }
    }

    transformed
}
