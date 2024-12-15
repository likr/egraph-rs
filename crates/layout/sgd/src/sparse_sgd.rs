use crate::Sgd;
use ndarray::prelude::*;
use ordered_float::OrderedFloat;
use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph_algorithm_shortest_path::{
    dijkstra_with_distance_matrix, multi_source_dijkstra, DistanceMatrix, SubDistanceMatrix,
};
use petgraph_drawing::{DrawingIndex, DrawingValue};
use rand::prelude::*;
use std::collections::{HashMap, HashSet};

pub struct SparseSgd<S> {
    node_pairs: Vec<(usize, usize, S, S, S, S)>,
}

impl<S> SparseSgd<S> {
    pub fn new<G, F>(graph: G, length: F, h: usize) -> Self
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
        S: DrawingValue,
    {
        let mut rng = rand::thread_rng();
        SparseSgd::new_with_rng(graph, length, h, &mut rng)
    }

    pub fn new_with_rng<G, F, R>(graph: G, length: F, h: usize, rng: &mut R) -> Self
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
        R: Rng,
        S: DrawingValue,
    {
        let mut length = length;
        let n = graph.node_count();
        let h = h.min(n);
        let (pivot, d) = Self::choose_pivot(graph, &mut length, h, rng);
        Self::new_with_pivot_and_distance_matrix(graph, length, &pivot, &d)
    }

    pub fn new_with_pivot<G, F>(graph: G, mut length: F, pivot: &[G::NodeId]) -> Self
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
        S: DrawingValue,
    {
        let d = multi_source_dijkstra(graph, &mut length, pivot);
        Self::new_with_pivot_and_distance_matrix(graph, &mut length, pivot, &d)
    }

    pub fn new_with_pivot_and_distance_matrix<G, F, D>(
        graph: G,
        mut length: F,
        pivot: &[G::NodeId],
        distance_matrix: &D,
    ) -> Self
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
        D: DistanceMatrix<G::NodeId, S>,
        S: DrawingValue,
    {
        let indices = graph
            .node_identifiers()
            .enumerate()
            .map(|(i, u)| (u, i))
            .collect::<HashMap<_, _>>();
        let n = indices.len();
        let h = pivot.len();
        let mut node_pairs = vec![];
        let mut edges = HashSet::new();
        for edge in graph.edge_references() {
            let i = indices[&edge.source()];
            let j = indices[&edge.target()];
            let dij = length(edge);
            let wij = S::one() / (dij * dij);
            node_pairs.push((i, j, dij, dij, wij, wij));
            edges.insert((i, j));
            edges.insert((j, i));
        }

        let r = (0..n)
            .map(|j| {
                (0..h)
                    .min_by_key(|&i| OrderedFloat(distance_matrix.get_by_index(i, j)))
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let mut r_nodes = vec![vec![]; h];
        for j in 0..n {
            r_nodes[r[j]].push(j);
        }

        for (k, &u) in pivot.iter().enumerate() {
            let p = indices[&u];
            for i in 0..n {
                if edges.contains(&(p, i)) || p == i {
                    continue;
                }
                let dpi = distance_matrix.get_by_index(k, i);
                let wpi = S::one() / (dpi * dpi);
                let spi = S::from_usize(
                    r_nodes[k]
                        .iter()
                        .filter(|&&j| {
                            S::from_usize(2).unwrap() * distance_matrix.get_by_index(k, j) <= dpi
                        })
                        .count(),
                )
                .unwrap();
                node_pairs.push((p, i, dpi, dpi, spi * wpi, S::zero()));
            }
        }
        SparseSgd { node_pairs }
    }

    pub fn choose_pivot<G, F, R>(
        graph: G,
        length: F,
        h: usize,
        rng: &mut R,
    ) -> (Vec<G::NodeId>, SubDistanceMatrix<G::NodeId, S>)
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
        R: Rng,
        S: DrawingValue,
    {
        max_min_random_sp(graph, length, h, rng)
    }
}

impl<S> Sgd<S> for SparseSgd<S> {
    fn node_pairs(&self) -> &Vec<(usize, usize, S, S, S, S)> {
        &self.node_pairs
    }

    fn node_pairs_mut(&mut self) -> &mut Vec<(usize, usize, S, S, S, S)> {
        &mut self.node_pairs
    }
}

fn max_min_random_sp<G, F, R, S>(
    graph: G,
    length: F,
    h: usize,
    rng: &mut R,
) -> (Vec<G::NodeId>, SubDistanceMatrix<G::NodeId, S>)
where
    G: IntoEdges + IntoNodeIdentifiers + NodeIndexable,
    G::NodeId: DrawingIndex + Ord,
    F: FnMut(G::EdgeRef) -> S,
    R: Rng,
    S: DrawingValue,
{
    let indices = graph
        .node_identifiers()
        .enumerate()
        .map(|(i, u)| (u, i))
        .collect::<HashMap<_, _>>();
    let nodes = graph.node_identifiers().collect::<Vec<_>>();
    let mut length = length;
    let n = indices.len();
    let mut pivot = vec![];
    pivot.push(nodes[rng.gen_range(0..n)]);
    let mut distance_matrix = SubDistanceMatrix::empty(graph);
    distance_matrix.push(pivot[0]);
    dijkstra_with_distance_matrix(graph, &mut length, pivot[0], &mut distance_matrix);
    let mut min_d = Array1::from_elem(n, S::infinity());
    for k in 1..h {
        for j in 0..n {
            min_d[j] = min_d[j].min(distance_matrix.get_by_index(k - 1, j));
        }
        pivot.push(nodes[proportional_sampling(&min_d, rng)]);
        distance_matrix.push(pivot[k]);
        dijkstra_with_distance_matrix(graph, &mut length, pivot[k], &mut distance_matrix);
    }
    (pivot, distance_matrix)
}

fn proportional_sampling<R, S>(values: &Array1<S>, rng: &mut R) -> usize
where
    R: Rng,
    S: DrawingValue,
{
    let n = values.len();
    let mut s = 0.;
    for i in 0..n {
        s += values[i].to_f32().unwrap();
    }
    if s == 0. {
        panic!("could not choice pivot");
    }
    let x = rng.gen_range(0.0..s);
    s = 0.;
    for i in 0..n {
        s += values[i].to_f32().unwrap();
        if x < s {
            return i;
        }
    }
    panic!("unreachable");
}
