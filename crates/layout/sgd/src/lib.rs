use ndarray::prelude::*;
use ordered_float::OrderedFloat;
use petgraph::{
    graph::{node_index, EdgeReference, IndexType},
    prelude::*,
    visit::IntoNodeIdentifiers,
    EdgeType,
};
use petgraph_algorithm_shortest_path::{dijkstra_with_distance_matrix, warshall_floyd};
use petgraph_layout_force_simulation::Coordinates;
use rand::prelude::*;
use std::{
    collections::{HashMap, HashSet},
    f32::INFINITY,
};

pub struct FullSgd {
    node_pairs: Vec<(usize, usize, f32, f32)>,
}

impl FullSgd {
    pub fn new<N, E, Ty: EdgeType, Ix: IndexType, F: FnMut(EdgeReference<'_, E, Ix>) -> f32>(
        graph: &Graph<N, E, Ty, Ix>,
        length: &mut F,
    ) -> FullSgd {
        let indices = graph
            .node_identifiers()
            .enumerate()
            .map(|(i, u)| (u, i))
            .collect::<HashMap<_, _>>();
        let n = indices.len();
        let d = warshall_floyd(graph, length);

        let mut node_pairs = vec![];
        for j in 1..n {
            for i in 0..j {
                let dij = d[[i, j]];
                let wij = 1. / (dij * dij);
                node_pairs.push((i, j, dij, wij));
                node_pairs.push((j, i, dij, wij));
            }
        }

        FullSgd { node_pairs }
    }
}

impl Sgd for FullSgd {
    fn node_pairs(&self) -> &Vec<(usize, usize, f32, f32)> {
        &self.node_pairs
    }

    fn node_pairs_mut(&mut self) -> &mut Vec<(usize, usize, f32, f32)> {
        &mut self.node_pairs
    }
}

pub struct SparseSgd {
    node_pairs: Vec<(usize, usize, f32, f32)>,
}

impl SparseSgd {
    pub fn new<N, E, Ty: EdgeType, Ix: IndexType, F: FnMut(EdgeReference<'_, E, Ix>) -> f32>(
        graph: &Graph<N, E, Ty, Ix>,
        length: &mut F,
        h: usize,
    ) -> SparseSgd {
        let mut rng = rand::thread_rng();
        SparseSgd::new_with_rng(graph, length, h, &mut rng)
    }

    pub fn new_with_rng<
        N,
        E,
        Ty: EdgeType,
        Ix: IndexType,
        F: FnMut(EdgeReference<'_, E, Ix>) -> f32,
        R: Rng,
    >(
        graph: &Graph<N, E, Ty, Ix>,
        length: &mut F,
        h: usize,
        rng: &mut R,
    ) -> SparseSgd {
        let indices = graph
            .node_identifiers()
            .enumerate()
            .map(|(i, u)| (u, i))
            .collect::<HashMap<_, _>>();
        let n = indices.len();
        let (pivot, d) = max_min_random_sp(graph, &indices, length, h, rng);

        let mut node_pairs = vec![];
        let mut edges = HashSet::new();
        for edge in graph.edge_references() {
            let i = indices[&edge.source()];
            let j = indices[&edge.target()];
            let dij = length(edge);
            let wij = 1. / (dij * dij);
            node_pairs.push((i, j, dij, wij));
            node_pairs.push((j, i, dij, wij));
            edges.insert((i, j));
            edges.insert((j, i));
        }

        let r = (0..n)
            .map(|i| (0..h).min_by_key(|&j| OrderedFloat(d[[i, j]])).unwrap())
            .collect::<Vec<_>>();
        let mut r_nodes = vec![vec![]; h];
        for i in 0..n {
            r_nodes[r[i]].push(i);
        }

        for (k, &j) in pivot.iter().enumerate() {
            for i in 0..n {
                if edges.contains(&(i, j)) || i == j {
                    continue;
                }
                let dij = d[[i, k]];
                let wij = 1. / (dij * dij);
                let sij = r_nodes[k]
                    .iter()
                    .filter(|&&l| 2. * d[[l, k]] <= dij)
                    .count() as f32;
                node_pairs.push((i, j, dij, sij * wij));
            }
        }
        SparseSgd { node_pairs }
    }
}

impl Sgd for SparseSgd {
    fn node_pairs(&self) -> &Vec<(usize, usize, f32, f32)> {
        &self.node_pairs
    }

    fn node_pairs_mut(&mut self) -> &mut Vec<(usize, usize, f32, f32)> {
        &mut self.node_pairs
    }
}

pub struct SgdScheduler {
    t: usize,
    t_max: usize,
    a: f32,
    b: f32,
}

impl SgdScheduler {
    pub fn run<F: FnMut(f32)>(&mut self, f: &mut F) {
        while !self.is_finished() {
            self.step(f)
        }
    }

    pub fn step<F: FnMut(f32)>(&mut self, f: &mut F) {
        let eta = self.a * (self.b * self.t as f32).exp();
        f(eta);
        self.t += 1;
    }

    pub fn is_finished(&self) -> bool {
        self.t >= self.t_max
    }
}

pub trait Sgd {
    fn node_pairs(&self) -> &Vec<(usize, usize, f32, f32)>;

    fn node_pairs_mut(&mut self) -> &mut Vec<(usize, usize, f32, f32)>;

    fn shuffle<R: Rng>(&mut self, rng: &mut R) {
        self.node_pairs_mut().shuffle(rng);
    }

    fn apply<Ix: IndexType>(&self, coordinates: &mut Coordinates<Ix>, eta: f32) {
        for &(i, j, dij, wij) in self.node_pairs().iter() {
            let mu = (eta * wij).min(1.);
            let dx = coordinates.points[i].x - coordinates.points[j].x;
            let dy = coordinates.points[i].y - coordinates.points[j].y;
            let norm = (dx * dx + dy * dy).sqrt().max(1.);
            let r = 0.5 * mu * (norm - dij) / norm;
            coordinates.points[i].x -= r * dx;
            coordinates.points[i].y -= r * dy;
        }
    }

    fn scheduler(&self, t_max: usize, epsilon: f32) -> SgdScheduler {
        let mut w_min = INFINITY;
        let mut w_max = 0.;
        for &(_, _, _, wij) in self.node_pairs().iter() {
            if wij == 0. {
                continue;
            }
            if wij < w_min {
                w_min = wij;
            }
            if wij > w_max {
                w_max = wij;
            }
        }
        let eta_max = 1. / w_min;
        let eta_min = epsilon / w_max;
        SgdScheduler {
            t: 0,
            t_max,
            a: eta_max,
            b: (eta_min / eta_max).ln() / (t_max - 1) as f32,
        }
    }
}

fn max_min_random_sp<
    N,
    E,
    Ty: EdgeType,
    Ix: IndexType,
    F: FnMut(EdgeReference<'_, E, Ix>) -> f32,
    R: Rng,
>(
    graph: &Graph<N, E, Ty, Ix>,
    indices: &HashMap<NodeIndex<Ix>, usize>,
    length: &mut F,
    h: usize,
    rng: &mut R,
) -> (Vec<usize>, Array2<f32>) {
    let n = graph.node_count();
    let mut pivot = vec![];
    pivot.push(rng.gen_range(0..n));
    let mut distance_matrix = Array2::from_elem((n, h), INFINITY);
    dijkstra_with_distance_matrix(
        graph,
        indices,
        length,
        node_index(pivot[0]),
        &mut distance_matrix,
        0,
    );
    let mut min_d = Array1::from_elem(n, INFINITY);
    for k in 1..h {
        for i in 0..n {
            min_d[i] = min_d[i].min(distance_matrix[[i, k - 1]]);
        }
        pivot.push(proportional_sampling(&min_d, rng));
        dijkstra_with_distance_matrix(
            graph,
            indices,
            length,
            node_index(pivot[k]),
            &mut distance_matrix,
            k,
        );
    }
    (pivot, distance_matrix)
}

fn proportional_sampling<R: Rng>(values: &Array1<f32>, rng: &mut R) -> usize {
    let n = values.len();
    let mut s = 0.;
    for i in 0..n {
        s += values[i];
    }
    let x = rng.gen_range(0.0..s);
    s = 0.;
    for i in 0..n {
        s += values[i];
        if x < s {
            return i;
        }
    }
    panic!("unreachable");
}
