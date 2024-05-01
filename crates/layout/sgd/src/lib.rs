use ndarray::prelude::*;
use ordered_float::OrderedFloat;
use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph_algorithm_shortest_path::{
    all_sources_dijkstra, dijkstra_with_distance_matrix, multi_source_dijkstra, DistanceMatrix,
    FullDistanceMatrix, SubDistanceMatrix,
};
use petgraph_drawing::{Delta, Drawing, DrawingIndex, Metric};
use rand::prelude::*;
use std::{
    collections::{HashMap, HashSet},
    f32::INFINITY,
};

pub struct FullSgd {
    node_pairs: Vec<(usize, usize, f32, f32)>,
}

impl FullSgd {
    pub fn new<G, F>(graph: G, length: F) -> FullSgd
    where
        G: IntoEdges + IntoNodeIdentifiers,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> f32,
    {
        let d = all_sources_dijkstra(graph, length);
        Self::new_with_distance_matrix(&d)
    }

    pub fn new_with_distance_matrix<N>(d: &FullDistanceMatrix<N, f32>) -> FullSgd
    where
        N: DrawingIndex,
    {
        let n = d.shape().0;
        let mut node_pairs = vec![];
        for j in 1..n {
            for i in 0..j {
                let dij = d.get_by_index(i, j);
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
    pub fn new<G, F>(graph: G, length: F, h: usize) -> SparseSgd
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> f32,
    {
        let mut rng = rand::thread_rng();
        SparseSgd::new_with_rng(graph, length, h, &mut rng)
    }

    pub fn new_with_rng<G, F, R>(graph: G, length: F, h: usize, rng: &mut R) -> SparseSgd
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> f32,
        R: Rng,
    {
        let mut length = length;
        let n = graph.node_count();
        let h = h.min(n);
        let (pivot, d) = Self::choose_pivot(graph, &mut length, h, rng);
        Self::new_with_pivot_and_distance_matrix(graph, length, &pivot, &d)
    }

    pub fn new_with_pivot<G, F>(graph: G, mut length: F, pivot: &[G::NodeId]) -> SparseSgd
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> f32,
    {
        let d = multi_source_dijkstra(graph, &mut length, pivot);
        Self::new_with_pivot_and_distance_matrix(graph, &mut length, pivot, &d)
    }

    pub fn new_with_pivot_and_distance_matrix<G, F, D>(
        graph: G,
        mut length: F,
        pivot: &[G::NodeId],
        distance_matrix: &D,
    ) -> SparseSgd
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> f32,
        D: DistanceMatrix<G::NodeId, f32>,
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
            let wij = 1. / (dij * dij);
            node_pairs.push((i, j, dij, wij));
            node_pairs.push((j, i, dij, wij));
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

        for (k, &i) in pivot.iter().enumerate() {
            let i = indices[&i];
            for j in 0..n {
                if edges.contains(&(i, j)) || i == j {
                    continue;
                }
                let dij = distance_matrix.get_by_index(k, i);
                let wij = 1. / (dij * dij);
                let sij = r_nodes[k]
                    .iter()
                    .filter(|&&l| 2. * distance_matrix.get_by_index(k, l) <= dij)
                    .count() as f32;
                node_pairs.push((i, j, dij, sij * wij));
            }
        }
        SparseSgd { node_pairs }
    }

    pub fn choose_pivot<G, F, R>(
        graph: G,
        length: F,
        h: usize,
        rng: &mut R,
    ) -> (Vec<G::NodeId>, SubDistanceMatrix<G::NodeId, f32>)
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> f32,
        R: Rng,
    {
        max_min_random_sp(graph, length, h, rng)
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

    fn apply<Diff, D, M>(&self, drawing: &mut D, eta: f32)
    where
        D: Drawing<Item = M>,
        Diff: Delta<S = f32>,
        M: Metric<D = Diff>,
    {
        for &(i, j, dij, wij) in self.node_pairs().iter() {
            let mu = (eta * wij).min(1.);
            let delta = drawing.delta(i, j);
            let norm = delta.norm();
            if norm > 0. {
                let r = 0.5 * mu * (norm - dij) / norm;
                *drawing.raw_entry_mut(i) += delta * -r;
            }
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

    fn update_distance<F>(&mut self, mut distance: F)
    where
        F: FnMut(usize, usize, f32, f32) -> f32,
    {
        for p in self.node_pairs_mut() {
            let (i, j, dij, wij) = p;
            p.2 = distance(*i, *j, *dij, *wij)
        }
    }

    fn update_weight<F>(&mut self, mut weight: F)
    where
        F: FnMut(usize, usize, f32, f32) -> f32,
    {
        for p in self.node_pairs_mut() {
            let (i, j, dij, wij) = p;
            p.3 = weight(*i, *j, *dij, *wij)
        }
    }
}

fn max_min_random_sp<G, F, R>(
    graph: G,
    length: F,
    h: usize,
    rng: &mut R,
) -> (Vec<G::NodeId>, SubDistanceMatrix<G::NodeId, f32>)
where
    G: IntoEdges + IntoNodeIdentifiers + NodeIndexable,
    G::NodeId: DrawingIndex + Ord,
    F: FnMut(G::EdgeRef) -> f32,
    R: Rng,
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
    let mut min_d = Array1::from_elem(n, INFINITY);
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

fn proportional_sampling<R>(values: &Array1<f32>, rng: &mut R) -> usize
where
    R: Rng,
{
    let n = values.len();
    let mut s = 0.;
    for i in 0..n {
        s += values[i];
    }
    if s == 0. {
        panic!("could not choice pivot");
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

pub struct DistanceAdjustedSgd<A>
where
    A: Sgd,
{
    pub alpha: f32,
    pub minimum_distance: f32,
    sgd: A,
    original_distance: HashMap<(usize, usize), f32>,
}

impl<A> DistanceAdjustedSgd<A>
where
    A: Sgd,
{
    pub fn new(sgd: A) -> DistanceAdjustedSgd<A> {
        let mut original_distance = HashMap::new();
        for p in sgd.node_pairs().iter() {
            original_distance.insert((p.0, p.1), p.2);
        }
        Self {
            alpha: 0.5,
            minimum_distance: 0.0,
            sgd,
            original_distance,
        }
    }

    pub fn apply_with_distance_adjustment<D, Diff, M>(&mut self, drawing: &mut D, eta: f32)
    where
        D: Drawing<Item = M>,
        Diff: Delta<S = f32>,
        M: Metric<D = Diff>,
    {
        self.sgd.apply(drawing, eta);
        self.sgd.update_distance(|i, j, _, w| {
            let delta = drawing.delta(i, j);
            let d1 = delta.norm();
            let d2 = self.original_distance[&(i, j)];
            let new_d = (self.alpha * w * d1 + 2. * (1. - self.alpha) * d2)
                / (self.alpha * w + 2. * (1. - self.alpha));
            new_d.clamp(self.minimum_distance, d2)
        });
        self.sgd.update_weight(|_, _, d, _| 1. / (d * d));
    }
}

impl<A> Sgd for DistanceAdjustedSgd<A>
where
    A: Sgd,
{
    fn node_pairs(&self) -> &Vec<(usize, usize, f32, f32)> {
        self.sgd.node_pairs()
    }

    fn node_pairs_mut(&mut self) -> &mut Vec<(usize, usize, f32, f32)> {
        self.sgd.node_pairs_mut()
    }
}
