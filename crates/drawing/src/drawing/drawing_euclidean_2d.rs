use crate::{DeltaEuclidean2d, Drawing, DrawingIndex, DrawingValue, MetricEuclidean2d};
use num_traits::{clamp, FloatConst, FromPrimitive};
use petgraph::visit::{IntoNeighbors, IntoNodeIdentifiers};
use std::collections::{HashMap, VecDeque};

/// Represents a drawing of items (nodes) in 2-dimensional Euclidean space.
///
/// This is a specialized version of `DrawingEuclidean` for 2D.
/// It implements the `Drawing` trait.
///
/// # Type Parameters
///
/// * `N`: The type used for indexing items (must implement `DrawingIndex`).
/// * `S`: The scalar type used for coordinates (must implement `DrawingValue`).
pub struct DrawingEuclidean2d<N, S> {
    /// A vector containing the unique identifiers (indices) of the items.
    indices: Vec<N>,
    coordinates: Vec<MetricEuclidean2d<S>>,
    index_map: HashMap<N, usize>,
}

impl<N, S> DrawingEuclidean2d<N, S>
where
    N: DrawingIndex,
    S: DrawingValue,
{
    pub fn new<G>(graph: G) -> Self
    where
        G: IntoNodeIdentifiers,
        G::NodeId: DrawingIndex + Into<N>,
        N: Copy,
        S: Default,
    {
        let indices = graph
            .node_identifiers()
            .map(|u| u.into())
            .collect::<Vec<N>>();
        Self::from_node_indices(&indices)
    }

    pub fn from_node_indices(indices: &[N]) -> Self
    where
        N: Copy,
        S: Default,
    {
        let indices = indices.to_vec();
        let index_map = indices
            .iter()
            .enumerate()
            .map(|(i, &u)| (u, i))
            .collect::<HashMap<_, _>>();
        let coordinates = vec![MetricEuclidean2d::default(); indices.len()];
        Self {
            indices,
            coordinates,
            index_map,
        }
    }

    pub fn x(&self, u: N) -> Option<S> {
        self.position(u).map(|p| p.0)
    }

    pub fn y(&self, u: N) -> Option<S> {
        self.position(u).map(|p| p.1)
    }

    pub fn set_x(&mut self, u: N, value: S) -> Option<()> {
        self.position_mut(u).map(|p| p.0 = value)
    }

    pub fn set_y(&mut self, u: N, value: S) -> Option<()> {
        self.position_mut(u).map(|p| p.1 = value)
    }

    pub fn centralize(&mut self)
    where
        S: FromPrimitive + Default,
    {
        let mut l = S::infinity();
        let mut r = S::neg_infinity();
        let mut t = S::infinity();
        let mut b = S::neg_infinity();
        for i in 0..self.len() {
            l = l.min(self.coordinates[i].0);
            r = r.max(self.coordinates[i].0);
            t = t.min(self.coordinates[i].1);
            b = b.max(self.coordinates[i].1);
        }
        let w = r - l;
        let h = b - t;
        for i in 0..self.len() {
            self.coordinates[i].0 -= l + w / S::from_f32(2.).unwrap();
            self.coordinates[i].1 -= t + h / S::from_f32(2.).unwrap();
        }
    }

    pub fn clamp_region(&mut self, x0: S, y0: S, x1: S, y1: S)
    where
        S: Default,
    {
        for i in 0..self.len() {
            self.coordinates[i].0 = clamp(self.coordinates[i].0, x0, x1);
            self.coordinates[i].1 = clamp(self.coordinates[i].1, y0, y1);
        }
    }

    pub fn initial_placement<G>(graph: G) -> Self
    where
        G: IntoNodeIdentifiers,
        G::NodeId: DrawingIndex + Into<N>,
        N: Copy,
        S: FloatConst + FromPrimitive + Default,
    {
        let nodes = graph.node_identifiers().collect::<Vec<_>>();
        Self::initial_placement_with_node_order(graph, &nodes)
    }

    pub fn initial_placement_with_node_order<G>(graph: G, nodes: &[G::NodeId]) -> Self
    where
        G: IntoNodeIdentifiers,
        G::NodeId: DrawingIndex + Into<N>,
        N: Copy,
        S: FloatConst + FromPrimitive + Default,
    {
        let mut drawing = Self::new(graph);
        for (i, &u) in nodes.iter().enumerate() {
            let r = S::from_usize(10).unwrap() * S::from_usize(i).unwrap().sqrt();
            let theta = S::PI()
                * (S::from_usize(3).unwrap() - S::from_usize(5).unwrap().sqrt())
                * (S::from_usize(i).unwrap());
            let x = r * theta.cos();
            let y = r * theta.sin();
            if let Some(p) = drawing.position_mut(u.into()) {
                *p = MetricEuclidean2d(x, y);
            }
        }
        drawing
    }

    pub fn initial_placement_with_bfs_order<G>(graph: G, s: G::NodeId) -> Self
    where
        G: IntoNeighbors + IntoNodeIdentifiers,
        G::NodeId: DrawingIndex + Into<N>,
        N: Copy,
        S: FloatConst + FromPrimitive + Default,
    {
        let mut queue = VecDeque::new();
        queue.push_back(s);
        let mut order = HashMap::new();
        order.insert(s, 0);
        let mut index = 1usize;
        while let Some(u) = queue.pop_front() {
            for v in graph.neighbors(u) {
                if let std::collections::hash_map::Entry::Vacant(e) = order.entry(v) {
                    queue.push_back(v);
                    e.insert(index);
                    index += 1;
                }
            }
        }
        let mut nodes = graph.node_identifiers().collect::<Vec<_>>();
        nodes.sort_by_key(|&u| order.get(&u).or(Some(&usize::MAX)));
        Self::initial_placement_with_node_order(graph, &nodes)
    }

    pub fn edge_segments(
        &self,
        u: N,
        v: N,
    ) -> Option<Vec<(MetricEuclidean2d<S>, MetricEuclidean2d<S>)>> {
        self.position(u)
            .zip(self.position(v))
            .map(|(&p, &q)| vec![(p, q)])
    }
}

impl<N, S> Drawing for DrawingEuclidean2d<N, S>
where
    N: DrawingIndex,
    S: DrawingValue,
{
    type Index = N;
    type Item = MetricEuclidean2d<S>;

    fn len(&self) -> usize {
        self.indices.len()
    }

    fn dimension(&self) -> usize {
        2
    }

    fn position(&self, u: Self::Index) -> Option<&Self::Item> {
        self.index_map.get(&u).map(|&i| &self.coordinates[i])
    }

    fn position_mut(&mut self, u: Self::Index) -> Option<&mut Self::Item> {
        self.index_map.get(&u).map(|&i| &mut self.coordinates[i])
    }

    fn node_id(&self, i: usize) -> &Self::Index {
        &self.indices[i]
    }

    fn index(&self, u: Self::Index) -> usize {
        self.index_map[&u]
    }

    fn raw_entry(&self, i: usize) -> &Self::Item {
        &self.coordinates[i]
    }

    fn raw_entry_mut(&mut self, i: usize) -> &mut Self::Item {
        &mut self.coordinates[i]
    }

    fn delta(&self, i: usize, j: usize) -> DeltaEuclidean2d<S> {
        self.raw_entry(i) - self.raw_entry(j)
    }
}
