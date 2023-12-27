use crate::{Difference, Drawing, DrawingIndex, DrawingValue, Metric};
use num_traits::{cast::FromPrimitive, clamp, float::FloatConst};
use petgraph::visit::{IntoNeighbors, IntoNodeIdentifiers};
use std::{
    collections::{HashMap, VecDeque},
    ops::{Add, AddAssign, Div, Mul, Sub, SubAssign},
};

#[derive(Copy, Clone, Debug, Default)]
pub struct Tuple2D<S>(pub S, pub S);

impl<S> Add for Tuple2D<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Tuple2D(self.0 + other.0, self.1 + other.1)
    }
}

impl<S> AddAssign for Tuple2D<S>
where
    S: DrawingValue,
{
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0;
        self.1 += other.1;
    }
}

impl<S> Sub for Tuple2D<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Tuple2D(self.0 - other.0, self.1 - other.1)
    }
}

impl<S> SubAssign for Tuple2D<S>
where
    S: DrawingValue,
{
    fn sub_assign(&mut self, other: Self) {
        self.0 -= other.0;
        self.1 -= other.1;
    }
}

impl<S> Mul<S> for Tuple2D<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn mul(self, other: S) -> Self {
        Tuple2D(self.0 * other, self.1 * other)
    }
}

impl<S> Div<S> for Tuple2D<S>
where
    S: DrawingValue,
{
    type Output = Self;

    fn div(self, other: S) -> Self {
        Tuple2D(self.0 / other, self.1 / other)
    }
}

impl<S> Difference for Tuple2D<S>
where
    S: DrawingValue,
{
    type S = S;
    fn norm(&self) -> Self::S {
        self.0.hypot(self.1)
    }
}

impl<S> Metric for Tuple2D<S>
where
    S: DrawingValue,
{
    type D = Tuple2D<S>;
}

pub type Drawing2D<N, S> = Drawing<N, Tuple2D<S>>;

impl<N, S> Drawing2D<N, S>
where
    N: DrawingIndex,
    S: DrawingValue,
{
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
            self.coordinates[i].0 -= l + w / S::from(2.).unwrap();
            self.coordinates[i].1 -= t + h / S::from(2.).unwrap();
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

    pub fn initial_placement<G>(graph: G) -> Drawing2D<N, S>
    where
        G: IntoNodeIdentifiers,
        G::NodeId: DrawingIndex + Into<N>,
        N: Copy,
        S: FloatConst + FromPrimitive + Default,
    {
        let nodes = graph.node_identifiers().collect::<Vec<_>>();
        Drawing::initial_placement_with_node_order(graph, &nodes)
    }

    pub fn initial_placement_with_node_order<G>(graph: G, nodes: &[G::NodeId]) -> Drawing2D<N, S>
    where
        G: IntoNodeIdentifiers,
        G::NodeId: DrawingIndex + Into<N>,
        N: Copy,
        S: FloatConst + FromPrimitive + Default,
    {
        let mut drawing = Drawing::new(graph);
        for (i, &u) in nodes.iter().enumerate() {
            let r = S::from_usize(10).unwrap() * S::from_usize(i).unwrap().sqrt();
            let theta = S::PI()
                * (S::from_usize(3).unwrap() - S::from_usize(5).unwrap().sqrt())
                * (S::from_usize(i).unwrap());
            let x = r * theta.cos();
            let y = r * theta.sin();
            drawing.position_mut(u.into()).map(|p| *p = Tuple2D(x, y));
        }
        drawing
    }

    pub fn initial_placement_with_bfs_order<G>(graph: G, s: G::NodeId) -> Drawing2D<N, S>
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
                if !order.contains_key(&v) {
                    queue.push_back(v);
                    order.insert(v, index);
                    index += 1;
                }
            }
        }
        let mut nodes = graph.node_identifiers().collect::<Vec<_>>();
        nodes.sort_by_key(|&u| order.get(&u).or(Some(&std::usize::MAX)));
        Drawing::initial_placement_with_node_order(graph, &nodes)
    }

    pub fn edge_segments(&self, u: N, v: N) -> Option<Vec<Tuple2D<S>>> {
        self.position(u)
            .zip(self.position(v))
            .map(|(&p, &q)| vec![p, q])
    }
}
