use petgraph_drawing::DrawingValue;
use std::{cmp::Ordering, collections::BTreeSet};

/// Represents a rectangle with its dimensions and node index
#[derive(Clone, Debug)]
pub(super) struct Rectangle<S> {
    /// Minimum x-coordinate
    pub(super) x_min: S,
    /// Maximum x-coordinate
    pub(super) x_max: S,
    /// Minimum y-coordinate
    pub(super) y_min: S,
    /// Maximum y-coordinate
    pub(super) y_max: S,
}

impl<S> Rectangle<S>
where
    S: DrawingValue,
{
    pub(super) fn new(x_min: S, x_max: S, y_min: S, y_max: S) -> Self {
        Rectangle {
            x_min,
            x_max,
            y_min,
            y_max,
        }
    }

    pub(super) fn cx(&self) -> S {
        (self.x_min + self.x_max) / (2.0).into()
    }

    pub(super) fn cy(&self) -> S {
        (self.y_min + self.y_max) / (2.0).into()
    }

    pub(super) fn width(&self) -> S {
        self.x_max - self.x_min
    }

    pub(super) fn height(&self) -> S {
        self.y_max - self.y_min
    }

    pub(super) fn overlap_x(&self, r: &Rectangle<S>) -> S {
        let ux = self.cx();
        let vx = r.cx();
        if ux <= vx && r.x_min < self.x_max {
            return self.x_max - r.x_min;
        }
        if vx <= ux && self.x_min < r.x_max {
            return r.x_max - self.x_min;
        }
        S::zero()
    }

    pub(super) fn overlap_y(&self, r: &Rectangle<S>) -> S {
        let uy = self.cy();
        let vy = r.cy();
        if uy <= vy && r.y_min < self.y_max {
            return self.y_max - r.y_min;
        }
        if vy <= uy && self.y_min < r.y_max {
            return r.y_max - self.y_min;
        }
        S::zero()
    }
}

/// Node structure used for the sweep line algorithm
#[derive(Debug, Clone)]
pub(super) struct Node<S> {
    /// Variable index
    pub(super) v: usize,
    /// Rectangle
    pub(super) r: Rectangle<S>,
    /// Position
    pub(super) pos: S,
    /// Previous nodes
    pub(super) prev: BTreeSet<NodeIndex<S>>,
    /// Next nodes
    pub(super) next: BTreeSet<NodeIndex<S>>,
}

impl<S> Node<S>
where
    S: DrawingValue,
{
    pub(super) fn new_x(v: usize, r: Rectangle<S>) -> Self {
        let pos = r.cx();
        Node {
            v,
            r,
            pos,
            prev: BTreeSet::new(),
            next: BTreeSet::new(),
        }
    }

    pub(super) fn new_y(v: usize, r: Rectangle<S>) -> Self {
        let pos = r.cy();
        Node {
            v,
            r,
            pos,
            prev: BTreeSet::new(),
            next: BTreeSet::new(),
        }
    }

    pub(super) fn index(&self) -> NodeIndex<S> {
        NodeIndex {
            pos: self.pos,
            index: self.v,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct NodeIndex<S> {
    pub(super) pos: S,
    pub(super) index: usize,
}

impl<S> Ord for NodeIndex<S>
where
    S: DrawingValue,
{
    fn cmp(&self, other: &Self) -> Ordering {
        match self.pos.partial_cmp(&other.pos) {
            Some(Ordering::Equal) | None => self.index.cmp(&other.index),
            Some(ord) => ord,
        }
    }
}

impl<S> PartialOrd for NodeIndex<S>
where
    S: DrawingValue,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<S> PartialEq for NodeIndex<S>
where
    S: DrawingValue,
{
    fn eq(&self, other: &Self) -> bool {
        (self.pos, self.index).eq(&(other.pos, other.index))
    }
}

impl<S> Eq for NodeIndex<S> where S: DrawingValue {}

/// Event for the sweepline algorithm
#[derive(Debug, Clone)]
pub(super) struct Event<S> {
    /// Position of the event along the sweep dimension
    pub(super) pos: S,
    /// Type of event (open or close)
    pub(super) is_open: bool,
    /// Associated node index
    pub(super) v: usize,
}

/// Find X-dimension neighbors
pub(super) fn find_x_neighbours<S: DrawingValue>(
    v: usize,
    scanline: &BTreeSet<NodeIndex<S>>,
    nodes: &mut [Node<S>],
) {
    let v_index = nodes[v].index();
    for r in scanline.range(v_index..).skip(1) {
        let u = r.index;
        let u_index = nodes[u].index();
        let u_over_v_x = nodes[u].r.overlap_x(&nodes[v].r);
        if u_over_v_x <= S::zero() || u_over_v_x <= nodes[u].r.overlap_y(&nodes[v].r) {
            nodes[v].next.insert(u_index);
            nodes[u].prev.insert(v_index);
        }
        if u_over_v_x <= S::zero() {
            break;
        }
    }
    let v_index = nodes[v].index();
    for r in scanline.range(..v_index).rev() {
        let u = r.index;
        let u_index = nodes[u].index();
        let u_over_v_x = nodes[u].r.overlap_x(&nodes[v].r);
        if u_over_v_x <= S::zero() || u_over_v_x <= nodes[u].r.overlap_y(&nodes[v].r) {
            nodes[v].prev.insert(u_index);
            nodes[u].next.insert(v_index);
        }
        if u_over_v_x <= S::zero() {
            break;
        }
    }
}

/// Find Y-dimension neighbors according to WebCola's algorithm
pub(super) fn find_y_neighbours<S: DrawingValue>(
    v: usize,
    scanline: &BTreeSet<NodeIndex<S>>,
    nodes: &mut [Node<S>],
) {
    let v_index = nodes[v].index();
    for r in scanline.range(v_index..).skip(1) {
        let u = r.index;
        let u_index = nodes[u].index();
        if nodes[u].r.overlap_x(&nodes[v].r) > S::zero() {
            nodes[v].next.insert(u_index);
            nodes[u].prev.insert(v_index);
        }
    }
    let v_index = nodes[v].index();
    for r in scanline.range(..v_index).rev() {
        let u = r.index;
        let u_index = nodes[u].index();
        if nodes[u].r.overlap_x(&nodes[v].r) > S::zero() {
            nodes[v].prev.insert(u_index);
            nodes[u].next.insert(v_index);
        }
    }
}
