//! Exact $O((|E|+K)\log|E|)$ sweep-line edge-crossing analysis, torus crossings, and
//! $O(M)$ deterministic edge-pair sampling.

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};
use std::f64::consts::FRAC_PI_2;
use std::hash::{BuildHasherDefault, Hasher};

use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, Layout2d, Scalar, wrap_torus_coord,
};

const NONE_U32: u32 = u32::MAX;

#[derive(Default)]
struct FastU64Hasher {
    state: u64,
}

impl Hasher for FastU64Hasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.state
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        let z = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.state = z ^ (z >> 32);
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut h = self.state;
        for &b in bytes {
            h = h.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ u64::from(b);
        }
        self.state = h;
    }
}

type PairSet = HashSet<u64, BuildHasherDefault<FastU64Hasher>>;

#[inline]
fn pack_pair(a: usize, b: usize) -> u64 {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    ((lo as u64) << 32) | (hi as u64)
}

#[inline]
fn unpack_pair(key: u64) -> (usize, usize) {
    ((key >> 32) as usize, (key & 0xFFFF_FFFF) as usize)
}

/// Details of a single crossing pair between two non-adjacent edges.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrossingPair {
    pub edge_a: usize,
    pub edge_b: usize,
    pub u1: u32,
    pub v1: u32,
    pub u2: u32,
    pub v2: u32,
    /// Acute crossing angle in `[0, pi/2]` radians, or `None` for zero-length segments.
    pub angle_rad: Option<f64>,
    /// $\cos^2(\theta)$ of the crossing angle (`0.0` when angle is undefined).
    pub cos_sq: f64,
}

/// Summary statistics of all edge crossings in a 2D Euclidean layout.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CrossingSummary {
    pub count: usize,
    /// Sum of $\cos^2(\theta)$ over crossing pairs (matches legacy `crossing_angle`).
    pub angle_cos_sq_sum: f64,
    /// Minimum acute crossing angle across all defined crossing pairs.
    pub min_angle_rad: Option<f64>,
}

/// Crossing pair of periodic `[0, 1)^2` torus edge sub-segments.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TorusCrossing {
    pub edge_a: usize,
    pub edge_b: usize,
    /// `[x1, y1, x2, y2]` coordinates of the first crossing sub-segment.
    pub seg_a: [f64; 4],
    /// `[x1, y1, x2, y2]` coordinates of the second crossing sub-segment.
    pub seg_b: [f64; 4],
}

#[derive(Debug, Clone, Copy)]
struct Seg {
    id: usize,
    u: u32,
    v: u32,
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    slope: f64,
    is_vertical: bool,
}

impl Seg {
    fn from_endpoints(id: usize, u: u32, v: u32, ax: f64, ay: f64, bx: f64, by: f64) -> Self {
        let (x1, y1, x2, y2) = if ax < bx || (ax == bx && ay <= by) {
            (ax, ay, bx, by)
        } else {
            (bx, by, ax, ay)
        };
        let is_vertical = x1 == x2;
        let slope = if is_vertical {
            0.0
        } else {
            (y2 - y1) / (x2 - x1)
        };
        Self {
            id,
            u,
            v,
            x1,
            y1,
            x2,
            y2,
            x_min: x1.min(x2),
            x_max: x1.max(x2),
            y_min: y1.min(y2),
            y_max: y1.max(y2),
            slope,
            is_vertical,
        }
    }

    #[inline]
    fn shares_endpoint(&self, other: &Self) -> bool {
        self.u == other.u
            || self.u == other.v
            || self.v == other.u
            || self.v == other.v
            || self.u == self.v
            || other.u == other.v
    }

    #[inline]
    fn y_at(&self, x: f64) -> f64 {
        if x <= self.x1 {
            self.y1
        } else if x >= self.x2 {
            self.y2
        } else {
            self.y1 + self.slope * (x - self.x1)
        }
    }
}

#[inline]
fn same_sign_nonzero(a: f64, b: f64) -> bool {
    (a > 0.0 && b > 0.0) || (a < 0.0 && b < 0.0)
}

#[inline]
fn segments_geometrically_intersect(a: &Seg, b: &Seg) -> bool {
    if a.x_max < b.x_min || b.x_max < a.x_min || a.y_max < b.y_min || b.y_max < a.y_min {
        return false;
    }
    let dx1 = a.x2 - a.x1;
    let dy1 = a.y2 - a.y1;
    let dx2 = b.x2 - b.x1;
    let dy2 = b.y2 - b.y1;

    let s1 = dx1 * (b.y1 - a.y1) - dy1 * (b.x1 - a.x1);
    let t1 = dx1 * (b.y2 - a.y1) - dy1 * (b.x2 - a.x1);
    if same_sign_nonzero(s1, t1) {
        return false;
    }

    let s2 = dx2 * (a.y1 - b.y1) - dy2 * (a.x1 - b.x1);
    let t2 = dx2 * (a.y2 - b.y1) - dy2 * (a.x2 - b.x1);
    !same_sign_nonzero(s2, t2)
}

#[inline]
fn segments_cross(a: &Seg, b: &Seg) -> bool {
    if a.shares_endpoint(b) {
        return false;
    }
    segments_geometrically_intersect(a, b)
}

#[inline]
fn seg_y_err(s: &Seg, x: f64) -> f64 {
    let x_scale = x.abs().max(s.x1.abs()).max(s.x2.abs());
    let y_scale = s.y1.abs().max(s.y2.abs());
    32.0 * f64::EPSILON * (s.slope.abs() * x_scale + y_scale + 1.0)
}

#[inline]
fn within_band(seed: &Seg, other: &Seg, x_evt: f64, eps_y: f64) -> bool {
    let tol = 2.0 * eps_y + seg_y_err(seed, x_evt) + seg_y_err(other, x_evt);
    (other.y_at(x_evt) - seed.y_at(x_evt)).abs() <= tol
}

#[inline]
fn intersection_x(a: &Seg, b: &Seg) -> Option<f64> {
    let dx_a = a.x2 - a.x1;
    let dx_b = b.x2 - b.x1;
    let (p, q) = if dx_a < dx_b || (dx_a == dx_b && a.id <= b.id) {
        (a, b)
    } else {
        (b, a)
    };
    let dx1 = p.x2 - p.x1;
    let dy1 = p.y2 - p.y1;
    let dx2 = q.x2 - q.x1;
    let dy2 = q.y2 - q.y1;
    let det = dx1 * dy2 - dy1 * dx2;
    if det == 0.0 {
        return None;
    }
    let t = ((q.x1 - p.x1) * dy2 - (q.y1 - p.y1) * dx2) / det;
    let x = p.x1 + t * dx1;
    let lo = p.x_min.max(q.x_min);
    let hi = p.x_max.min(q.x_max);
    Some(x.clamp(lo, hi))
}

/// Exact post-`x_evt` vertical order of two active non-vertical segments spanning `x_evt`.
///
/// Uses bounding-box y-disjointness and exact endpoint orientation signs (`s1, t1, s2, t2`) for
/// non-intersecting pairs (avoiding `y_at(x_evt)` cancellation and scalar `dt_probe` overshoot),
/// shared-endpoint slope checks, scale-safe `det == 0.0` fallback, and canonicalized
/// `intersection_x` to separate completed crossings (`xi <= x_evt`) from future ones (`xi > x_evt`).
#[inline]
fn cmp_post_x(a: &Seg, b: &Seg, x_evt: f64) -> Ordering {
    if a.id == b.id {
        return Ordering::Equal;
    }
    if a.y_max < b.y_min {
        return Ordering::Less;
    }
    if b.y_max < a.y_min {
        return Ordering::Greater;
    }

    let dx1 = a.x2 - a.x1;
    let dy1 = a.y2 - a.y1;
    let dx2 = b.x2 - b.x1;
    let dy2 = b.y2 - b.y1;

    let s1 = dx1 * (b.y1 - a.y1) - dy1 * (b.x1 - a.x1);
    let t1 = dx1 * (b.y2 - a.y1) - dy1 * (b.x2 - a.x1);
    if same_sign_nonzero(s1, t1) {
        return if s1 > 0.0 {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }

    let s2 = dx2 * (a.y1 - b.y1) - dy2 * (a.x1 - b.x1);
    let t2 = dx2 * (a.y2 - b.y1) - dy2 * (a.x2 - b.x1);
    if same_sign_nonzero(s2, t2) {
        return if s2 > 0.0 {
            Ordering::Greater
        } else {
            Ordering::Less
        };
    }

    if a.x2 == b.x2 && a.y2 == b.y2 && a.x2 > x_evt {
        return b.slope.total_cmp(&a.slope).then_with(|| a.id.cmp(&b.id));
    }
    if a.x1 == b.x1 && a.y1 == b.y1 && a.x1 <= x_evt {
        return a.slope.total_cmp(&b.slope).then_with(|| a.id.cmp(&b.id));
    }

    let det = dx1 * dy2 - dy1 * dx2;
    if det == 0.0 {
        let ya = a.y_at(x_evt);
        let yb = b.y_at(x_evt);
        let x_scale = x_evt
            .abs()
            .max(a.x1.abs())
            .max(a.x2.abs())
            .max(b.x1.abs())
            .max(b.x2.abs());
        let y_scale = a.y1.abs().max(a.y2.abs()).max(b.y1.abs()).max(b.y2.abs());
        let rel_tol = 32.0 * f64::EPSILON * (a.slope.abs().max(b.slope.abs()) * x_scale + y_scale);
        if (ya - yb).abs() > rel_tol {
            return ya.total_cmp(&yb);
        }
        return a.slope.total_cmp(&b.slope).then_with(|| a.id.cmp(&b.id));
    }

    let pdx = dx1.min(dx2);
    let x_scale_evt = x_evt.abs().max(1.0);
    let post_crossing = if pdx <= 64.0 * f64::EPSILON * x_scale_evt {
        // Near-vertical segment (dx spans only a few ULPs): `p.x1 + t * pdx` loses sub-ULP
        // position along `p`, so compare normalized `t_int` vs `t_evt` directly.
        let (p, q) = if dx1 < dx2 || (dx1 == dx2 && a.id <= b.id) {
            (a, b)
        } else {
            (b, a)
        };
        let p_dx = p.x2 - p.x1;
        let p_dy = p.y2 - p.y1;
        let q_dx = q.x2 - q.x1;
        let q_dy = q.y2 - q.y1;
        let p_det = p_dx * q_dy - p_dy * q_dx;
        let t_int = ((q.x1 - p.x1) * q_dy - (q.y1 - p.y1) * q_dx) / p_det;
        let t_evt = (x_evt - p.x1) / p_dx;
        t_int <= t_evt
    } else {
        let xi = intersection_x(a, b).unwrap_or(x_evt);
        xi <= x_evt
    };
    if post_crossing {
        a.slope
            .total_cmp(&b.slope)
            .then_with(|| 0.0_f64.total_cmp(&det))
            .then_with(|| a.id.cmp(&b.id))
    } else {
        b.slope
            .total_cmp(&a.slope)
            .then_with(|| det.total_cmp(&0.0))
            .then_with(|| a.id.cmp(&b.id))
    }
}

#[inline]
fn crossing_angle_metrics(a: &Seg, b: &Seg) -> (Option<f64>, f64) {
    let dx1 = a.x2 - a.x1;
    let dy1 = a.y2 - a.y1;
    let dx2 = b.x2 - b.x1;
    let dy2 = b.y2 - b.y1;
    let len1 = dx1.hypot(dy1);
    let len2 = dx2.hypot(dy2);
    if len1 == 0.0 || len2 == 0.0 {
        return (None, 0.0);
    }
    let ux1 = dx1 / len1;
    let uy1 = dy1 / len1;
    let ux2 = dx2 / len2;
    let uy2 = dy2 / len2;
    let cos_val = (ux1 * ux2 + uy1 * uy2).clamp(-1.0, 1.0);
    let acute = cos_val.abs().acos().min(FRAC_PI_2);
    (Some(acute), cos_val * cos_val)
}

#[derive(Debug, Clone, Copy)]
struct AvlNode {
    parent: u32,
    left: u32,
    right: u32,
    height: u16,
    size: u32,
    edge_idx: usize,
}

/// Arena-backed AVL tree maintaining active non-vertical segments ordered by `(y(x), slope, id)`.
///
/// Storing `node_of_edge` lets segment removal at right endpoints run in $O(\log |A|)$ by handle
/// without relying on floating-point `y(x)` search, and swaps adjacent crossing segments in $O(1)$
/// by exchanging edge payloads.
struct ActiveSegmentTree {
    nodes: Vec<AvlNode>,
    free_list: Vec<u32>,
    root: u32,
    node_of_edge: Vec<u32>,
}

impl ActiveSegmentTree {
    fn new(num_edges: usize) -> Self {
        Self {
            nodes: Vec::new(),
            free_list: Vec::new(),
            root: NONE_U32,
            node_of_edge: vec![NONE_U32; num_edges],
        }
    }

    #[inline]
    fn is_active(&self, edge_idx: usize) -> bool {
        self.node_of_edge[edge_idx] != NONE_U32
    }

    #[inline]
    fn height(&self, n: u32) -> u16 {
        if n == NONE_U32 {
            0
        } else {
            self.nodes[n as usize].height
        }
    }

    #[inline]
    fn size(&self, n: u32) -> u32 {
        if n == NONE_U32 {
            0
        } else {
            self.nodes[n as usize].size
        }
    }

    #[inline]
    fn update(&mut self, n: u32) {
        let l = self.nodes[n as usize].left;
        let r = self.nodes[n as usize].right;
        self.nodes[n as usize].height = 1 + self.height(l).max(self.height(r));
        self.nodes[n as usize].size = 1 + self.size(l) + self.size(r);
    }

    #[inline]
    fn balance_factor(&self, n: u32) -> i32 {
        self.height(self.nodes[n as usize].left) as i32
            - self.height(self.nodes[n as usize].right) as i32
    }

    fn alloc_node(&mut self, parent: u32, edge_idx: usize) -> u32 {
        let node = AvlNode {
            parent,
            left: NONE_U32,
            right: NONE_U32,
            height: 1,
            size: 1,
            edge_idx,
        };
        if let Some(id) = self.free_list.pop() {
            self.nodes[id as usize] = node;
            id
        } else {
            let id = self.nodes.len() as u32;
            self.nodes.push(node);
            id
        }
    }

    fn rotate_left(&mut self, x: u32) -> u32 {
        let y = self.nodes[x as usize].right;
        let t2 = self.nodes[y as usize].left;
        let p = self.nodes[x as usize].parent;

        self.nodes[y as usize].left = x;
        self.nodes[x as usize].parent = y;
        self.nodes[x as usize].right = t2;
        if t2 != NONE_U32 {
            self.nodes[t2 as usize].parent = x;
        }
        self.nodes[y as usize].parent = p;
        if p == NONE_U32 {
            self.root = y;
        } else if self.nodes[p as usize].left == x {
            self.nodes[p as usize].left = y;
        } else {
            self.nodes[p as usize].right = y;
        }
        self.update(x);
        self.update(y);
        y
    }

    fn rotate_right(&mut self, y: u32) -> u32 {
        let x = self.nodes[y as usize].left;
        let t2 = self.nodes[x as usize].right;
        let p = self.nodes[y as usize].parent;

        self.nodes[x as usize].right = y;
        self.nodes[y as usize].parent = x;
        self.nodes[y as usize].left = t2;
        if t2 != NONE_U32 {
            self.nodes[t2 as usize].parent = y;
        }
        self.nodes[x as usize].parent = p;
        if p == NONE_U32 {
            self.root = x;
        } else if self.nodes[p as usize].left == y {
            self.nodes[p as usize].left = x;
        } else {
            self.nodes[p as usize].right = x;
        }
        self.update(y);
        self.update(x);
        x
    }

    fn rebalance_up(&mut self, mut cur: u32) {
        while cur != NONE_U32 {
            self.update(cur);
            let bf = self.balance_factor(cur);
            if bf > 1 {
                let left = self.nodes[cur as usize].left;
                if self.balance_factor(left) < 0 {
                    self.rotate_left(left);
                }
                cur = self.rotate_right(cur);
            } else if bf < -1 {
                let right = self.nodes[cur as usize].right;
                if self.balance_factor(right) > 0 {
                    self.rotate_right(right);
                }
                cur = self.rotate_left(cur);
            }
            cur = self.nodes[cur as usize].parent;
        }
    }

    fn insert(&mut self, edge_idx: usize, sweep_x: f64, segs: &[Seg]) {
        if self.node_of_edge[edge_idx] != NONE_U32 {
            return;
        }
        if self.root == NONE_U32 {
            let id = self.alloc_node(NONE_U32, edge_idx);
            self.root = id;
            self.node_of_edge[edge_idx] = id;
            return;
        }

        let s_new = &segs[edge_idx];
        let mut cur = self.root;
        loop {
            let cur_edge = self.nodes[cur as usize].edge_idx;
            let s_cur = &segs[cur_edge];
            let ord = cmp_post_x(s_new, s_cur, sweep_x);
            if ord == Ordering::Less {
                let left = self.nodes[cur as usize].left;
                if left == NONE_U32 {
                    let id = self.alloc_node(cur, edge_idx);
                    self.nodes[cur as usize].left = id;
                    self.node_of_edge[edge_idx] = id;
                    self.rebalance_up(cur);
                    return;
                }
                cur = left;
            } else {
                let right = self.nodes[cur as usize].right;
                if right == NONE_U32 {
                    let id = self.alloc_node(cur, edge_idx);
                    self.nodes[cur as usize].right = id;
                    self.node_of_edge[edge_idx] = id;
                    self.rebalance_up(cur);
                    return;
                }
                cur = right;
            }
        }
    }

    fn leftmost(&self, mut n: u32) -> u32 {
        while n != NONE_U32 {
            let l = self.nodes[n as usize].left;
            if l == NONE_U32 {
                break;
            }
            n = l;
        }
        n
    }

    fn rightmost(&self, mut n: u32) -> u32 {
        while n != NONE_U32 {
            let r = self.nodes[n as usize].right;
            if r == NONE_U32 {
                break;
            }
            n = r;
        }
        n
    }

    fn prev_node(&self, mut n: u32) -> u32 {
        if n == NONE_U32 {
            return NONE_U32;
        }
        let l = self.nodes[n as usize].left;
        if l != NONE_U32 {
            return self.rightmost(l);
        }
        let mut p = self.nodes[n as usize].parent;
        while p != NONE_U32 && self.nodes[p as usize].left == n {
            n = p;
            p = self.nodes[p as usize].parent;
        }
        p
    }

    fn next_node(&self, mut n: u32) -> u32 {
        if n == NONE_U32 {
            return NONE_U32;
        }
        let r = self.nodes[n as usize].right;
        if r != NONE_U32 {
            return self.leftmost(r);
        }
        let mut p = self.nodes[n as usize].parent;
        while p != NONE_U32 && self.nodes[p as usize].right == n {
            n = p;
            p = self.nodes[p as usize].parent;
        }
        p
    }

    #[inline]
    fn prev_edge(&self, edge_idx: usize) -> Option<usize> {
        let n = self.node_of_edge[edge_idx];
        if n == NONE_U32 {
            return None;
        }
        let p = self.prev_node(n);
        if p == NONE_U32 {
            None
        } else {
            Some(self.nodes[p as usize].edge_idx)
        }
    }

    #[inline]
    fn next_edge(&self, edge_idx: usize) -> Option<usize> {
        let n = self.node_of_edge[edge_idx];
        if n == NONE_U32 {
            return None;
        }
        let nx = self.next_node(n);
        if nx == NONE_U32 {
            None
        } else {
            Some(self.nodes[nx as usize].edge_idx)
        }
    }

    fn rank_of_edge(&self, edge_idx: usize) -> Option<usize> {
        let mut cur = self.node_of_edge[edge_idx];
        if cur == NONE_U32 {
            return None;
        }
        let mut rank = self.size(self.nodes[cur as usize].left) as usize;
        while self.nodes[cur as usize].parent != NONE_U32 {
            let p = self.nodes[cur as usize].parent;
            if self.nodes[p as usize].right == cur {
                rank += 1 + self.size(self.nodes[p as usize].left) as usize;
            }
            cur = p;
        }
        Some(rank)
    }

    fn swap_adjacent_edges(&mut self, e1: usize, e2: usize) {
        let n1 = self.node_of_edge[e1];
        let n2 = self.node_of_edge[e2];
        debug_assert!(n1 != NONE_U32 && n2 != NONE_U32);
        self.nodes[n1 as usize].edge_idx = e2;
        self.nodes[n2 as usize].edge_idx = e1;
        self.node_of_edge[e1] = n2;
        self.node_of_edge[e2] = n1;
    }

    /// Removes `edge_idx` by handle and returns the newly adjacent `(prev, next)` edge indices.
    fn remove_by_edge(&mut self, edge_idx: usize) -> (Option<usize>, Option<usize>) {
        let n = self.node_of_edge[edge_idx];
        if n == NONE_U32 {
            return (None, None);
        }
        let prev = self.prev_edge(edge_idx);
        let next = self.next_edge(edge_idx);

        // If node has two children, swap payload with its successor so deletion has <= 1 child.
        let mut target = n;
        if self.nodes[n as usize].left != NONE_U32 && self.nodes[n as usize].right != NONE_U32 {
            let succ = self.leftmost(self.nodes[n as usize].right);
            let succ_edge = self.nodes[succ as usize].edge_idx;
            self.nodes[n as usize].edge_idx = succ_edge;
            self.node_of_edge[succ_edge] = n;
            target = succ;
        }

        let child = if self.nodes[target as usize].left != NONE_U32 {
            self.nodes[target as usize].left
        } else {
            self.nodes[target as usize].right
        };
        let parent = self.nodes[target as usize].parent;
        if child != NONE_U32 {
            self.nodes[child as usize].parent = parent;
        }
        if parent == NONE_U32 {
            self.root = child;
        } else if self.nodes[parent as usize].left == target {
            self.nodes[parent as usize].left = child;
        } else {
            self.nodes[parent as usize].right = child;
        }

        self.node_of_edge[edge_idx] = NONE_U32;
        self.free_list.push(target);
        if parent != NONE_U32 {
            self.rebalance_up(parent);
        }

        (prev, next)
    }

    fn lower_bound_y(&self, target_y: f64, sweep_x: f64, segs: &[Seg]) -> Option<usize> {
        let mut cur = self.root;
        let mut candidate = NONE_U32;
        while cur != NONE_U32 {
            let e = self.nodes[cur as usize].edge_idx;
            let y = segs[e].y_at(sweep_x);
            if y >= target_y {
                candidate = cur;
                cur = self.nodes[cur as usize].left;
            } else {
                cur = self.nodes[cur as usize].right;
            }
        }
        if candidate == NONE_U32 {
            None
        } else {
            Some(self.nodes[candidate as usize].edge_idx)
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum EventKind {
    Intersection { e1: usize, e2: usize },
    Start { edge: usize },
    Vertical { v_idx: usize },
    End { edge: usize },
}

impl EventKind {
    #[inline]
    fn priority(self) -> u8 {
        match self {
            Self::Intersection { .. } => 0,
            Self::Start { .. } => 1,
            Self::Vertical { .. } => 2,
            Self::End { .. } => 3,
        }
    }

    #[inline]
    fn tie_key(self) -> (usize, usize) {
        match self {
            Self::Intersection { e1, e2 } => (e1, e2),
            Self::Start { edge } | Self::Vertical { v_idx: edge } | Self::End { edge } => (edge, 0),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct SweepEvent {
    x: f64,
    kind: EventKind,
}

impl PartialEq for SweepEvent {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for SweepEvent {}

impl PartialOrd for SweepEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SweepEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        self.x
            .total_cmp(&other.x)
            .then_with(|| self.kind.priority().cmp(&other.kind.priority()))
            .then_with(|| self.kind.tie_key().cmp(&other.kind.tie_key()))
    }
}

/// Exact $O((|E|+K)\log|E|)$ x-sweep edge-crossing counter, angle evaluator, and pair finder.
pub struct SweepLineCrossings;

impl SweepLineCrossings {
    /// Counts all crossing edge pairs in `graph` under `layout`.
    pub fn count<W, S: Scalar>(graph: &CsrGraph<W>, layout: &Layout2d<S>) -> CoreResult<usize> {
        let edges = extract_csr_edges(graph);
        Self::count_edges(&edges, layout)
    }

    /// Computes crossing count and crossing angle statistics in `graph` under `layout`.
    pub fn summarize<W, S: Scalar>(
        graph: &CsrGraph<W>,
        layout: &Layout2d<S>,
    ) -> CoreResult<CrossingSummary> {
        let edges = extract_csr_edges(graph);
        Self::summarize_edges(&edges, layout)
    }

    /// Finds all crossing edge pairs in `graph` under `layout`, sorted deterministically.
    pub fn find_crossings<W, S: Scalar>(
        graph: &CsrGraph<W>,
        layout: &Layout2d<S>,
    ) -> CoreResult<Vec<CrossingPair>> {
        let edges = extract_csr_edges(graph);
        Self::find_edge_crossings(&edges, layout)
    }

    /// Counts all crossing pairs among `edges` under `layout`.
    pub fn count_edges<S: Scalar>(edges: &[(u32, u32)], layout: &Layout2d<S>) -> CoreResult<usize> {
        let segs = build_segments(edges, layout)?;
        let mut count = 0usize;
        sweep_segments_with(&segs, |_, _| {
            count += 1;
        });
        Ok(count)
    }

    /// Summarizes crossing count and crossing angles among `edges` under `layout`.
    pub fn summarize_edges<S: Scalar>(
        edges: &[(u32, u32)],
        layout: &Layout2d<S>,
    ) -> CoreResult<CrossingSummary> {
        let segs = build_segments(edges, layout)?;
        let mut count = 0usize;
        let mut cos_sq_sum = 0.0_f64;
        let mut cos_sq_comp = 0.0_f64;
        let mut min_angle_rad: Option<f64> = None;

        sweep_segments_with(&segs, |a, b| {
            count += 1;
            let (angle_rad, cos_sq) = crossing_angle_metrics(&segs[a], &segs[b]);
            // Use Neumaier compensated summation so accumulation is deterministic (independent
            // of hash-table bucket order) and keeps floating-point error at O(eps) rather than
            // O(K * eps) on high-crossing graphs. Note that IEEE-754 addition order differs from
            // sorting all pairs by (edge_a, edge_b) first, so angle_cos_sq_sum agrees within
            // O(eps * K) while count and min_angle_rad are bit-exact.
            let t = cos_sq_sum + cos_sq;
            cos_sq_comp += if cos_sq_sum.abs() >= cos_sq.abs() {
                (cos_sq_sum - t) + cos_sq
            } else {
                (cos_sq - t) + cos_sq_sum
            };
            cos_sq_sum = t;
            if let Some(ang) = angle_rad {
                min_angle_rad = Some(match min_angle_rad {
                    Some(prev) => prev.min(ang),
                    None => ang,
                });
            }
        });

        Ok(CrossingSummary {
            count,
            angle_cos_sq_sum: cos_sq_sum + cos_sq_comp,
            min_angle_rad,
        })
    }

    /// Finds all crossing pairs among `edges` under `layout`, sorted by `(edge_a, edge_b)`.
    pub fn find_edge_crossings<S: Scalar>(
        edges: &[(u32, u32)],
        layout: &Layout2d<S>,
    ) -> CoreResult<Vec<CrossingPair>> {
        let segs = build_segments(edges, layout)?;
        let reported = sweep_segments(&segs);
        Ok(finalize_pairs(&reported, &segs))
    }
}

fn extract_csr_edges<W>(graph: &CsrGraph<W>) -> Vec<(u32, u32)> {
    let n = graph.num_nodes();
    let directed = graph.is_directed();
    let mut edges = Vec::with_capacity(graph.num_edges());
    for u in 0..n {
        let u32_u = u as u32;
        for &v in graph.neighbors(u) {
            if directed || u32_u < v {
                edges.push((u32_u, v));
            }
        }
    }
    edges
}

fn build_segments<S: Scalar>(edges: &[(u32, u32)], layout: &Layout2d<S>) -> CoreResult<Vec<Seg>> {
    if edges.len() > u32::MAX as usize {
        return Err(CoreError::invalid_input("edge count exceeds u32::MAX"));
    }
    layout.validate_finite()?;
    let n = layout.len();
    let xs = layout.x();
    let ys = layout.y();
    let mut segs = Vec::with_capacity(edges.len());

    for (id, &(u, v)) in edges.iter().enumerate() {
        let ui = u as usize;
        let vi = v as usize;
        if ui >= n {
            return Err(CoreError::index_out_of_bounds(
                ui,
                n,
                "SweepLineCrossings::edge_source",
            ));
        }
        if vi >= n {
            return Err(CoreError::index_out_of_bounds(
                vi,
                n,
                "SweepLineCrossings::edge_target",
            ));
        }
        segs.push(Seg::from_endpoints(
            id,
            u,
            v,
            xs[ui].to_f64(),
            ys[ui].to_f64(),
            xs[vi].to_f64(),
            ys[vi].to_f64(),
        ));
    }
    Ok(segs)
}

#[derive(Debug, Clone, Copy)]
struct OrderedF64(f64);

impl PartialEq for OrderedF64 {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0) == Ordering::Equal
    }
}

impl Eq for OrderedF64 {}

impl PartialOrd for OrderedF64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

trait CrossingCollector {
    fn on_vertical_pair(&mut self, a: usize, b: usize);
    fn on_swap_pair(&mut self, s1: &Seg, s2: &Seg, sweep_x: f64, eps_y: f64, dt_probe: f64);
    fn on_band_pair(
        &mut self,
        s_lo: &Seg,
        s_hi: &Seg,
        sweep_x: f64,
        eps_y: f64,
        dt_probe: f64,
        opt_xi: Option<f64>,
    );
    fn evict_before(&mut self, x_evt: f64);
    fn flush_at_x(&mut self, sweep_x: f64, next_x: f64, segs: &[Seg], eps_y: f64, dt_probe: f64);
}

struct InlinePairSet {
    inline: [u64; 8],
    inline_len: usize,
    overflow: PairSet,
}

impl InlinePairSet {
    fn new() -> Self {
        Self {
            inline: [0; 8],
            inline_len: 0,
            overflow: PairSet::default(),
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.inline_len == 0 && self.overflow.is_empty()
    }

    #[inline]
    fn len(&self) -> usize {
        if self.overflow.is_empty() {
            self.inline_len
        } else {
            self.overflow.len()
        }
    }

    #[inline]
    fn clear(&mut self) {
        self.inline_len = 0;
        if !self.overflow.is_empty() {
            self.overflow.clear();
        }
    }

    #[inline]
    fn contains(&self, pair: u64) -> bool {
        if self.overflow.is_empty() {
            self.inline[..self.inline_len].contains(&pair)
        } else {
            self.overflow.contains(&pair)
        }
    }

    #[inline]
    fn insert(&mut self, pair: u64) -> bool {
        if self.overflow.is_empty() {
            if self.inline[..self.inline_len].contains(&pair) {
                return false;
            }
            if self.inline_len < self.inline.len() {
                self.inline[self.inline_len] = pair;
                self.inline_len += 1;
                return true;
            }
            for &p in &self.inline {
                self.overflow.insert(p);
            }
            self.inline_len = 0;
        }
        self.overflow.insert(pair)
    }

    #[inline]
    fn remove(&mut self, pair: u64) {
        if self.overflow.is_empty() {
            if let Some(pos) = self.inline[..self.inline_len]
                .iter()
                .position(|&p| p == pair)
            {
                self.inline_len -= 1;
                self.inline[pos] = self.inline[self.inline_len];
            }
        } else {
            self.overflow.remove(&pair);
        }
    }
}

struct PairSetCollector {
    reported: PairSet,
}

impl CrossingCollector for PairSetCollector {
    #[inline]
    fn on_vertical_pair(&mut self, a: usize, b: usize) {
        self.reported.insert(pack_pair(a, b));
    }

    #[inline]
    fn on_swap_pair(&mut self, s1: &Seg, s2: &Seg, _sweep_x: f64, _eps_y: f64, _dt_probe: f64) {
        if !s1.shares_endpoint(s2) && segments_geometrically_intersect(s1, s2) {
            self.reported.insert(pack_pair(s1.id, s2.id));
        }
    }

    #[inline]
    fn on_band_pair(
        &mut self,
        s_lo: &Seg,
        s_hi: &Seg,
        _sweep_x: f64,
        _eps_y: f64,
        _dt_probe: f64,
        _opt_xi: Option<f64>,
    ) {
        self.reported.insert(pack_pair(s_lo.id, s_hi.id));
    }

    #[inline]
    fn evict_before(&mut self, _x_evt: f64) {}

    #[inline]
    fn flush_at_x(
        &mut self,
        _sweep_x: f64,
        _next_x: f64,
        _segs: &[Seg],
        _eps_y: f64,
        _dt_probe: f64,
    ) {
    }
}

struct StreamingCollector<F: FnMut(usize, usize)> {
    on_crossing: F,
    at_x_inline: [(u64, u32, u32, f64); 8],
    at_x_inline_len: usize,
    at_x_overflow_set: PairSet,
    at_x_overflow_list: Vec<(u64, u32, u32, f64)>,
    active_band_reported: InlinePairSet,
    evict_heap: BinaryHeap<Reverse<(OrderedF64, u64)>>,
    peak_active_band: usize,
}

impl<F: FnMut(usize, usize)> StreamingCollector<F> {
    fn new(on_crossing: F) -> Self {
        Self {
            on_crossing,
            at_x_inline: [(0, 0, 0, f64::NAN); 8],
            at_x_inline_len: 0,
            at_x_overflow_set: PairSet::default(),
            at_x_overflow_list: Vec::new(),
            active_band_reported: InlinePairSet::new(),
            evict_heap: BinaryHeap::new(),
            peak_active_band: 0,
        }
    }

    #[inline]
    fn contains_at_x(&self, pair: u64) -> bool {
        if self.at_x_overflow_list.is_empty() {
            self.at_x_inline[..self.at_x_inline_len]
                .iter()
                .any(|&(p, _, _, _)| p == pair)
        } else {
            self.at_x_overflow_set.contains(&pair)
        }
    }

    #[inline]
    fn is_reported(&self, pair: u64) -> bool {
        self.contains_at_x(pair)
            || (!self.active_band_reported.is_empty() && self.active_band_reported.contains(pair))
    }

    #[inline]
    fn insert_at_x(&mut self, pair: u64, a_id: u32, b_id: u32, x_base: f64) {
        if self.at_x_overflow_list.is_empty() {
            if self.at_x_inline_len < self.at_x_inline.len() {
                self.at_x_inline[self.at_x_inline_len] = (pair, a_id, b_id, x_base);
                self.at_x_inline_len += 1;
                return;
            }
            for &(p, aid, bid, xb) in &self.at_x_inline {
                self.at_x_overflow_set.insert(p);
                self.at_x_overflow_list.push((p, aid, bid, xb));
            }
            self.at_x_inline_len = 0;
        }
        if self.at_x_overflow_set.insert(pair) {
            self.at_x_overflow_list.push((pair, a_id, b_id, x_base));
        }
    }

    #[inline]
    fn emit_pair_unchecked(&mut self, pair: u64, a: usize, b: usize, x_base: f64) {
        self.insert_at_x(pair, a as u32, b as u32, x_base);
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        (self.on_crossing)(lo, hi);
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn promote_band_pair(
    pair: u64,
    a_id: u32,
    b_id: u32,
    x_base: f64,
    sweep_x: f64,
    next_x: f64,
    segs: &[Seg],
    eps_y: f64,
    dt_probe: f64,
    active_band_reported: &mut InlinePairSet,
    evict_heap: &mut BinaryHeap<Reverse<(OrderedF64, u64)>>,
) {
    if x_base.is_nan() {
        return;
    }
    let a = &segs[a_id as usize];
    let b = &segs[b_id as usize];
    let r = a.x2.min(b.x2);
    if next_x > r || sweep_x >= r {
        return;
    }
    let ds = (a.slope - b.slope).abs();
    let x_evict = if ds == 0.0 {
        r
    } else {
        let dx_rem = next_x - x_base;
        let dy = (a.y_at(sweep_x) - b.y_at(sweep_x)).abs();
        let margin = dy + 2.0 * eps_y + 2.0 * (seg_y_err(a, r) + seg_y_err(b, r));
        if dx_rem > dt_probe {
            let rhs = (dx_rem - dt_probe) * ds;
            if margin < rhs * (1.0 - 4.0 * f64::EPSILON) {
                return;
            }
        }
        r.min(x_base + margin / ds + dt_probe)
    };
    if x_evict >= next_x && active_band_reported.insert(pair) {
        evict_heap.push(Reverse((OrderedF64(x_evict), pair)));
    }
}

impl<F: FnMut(usize, usize)> CrossingCollector for StreamingCollector<F> {
    #[inline]
    fn on_vertical_pair(&mut self, a: usize, b: usize) {
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        (self.on_crossing)(lo, hi);
    }

    #[inline]
    fn on_swap_pair(&mut self, s1: &Seg, s2: &Seg, sweep_x: f64, _eps_y: f64, _dt_probe: f64) {
        if !s1.shares_endpoint(s2) && segments_geometrically_intersect(s1, s2) {
            if s1.slope == s2.slope
                && (s1.x2 - s1.x1) * (s2.y2 - s2.y1) - (s1.y2 - s1.y1) * (s2.x2 - s2.x1) == 0.0
            {
                // Collinear/parallel pairs with det == 0.0 are emitted exclusively at `sweep_x == l`
                // inside `on_band_pair` without cross-x_evt tracking.
                return;
            }
            let pair = pack_pair(s1.id, s2.id);
            if !self.is_reported(pair) {
                let x_base = if s1.slope < s2.slope {
                    intersection_x(s1, s2).map_or(sweep_x, |xi| sweep_x.max(xi))
                } else {
                    sweep_x
                };
                self.emit_pair_unchecked(pair, s1.id, s2.id, x_base);
            }
        }
    }

    #[inline]
    fn on_band_pair(
        &mut self,
        s_lo: &Seg,
        s_hi: &Seg,
        sweep_x: f64,
        eps_y: f64,
        _dt_probe: f64,
        opt_xi: Option<f64>,
    ) {
        match opt_xi {
            None => {
                // det == 0.0 (collinear / parallel, or sub-1e-160 underflow).
                let l = s_lo.x1.max(s_hi.x1);
                if sweep_x == l {
                    let pair = pack_pair(s_lo.id, s_hi.id);
                    if !self.is_reported(pair) {
                        let x_base = if s_lo.slope == s_hi.slope {
                            f64::NAN
                        } else {
                            sweep_x
                        };
                        self.emit_pair_unchecked(pair, s_lo.id, s_hi.id, x_base);
                    }
                }
            }
            Some(xi) => {
                if xi < sweep_x && s_lo.slope <= s_hi.slope {
                    return;
                }
                if xi == sweep_x || within_band(s_lo, s_hi, sweep_x, eps_y) {
                    let pair = pack_pair(s_lo.id, s_hi.id);
                    if !self.is_reported(pair) {
                        self.emit_pair_unchecked(pair, s_lo.id, s_hi.id, sweep_x.max(xi));
                    }
                }
            }
        }
    }

    fn evict_before(&mut self, x_evt: f64) {
        while let Some(&Reverse((OrderedF64(ex), pair))) = self.evict_heap.peek() {
            if ex < x_evt {
                self.evict_heap.pop();
                self.active_band_reported.remove(pair);
            } else {
                break;
            }
        }
    }

    fn flush_at_x(&mut self, sweep_x: f64, next_x: f64, segs: &[Seg], eps_y: f64, dt_probe: f64) {
        if self.at_x_inline_len == 0 && self.at_x_overflow_list.is_empty() {
            return;
        }
        if next_x == f64::INFINITY {
            self.at_x_inline_len = 0;
            if !self.at_x_overflow_list.is_empty() {
                self.at_x_overflow_set.clear();
                self.at_x_overflow_list.clear();
            }
            return;
        }
        if self.at_x_overflow_list.is_empty() {
            for i in 0..self.at_x_inline_len {
                let (pair, a_id, b_id, x_base) = self.at_x_inline[i];
                promote_band_pair(
                    pair,
                    a_id,
                    b_id,
                    x_base,
                    sweep_x,
                    next_x,
                    segs,
                    eps_y,
                    dt_probe,
                    &mut self.active_band_reported,
                    &mut self.evict_heap,
                );
            }
            self.at_x_inline_len = 0;
        } else {
            for &(pair, a_id, b_id, x_base) in &self.at_x_overflow_list {
                promote_band_pair(
                    pair,
                    a_id,
                    b_id,
                    x_base,
                    sweep_x,
                    next_x,
                    segs,
                    eps_y,
                    dt_probe,
                    &mut self.active_band_reported,
                    &mut self.evict_heap,
                );
            }
            self.at_x_overflow_set.clear();
            self.at_x_overflow_list.clear();
        }
        self.peak_active_band = self.peak_active_band.max(self.active_band_reported.len());
    }
}

fn sweep_segments(segs: &[Seg]) -> PairSet {
    let mut collector = PairSetCollector {
        reported: PairSet::default(),
    };
    sweep_segments_core(segs, &mut collector);
    collector.reported
}

fn sweep_segments_with<F: FnMut(usize, usize)>(segs: &[Seg], on_crossing: F) {
    let _ = sweep_segments_with_stats(segs, on_crossing);
}

fn sweep_segments_with_stats<F: FnMut(usize, usize)>(segs: &[Seg], on_crossing: F) -> usize {
    let mut collector = StreamingCollector::new(on_crossing);
    sweep_segments_core(segs, &mut collector);
    collector.peak_active_band
}

fn sweep_segments_core<C: CrossingCollector>(segs: &[Seg], collector: &mut C) {
    if segs.len() < 2 {
        return;
    }

    let mut max_coord = 1.0_f64;
    let mut non_vert_count = 0usize;
    let mut vert_indices = Vec::new();

    for s in segs {
        if s.u == s.v {
            continue;
        }
        max_coord = max_coord
            .max(s.x1.abs())
            .max(s.x2.abs())
            .max(s.y1.abs())
            .max(s.y2.abs());
        if s.is_vertical {
            vert_indices.push(s.id);
        } else {
            non_vert_count += 1;
        }
    }

    let mut max_seg_y_err = 0.0_f64;
    let eps_y = 1e-10 * max_coord + 1e-12;
    for s in segs {
        if s.u != s.v && !s.is_vertical {
            max_seg_y_err =
                max_seg_y_err.max(seg_y_err(s, s.x_max).min((s.y_max - s.y_min) + eps_y));
        }
    }

    // 1. Check vertical-vs-vertical crossings via 1D sweep grouped by exact x coordinate.
    if vert_indices.len() >= 2 {
        vert_indices.sort_unstable_by(|&a, &b| {
            segs[a]
                .x1
                .total_cmp(&segs[b].x1)
                .then_with(|| segs[a].y_min.total_cmp(&segs[b].y_min))
                .then_with(|| segs[a].y_max.total_cmp(&segs[b].y_max))
                .then_with(|| a.cmp(&b))
        });

        let mut i = 0;
        let mut active_y: Vec<usize> = Vec::new();
        while i < vert_indices.len() {
            let x_cur = segs[vert_indices[i]].x1;
            let mut j = i;
            while j < vert_indices.len() && segs[vert_indices[j]].x1 == x_cur {
                j += 1;
            }
            if j - i >= 2 {
                active_y.clear();
                for &v_id in &vert_indices[i..j] {
                    let s_cur = &segs[v_id];
                    active_y.retain(|&prev_id| segs[prev_id].y_max >= s_cur.y_min);
                    for &prev_id in &active_y {
                        if segments_cross(s_cur, &segs[prev_id]) {
                            collector.on_vertical_pair(v_id, prev_id);
                        }
                    }
                    active_y.push(v_id);
                }
            }
            i = j;
        }
    }

    if non_vert_count == 0 {
        return;
    }

    // 2. X-sweep over non-vertical segments + vertical queries at exact x_v.
    let mut events: BinaryHeap<Reverse<SweepEvent>> =
        BinaryHeap::with_capacity(non_vert_count * 2 + vert_indices.len());
    for s in segs {
        if s.u == s.v || s.is_vertical {
            continue;
        }
        events.push(Reverse(SweepEvent {
            x: s.x1,
            kind: EventKind::Start { edge: s.id },
        }));
        events.push(Reverse(SweepEvent {
            x: s.x2,
            kind: EventKind::End { edge: s.id },
        }));
    }
    for &v_idx in &vert_indices {
        events.push(Reverse(SweepEvent {
            x: segs[v_idx].x1,
            kind: EventKind::Vertical { v_idx },
        }));
    }

    let mut tree = ActiveSegmentTree::new(segs.len());
    let mut started_batch: Vec<usize> = Vec::new();
    let mut end_batch: Vec<usize> = Vec::new();
    let mut vert_batch: Vec<usize> = Vec::new();
    let mut touched_edges: Vec<usize> = Vec::new();
    let mut scratch = BandScratch::new(segs.len(), (max_coord * 1e-11).max(1e-12));

    while let Some(Reverse(first)) = events.peek().copied() {
        let x_evt = first.x;
        if x_evt != scratch.last_int_x {
            scratch.processed_int_at_x.clear();
            scratch.scheduled_at_x.clear();
            scratch.last_int_x = x_evt;
            collector.evict_before(x_evt);
        }
        started_batch.clear();
        end_batch.clear();
        vert_batch.clear();
        touched_edges.clear();

        // Drain all events sharing exact x_evt, processing Intersections first so crossing
        // active pairs settle into post-crossing slope order before Starts/Verticals.
        while let Some(Reverse(ev)) = events.peek().copied() {
            if ev.x != x_evt {
                break;
            }
            events.pop();
            match ev.kind {
                EventKind::Intersection { e1, e2 } => {
                    handle_intersection(
                        e1,
                        e2,
                        x_evt,
                        eps_y,
                        segs,
                        &mut tree,
                        &mut events,
                        collector,
                        &mut touched_edges,
                        &mut scratch,
                    );
                }
                EventKind::Start { edge } => started_batch.push(edge),
                EventKind::Vertical { v_idx } => vert_batch.push(v_idx),
                EventKind::End { edge } => end_batch.push(edge),
            }
        }

        // Insert all segments starting at x_evt before checking clusters so co-located stars,
        // ending tips, and passing segments at (x_evt, y) are evaluated simultaneously.
        for &edge in &started_batch {
            tree.insert(edge, x_evt, segs);
        }
        touched_edges.extend_from_slice(&started_batch);
        touched_edges.extend_from_slice(&end_batch);

        if !touched_edges.is_empty() {
            check_eps_bands_at_x(
                &touched_edges,
                x_evt,
                eps_y,
                segs,
                &tree,
                &mut events,
                collector,
                &mut scratch,
            );
        }

        // Check vertical segments against active non-vertical segments, accounting for slope ULP drift.
        let vert_margin = eps_y + max_seg_y_err;
        for &v_idx in &vert_batch {
            let v_seg = &segs[v_idx];
            let lo = v_seg.y_min - vert_margin;
            let hi = v_seg.y_max + vert_margin;
            if let Some(mut cur_edge) = tree.lower_bound_y(lo, x_evt, segs) {
                while let Some(prev) = tree.prev_edge(cur_edge) {
                    if segs[prev].y_at(x_evt) < lo {
                        break;
                    }
                    cur_edge = prev;
                }
                loop {
                    let s_cur = &segs[cur_edge];
                    if s_cur.y_at(x_evt) > hi {
                        break;
                    }
                    if segments_cross(v_seg, s_cur) {
                        collector.on_vertical_pair(v_idx, cur_edge);
                    }
                    let Some(nx) = tree.next_edge(cur_edge) else {
                        break;
                    };
                    cur_edge = nx;
                }
            }
        }

        // Remove ending segments, collect surviving touched peers, and enforce post-x_evt
        // slope order across each local cluster so shared-coord ties never block future pairs.
        scratch.surviving_touched.clear();
        for &e in &touched_edges {
            if tree.is_active(e) {
                scratch.surviving_touched.push(e);
            }
        }
        for &edge in &end_batch {
            let (prev, next) = tree.remove_by_edge(edge);
            if let (Some(p), Some(n)) = (prev, next) {
                check_adjacent_and_schedule(
                    p,
                    n,
                    x_evt,
                    eps_y,
                    segs,
                    &mut events,
                    collector,
                    scratch.dt_probe,
                    &scratch.processed_int_at_x,
                    &mut scratch.scheduled_at_x,
                );
                scratch.surviving_touched.push(p);
                scratch.surviving_touched.push(n);
            }
        }
        if !scratch.surviving_touched.is_empty() {
            reorder_surviving_bands_at_x(
                x_evt,
                eps_y,
                segs,
                &mut tree,
                &mut events,
                collector,
                &mut scratch,
            );
        }

        let next_x = events.peek().map_or(f64::INFINITY, |e| e.0.x);
        if next_x != x_evt {
            collector.flush_at_x(x_evt, next_x, segs, eps_y, scratch.dt_probe);
        }
    }
}

struct BandScratch {
    dt_probe: f64,
    cluster: Vec<(usize, usize)>,
    hub_buf: Vec<(usize, usize)>,
    rest_buf: Vec<(usize, usize)>,
    cur_buf: Vec<(usize, usize)>,
    band_edges: Vec<usize>,
    band_nodes: Vec<u32>,
    surviving_touched: Vec<usize>,
    visited_stamp: Vec<u64>,
    visited_err: Vec<f64>,
    stamp: u64,
    processed_int_at_x: InlinePairSet,
    scheduled_at_x: InlinePairSet,
    last_int_x: f64,
}

impl BandScratch {
    fn new(num_edges: usize, dt_probe: f64) -> Self {
        Self {
            dt_probe,
            cluster: Vec::new(),
            hub_buf: Vec::new(),
            rest_buf: Vec::new(),
            cur_buf: Vec::new(),
            band_edges: Vec::new(),
            band_nodes: Vec::new(),
            surviving_touched: Vec::new(),
            visited_stamp: vec![0; num_edges],
            visited_err: vec![0.0; num_edges],
            stamp: 0,
            processed_int_at_x: InlinePairSet::new(),
            scheduled_at_x: InlinePairSet::new(),
            last_int_x: f64::NAN,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_flipped_pairs<C: CrossingCollector>(
    x_evt: f64,
    eps_y: f64,
    segs: &[Seg],
    collector: &mut C,
    scratch: &mut BandScratch,
) {
    let k = scratch.cluster.len();
    if k <= 1 {
        return;
    }
    if k <= 64 {
        for i in 0..k {
            let (pos_u, edge_u) = scratch.cluster[i];
            for j in (i + 1)..k {
                let (pos_v, edge_v) = scratch.cluster[j];
                if pos_u > pos_v {
                    collector.on_swap_pair(
                        &segs[edge_v],
                        &segs[edge_u],
                        x_evt,
                        eps_y,
                        scratch.dt_probe,
                    );
                }
            }
        }
        return;
    }

    // Bottom-up merge sort on `orig_pos` using `cur_buf` / `rest_buf` so k > 64 remains
    // O(k log k + inversions) without per-event allocations.
    scratch.cur_buf.clear();
    scratch.cur_buf.extend_from_slice(&scratch.cluster);
    scratch.rest_buf.resize(k, (0, 0));

    let mut width = 1usize;
    while width < k {
        let mut lo = 0usize;
        while lo < k {
            let mid = (lo + width).min(k);
            let hi = (lo + 2 * width).min(k);
            let mut l = lo;
            let mut r = mid;
            let mut out = lo;
            while l < mid && r < hi {
                if scratch.cur_buf[l].0 <= scratch.cur_buf[r].0 {
                    scratch.rest_buf[out] = scratch.cur_buf[l];
                    l += 1;
                } else {
                    let (_, edge_v) = scratch.cur_buf[r];
                    for &(_, edge_u) in &scratch.cur_buf[l..mid] {
                        collector.on_swap_pair(
                            &segs[edge_v],
                            &segs[edge_u],
                            x_evt,
                            eps_y,
                            scratch.dt_probe,
                        );
                    }
                    scratch.rest_buf[out] = scratch.cur_buf[r];
                    r += 1;
                }
                out += 1;
            }
            while l < mid {
                scratch.rest_buf[out] = scratch.cur_buf[l];
                l += 1;
                out += 1;
            }
            while r < hi {
                scratch.rest_buf[out] = scratch.cur_buf[r];
                r += 1;
                out += 1;
            }
            lo += 2 * width;
        }
        std::mem::swap(&mut scratch.cur_buf, &mut scratch.rest_buf);
        width *= 2;
    }
}

/// Sorts `scratch.cluster` into exact post-`x_evt` order via `cmp_post_x`, resolves any local
/// cyclic adjacent inversion spans by `(slope, id)`, notifies `collector.on_swap_pair` for every
/// flipped pair, and writes the reordered edge indices back to `tree` and `scratch.band_edges`.
#[allow(clippy::too_many_arguments)]
fn sort_band_post_x<C: CrossingCollector>(
    x_evt: f64,
    eps_y: f64,
    segs: &[Seg],
    tree: &mut ActiveSegmentTree,
    collector: &mut C,
    scratch: &mut BandScratch,
) {
    let k = scratch.cluster.len();
    if k <= 32 {
        // Insertion sort tolerates rare O(ULP) 3-cycles in `cmp_post_x` without triggering
        // `slice::sort_unstable_by` total-order assertions.
        for i in 1..k {
            let key = scratch.cluster[i];
            let mut j = i;
            while j > 0
                && cmp_post_x(&segs[key.1], &segs[scratch.cluster[j - 1].1], x_evt)
                    == Ordering::Less
            {
                scratch.cluster[j] = scratch.cluster[j - 1];
                j -= 1;
            }
            scratch.cluster[j] = key;
        }
    } else {
        // Bottom-up merge sort using `cur_buf` / `rest_buf` (cycle-safe, zero allocations).
        scratch.cur_buf.clear();
        scratch.cur_buf.extend_from_slice(&scratch.cluster);
        scratch.rest_buf.resize(k, (0, 0));
        let mut width = 1usize;
        while width < k {
            let mut lo = 0usize;
            while lo < k {
                let mid = (lo + width).min(k);
                let hi = (lo + 2 * width).min(k);
                let mut l = lo;
                let mut r = mid;
                let mut out = lo;
                while l < mid && r < hi {
                    if cmp_post_x(
                        &segs[scratch.cur_buf[l].1],
                        &segs[scratch.cur_buf[r].1],
                        x_evt,
                    ) != Ordering::Greater
                    {
                        scratch.rest_buf[out] = scratch.cur_buf[l];
                        l += 1;
                    } else {
                        scratch.rest_buf[out] = scratch.cur_buf[r];
                        r += 1;
                    }
                    out += 1;
                }
                while l < mid {
                    scratch.rest_buf[out] = scratch.cur_buf[l];
                    l += 1;
                    out += 1;
                }
                while r < hi {
                    scratch.rest_buf[out] = scratch.cur_buf[r];
                    r += 1;
                    out += 1;
                }
                lo += 2 * width;
            }
            std::mem::swap(&mut scratch.cur_buf, &mut scratch.rest_buf);
            width *= 2;
        }
        scratch.cluster.clear();
        scratch.cluster.extend_from_slice(&scratch.cur_buf);
    }
    let mut pass = 0usize;
    loop {
        let mut had_inversion = false;
        let mut i = 0usize;
        while i + 1 < k {
            if cmp_post_x(
                &segs[scratch.cluster[i].1],
                &segs[scratch.cluster[i + 1].1],
                x_evt,
            ) == Ordering::Greater
            {
                had_inversion = true;
                let lo = i;
                let mut hi = i + 1;
                while hi + 1 < k
                    && cmp_post_x(
                        &segs[scratch.cluster[hi].1],
                        &segs[scratch.cluster[hi + 1].1],
                        x_evt,
                    ) == Ordering::Greater
                {
                    hi += 1;
                }
                if hi > lo + 1 {
                    scratch.cluster[lo..=hi].sort_unstable_by(|a, b| {
                        segs[a.1]
                            .slope
                            .total_cmp(&segs[b.1].slope)
                            .then_with(|| a.1.cmp(&b.1))
                    });
                } else {
                    scratch.cluster.swap(lo, hi);
                }
                i = hi + 1;
            } else {
                i += 1;
            }
        }
        if !had_inversion {
            break;
        }
        pass += 1;
        if pass >= k {
            scratch.cluster.sort_unstable_by(|a, b| {
                segs[a.1]
                    .slope
                    .total_cmp(&segs[b.1].slope)
                    .then_with(|| a.1.cmp(&b.1))
            });
            break;
        }
    }

    emit_flipped_pairs(x_evt, eps_y, segs, collector, scratch);

    scratch.band_edges.clear();
    for (&node_id, &(_, edge_id)) in scratch.band_nodes.iter().zip(scratch.cluster.iter()) {
        tree.nodes[node_id as usize].edge_idx = edge_id;
        tree.node_of_edge[edge_id] = node_id;
        scratch.band_edges.push(edge_id);
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_intersection<C: CrossingCollector>(
    e1: usize,
    e2: usize,
    x_evt: f64,
    eps_y: f64,
    segs: &[Seg],
    tree: &mut ActiveSegmentTree,
    events: &mut BinaryHeap<Reverse<SweepEvent>>,
    collector: &mut C,
    touched_edges: &mut Vec<usize>,
    scratch: &mut BandScratch,
) {
    if !tree.is_active(e1) || !tree.is_active(e2) {
        return;
    }
    let Some(r1) = tree.rank_of_edge(e1) else {
        return;
    };
    let Some(r2) = tree.rank_of_edge(e2) else {
        return;
    };
    if r1 >= r2 {
        return;
    }

    if r1 + 1 == r2 {
        tree.swap_adjacent_edges(e1, e2);
        collector.on_swap_pair(&segs[e1], &segs[e2], x_evt, eps_y, scratch.dt_probe);
        scratch.processed_int_at_x.insert(pack_pair(e1, e2));
        // After swap, `e2` precedes `e1`.
        let left = tree.prev_edge(e2);
        let right = tree.next_edge(e1);
        let s1 = &segs[e1];
        let s2 = &segs[e2];
        let y1 = s1.y_at(x_evt);
        let y2 = s2.y_at(x_evt);
        let err1 = seg_y_err(s1, x_evt);
        let err2 = seg_y_err(s2, x_evt);
        let left_isolated = match left {
            None => true,
            Some(l) => {
                let sl = &segs[l];
                let yl = sl.y_at(x_evt);
                let err_l = seg_y_err(sl, x_evt);
                y2 - yl > 2.0 * eps_y + err_l + err2 && y1 - yl > 2.0 * eps_y + err_l + err1
            }
        };
        let right_isolated = match right {
            None => true,
            Some(r) => {
                let sr = &segs[r];
                let yr = sr.y_at(x_evt);
                let err_r = seg_y_err(sr, x_evt);
                yr - y1 > 2.0 * eps_y + err1 + err_r && yr - y2 > 2.0 * eps_y + err2 + err_r
            }
        };
        if !left_isolated || !right_isolated {
            touched_edges.push(e1);
            touched_edges.push(e2);
        }
        if let Some(left) = left {
            check_adjacent_and_schedule(
                left,
                e2,
                x_evt,
                eps_y,
                segs,
                events,
                collector,
                scratch.dt_probe,
                &scratch.processed_int_at_x,
                &mut scratch.scheduled_at_x,
            );
        }
        if let Some(right) = right {
            check_adjacent_and_schedule(
                e1,
                right,
                x_evt,
                eps_y,
                segs,
                events,
                collector,
                scratch.dt_probe,
                &scratch.processed_int_at_x,
                &mut scratch.scheduled_at_x,
            );
        }
        return;
    }

    // Multi-way intersection or 1-ULP tie: sort [e1..=e2] by exact post-x_evt order.
    scratch.band_nodes.clear();
    scratch.band_edges.clear();
    let mut cur_edge = e1;
    let span = r2 - r1;
    for _ in 0..=span {
        scratch.band_nodes.push(tree.node_of_edge[cur_edge]);
        scratch.band_edges.push(cur_edge);
        if cur_edge == e2 {
            break;
        }
        let Some(nx) = tree.next_edge(cur_edge) else {
            break;
        };
        cur_edge = nx;
    }

    if scratch.band_edges.last() != Some(&e2) {
        return;
    }

    scratch.processed_int_at_x.insert(pack_pair(e1, e2));
    touched_edges.push(e1);
    touched_edges.push(e2);

    // Check pairwise crossings inside the multi-way band while pruning shared-hub star pairs.
    scratch.cluster.clear();
    for (pos, &edge) in scratch.band_edges.iter().enumerate() {
        scratch.cluster.push((pos, edge));
    }
    check_band_pairs(x_evt, eps_y, segs, events, collector, scratch);
    sort_band_post_x(x_evt, eps_y, segs, tree, collector, scratch);

    if let (Some(&first_e), Some(&last_e)) = (scratch.band_edges.first(), scratch.band_edges.last())
    {
        if let Some(left) = tree.prev_edge(first_e) {
            check_adjacent_and_schedule(
                left,
                first_e,
                x_evt,
                eps_y,
                segs,
                events,
                collector,
                scratch.dt_probe,
                &scratch.processed_int_at_x,
                &mut scratch.scheduled_at_x,
            );
        }
        if let Some(right) = tree.next_edge(last_e) {
            check_adjacent_and_schedule(
                last_e,
                right,
                x_evt,
                eps_y,
                segs,
                events,
                collector,
                scratch.dt_probe,
                &scratch.processed_int_at_x,
                &mut scratch.scheduled_at_x,
            );
        }
    }
    for win in scratch.band_edges.windows(2) {
        scratch.processed_int_at_x.insert(pack_pair(win[0], win[1]));
        check_adjacent_and_schedule(
            win[0],
            win[1],
            x_evt,
            eps_y,
            segs,
            events,
            collector,
            scratch.dt_probe,
            &scratch.processed_int_at_x,
            &mut scratch.scheduled_at_x,
        );
    }
}

fn collect_band_around_seed(
    seed: usize,
    x_evt: f64,
    eps_y: f64,
    segs: &[Seg],
    tree: &ActiveSegmentTree,
    out_edges: &mut Vec<usize>,
) {
    out_edges.clear();
    let s_seed = &segs[seed];
    let mut left_most = seed;
    let mut cur = seed;
    while let Some(prev) = tree.prev_edge(cur) {
        if !within_band(s_seed, &segs[prev], x_evt, eps_y) {
            break;
        }
        left_most = prev;
        cur = prev;
    }

    out_edges.push(left_most);
    let mut cur_r = left_most;
    while let Some(nx) = tree.next_edge(cur_r) {
        if !within_band(s_seed, &segs[nx], x_evt, eps_y) {
            break;
        }
        out_edges.push(nx);
        cur_r = nx;
    }
}

#[allow(clippy::too_many_arguments)]
fn check_eps_bands_at_x<C: CrossingCollector>(
    touched_edges: &[usize],
    x_evt: f64,
    eps_y: f64,
    segs: &[Seg],
    tree: &ActiveSegmentTree,
    events: &mut BinaryHeap<Reverse<SweepEvent>>,
    collector: &mut C,
    scratch: &mut BandScratch,
) {
    scratch.stamp = scratch.stamp.wrapping_add(1).max(1);
    let cur_stamp = scratch.stamp;

    for &seed in touched_edges {
        if !tree.is_active(seed) {
            continue;
        }
        let seed_err = seg_y_err(&segs[seed], x_evt);
        if scratch.visited_stamp[seed] == cur_stamp && seed_err <= scratch.visited_err[seed] + eps_y
        {
            continue;
        }

        collect_band_around_seed(seed, x_evt, eps_y, segs, tree, &mut scratch.band_edges);
        for &e in &scratch.band_edges {
            if scratch.visited_stamp[e] != cur_stamp || seed_err > scratch.visited_err[e] {
                scratch.visited_stamp[e] = cur_stamp;
                scratch.visited_err[e] = seed_err;
            }
        }

        let first_edge = scratch.band_edges[0];
        let last_edge = scratch.band_edges[scratch.band_edges.len() - 1];
        if let Some(left) = tree.prev_edge(first_edge) {
            check_adjacent_and_schedule(
                left,
                first_edge,
                x_evt,
                eps_y,
                segs,
                events,
                collector,
                scratch.dt_probe,
                &scratch.processed_int_at_x,
                &mut scratch.scheduled_at_x,
            );
        }
        if let Some(right) = tree.next_edge(last_edge) {
            check_adjacent_and_schedule(
                last_edge,
                right,
                x_evt,
                eps_y,
                segs,
                events,
                collector,
                scratch.dt_probe,
                &scratch.processed_int_at_x,
                &mut scratch.scheduled_at_x,
            );
        }

        if scratch.band_edges.len() >= 2 {
            scratch.cluster.clear();
            for (pos, &e) in scratch.band_edges.iter().enumerate() {
                scratch.cluster.push((pos, e));
            }
            check_band_pairs(x_evt, eps_y, segs, events, collector, scratch);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn reorder_surviving_bands_at_x<C: CrossingCollector>(
    x_evt: f64,
    eps_y: f64,
    segs: &[Seg],
    tree: &mut ActiveSegmentTree,
    events: &mut BinaryHeap<Reverse<SweepEvent>>,
    collector: &mut C,
    scratch: &mut BandScratch,
) {
    scratch.stamp = scratch.stamp.wrapping_add(1).max(1);
    let cur_stamp = scratch.stamp;

    for idx in 0..scratch.surviving_touched.len() {
        let seed = scratch.surviving_touched[idx];
        if !tree.is_active(seed) {
            continue;
        }
        let seed_err = seg_y_err(&segs[seed], x_evt);
        if scratch.visited_stamp[seed] == cur_stamp && seed_err <= scratch.visited_err[seed] + eps_y
        {
            continue;
        }

        collect_band_around_seed(seed, x_evt, eps_y, segs, tree, &mut scratch.band_edges);
        for &e in &scratch.band_edges {
            if scratch.visited_stamp[e] != cur_stamp || seed_err > scratch.visited_err[e] {
                scratch.visited_stamp[e] = cur_stamp;
                scratch.visited_err[e] = seed_err;
            }
        }
        if scratch.band_edges.len() < 2 {
            continue;
        }

        let is_sorted = scratch
            .band_edges
            .windows(2)
            .all(|w| cmp_post_x(&segs[w[0]], &segs[w[1]], x_evt) != Ordering::Greater);
        if is_sorted {
            continue;
        }

        // Check all band pairs before permuting so near-coincident crossings are never lost.
        scratch.cluster.clear();
        scratch.band_nodes.clear();
        for (pos, &e) in scratch.band_edges.iter().enumerate() {
            scratch.cluster.push((pos, e));
            scratch.band_nodes.push(tree.node_of_edge[e]);
        }
        check_band_pairs(x_evt, eps_y, segs, events, collector, scratch);
        sort_band_post_x(x_evt, eps_y, segs, tree, collector, scratch);

        if let (Some(&first_e), Some(&last_e)) =
            (scratch.band_edges.first(), scratch.band_edges.last())
        {
            if let Some(left) = tree.prev_edge(first_e) {
                check_adjacent_and_schedule(
                    left,
                    first_e,
                    x_evt,
                    eps_y,
                    segs,
                    events,
                    collector,
                    scratch.dt_probe,
                    &scratch.processed_int_at_x,
                    &mut scratch.scheduled_at_x,
                );
            }
            if let Some(right) = tree.next_edge(last_e) {
                check_adjacent_and_schedule(
                    last_e,
                    right,
                    x_evt,
                    eps_y,
                    segs,
                    events,
                    collector,
                    scratch.dt_probe,
                    &scratch.processed_int_at_x,
                    &mut scratch.scheduled_at_x,
                );
            }
        }
        for win in scratch.band_edges.windows(2) {
            scratch.processed_int_at_x.insert(pack_pair(win[0], win[1]));
            check_adjacent_and_schedule(
                win[0],
                win[1],
                x_evt,
                eps_y,
                segs,
                events,
                collector,
                scratch.dt_probe,
                &scratch.processed_int_at_x,
                &mut scratch.scheduled_at_x,
            );
        }
    }
}

/// Checks all crossing pairs inside `scratch.cluster` while pruning same-vertex star fans
/// via 4-slot Misra-Gries heavy-hitter detection so pure or nested stars remain $O(|C|)$.
fn check_band_pairs<C: CrossingCollector>(
    x_evt: f64,
    eps_y: f64,
    segs: &[Seg],
    events: &mut BinaryHeap<Reverse<SweepEvent>>,
    collector: &mut C,
    scratch: &mut BandScratch,
) {
    scratch.cur_buf.clear();
    scratch.cur_buf.extend_from_slice(&scratch.cluster);

    while scratch.cur_buf.len() >= 2 {
        let n = scratch.cur_buf.len();
        if n > 32
            && let Some(hub_v) = dominant_shared_vertex(&scratch.cur_buf, segs)
        {
            scratch.hub_buf.clear();
            scratch.rest_buf.clear();
            for &(pos, e) in &scratch.cur_buf {
                let s = &segs[e];
                if s.u == hub_v || s.v == hub_v {
                    scratch.hub_buf.push((pos, e));
                } else {
                    scratch.rest_buf.push((pos, e));
                }
            }
            // Segments in `hub_buf` share `hub_v` at the cluster point and cannot report crossings.
            for &(pos_h, edge_h) in &scratch.hub_buf {
                for &(pos_r, edge_r) in &scratch.rest_buf {
                    let (lo, hi) = if pos_h < pos_r {
                        (edge_h, edge_r)
                    } else {
                        (edge_r, edge_h)
                    };
                    check_adjacent_and_schedule(
                        lo,
                        hi,
                        x_evt,
                        eps_y,
                        segs,
                        events,
                        collector,
                        scratch.dt_probe,
                        &scratch.processed_int_at_x,
                        &mut scratch.scheduled_at_x,
                    );
                }
            }
            std::mem::swap(&mut scratch.cur_buf, &mut scratch.rest_buf);
            continue;
        }

        for i in 0..n {
            let (pos_a, edge_a) = scratch.cur_buf[i];
            for j in (i + 1)..n {
                let (pos_b, edge_b) = scratch.cur_buf[j];
                let (lo, hi) = if pos_a < pos_b {
                    (edge_a, edge_b)
                } else {
                    (edge_b, edge_a)
                };
                check_adjacent_and_schedule(
                    lo,
                    hi,
                    x_evt,
                    eps_y,
                    segs,
                    events,
                    collector,
                    scratch.dt_probe,
                    &scratch.processed_int_at_x,
                    &mut scratch.scheduled_at_x,
                );
            }
        }
        break;
    }
}

fn dominant_shared_vertex(items: &[(usize, usize)], segs: &[Seg]) -> Option<u32> {
    let mut slots: [(Option<u32>, usize); 4] = [(None, 0); 4];
    for &(_, e) in items {
        let s = &segs[e];
        for v in [s.u, s.v] {
            if let Some((_, cnt)) = slots.iter_mut().find(|(c, _)| *c == Some(v)) {
                *cnt += 1;
            } else if let Some(slot) = slots.iter_mut().find(|(_, cnt)| *cnt == 0) {
                *slot = (Some(v), 1);
            } else {
                for slot in &mut slots {
                    slot.1 -= 1;
                    if slot.1 == 0 {
                        slot.0 = None;
                    }
                }
            }
        }
    }

    let mut best_v = None;
    let mut best_cnt = 0usize;
    for (cand, _) in slots {
        let Some(v) = cand else {
            continue;
        };
        let cnt = items
            .iter()
            .filter(|&&(_, e)| segs[e].u == v || segs[e].v == v)
            .count();
        if cnt > best_cnt {
            best_cnt = cnt;
            best_v = Some(v);
        }
    }
    if best_cnt >= 4 { best_v } else { None }
}

#[allow(clippy::too_many_arguments)]
fn check_adjacent_and_schedule<C: CrossingCollector>(
    lower: usize,
    upper: usize,
    sweep_x: f64,
    eps_y: f64,
    segs: &[Seg],
    events: &mut BinaryHeap<Reverse<SweepEvent>>,
    collector: &mut C,
    dt_probe: f64,
    processed_int_at_x: &InlinePairSet,
    scheduled_at_x: &mut InlinePairSet,
) {
    let s_lo = &segs[lower];
    let s_hi = &segs[upper];
    let shares_ep = s_lo.shares_endpoint(s_hi);
    if shares_ep && s_lo.slope <= s_hi.slope {
        return;
    }
    if !segments_geometrically_intersect(s_lo, s_hi) {
        return;
    }
    let can_schedule = s_lo.x2 > sweep_x
        && s_hi.x2 > sweep_x
        && s_lo.slope > s_hi.slope
        && !(s_lo.x2 == s_hi.x2 && s_lo.y2 == s_hi.y2);
    if shares_ep && !can_schedule {
        return;
    }
    let opt_xi = intersection_x(s_lo, s_hi);
    if !shares_ep {
        collector.on_band_pair(s_lo, s_hi, sweep_x, eps_y, dt_probe, opt_xi);
    }
    // Schedule interior swaps even when segments share graph vertex IDs (e.g. wrapped torus pieces)
    // so sweep order stays valid past interior crossings, while skipping shared right tips.
    if can_schedule && let Some(x_int) = opt_xi {
        let x_push = x_int.max(sweep_x);
        let pair = pack_pair(lower, upper);
        let should_push = if x_push > sweep_x {
            scheduled_at_x.insert(pair)
        } else {
            !processed_int_at_x.contains(pair) && scheduled_at_x.insert(pair)
        };
        if should_push {
            events.push(Reverse(SweepEvent {
                x: x_push,
                kind: EventKind::Intersection {
                    e1: lower,
                    e2: upper,
                },
            }));
        }
    }
}

fn finalize_pairs(reported: &PairSet, segs: &[Seg]) -> Vec<CrossingPair> {
    let mut pairs: Vec<CrossingPair> = reported
        .iter()
        .map(|&key| {
            let (a, b) = unpack_pair(key);
            let sa = &segs[a];
            let sb = &segs[b];
            let (angle_rad, cos_sq) = crossing_angle_metrics(sa, sb);
            CrossingPair {
                edge_a: a,
                edge_b: b,
                u1: sa.u,
                v1: sa.v,
                u2: sb.u,
                v2: sb.v,
                angle_rad,
                cos_sq,
            }
        })
        .collect();
    pairs.sort_unstable_by_key(|p| (p.edge_a, p.edge_b));
    pairs
}

/// Splits a 2D torus edge `(x1, y1) -> (x2, y2)` into 1, 2, or 3 Euclidean segments on `[0, 1]^2`
/// matching legacy `DrawingTorus2d::edge_segments` wrap-around conventions.
pub fn torus_edge_segments<S: Scalar>(x1: S, y1: S, x2: S, y2: S) -> CoreResult<Vec<[f64; 4]>> {
    if !x1.is_finite() || !y1.is_finite() || !x2.is_finite() || !y2.is_finite() {
        return Err(CoreError::non_finite("torus_edge_segments"));
    }
    let px = wrap_torus_coord(x1).to_f64();
    let py = wrap_torus_coord(y1).to_f64();
    let qx = wrap_torus_coord(x2).to_f64();
    let qy = wrap_torus_coord(y2).to_f64();

    // Find periodic shift (dx, dy) in {-1, 0, +1}^2 minimizing Euclidean distance (px + dx - qx, py + dy - qy).
    let mut best_d = f64::INFINITY;
    let mut best_dx = 0_i32;
    let mut best_dy = 0_i32;
    for dy in -1..=1 {
        let y_shift = py + dy as f64;
        for dx in -1..=1 {
            let x_shift = px + dx as f64;
            let d = (x_shift - qx).hypot(y_shift - qy);
            if d < best_d {
                best_d = d;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }

    if best_dx == 0 && best_dy == 0 {
        return Ok(vec![[px, py, qx, qy]]);
    }

    let torus_max = 1.0 - f64::EPSILON;

    if best_dx == 0 {
        let (x0, y0, x1_w, y1_w) = if py < qy {
            (px, py, qx, qy)
        } else {
            (qx, qy, px, py)
        };
        let denom = y0 - y1_w + 1.0;
        let x2_cross = if denom.abs() <= 1e-15 {
            0.5 * (x0 + x1_w)
        } else {
            (y0 * x1_w - y1_w * x0 + x0) / denom
        };
        let xw = wrap_torus_coord(x2_cross);
        return Ok(vec![[x0, y0, xw, 0.0], [xw, torus_max, x1_w, y1_w]]);
    }

    if best_dy == 0 {
        let (x0, y0, x1_w, y1_w) = if px < qx {
            (px, py, qx, qy)
        } else {
            (qx, qy, px, py)
        };
        let denom = x0 - x1_w + 1.0;
        let y2_cross = if denom.abs() <= 1e-15 {
            0.5 * (y0 + y1_w)
        } else {
            (x0 * y1_w - x1_w * y0 + y0) / denom
        };
        let yw = wrap_torus_coord(y2_cross);
        return Ok(vec![[x0, y0, 0.0, yw], [torus_max, yw, x1_w, y1_w]]);
    }

    let (x0, y0, x1_w, y1_w) = if px < qx {
        (px, py, qx, qy)
    } else {
        (qx, qy, px, py)
    };
    let cx = x0 - x1_w + 1.0;
    let opp = best_dx * best_dy < 0;
    let cy = if opp {
        y0 - y1_w - 1.0
    } else {
        y0 - y1_w + 1.0
    };
    let x2_cross = if opp {
        (cy * x0 - cx * y0 + cx) / cy
    } else {
        (cy * x0 - cx * y0) / cy
    };
    let y2_cross = (cx * y0 - cy * x0) / cx;

    if opp {
        if x2_cross < 0.0 {
            let yw = wrap_torus_coord(y2_cross);
            let xw = wrap_torus_coord(x2_cross + 1.0);
            Ok(vec![
                [x0, y0, 0.0, yw],
                [torus_max, yw, xw, torus_max],
                [xw, 0.0, x1_w, y1_w],
            ])
        } else {
            let xw = wrap_torus_coord(x2_cross);
            let yw = wrap_torus_coord(y2_cross + 1.0);
            Ok(vec![
                [x0, y0, xw, torus_max],
                [xw, 0.0, 0.0, yw],
                [torus_max, yw, x1_w, y1_w],
            ])
        }
    } else if y2_cross < 0.0 {
        let xw = wrap_torus_coord(x2_cross);
        let yw = wrap_torus_coord(y2_cross + 1.0);
        Ok(vec![
            [x0, y0, xw, 0.0],
            [xw, torus_max, 0.0, yw],
            [torus_max, yw, x1_w, y1_w],
        ])
    } else {
        let yw = wrap_torus_coord(y2_cross);
        let xw = wrap_torus_coord(x2_cross + 1.0);
        Ok(vec![
            [x0, y0, 0.0, yw],
            [torus_max, yw, xw, 0.0],
            [xw, torus_max, x1_w, y1_w],
        ])
    }
}

/// Finds all crossing edge segment pairs for `graph` laid out on the flat unit torus `[0, 1)^2`.
pub fn torus_crossing_edges<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
) -> CoreResult<Vec<TorusCrossing>> {
    let edges = extract_csr_edges(graph);
    torus_crossing_edges_from_edges(&edges, layout)
}

/// Finds all crossing edge segment pairs among `edges` on the flat unit torus `[0, 1)^2`.
pub fn torus_crossing_edges_from_edges<S: Scalar>(
    edges: &[(u32, u32)],
    layout: &Layout2d<S>,
) -> CoreResult<Vec<TorusCrossing>> {
    layout.validate_finite()?;
    let n = layout.len();
    let xs = layout.x();
    let ys = layout.y();

    let mut sub_segs: Vec<Seg> = Vec::with_capacity(edges.len());
    let mut orig_edge_of_sub: Vec<usize> = Vec::with_capacity(edges.len());

    for (edge_id, &(u, v)) in edges.iter().enumerate() {
        let ui = u as usize;
        let vi = v as usize;
        if ui >= n {
            return Err(CoreError::index_out_of_bounds(
                ui,
                n,
                "torus_crossing_edges_from_edges",
            ));
        }
        if vi >= n {
            return Err(CoreError::index_out_of_bounds(
                vi,
                n,
                "torus_crossing_edges_from_edges",
            ));
        }
        let pieces = torus_edge_segments(xs[ui], ys[ui], xs[vi], ys[vi])?;
        for piece in pieces {
            let sub_id = sub_segs.len();
            sub_segs.push(Seg::from_endpoints(
                sub_id, u, v, piece[0], piece[1], piece[2], piece[3],
            ));
            orig_edge_of_sub.push(edge_id);
        }
    }

    if sub_segs.len() > u32::MAX as usize {
        return Err(CoreError::invalid_input("edge count exceeds u32::MAX"));
    }
    let pairs = sweep_segments(&sub_segs);
    let mut out = Vec::with_capacity(pairs.len());
    for key in pairs {
        let (sub_a, sub_b) = unpack_pair(key);
        let sa = &sub_segs[sub_a];
        let sb = &sub_segs[sub_b];
        let ea = orig_edge_of_sub[sub_a];
        let eb = orig_edge_of_sub[sub_b];
        if ea <= eb {
            out.push(TorusCrossing {
                edge_a: ea,
                edge_b: eb,
                seg_a: [sa.x1, sa.y1, sa.x2, sa.y2],
                seg_b: [sb.x1, sb.y1, sb.x2, sb.y2],
            });
        } else {
            out.push(TorusCrossing {
                edge_a: eb,
                edge_b: ea,
                seg_a: [sb.x1, sb.y1, sb.x2, sb.y2],
                seg_b: [sa.x1, sa.y1, sa.x2, sa.y2],
            });
        }
    }
    out.sort_unstable_by(|a, b| {
        a.edge_a
            .cmp(&b.edge_a)
            .then_with(|| a.edge_b.cmp(&b.edge_b))
            .then_with(|| a.seg_a[0].total_cmp(&b.seg_a[0]))
            .then_with(|| a.seg_a[1].total_cmp(&b.seg_a[1]))
            .then_with(|| a.seg_b[0].total_cmp(&b.seg_b[0]))
            .then_with(|| a.seg_b[1].total_cmp(&b.seg_b[1]))
    });
    Ok(out)
}

/// Estimates total edge crossings in $O(M)$ time via deterministic edge-pair sampling.
///
/// When the total number of distinct edge pairs $\binom{|E|}{2}$ is at most `num_samples`,
/// evaluates all pairs deterministically and returns the exact crossing count in $O(|E|^2) \le O(M)$.
pub fn sampled_crossing_number<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    num_samples: usize,
    rng: &mut DeterministicRng,
) -> CoreResult<f64> {
    let edges = extract_csr_edges(graph);
    sampled_crossing_number_edges(&edges, layout, num_samples, rng)
}

/// Edge-slice variant of [`sampled_crossing_number`].
pub fn sampled_crossing_number_edges<S: Scalar>(
    edges: &[(u32, u32)],
    layout: &Layout2d<S>,
    num_samples: usize,
    rng: &mut DeterministicRng,
) -> CoreResult<f64> {
    if num_samples == 0 {
        return Err(CoreError::invalid_input("num_samples must be >= 1"));
    }
    let segs = build_segments(edges, layout)?;
    let non_loops: Vec<&Seg> = segs.iter().filter(|s| s.u != s.v).collect();
    let m = non_loops.len();
    if m < 2 {
        return Ok(0.0);
    }

    let total_pairs = (m as u128 * (m as u128 - 1)) / 2;
    if total_pairs <= num_samples as u128 {
        let mut exact = 0usize;
        for i in 0..m {
            for j in (i + 1)..m {
                if segments_cross(non_loops[i], non_loops[j]) {
                    exact += 1;
                }
            }
        }
        return Ok(exact as f64);
    }

    let mut hits = 0usize;
    for _ in 0..num_samples {
        let i = rng.gen_range_usize(m);
        let mut j = rng.gen_range_usize(m - 1);
        if j >= i {
            j += 1;
        }
        if segments_cross(non_loops[i], non_loops[j]) {
            hits += 1;
        }
    }

    Ok((hits as f64 / num_samples as f64) * (total_pairs as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute_force_pairs(edges: &[(u32, u32)], layout: &Layout2d<f64>) -> Vec<(usize, usize)> {
        let segs = build_segments(edges, layout).unwrap();
        let mut out = Vec::new();
        for i in 0..segs.len() {
            for j in (i + 1)..segs.len() {
                if segments_cross(&segs[i], &segs[j]) {
                    out.push((i, j));
                }
            }
        }
        out
    }

    fn assert_matches_brute_force(label: &str, edges: &[(u32, u32)], layout: &Layout2d<f64>) {
        let expected = brute_force_pairs(edges, layout);
        let got_pairs = SweepLineCrossings::find_edge_crossings(edges, layout).unwrap();
        let got_indices: Vec<(usize, usize)> =
            got_pairs.iter().map(|p| (p.edge_a, p.edge_b)).collect();
        assert_eq!(got_indices, expected, "{label}: crossing mismatch");
        assert_eq!(
            SweepLineCrossings::count_edges(edges, layout).unwrap(),
            expected.len(),
            "{label}: count_edges mismatch"
        );
        let summary = SweepLineCrossings::summarize_edges(edges, layout).unwrap();
        assert_eq!(summary.count, expected.len());
        let expected_cos_sq: f64 = got_pairs.iter().map(|p| p.cos_sq).sum();
        assert!(
            (summary.angle_cos_sq_sum - expected_cos_sq).abs()
                <= 1e-12 * (expected.len().max(1) as f64),
            "{label}: angle_cos_sq_sum mismatch"
        );
        let expected_min_ang = got_pairs
            .iter()
            .filter_map(|p| p.angle_rad)
            .reduce(f64::min);
        assert_eq!(summary.min_angle_rad, expected_min_ang);
    }

    #[test]
    fn test_sweep_line_planar_nonplanar_grid_and_star() {
        // 1. Planar 4x4 grid (0 crossings).
        let mut grid_x = Vec::new();
        let mut grid_y = Vec::new();
        let mut grid_edges = Vec::new();
        for r in 0..4u32 {
            for c in 0..4u32 {
                let id = r * 4 + c;
                grid_x.push(c as f64);
                grid_y.push(r as f64);
                if c + 1 < 4 {
                    grid_edges.push((id, id + 1));
                }
                if r + 1 < 4 {
                    grid_edges.push((id, id + 4));
                }
            }
        }
        let grid_layout = Layout2d::from_xy(grid_x, grid_y).unwrap();
        assert_matches_brute_force("grid_4x4", &grid_edges, &grid_layout);
        assert_eq!(
            SweepLineCrossings::count_edges(&grid_edges, &grid_layout).unwrap(),
            0
        );

        // 2. K5 pentagon + pentagram (non-planar, 5 crossings).
        let k5_x: Vec<f64> = (0..5)
            .map(|i| (i as f64 * std::f64::consts::TAU / 5.0).cos())
            .collect();
        let k5_y: Vec<f64> = (0..5)
            .map(|i| (i as f64 * std::f64::consts::TAU / 5.0).sin())
            .collect();
        let mut k5_edges = Vec::new();
        for u in 0..5u32 {
            for v in (u + 1)..5u32 {
                k5_edges.push((u, v));
            }
        }
        let k5_layout = Layout2d::from_xy(k5_x, k5_y).unwrap();
        assert_matches_brute_force("K5", &k5_edges, &k5_layout);
        assert_eq!(
            SweepLineCrossings::count_edges(&k5_edges, &k5_layout).unwrap(),
            5
        );

        // 3. K_{3,3} bipartite layout (9 crossings).
        let k33_layout = Layout2d::from_xy(
            vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        )
        .unwrap();
        let mut k33_edges = Vec::new();
        for u in 0..3u32 {
            for v in 3..6u32 {
                k33_edges.push((u, v));
            }
        }
        assert_matches_brute_force("K3,3", &k33_edges, &k33_layout);
        assert_eq!(
            SweepLineCrossings::count_edges(&k33_edges, &k33_layout).unwrap(),
            9
        );

        // 4. Star graph with 64 rays sharing center vertex 0 (0 crossings).
        let mut star_x = vec![0.0];
        let mut star_y = vec![0.0];
        let mut star_edges = Vec::new();
        for i in 1..=64u32 {
            let ang = (i as f64) * std::f64::consts::TAU / 64.0;
            star_x.push(ang.cos());
            star_y.push(ang.sin());
            star_edges.push((0, i));
        }
        let star_layout = Layout2d::from_xy(star_x, star_y).unwrap();
        assert_matches_brute_force("star_64", &star_edges, &star_layout);
        assert_eq!(
            SweepLineCrossings::count_edges(&star_edges, &star_layout).unwrap(),
            0
        );
    }

    #[test]
    fn test_sweep_line_degeneracies_horizontal_vertical_collinear_and_shared_coords() {
        // Horizontal, vertical, collinear overlapping, collinear disjoint, shared endpoints,
        // coincident distinct-ID endpoints, and multi-way intersection at (1.0, 1.0).
        let layout = Layout2d::from_xy(
            vec![
                0.0, 3.0, 1.0, 4.0, 5.0, 6.0, // horizontal / collinear on y=0
                1.0, 1.0, 1.0, 1.0, // vertical overlapping on x=1
                0.0, 2.0, 0.0, 2.0, 0.0, 2.0, // multi-way X crossing at (1, 1)
                0.0, 1.0, // coincident distinct vertex at (0, 0) & T-junction at (1, 0)
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                -1.0, 2.0, 0.5, 3.0, //
                0.0, 2.0, 2.0, 0.0, 1.0, 1.0, //
                0.0, 0.0, //
            ],
        )
        .unwrap();
        let edges = vec![
            (0, 1),   // (0,0)-(3,0)
            (2, 3),   // (1,0)-(4,0) collinear overlap with (0,1)
            (4, 5),   // (5,0)-(6,0) collinear disjoint
            (6, 7),   // vertical (1,-1)-(1,2)
            (8, 9),   // vertical (1,0.5)-(1,3) overlaps (6,7)
            (10, 11), // diag (0,0)-(2,2)
            (12, 13), // diag (0,2)-(2,0)
            (14, 15), // horiz (0,1)-(2,1) multi-way through (1,1)
            (16, 17), // coincident vertex (0,0) to T-junction (1,0)
            (0, 10),  // self-loop-like shared coord with distinct IDs
            (1, 2),   // shared endpoint test
        ];
        assert_matches_brute_force("degenerate_suite", &edges, &layout);

        // Bridged-hub cluster at (1.5, 0.75): a star at `hub`, an edge bridging across to
        // a non-incident chord, plus passing/ending/starting edges meeting at the exact hub point.
        let bridge_layout = Layout2d::from_xy(
            vec![
                1.5, 2.5, 2.5, 2.25, // hub=0 and 3 rays extending right
                0.5, 2.5, // 4->5 passes through (1.5, 0.75) without sharing hub
                0.75, 1.5, // 6->7 ends at (1.5, 0.75) on distinct vertex
                1.0, 2.0, // 8->9 passes through (1.5, 0.75) without sharing hub
                1.5, 2.5, // 10->11 starts at (1.5, 0.75) on distinct vertex
            ],
            vec![
                0.75, 1.75, -0.25, 0.75, //
                0.25, 1.25, //
                0.0, 0.75, //
                1.25, 0.25, //
                0.75, 1.25, //
            ],
        )
        .unwrap();
        let bridge_edges = vec![
            (0, 1),
            (0, 2),
            (0, 3),
            (4, 5),
            (6, 7),
            (8, 9),
            (10, 11),
            (3, 8), // shares vertex 3 with (0, 3) and vertex 8 with (8, 9)
        ];
        assert_matches_brute_force("bridged_hub_cluster", &bridge_edges, &bridge_layout);

        // Random dense stress test across 24 quantized seeds forcing dense multi-way ties.
        for seed in 1..=24u64 {
            let mut rng = DeterministicRng::from_seed_u64(seed);
            let n = 25usize;
            // Include quantized grid coordinates to force frequent collinear/vertical/shared ties.
            let xs: Vec<f64> = (0..n)
                .map(|_| (rng.gen_range_usize(11) as f64) * 0.25)
                .collect();
            let ys: Vec<f64> = (0..n)
                .map(|_| (rng.gen_range_usize(11) as f64) * 0.25)
                .collect();
            let l = Layout2d::from_xy(xs, ys).unwrap();
            let mut es = Vec::new();
            for _ in 0..45 {
                let u = rng.gen_range_usize(n) as u32;
                let v = rng.gen_range_usize(n) as u32;
                es.push((u, v));
            }
            assert_matches_brute_force(&format!("random_quantized_seed_{seed}"), &es, &l);
        }

        // Large 1024-ray star centered at (0, 0) verifying O(|E| log |E|) Misra-Gries hub pruning.
        let mut big_x = vec![0.0];
        let mut big_y = vec![0.0];
        let mut big_edges = Vec::with_capacity(1024);
        for i in 1..=1024u32 {
            let theta = (i as f64) * (2.0 * std::f64::consts::PI / 1024.0);
            big_x.push(theta.cos());
            big_y.push(theta.sin());
            big_edges.push((0, i));
        }
        let big_star = Layout2d::from_xy(big_x, big_y).unwrap();
        assert_eq!(
            SweepLineCrossings::count_edges(&big_edges, &big_star).unwrap(),
            0
        );
    }

    #[test]
    fn test_sweep_line_angles_csr_torus_and_sampling() {
        // Right-angle crossing gives cos_sq == 0 and min_angle == PI/2.
        let layout =
            Layout2d::from_xy(vec![-1.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, -1.0, 1.0]).unwrap();
        let csr = CsrGraph::<f64>::from_edges(4, &[(0, 1), (2, 3)], None, false, false).unwrap();
        let summary = SweepLineCrossings::summarize(&csr, &layout).unwrap();
        assert_eq!(summary.count, 1);
        assert!(summary.angle_cos_sq_sum.abs() < 1e-12);
        assert!((summary.min_angle_rad.unwrap() - FRAC_PI_2).abs() < 1e-12);

        // Torus crossing test: two wrapping edges that cross after periodic boundary split.
        let torus_layout =
            Layout2d::from_xy(vec![0.9, 0.1, 0.9, 0.1], vec![0.1, 0.9, 0.9, 0.1]).unwrap();
        let torus_csr =
            CsrGraph::<f64>::from_edges(4, &[(0, 1), (2, 3)], None, false, false).unwrap();
        let torus_crossings = torus_crossing_edges(&torus_csr, &torus_layout).unwrap();
        assert!(!torus_crossings.is_empty());

        // Sampled crossing number determinism and unbiased convergence on a dense bipartite layout.
        let mut bx = Vec::new();
        let mut by = Vec::new();
        let mut bedges = Vec::new();
        for i in 0..10u32 {
            bx.push(i as f64);
            by.push(0.0);
        }
        for j in 0..10u32 {
            bx.push(j as f64);
            by.push(1.0);
        }
        for u in 0..10u32 {
            for v in 10..20u32 {
                bedges.push((u, v));
            }
        }
        let blayout = Layout2d::from_xy(bx, by).unwrap();
        let exact = SweepLineCrossings::count_edges(&bedges, &blayout).unwrap() as f64;
        let mut r1 = DeterministicRng::from_seed_u64(77);
        let mut r2 = DeterministicRng::from_seed_u64(77);
        let est1 = sampled_crossing_number_edges(&bedges, &blayout, 4000, &mut r1).unwrap();
        let est2 = sampled_crossing_number_edges(&bedges, &blayout, 4000, &mut r2).unwrap();
        assert_eq!(est1, est2);
        assert!(
            (est1 - exact).abs() / exact < 0.08,
            "est1={est1}, exact={exact}"
        );

        // Validation errors.
        let bad = Layout2d::from_xy(vec![f64::NAN, 1.0], vec![0.0, 0.0]).unwrap();
        assert!(matches!(
            SweepLineCrossings::count_edges(&[(0, 1)], &bad),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            sampled_crossing_number_edges(&[(0, 1)], &layout, 0, &mut r1),
            Err(CoreError::InvalidInput(_))
        ));
    }

    #[test]
    fn test_sweep_line_large_tiny_steep_scales_and_repros() {
        // 1. Large 1e8 scale: multi-way x_probe must not underflow to x_evt.
        let large_layout = Layout2d::from_xy(
            vec![
                1e8 + 1.0,
                1e8 + 5.0,
                1e8,
                1e8 + 5.0,
                1e8 + 3.0,
                1e8 + 4.0,
                1e8 + 5.0,
                1e8 + 4.0,
            ],
            vec![
                1e8 + 3.0,
                1e8 + 5.0,
                1e8,
                1e8 + 6.0,
                1e8 + 6.0,
                1e8 + 3.0,
                1e8 + 2.0,
                1e8 + 3.0,
            ],
        )
        .unwrap();
        let large_edges = vec![(0, 1), (2, 3), (4, 5), (6, 7)];
        assert_matches_brute_force("large_1e8_repro", &large_edges, &large_layout);
        assert_eq!(
            SweepLineCrossings::count_edges(&large_edges, &large_layout).unwrap(),
            4
        );

        for seed in 1..=30u64 {
            let mut rng = DeterministicRng::from_seed_u64(1000 + seed);
            let n = 25usize;
            let xs: Vec<f64> = (0..n)
                .map(|_| 1e8 + rng.gen_range_usize(11) as f64)
                .collect();
            let ys: Vec<f64> = (0..n)
                .map(|_| 1e8 + rng.gen_range_usize(11) as f64)
                .collect();
            let l = Layout2d::from_xy(xs, ys).unwrap();
            let mut es = Vec::new();
            for _ in 0..45 {
                let u = rng.gen_range_usize(n) as u32;
                let v = rng.gen_range_usize(n) as u32;
                es.push((u, v));
            }
            assert_matches_brute_force(&format!("large_1e8_seed_{seed}"), &es, &l);
        }

        // 2. Tiny 1e-8 scale: co-located Start/End ULP tie and post-end_batch slope ordering.
        let mut tx = vec![0.0; 29];
        let mut ty = vec![0.0; 29];
        let coords = [
            (10usize, (2e-8, 6e-8)),
            (14, (5e-8, 0.0)),
            (3, (0.0, 3e-8)),
            (24, (3e-8, 4e-8)),
            (25, (2e-8, 2e-8)),
            (6, (4e-8, 5e-8)),
            (13, (6e-8, 3e-8)),
            (28, (3e-8, 4e-8)),
            (2, (5e-8, 2e-8)),
            (9, (1e-8, 3e-8)),
            (1, (4e-8, 6e-8)),
            (11, (1e-8, 3e-8)),
        ];
        for (idx, (x, y)) in coords {
            tx[idx] = x;
            ty[idx] = y;
        }
        let tiny_layout = Layout2d::from_xy(tx, ty).unwrap();
        let tiny_edges = vec![
            (10, 14),
            (3, 24),
            (24, 25),
            (3, 6),
            (13, 28),
            (2, 9),
            (1, 11),
        ];
        assert_matches_brute_force("tiny_1e-8_repro", &tiny_edges, &tiny_layout);

        // Also test `(k as f64) * 1e-8` construction (1-ULP difference on 6e-8) matching qa-m1's seed 13.
        let mut sx = vec![0.0; 29];
        let mut sy = vec![0.0; 29];
        let scaled = [
            (10usize, (2, 6)),
            (14, (5, 0)),
            (3, (0, 3)),
            (24, (3, 4)),
            (25, (2, 2)),
            (6, (4, 5)),
            (13, (6, 3)),
            (28, (3, 4)),
            (2, (5, 2)),
            (9, (1, 3)),
            (1, (4, 6)),
            (11, (1, 3)),
        ];
        for (idx, (ix, iy)) in scaled {
            sx[idx] = (ix as f64) * 1e-8;
            sy[idx] = (iy as f64) * 1e-8;
        }
        let tiny_scaled_layout = Layout2d::from_xy(sx, sy).unwrap();
        assert_matches_brute_force("tiny_1e-8_scaled_repro", &tiny_edges, &tiny_scaled_layout);

        for seed in 1..=30u64 {
            let mut rng = DeterministicRng::from_seed_u64(2000 + seed);
            let n = 30usize;
            let xs: Vec<f64> = (0..n)
                .map(|_| (rng.gen_range_usize(11) as f64) * 1e-8)
                .collect();
            let ys: Vec<f64> = (0..n)
                .map(|_| (rng.gen_range_usize(11) as f64) * 1e-8)
                .collect();
            let l = Layout2d::from_xy(xs, ys).unwrap();
            let mut es = Vec::new();
            for _ in 0..50 {
                let u = rng.gen_range_usize(n) as u32;
                let v = rng.gen_range_usize(n) as u32;
                es.push((u, v));
            }
            assert_matches_brute_force(&format!("tiny_1e-8_seed_{seed}"), &es, &l);
        }

        // 3. Steep-slope x=100, dx=1e-8 ULP drift repro + stress seeds.
        let steep_layout = Layout2d::from_xy(
            vec![
                100.0 + 2e-8,
                100.0 + 3e-8,
                100.0,
                100.0 + 2e-8,
                100.0 + 1e-8,
                100.0 + 3e-8,
            ],
            vec![3.5, 9.0, 6.0, 3.5, 1.5, 5.5],
        )
        .unwrap();
        let steep_edges = vec![(0, 1), (2, 3), (4, 5), (5, 4)];
        assert_matches_brute_force("steep_slope_repro", &steep_edges, &steep_layout);
        assert_eq!(
            SweepLineCrossings::count_edges(&steep_edges, &steep_layout).unwrap(),
            5
        );

        for seed in 1..=30u64 {
            let mut rng = DeterministicRng::from_seed_u64(3000 + seed);
            let n = 20usize;
            let xs: Vec<f64> = (0..n)
                .map(|_| 100.0 + (rng.gen_range_usize(7) as f64) * 1e-8)
                .collect();
            let ys: Vec<f64> = (0..n)
                .map(|_| (rng.gen_range_usize(20) as f64) * 0.5)
                .collect();
            let l = Layout2d::from_xy(xs, ys).unwrap();
            let mut es = Vec::new();
            for _ in 0..40 {
                let u = rng.gen_range_usize(n) as u32;
                let v = rng.gen_range_usize(n) as u32;
                es.push((u, v));
            }
            assert_matches_brute_force(&format!("steep_slope_seed_{seed}"), &es, &l);
        }
    }

    #[test]
    fn test_torus_edge_segments_bounds_and_crossing_parity() {
        // Explicit check that shared-vertex wrapped pieces swap inside (0, 1)^2 and never distort
        // subsequent crossings with unrelated edges, and all split coordinates stay in [0, 1)^2.
        let adj_wrap_layout = Layout2d::from_xy(
            vec![0.1, 0.9, 0.95, 0.92, 0.92],
            vec![0.2, 0.85, 0.15, 0.4, 0.89],
        )
        .unwrap();
        let adj_wrap_edges = vec![(0, 1), (0, 2), (3, 4)];
        for extreme in [
            (0.48_f64, 0.95, 0.99, 0.01),
            (0.02, 0.98, 0.99, 0.48),
            (0.45, 0.05, 0.96, 0.98),
        ] {
            for p in torus_edge_segments(extreme.0, extreme.1, extreme.2, extreme.3).unwrap() {
                for coord in p {
                    assert!(
                        coord.is_finite() && (0.0..1.0).contains(&coord),
                        "extreme torus segment coord out of [0, 1): {p:?}"
                    );
                }
            }
        }
        for &(u, v) in &adj_wrap_edges {
            let pieces = torus_edge_segments(
                adj_wrap_layout.x()[u as usize],
                adj_wrap_layout.y()[u as usize],
                adj_wrap_layout.x()[v as usize],
                adj_wrap_layout.y()[v as usize],
            )
            .unwrap();
            for p in pieces {
                for coord in p {
                    assert!(
                        coord.is_finite() && (0.0..1.0).contains(&coord),
                        "torus segment coord out of [0, 1): {p:?}"
                    );
                }
            }
        }
        let got_adj = torus_crossing_edges_from_edges(&adj_wrap_edges, &adj_wrap_layout).unwrap();
        assert!(!got_adj.is_empty());

        // Compare torus_crossing_edges_from_edges against O(|S|^2) brute force across 30 random seeds.
        for seed in 1..=30u64 {
            let mut rng = DeterministicRng::from_seed_u64(4000 + seed);
            let n = 18usize;
            let xs: Vec<f64> = (0..n)
                .map(|_| (rng.gen_range_usize(33) as f64) / 33.0)
                .collect();
            let ys: Vec<f64> = (0..n)
                .map(|_| (rng.gen_range_usize(33) as f64) / 33.0)
                .collect();
            let layout = Layout2d::from_xy(xs, ys).unwrap();
            let mut edges = Vec::new();
            for _ in 0..32 {
                let u = rng.gen_range_usize(n) as u32;
                let v = rng.gen_range_usize(n) as u32;
                if u != v {
                    edges.push((u, v));
                }
            }

            let mut sub_segs = Vec::new();
            let mut orig_edge = Vec::new();
            for (eid, &(u, v)) in edges.iter().enumerate() {
                let pieces = torus_edge_segments(
                    layout.x()[u as usize],
                    layout.y()[u as usize],
                    layout.x()[v as usize],
                    layout.y()[v as usize],
                )
                .unwrap();
                for p in pieces {
                    for coord in p {
                        assert!(
                            coord.is_finite() && (0.0..1.0).contains(&coord),
                            "seed {seed}: torus segment coord out of [0, 1): {p:?}"
                        );
                    }
                    let id = sub_segs.len();
                    sub_segs.push(Seg::from_endpoints(id, u, v, p[0], p[1], p[2], p[3]));
                    orig_edge.push(eid);
                }
            }

            let mut expected = Vec::new();
            for i in 0..sub_segs.len() {
                for j in (i + 1)..sub_segs.len() {
                    if segments_cross(&sub_segs[i], &sub_segs[j]) {
                        let ea = orig_edge[i];
                        let eb = orig_edge[j];
                        let sa = &sub_segs[i];
                        let sb = &sub_segs[j];
                        if ea <= eb {
                            expected.push(TorusCrossing {
                                edge_a: ea,
                                edge_b: eb,
                                seg_a: [sa.x1, sa.y1, sa.x2, sa.y2],
                                seg_b: [sb.x1, sb.y1, sb.x2, sb.y2],
                            });
                        } else {
                            expected.push(TorusCrossing {
                                edge_a: eb,
                                edge_b: ea,
                                seg_a: [sb.x1, sb.y1, sb.x2, sb.y2],
                                seg_b: [sa.x1, sa.y1, sa.x2, sa.y2],
                            });
                        }
                    }
                }
            }
            expected.sort_unstable_by(|a, b| {
                a.edge_a
                    .cmp(&b.edge_a)
                    .then_with(|| a.edge_b.cmp(&b.edge_b))
                    .then_with(|| a.seg_a[0].total_cmp(&b.seg_a[0]))
                    .then_with(|| a.seg_a[1].total_cmp(&b.seg_a[1]))
                    .then_with(|| a.seg_b[0].total_cmp(&b.seg_b[0]))
                    .then_with(|| a.seg_b[1].total_cmp(&b.seg_b[1]))
            });

            let got = torus_crossing_edges_from_edges(&edges, &layout).unwrap();
            assert_eq!(
                got.len(),
                expected.len(),
                "torus seed {seed}: count mismatch"
            );
            assert_eq!(got, expected, "torus seed {seed}: pair mismatch");
        }
    }

    fn assert_streaming_matches_find(label: &str, edges: &[(u32, u32)], layout: &Layout2d<f64>) {
        let pairs = SweepLineCrossings::find_edge_crossings(edges, layout).unwrap();
        let count = SweepLineCrossings::count_edges(edges, layout).unwrap();
        let summary = SweepLineCrossings::summarize_edges(edges, layout).unwrap();
        assert_eq!(
            count,
            pairs.len(),
            "{label}: count_edges vs find_edge_crossings"
        );
        assert_eq!(
            summary.count,
            pairs.len(),
            "{label}: summary.count vs find_edge_crossings"
        );
        let expected_min_ang = pairs.iter().filter_map(|p| p.angle_rad).reduce(f64::min);
        assert_eq!(
            summary.min_angle_rad, expected_min_ang,
            "{label}: min_angle_rad mismatch"
        );
        let expected_cos_sq: f64 = pairs.iter().map(|p| p.cos_sq).sum();
        assert!(
            (summary.angle_cos_sq_sum - expected_cos_sq).abs()
                <= 1e-12 * (pairs.len().max(1) as f64),
            "{label}: angle_cos_sq_sum mismatch (got={}, expected={})",
            summary.angle_cos_sq_sum,
            expected_cos_sq
        );
    }

    #[test]
    fn test_streaming_sweep_collinear_zero_length_shared_tips_and_eps_bands() {
        // 1. Collinear overlaps (horizontal, vertical, diagonal), touching tips (a.x2 == b.x1),
        // containment, duplicate coordinates with distinct vertex IDs, zero-length segments,
        // self-loops, and shared right endpoints.
        let layout = Layout2d::from_xy(
            vec![
                0.0, 2.0, 2.0, 4.0, 1.0, 3.0, 0.0, 5.0, // horizontal collinear on y=0
                0.0, 2.0, 1.0, 3.0, 2.0, 4.0, // diagonal collinear on y=x
                1.5, 1.5, 1.5, 1.5, // zero-length segment (14,15) and vertical (16,17)
                0.5, 1.5, // zero-length at (0.5, 0.5) on y=x
                3.0, 5.0, 3.0, 5.0, // shared right endpoint at (5, 2)
                0.0, 4.0, // duplicate diagonal with distinct vertex IDs
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 2.0, 1.0, 3.0, 2.0, 4.0, //
                0.5, 0.5, -1.0, 4.0, //
                0.5, 0.5, //
                1.0, 2.0, 3.0, 2.0, //
                0.0, 4.0, //
            ],
        )
        .unwrap();
        let edges = vec![
            (0, 1),   // [0,2] on y=0
            (2, 3),   // [2,4] touching tip on y=0
            (4, 5),   // [1,3] partial overlap with (0,1) and (2,3)
            (6, 7),   // [0,5] containment on y=0
            (8, 9),   // [0,2] on y=x
            (10, 11), // [1,3] on y=x
            (12, 13), // [2,4] touching tip on y=x
            (14, 15), // zero-length at (1.5, 0.5)
            (16, 17), // vertical x=1.5 crossing y=0, y=x, and zero-length (14,15)
            (18, 19), // zero-length at (0.5, 0.5) on y=x
            (20, 21), // ends at (5, 2)
            (22, 23), // shares right endpoint (5, 2) with (20, 21) via distinct IDs
            (24, 25), // [0,4] duplicate diagonal
            (0, 0),   // self-loop
            (14, 14), // self-loop on zero-length vertex
        ];
        assert_matches_brute_force("collinear_zero_shared_tips", &edges, &layout);
        assert_streaming_matches_find("collinear_zero_shared_tips_find", &edges, &layout);

        // 2. Near-parallel eps_y band stress at y = 100.0 (|slope_a - slope_b| ~ 1e-15)
        // plus crossing transverse segments and identical x_int ties.
        let mut bx = Vec::new();
        let mut by = Vec::new();
        let mut bedges = Vec::new();
        for i in 0..8u32 {
            let x0 = 0.0;
            let x1 = 10.0;
            let dy = (i as f64) * 1e-13;
            let y0 = 100.0 + dy;
            let y1 = 100.0 + (7 - i) as f64 * 1e-13;
            let u = bx.len() as u32;
            bx.push(x0);
            by.push(y0);
            bx.push(x1);
            by.push(y1);
            bedges.push((u, u + 1));
        }
        // Add transverse segments crossing the band at distinct and shared x positions.
        for k in 0..6u32 {
            let u = bx.len() as u32;
            let xc = 1.0 + (k as f64) * 1.5;
            bx.push(xc - 0.5);
            by.push(99.0);
            bx.push(xc + 0.5);
            by.push(101.0);
            bedges.push((u, u + 1));
        }
        let band_layout = Layout2d::from_xy(bx, by).unwrap();
        assert_matches_brute_force("near_parallel_eps_band_y100", &bedges, &band_layout);
        assert_streaming_matches_find("near_parallel_eps_band_y100_find", &bedges, &band_layout);

        // 3. Deferred crossings (x_int > sweep_x with intermediate segments starting/ending)
        // and early dt_probe swap stress.
        for seed in 1..=20u64 {
            let mut rng = DeterministicRng::from_seed_u64(9000 + seed);
            let n = 32usize;
            let xs: Vec<f64> = (0..n)
                .map(|_| rng.gen_range_usize(1000) as f64 * 1e-3)
                .collect();
            let ys: Vec<f64> = (0..n)
                .map(|_| 50.0 + rng.gen_range_usize(1000) as f64 * 1e-12)
                .collect();
            let l = Layout2d::from_xy(xs, ys).unwrap();
            let mut es = Vec::new();
            for _ in 0..60 {
                let u = rng.gen_range_usize(n) as u32;
                let v = rng.gen_range_usize(n) as u32;
                es.push((u, v));
            }
            assert_matches_brute_force(&format!("eps_band_random_seed_{seed}"), &es, &l);
        }
    }

    #[test]
    fn test_streaming_sweep_dense_high_crossing_and_bounded_memory() {
        // 1. K_{40,40} bipartite layout on parallel lines x=0 and x=1 (many identical x_int ties).
        let mut bx = Vec::with_capacity(80);
        let mut by = Vec::with_capacity(80);
        for i in 0..40u32 {
            bx.push(0.0);
            by.push(i as f64);
        }
        for j in 0..40u32 {
            bx.push(1.0);
            by.push(j as f64);
        }
        let mut bedges = Vec::with_capacity(1600);
        for u in 0..40u32 {
            for v in 40..80u32 {
                bedges.push((u, v));
            }
        }
        let blayout = Layout2d::from_xy(bx, by).unwrap();
        assert_streaming_matches_find("K_40_40", &bedges, &blayout);

        let bsegs = build_segments(&bedges, &blayout).unwrap();
        let mut bcount = 0usize;
        let b_peak = sweep_segments_with_stats(&bsegs, |_, _| {
            bcount += 1;
        });
        // Exact analytical crossing count for K_{40,40}: C(40,2)^2 = 780^2 = 608,400.
        assert_eq!(bcount, 780 * 780);
        assert!(
            b_peak * 100 < bcount,
            "K_{{40,40}} active band tracking ({b_peak}) must stay << total crossings ({bcount})"
        );

        // 2. K_50 convex circle layout (C(50, 4) = 230,300 crossings).
        let k50_x: Vec<f64> = (0..50)
            .map(|i| (i as f64 * std::f64::consts::TAU / 50.0).cos())
            .collect();
        let k50_y: Vec<f64> = (0..50)
            .map(|i| (i as f64 * std::f64::consts::TAU / 50.0).sin())
            .collect();
        let mut k50_edges = Vec::with_capacity(50 * 49 / 2);
        for u in 0..50u32 {
            for v in (u + 1)..50u32 {
                k50_edges.push((u, v));
            }
        }
        let k50_layout = Layout2d::from_xy(k50_x, k50_y).unwrap();
        assert_streaming_matches_find("K_50_circle", &k50_edges, &k50_layout);

        let k50_segs = build_segments(&k50_edges, &k50_layout).unwrap();
        let mut k50_count = 0usize;
        let k50_peak = sweep_segments_with_stats(&k50_segs, |_, _| {
            k50_count += 1;
        });
        assert_eq!(k50_count, 230_300);
        assert!(
            k50_peak * 20 < k50_count,
            "K_50 active band tracking ({k50_peak}) must stay << total crossings ({k50_count})"
        );

        // 3. Random dense graphs (N=80, M=600) verifying CSR count/summarize parity and tiny active band.
        for seed in 1..=5u64 {
            let mut rng = DeterministicRng::from_seed_u64(7000 + seed);
            let n = 80usize;
            let xs: Vec<f64> = (0..n).map(|_| rng.next_f64() * 100.0).collect();
            let ys: Vec<f64> = (0..n).map(|_| rng.next_f64() * 100.0).collect();
            let layout = Layout2d::from_xy(xs, ys).unwrap();
            let mut edges = Vec::with_capacity(600);
            for _ in 0..600 {
                let u = rng.gen_range_usize(n) as u32;
                let v = rng.gen_range_usize(n) as u32;
                if u != v {
                    edges.push((u, v));
                }
            }
            assert_streaming_matches_find(&format!("random_dense_seed_{seed}"), &edges, &layout);

            let segs = build_segments(&edges, &layout).unwrap();
            let mut c = 0usize;
            let peak = sweep_segments_with_stats(&segs, |_, _| {
                c += 1;
            });
            assert!(c > 10_000, "expected high-crossing dense layout, got {c}");
            assert!(
                peak * 100 < c,
                "random_dense_seed_{seed}: active band ({peak}) must stay << total crossings ({c})"
            );

            // Also verify CsrGraph entrypoints count() and summarize() match find_crossings().
            let csr = CsrGraph::<f64>::from_edges(n, &edges, None, false, true).unwrap();
            let csr_pairs = SweepLineCrossings::find_crossings(&csr, &layout).unwrap();
            let csr_count = SweepLineCrossings::count(&csr, &layout).unwrap();
            let csr_summary = SweepLineCrossings::summarize(&csr, &layout).unwrap();
            assert_eq!(csr_count, csr_pairs.len());
            assert_eq!(csr_summary.count, csr_pairs.len());
            assert_eq!(
                csr_summary.min_angle_rad,
                csr_pairs
                    .iter()
                    .filter_map(|p| p.angle_rad)
                    .reduce(f64::min)
            );
        }
    }

    fn torus_brute_force(edges: &[(u32, u32)], layout: &Layout2d<f64>) -> Vec<TorusCrossing> {
        let mut sub_segs = Vec::new();
        let mut orig_edge = Vec::new();
        for (eid, &(u, v)) in edges.iter().enumerate() {
            let pieces = torus_edge_segments(
                layout.x()[u as usize],
                layout.y()[u as usize],
                layout.x()[v as usize],
                layout.y()[v as usize],
            )
            .unwrap();
            for p in pieces {
                let id = sub_segs.len();
                sub_segs.push(Seg::from_endpoints(id, u, v, p[0], p[1], p[2], p[3]));
                orig_edge.push(eid);
            }
        }
        let mut expected = Vec::new();
        for i in 0..sub_segs.len() {
            for j in (i + 1)..sub_segs.len() {
                if segments_cross(&sub_segs[i], &sub_segs[j]) {
                    let ea = orig_edge[i];
                    let eb = orig_edge[j];
                    let sa = &sub_segs[i];
                    let sb = &sub_segs[j];
                    if ea <= eb {
                        expected.push(TorusCrossing {
                            edge_a: ea,
                            edge_b: eb,
                            seg_a: [sa.x1, sa.y1, sa.x2, sa.y2],
                            seg_b: [sb.x1, sb.y1, sb.x2, sb.y2],
                        });
                    } else {
                        expected.push(TorusCrossing {
                            edge_a: eb,
                            edge_b: ea,
                            seg_a: [sb.x1, sb.y1, sb.x2, sb.y2],
                            seg_b: [sa.x1, sa.y1, sa.x2, sa.y2],
                        });
                    }
                }
            }
        }
        expected.sort_unstable_by(|a, b| {
            a.edge_a
                .cmp(&b.edge_a)
                .then_with(|| a.edge_b.cmp(&b.edge_b))
                .then_with(|| a.seg_a[0].total_cmp(&b.seg_a[0]))
                .then_with(|| a.seg_a[1].total_cmp(&b.seg_a[1]))
                .then_with(|| a.seg_b[0].total_cmp(&b.seg_b[0]))
                .then_with(|| a.seg_b[1].total_cmp(&b.seg_b[1]))
        });
        expected
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn test_challenger_reproducers_a_b_c_d() {
        // Reproducer A: 2-edge ULP drift at xi + past-crossing on_band_pair double-count guard.
        let layout_a = Layout2d::from_xy(
            vec![
                -3.04805189247130648e-5,
                5.00000097576867404e1,
                3.31165387303849953e-5,
                4.99999652101946026e1,
            ],
            vec![
                -7.55507051547136343e-6,
                2.00000408856693603e1,
                3.83695954409438164e-5,
                1.99999834070688145e1,
            ],
        )
        .unwrap();
        let edges_a = vec![(0, 1), (2, 3)];
        assert_matches_brute_force("reproducer_A", &edges_a, &layout_a);
        assert_streaming_matches_find("reproducer_A_stream", &edges_a, &layout_a);
        assert_eq!(
            SweepLineCrossings::count_edges(&edges_a, &layout_a).unwrap(),
            1
        );

        // Reproducer B: 4-edge cluster where reverting completed swap blocked (0, 2).
        let layout_b = Layout2d::from_xy(
            vec![
                4.99999832431687921e1,
                9.99999952436454294e1,
                4.99999784950573343e1,
                1.00000017056024674e2,
                -9.83780916952248230e-7,
                4.99999681576062898e1,
            ],
            vec![
                8.00000140893703815e1,
                1.69716668579220089e-5,
                7.99999695116295868e1,
                4.72816566238181744e-5,
                4.90065288854551163e-5,
                1.99999520082606139e1,
            ],
        )
        .unwrap();
        let edges_b = vec![(0, 1), (2, 3), (4, 3), (5, 1)];
        assert_matches_brute_force("reproducer_B", &edges_b, &layout_b);
        assert_streaming_matches_find("reproducer_B_stream", &edges_b, &layout_b);
        assert_eq!(
            SweepLineCrossings::count_edges(&edges_b, &layout_b).unwrap(),
            3
        );

        // Reproducer C: 8-edge steep cluster at x ~ 1e6, dx ~ 4e-8.
        let mut xc = vec![0.0; 20];
        let mut yc = vec![0.0; 20];
        let pts_c = [
            (9usize, 1.00000000000005995e6, 2.5),
            (6, 1.00000000000002002e6, 9.5),
            (18, 1.00000000000001001e6, 4.0),
            (3, 1.00000000000004994e6, 5.5),
            (2, 1.00000000000000000e6, 0.0),
            (8, 1.00000000000004994e6, 4.0),
            (10, 1.00000000000003004e6, 4.0),
            (14, 1.00000000000002002e6, 4.0),
            (5, 1.00000000000001001e6, 7.0),
            (11, 1.00000000000004994e6, 8.0),
            (7, 1.00000000000003004e6, 7.5),
            (19, 1.00000000000004005e6, 8.0),
            (0, 1.00000000000003004e6, 2.5),
        ];
        for (idx, px, py) in pts_c {
            xc[idx] = px;
            yc[idx] = py;
        }
        let layout_c = Layout2d::from_xy(xc, yc).unwrap();
        let edges_c = vec![
            (9, 6),
            (18, 3),
            (2, 8),
            (6, 10),
            (14, 5),
            (18, 6),
            (11, 7),
            (6, 19),
        ];
        assert_matches_brute_force("reproducer_C", &edges_c, &layout_c);
        assert_streaming_matches_find("reproducer_C_stream", &edges_c, &layout_c);

        // Reproducer D: 5-edge steep cluster with independent vertices at x ~ 1e6.
        let layout_d = Layout2d::from_xy(
            vec![
                1.00000000000001001e6,
                1.00000000000004994e6,
                1.00000000000001001e6,
                1.00000000000004005e6,
                1.00000000000001001e6,
                1.00000000000003004e6,
                1.00000000000001001e6,
                1.00000000000002002e6,
                1.00000000000002002e6,
                1.00000000000003004e6,
            ],
            vec![2.0, 8.0, 7.0, 8.0, 4.0, 2.5, 4.0, 9.5, 4.0, 4.0],
        )
        .unwrap();
        let edges_d = vec![(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)];
        assert_matches_brute_force("reproducer_D", &edges_d, &layout_d);
        assert_streaming_matches_find("reproducer_D_stream", &edges_d, &layout_d);

        // Large non-hub multi-way intersection (70 segments crossing at (1.0, 1.0)) exercising
        // the k > 64 merge-sort inversion path in `emit_flipped_pairs`.
        let k_multi = 70usize;
        let mut mx = Vec::with_capacity(k_multi * 2);
        let mut my = Vec::with_capacity(k_multi * 2);
        let mut medges = Vec::with_capacity(k_multi);
        for i in 0..k_multi {
            let slope = -1.0 + 2.0 * (i as f64) / ((k_multi - 1) as f64);
            mx.push(0.0);
            my.push(1.0 - slope);
            mx.push(2.0);
            my.push(1.0 + slope);
            let u = (2 * i) as u32;
            medges.push((u, u + 1));
        }
        let mlayout = Layout2d::from_xy(mx, my).unwrap();
        assert_matches_brute_force("multiway_70_non_hub", &medges, &mlayout);
        assert_streaming_matches_find("multiway_70_non_hub_stream", &medges, &mlayout);
    }

    #[test]
    fn test_torus_steep_and_clustered_parity() {
        // Steep wrapped torus sub-segments (dx ~ 1e-8 across periodic boundary).
        for seed in 1..=25u64 {
            let mut rng = DeterministicRng::from_seed_u64(5000 + seed);
            let n = 20usize;
            let base_x = 0.99999998;
            let xs: Vec<f64> = (0..n)
                .map(|i| {
                    if i % 2 == 0 {
                        (base_x + (rng.gen_range_usize(10) as f64) * 1e-9).rem_euclid(1.0)
                    } else {
                        ((rng.gen_range_usize(10) as f64) * 1e-9).rem_euclid(1.0)
                    }
                })
                .collect();
            let ys: Vec<f64> = (0..n)
                .map(|_| (rng.gen_range_usize(20) as f64) / 20.0)
                .collect();
            let layout = Layout2d::from_xy(xs, ys).unwrap();
            let mut edges = Vec::new();
            for _ in 0..35 {
                let u = rng.gen_range_usize(n) as u32;
                let v = rng.gen_range_usize(n) as u32;
                if u != v {
                    edges.push((u, v));
                }
            }
            let expected = torus_brute_force(&edges, &layout);
            let got = torus_crossing_edges_from_edges(&edges, &layout).unwrap();
            assert_eq!(got, expected, "torus steep seed {seed}");
        }

        // Clustered torus layout (4 tight clusters of radius 1e-4 on [0, 1)^2).
        for seed in 1..=25u64 {
            let mut rng = DeterministicRng::from_seed_u64(6000 + seed);
            let centers = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)];
            let n = 24usize;
            let mut xs = Vec::with_capacity(n);
            let mut ys = Vec::with_capacity(n);
            for i in 0..n {
                let (cx, cy) = centers[i % 4];
                xs.push((cx + (rng.next_f64() - 0.5) * 1e-4).rem_euclid(1.0));
                ys.push((cy + (rng.next_f64() - 0.5) * 1e-4).rem_euclid(1.0));
            }
            let layout = Layout2d::from_xy(xs, ys).unwrap();
            let mut edges = Vec::new();
            for _ in 0..45 {
                let u = rng.gen_range_usize(n) as u32;
                let v = rng.gen_range_usize(n) as u32;
                if u != v {
                    edges.push((u, v));
                }
            }
            let expected = torus_brute_force(&edges, &layout);
            let got = torus_crossing_edges_from_edges(&edges, &layout).unwrap();
            assert_eq!(got, expected, "torus clustered seed {seed}");
        }
    }

    #[test]
    fn test_randomized_clustered_and_steep_stress_regression() {
        // 1. 40 clustered seeds (4 tight clusters of radius 1e-4 with inter-cluster chords).
        for seed in 1..=40u64 {
            let mut rng = DeterministicRng::from_seed_u64(10_000 + seed);
            let centers = [(0.0, 0.0), (50.0, 20.0), (50.0, 80.0), (100.0, 0.0)];
            let n = 28usize;
            let mut xs = Vec::with_capacity(n);
            let mut ys = Vec::with_capacity(n);
            for i in 0..n {
                let (cx, cy) = centers[i % 4];
                xs.push(cx + (rng.next_f64() - 0.5) * 1e-4);
                ys.push(cy + (rng.next_f64() - 0.5) * 1e-4);
            }
            let layout = Layout2d::from_xy(xs, ys).unwrap();
            let mut edges = Vec::new();
            for _ in 0..80 {
                let u = rng.gen_range_usize(n) as u32;
                let v = rng.gen_range_usize(n) as u32;
                edges.push((u, v));
            }
            assert_matches_brute_force(&format!("clustered_seed_{seed}"), &edges, &layout);
            assert_streaming_matches_find(
                &format!("clustered_seed_{seed}_stream"),
                &edges,
                &layout,
            );
        }

        // 2. 40 steep seeds at x_off = 100 and x_off = 1e6 with dx ~ 1e-8.
        for seed in 1..=40u64 {
            let mut rng = DeterministicRng::from_seed_u64(20_000 + seed);
            let x_off = if seed <= 20 { 100.0 } else { 1e6 };
            let n = 24usize;
            let xs: Vec<f64> = (0..n)
                .map(|_| x_off + (rng.gen_range_usize(8) as f64) * 1e-8)
                .collect();
            let ys: Vec<f64> = (0..n)
                .map(|_| (rng.gen_range_usize(24) as f64) * 0.5)
                .collect();
            let layout = Layout2d::from_xy(xs, ys).unwrap();
            let mut edges = Vec::new();
            for _ in 0..55 {
                let u = rng.gen_range_usize(n) as u32;
                let v = rng.gen_range_usize(n) as u32;
                edges.push((u, v));
            }
            assert_matches_brute_force(&format!("steep_stress_seed_{seed}"), &edges, &layout);
            assert_streaming_matches_find(
                &format!("steep_stress_seed_{seed}_stream"),
                &edges,
                &layout,
            );
        }

        // 3. 15 tiny 1e-165 scale seeds (det underflows to 0.0) verifying det == 0.0 fallback.
        for seed in 1..=15u64 {
            let mut rng = DeterministicRng::from_seed_u64(30_000 + seed);
            let n = 20usize;
            let xs: Vec<f64> = (0..n)
                .map(|_| (rng.gen_range_usize(11) as f64) * 1e-165)
                .collect();
            let ys: Vec<f64> = (0..n)
                .map(|_| (rng.gen_range_usize(11) as f64) * 1e-165)
                .collect();
            let layout = Layout2d::from_xy(xs, ys).unwrap();
            let mut edges = Vec::new();
            for _ in 0..40 {
                let u = rng.gen_range_usize(n) as u32;
                let v = rng.gen_range_usize(n) as u32;
                edges.push((u, v));
            }
            assert_matches_brute_force(&format!("tiny_1e-165_seed_{seed}"), &edges, &layout);
            assert_streaming_matches_find(
                &format!("tiny_1e-165_seed_{seed}_stream"),
                &edges,
                &layout,
            );
        }
    }
}
