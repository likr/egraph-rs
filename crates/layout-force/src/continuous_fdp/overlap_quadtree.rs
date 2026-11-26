const OVERLAP_DPRIME_MIN: f64 = 1e-6;
const OVERLAP_REPULSION_K: f64 = 100.0;
const MAX_OVERLAP_QUAD_DEPTH: usize = 32;
const OVERLAP_QUAD_COINCIDENT_SQ: f64 = 1e-12;

#[derive(Debug, Clone, Copy)]
struct OverlapQuadNode {
    center_x: f64,
    center_y: f64,
    width: f64,
    height: f64,
    count: usize,
    total_mass: f64,
    com_x: f64,
    com_y: f64,
    max_radius: f64,
    avg_radius: f64,
    leaf_head: u32,
    first_child: u32,
}

impl OverlapQuadNode {
    #[inline]
    fn new(center_x: f64, center_y: f64, width: f64, height: f64) -> Self {
        Self {
            center_x,
            center_y,
            width,
            height,
            count: 0,
            total_mass: 0.0,
            com_x: 0.0,
            com_y: 0.0,
            max_radius: 0.0,
            avg_radius: 0.0,
            leaf_head: u32::MAX,
            first_child: u32::MAX,
        }
    }

    #[inline]
    fn quadrant(&self, x: f64, y: f64) -> usize {
        let right = usize::from(x >= self.center_x);
        let bottom = usize::from(y >= self.center_y) * 2;
        right + bottom
    }

    #[inline]
    fn is_unsplitable_at_depth(&self, depth: usize) -> bool {
        if depth >= MAX_OVERLAP_QUAD_DEPTH {
            return true;
        }
        let qw = self.width * 0.25;
        let qh = self.height * 0.25;
        qw <= 0.0
            || qh <= 0.0
            || (self.center_x - qw) == self.center_x
            || (self.center_x + qw) == self.center_x
            || (self.center_y - qh) == self.center_y
            || (self.center_y + qh) == self.center_y
    }
}

/// Radius-aware flat arena Barnes-Hut quadtree for Jacomy (2014) `prevent_overlap` repulsion.
///
/// Tracks `max_radius` per quadtree cell so multipole approximation is only used when
/// `max_dim < theta * d` AND `d > 2 * (r_i + cell.max_radius)` (strictly well-separated,
/// guaranteeing `d' > 0`), while descending to exact pairwise `d' = d - r_i - r_j` checks
/// for all nearby or overlapping pairs.
#[derive(Debug, Clone, Default)]
pub(super) struct OverlapQuadtree {
    nodes: Vec<OverlapQuadNode>,
    leaf_of_point: Vec<u32>,
    next_in_leaf: Vec<u32>,
}

impl OverlapQuadtree {
    pub(super) fn build(&mut self, xs: &[f64], ys: &[f64], masses: &[f64], radii: &[f64]) {
        self.nodes.clear();
        self.leaf_of_point.clear();
        self.next_in_leaf.clear();

        let n = xs.len();
        if n == 0 {
            return;
        }

        let mut min_x = xs[0];
        let mut max_x = xs[0];
        let mut min_y = ys[0];
        let mut max_y = ys[0];
        for i in 1..n {
            min_x = min_x.min(xs[i]);
            max_x = max_x.max(xs[i]);
            min_y = min_y.min(ys[i]);
            max_y = max_y.max(ys[i]);
        }

        self.leaf_of_point.resize(n, u32::MAX);
        self.next_in_leaf.resize(n, u32::MAX);

        let center_x = 0.5 * (min_x + max_x);
        let center_y = 0.5 * (min_y + max_y);
        let width = ((max_x - min_x) * 1.01).max(1e-4);
        let height = ((max_y - min_y) * 1.01).max(1e-4);
        self.nodes
            .push(OverlapQuadNode::new(center_x, center_y, width, height));

        for i in 0..n {
            self.insert(i, xs, ys, masses, radii);
        }
    }

    fn insert(&mut self, point_idx: usize, xs: &[f64], ys: &[f64], masses: &[f64], radii: &[f64]) {
        let x = xs[point_idx];
        let y = ys[point_idx];
        let mass = masses[point_idx];
        let rad = radii[point_idx];
        let mut cur = 0usize;
        let mut depth = 0usize;

        loop {
            let node = self.nodes[cur];
            if node.count == 0 {
                let slot = &mut self.nodes[cur];
                slot.count = 1;
                slot.total_mass = mass;
                slot.com_x = x;
                slot.com_y = y;
                slot.max_radius = rad;
                slot.avg_radius = rad;
                slot.leaf_head = point_idx as u32;
                self.leaf_of_point[point_idx] = cur as u32;
                self.next_in_leaf[point_idx] = u32::MAX;
                return;
            }

            let old_c = node.count as f64;
            let new_c = (node.count + 1) as f64;
            let old_m = node.total_mass;
            let new_m = old_m + mass;
            {
                let slot = &mut self.nodes[cur];
                if new_m > 0.0 {
                    slot.com_x = (slot.com_x * old_m + x * mass) / new_m;
                    slot.com_y = (slot.com_y * old_m + y * mass) / new_m;
                    slot.avg_radius = (slot.avg_radius * old_m + rad * mass) / new_m;
                } else {
                    slot.com_x = (slot.com_x * old_c + x) / new_c;
                    slot.com_y = (slot.com_y * old_c + y) / new_c;
                    slot.avg_radius = (slot.avg_radius * old_c + rad) / new_c;
                }
                slot.max_radius = slot.max_radius.max(rad);
                slot.total_mass = new_m;
                slot.count += 1;
            }

            if node.first_child == u32::MAX {
                let ref_idx = node.leaf_head as usize;
                let rx = xs[ref_idx];
                let ry = ys[ref_idx];
                let dist_sq = (rx - x) * (rx - x) + (ry - y) * (ry - y);
                if !dist_sq.is_finite()
                    || dist_sq < OVERLAP_QUAD_COINCIDENT_SQ
                    || node.is_unsplitable_at_depth(depth)
                {
                    self.next_in_leaf[point_idx] = self.nodes[cur].leaf_head;
                    self.nodes[cur].leaf_head = point_idx as u32;
                    self.leaf_of_point[point_idx] = cur as u32;
                    return;
                }

                let head = self.nodes[cur].leaf_head;
                self.nodes[cur].leaf_head = u32::MAX;
                let base_child = self.subdivide(cur);

                let mut existing = head;
                while existing != u32::MAX {
                    let e_idx = existing as usize;
                    let next = self.next_in_leaf[e_idx];
                    self.next_in_leaf[e_idx] = u32::MAX;
                    let ex = xs[e_idx];
                    let ey = ys[e_idx];
                    let em = masses[e_idx];
                    let er = radii[e_idx];
                    let q = self.nodes[cur].quadrant(ex, ey);
                    let child = base_child + q;
                    let c_node = &mut self.nodes[child];
                    if c_node.count == 0 {
                        c_node.count = 1;
                        c_node.total_mass = em;
                        c_node.com_x = ex;
                        c_node.com_y = ey;
                        c_node.max_radius = er;
                        c_node.avg_radius = er;
                        c_node.leaf_head = e_idx as u32;
                        self.leaf_of_point[e_idx] = child as u32;
                    } else {
                        let oc = c_node.count as f64;
                        let nc = (c_node.count + 1) as f64;
                        let om = c_node.total_mass;
                        let nm = om + em;
                        if nm > 0.0 {
                            c_node.com_x = (c_node.com_x * om + ex * em) / nm;
                            c_node.com_y = (c_node.com_y * om + ey * em) / nm;
                            c_node.avg_radius = (c_node.avg_radius * om + er * em) / nm;
                        } else {
                            c_node.com_x = (c_node.com_x * oc + ex) / nc;
                            c_node.com_y = (c_node.com_y * oc + ey) / nc;
                            c_node.avg_radius = (c_node.avg_radius * oc + er) / nc;
                        }
                        c_node.max_radius = c_node.max_radius.max(er);
                        c_node.total_mass = nm;
                        c_node.count += 1;
                        self.next_in_leaf[e_idx] = c_node.leaf_head;
                        c_node.leaf_head = e_idx as u32;
                        self.leaf_of_point[e_idx] = child as u32;
                    }
                    existing = next;
                }

                let q_new = self.nodes[cur].quadrant(x, y);
                cur = base_child + q_new;
                depth += 1;
            } else {
                let q = node.quadrant(x, y);
                cur = (node.first_child as usize) + q;
                depth += 1;
            }
        }
    }

    fn subdivide(&mut self, parent_idx: usize) -> usize {
        let p = self.nodes[parent_idx];
        let hw = p.width * 0.5;
        let hh = p.height * 0.5;
        let qw = p.width * 0.25;
        let qh = p.height * 0.25;
        let base = self.nodes.len();
        self.nodes[parent_idx].first_child = base as u32;
        self.nodes.push(OverlapQuadNode::new(
            p.center_x - qw,
            p.center_y - qh,
            hw,
            hh,
        ));
        self.nodes.push(OverlapQuadNode::new(
            p.center_x + qw,
            p.center_y - qh,
            hw,
            hh,
        ));
        self.nodes.push(OverlapQuadNode::new(
            p.center_x - qw,
            p.center_y + qh,
            hw,
            hh,
        ));
        self.nodes.push(OverlapQuadNode::new(
            p.center_x + qw,
            p.center_y + qh,
            hw,
            hh,
        ));
        base
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn compute_repulsion(
        &self,
        xs: &[f64],
        ys: &[f64],
        masses: &[f64],
        radii: &[f64],
        theta: f64,
        scaling_ratio: f64,
        seed: u64,
        iter: usize,
        out_fx: &mut [f64],
        out_fy: &mut [f64],
    ) {
        let n = xs.len();
        if n < 2 || self.nodes.is_empty() {
            out_fx.fill(0.0);
            out_fy.fill(0.0);
            return;
        }
        let theta_sq = theta * theta;
        let mut stack = [0u32; 128];

        for i in 0..n {
            let x = xs[i];
            let y = ys[i];
            let mi = masses[i];
            let ri = radii[i];
            let mut sp = 1usize;
            stack[0] = 0;
            let mut fx = 0.0_f64;
            let mut fy = 0.0_f64;

            while sp > 0 {
                sp -= 1;
                let node_idx = stack[sp];
                let node = &self.nodes[node_idx as usize];
                if node.count == 0 || node.total_mass == 0.0 {
                    continue;
                }

                if node.first_child == u32::MAX {
                    let mut p = node.leaf_head;
                    while p != u32::MAX {
                        let other = p as usize;
                        if other != i {
                            let (px, py) = overlap_repulsion_pair(
                                x - xs[other],
                                y - ys[other],
                                mi,
                                masses[other],
                                ri + radii[other],
                                scaling_ratio,
                                seed,
                                iter,
                                i,
                                other,
                            );
                            fx += px;
                            fy += py;
                        }
                        p = self.next_in_leaf[other];
                    }
                    continue;
                }

                let dx = x - node.com_x;
                let dy = y - node.com_y;
                let dist_sq = dx * dx + dy * dy;
                let max_dim = node.width.max(node.height);
                if theta > 0.0 && (max_dim * max_dim) < theta_sq * dist_sq {
                    let d = dist_sq.sqrt();
                    let min_sep = 2.0 * (ri + node.max_radius);
                    if d > min_sep && d > 1e-12 {
                        let d_prime = (d - ri - node.avg_radius).max(OVERLAP_DPRIME_MIN);
                        let mult = (scaling_ratio * mi * node.total_mass) / (d * d_prime);
                        fx += mult * dx;
                        fy += mult * dy;
                        continue;
                    }
                }
                let base = node.first_child;
                for c in 0..4 {
                    let child = base + c;
                    if self.nodes[child as usize].count > 0 {
                        stack[sp] = child;
                        sp += 1;
                    }
                }
            }

            out_fx[i] = fx;
            out_fy[i] = fy;
        }
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn overlap_repulsion_pair(
    dx: f64,
    dy: f64,
    mi: f64,
    mj: f64,
    r_sum: f64,
    scaling_ratio: f64,
    seed: u64,
    iter: usize,
    i: usize,
    j: usize,
) -> (f64, f64) {
    let d = dx.hypot(dy);
    let d_prime = d - r_sum;
    if d_prime > 0.0 {
        let d_prime_safe = d_prime.max(OVERLAP_DPRIME_MIN);
        let mult = (scaling_ratio * mi * mj) / (d * d_prime_safe);
        (mult * dx, mult * dy)
    } else if d_prime < 0.0 {
        let (ux, uy) = if d > 1e-12 {
            (dx / d, dy / d)
        } else {
            deterministic_pair_unit_vector(seed, iter, i, j)
        };
        let mag = OVERLAP_REPULSION_K * mi * mj;
        (mag * ux, mag * uy)
    } else {
        (0.0, 0.0)
    }
}

#[inline]
fn deterministic_pair_unit_vector(seed: u64, iter: usize, i: usize, j: usize) -> (f64, f64) {
    let (a, b) = if i <= j {
        (i as u64, j as u64)
    } else {
        (j as u64, i as u64)
    };
    let mut z = seed
        .wrapping_add((iter as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add(a.wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add(b.wrapping_mul(0x94D0_49BB_1331_11EB));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let u = ((z >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
    let angle = u * std::f64::consts::TAU;
    let sign = if i <= j { 1.0 } else { -1.0 };
    (sign * angle.cos(), sign * angle.sin())
}
