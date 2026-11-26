const MAX_QUAD_DEPTH: usize = 32;
const COINCIDENT_DIST_SQ: f64 = 1e-12;

#[derive(Debug, Clone, Copy)]
struct GenQuadNode {
    center_x: f64,
    center_y: f64,
    width: f64,
    height: f64,
    count: usize,
    total_mass: f64,
    com_x: f64,
    com_y: f64,
    leaf_head: u32,
    first_child: u32,
}

impl GenQuadNode {
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
        if depth >= MAX_QUAD_DEPTH {
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

/// Flat arena Barnes-Hut quadtree for generalized $d^r$ repulsion ($r \neq -1$).
#[derive(Debug, Clone, Default)]
pub(super) struct GenRepulsionQuadtree {
    nodes: Vec<GenQuadNode>,
    leaf_of_point: Vec<u32>,
    next_in_leaf: Vec<u32>,
}

impl GenRepulsionQuadtree {
    pub(super) fn build(&mut self, xs: &[f64], ys: &[f64], masses: &[f64]) {
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
            .push(GenQuadNode::new(center_x, center_y, width, height));

        for i in 0..n {
            self.insert(i, xs, ys, masses);
        }
    }

    fn insert(&mut self, point_idx: usize, xs: &[f64], ys: &[f64], masses: &[f64]) {
        let x = xs[point_idx];
        let y = ys[point_idx];
        let mass = masses[point_idx];
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
                } else {
                    slot.com_x = (slot.com_x * old_c + x) / new_c;
                    slot.com_y = (slot.com_y * old_c + y) / new_c;
                }
                slot.total_mass = new_m;
                slot.count += 1;
            }

            if node.first_child == u32::MAX {
                let ref_idx = node.leaf_head as usize;
                let rx = xs[ref_idx];
                let ry = ys[ref_idx];
                let dist_sq = (rx - x) * (rx - x) + (ry - y) * (ry - y);
                if !dist_sq.is_finite()
                    || dist_sq < COINCIDENT_DIST_SQ
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
                    let q = self.nodes[cur].quadrant(ex, ey);
                    let child = base_child + q;
                    let c_node = &mut self.nodes[child];
                    if c_node.count == 0 {
                        c_node.count = 1;
                        c_node.total_mass = em;
                        c_node.com_x = ex;
                        c_node.com_y = ey;
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
                        } else {
                            c_node.com_x = (c_node.com_x * oc + ex) / nc;
                            c_node.com_y = (c_node.com_y * oc + ey) / nc;
                        }
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
        self.nodes
            .push(GenQuadNode::new(p.center_x - qw, p.center_y - qh, hw, hh));
        self.nodes
            .push(GenQuadNode::new(p.center_x + qw, p.center_y - qh, hw, hh));
        self.nodes
            .push(GenQuadNode::new(p.center_x - qw, p.center_y + qh, hw, hh));
        self.nodes
            .push(GenQuadNode::new(p.center_x + qw, p.center_y + qh, hw, hh));
        base
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn compute_general_repulsion(
        &self,
        xs: &[f64],
        ys: &[f64],
        masses: &[f64],
        theta: f64,
        repulsion_exponent: f64,
        eps_sq: f64,
        out_x: &mut [f64],
        out_y: &mut [f64],
    ) {
        let n = xs.len();
        if n < 2 || self.nodes.is_empty() {
            out_x.fill(0.0);
            out_y.fill(0.0);
            return;
        }
        let theta_sq = theta * theta;
        let pow_exp = 0.5 * (repulsion_exponent - 1.0);
        let mut stack = [0u32; 128];

        for i in 0..n {
            let x = xs[i];
            let y = ys[i];
            let own_leaf = self.leaf_of_point[i];
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
                    if node_idx == own_leaf {
                        let mut p = node.leaf_head;
                        while p != u32::MAX {
                            let other = p as usize;
                            if other != i {
                                let dx = x - xs[other];
                                let dy = y - ys[other];
                                let d_sq = dx * dx + dy * dy + eps_sq;
                                let mult = masses[other] * d_sq.powf(pow_exp);
                                fx += mult * dx;
                                fy += mult * dy;
                            }
                            p = self.next_in_leaf[other];
                        }
                    } else {
                        let dx = x - node.com_x;
                        let dy = y - node.com_y;
                        let d_sq = dx * dx + dy * dy + eps_sq;
                        let mult = node.total_mass * d_sq.powf(pow_exp);
                        fx += mult * dx;
                        fy += mult * dy;
                    }
                    continue;
                }

                let dx = x - node.com_x;
                let dy = y - node.com_y;
                let dist_sq = dx * dx + dy * dy;
                let max_dim = node.width.max(node.height);
                if (max_dim * max_dim) < theta_sq * dist_sq {
                    let mult = node.total_mass * (dist_sq + eps_sq).powf(pow_exp);
                    fx += mult * dx;
                    fy += mult * dy;
                } else {
                    let base = node.first_child;
                    for c in 0..4 {
                        let child = base + c;
                        if self.nodes[child as usize].count > 0 {
                            stack[sp] = child;
                            sp += 1;
                        }
                    }
                }
            }

            out_x[i] = fx;
            out_y[i] = fy;
        }
    }
}
