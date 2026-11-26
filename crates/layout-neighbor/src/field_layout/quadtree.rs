use egraph_core::{CoreError, CoreResult, Scalar};

const QUADTREE_MAX_DEPTH: usize = 32;
const QUADTREE_COINCIDENT_DIST_SQ: f64 = 1e-12;

#[derive(Debug, Clone, Copy)]
pub(super) struct FieldQuadNode {
    center_x: f64,
    center_y: f64,
    width: f64,
    height: f64,
    count: usize,
    com_x: f64,
    com_y: f64,
    leaf_head: u32,
    first_child: u32,
}

impl FieldQuadNode {
    #[inline]
    fn new(center_x: f64, center_y: f64, width: f64, height: f64) -> Self {
        Self {
            center_x,
            center_y,
            width,
            height,
            count: 0,
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
        if depth >= QUADTREE_MAX_DEPTH {
            return true;
        }
        let quarter_w = self.width * 0.25;
        let quarter_h = self.height * 0.25;
        quarter_w <= 0.0
            || quarter_h <= 0.0
            || (self.center_x - quarter_w) == self.center_x
            || (self.center_x + quarter_w) == self.center_x
            || (self.center_y - quarter_h) == self.center_y
            || (self.center_y + quarter_h) == self.center_y
    }
}

/// Reusable arena quadtree for arbitrary isotropic radial kernels $K(d^2)$ in Barnes-Hut mode.
#[derive(Debug, Clone, Default)]
pub(super) struct FieldQuadtree {
    pub(super) nodes: Vec<FieldQuadNode>,
    coords_x: Vec<f64>,
    coords_y: Vec<f64>,
    leaf_of_point: Vec<u32>,
    next_in_leaf: Vec<u32>,
}

impl FieldQuadtree {
    pub(super) fn new() -> Self {
        Self::default()
    }

    fn clear(&mut self) {
        self.nodes.clear();
        self.coords_x.clear();
        self.coords_y.clear();
        self.leaf_of_point.clear();
        self.next_in_leaf.clear();
    }

    pub(super) fn build_xy<S: Scalar>(&mut self, xs: &[S], ys: &[S]) -> CoreResult<()> {
        if xs.len() != ys.len() {
            return Err(CoreError::dimension_mismatch(
                xs.len(),
                ys.len(),
                "FieldQuadtree::build_xy",
            ));
        }
        let n = xs.len();
        for i in 0..n {
            if !xs[i].is_finite() || !ys[i].is_finite() {
                return Err(CoreError::non_finite("FieldQuadtree::build_xy"));
            }
        }
        self.clear();
        if n == 0 {
            return Ok(());
        }

        self.coords_x.reserve(n);
        self.coords_y.reserve(n);
        let mut min_x = xs[0].to_f64();
        let mut max_x = min_x;
        let mut min_y = ys[0].to_f64();
        let mut max_y = min_y;

        for i in 0..n {
            let x = xs[i].to_f64();
            let y = ys[i].to_f64();
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
            self.coords_x.push(x);
            self.coords_y.push(y);
        }

        self.leaf_of_point.resize(n, u32::MAX);
        self.next_in_leaf.resize(n, u32::MAX);

        let center_x = 0.5 * (min_x + max_x);
        let center_y = 0.5 * (min_y + max_y);
        let width = ((max_x - min_x) * 1.01).max(1e-4);
        let height = ((max_y - min_y) * 1.01).max(1e-4);
        self.nodes
            .push(FieldQuadNode::new(center_x, center_y, width, height));

        for i in 0..n {
            self.insert_iterative(i, self.coords_x[i], self.coords_y[i]);
        }
        Ok(())
    }

    fn insert_iterative(&mut self, point_idx: usize, x: f64, y: f64) {
        let mut cur = 0usize;
        let mut depth = 0usize;

        loop {
            let node = self.nodes[cur];
            if node.count == 0 {
                let slot = &mut self.nodes[cur];
                slot.count = 1;
                slot.com_x = x;
                slot.com_y = y;
                slot.leaf_head = point_idx as u32;
                self.leaf_of_point[point_idx] = cur as u32;
                self.next_in_leaf[point_idx] = u32::MAX;
                return;
            }

            let old_c = node.count as f64;
            let new_c = (node.count + 1) as f64;
            {
                let slot = &mut self.nodes[cur];
                slot.com_x = (slot.com_x * old_c + x) / new_c;
                slot.com_y = (slot.com_y * old_c + y) / new_c;
                slot.count += 1;
            }

            if node.first_child == u32::MAX {
                let ref_idx = node.leaf_head as usize;
                let rx = self.coords_x[ref_idx];
                let ry = self.coords_y[ref_idx];
                let dist_sq = (rx - x) * (rx - x) + (ry - y) * (ry - y);
                if !dist_sq.is_finite()
                    || dist_sq < QUADTREE_COINCIDENT_DIST_SQ
                    || node.is_unsplitable_at_depth(depth)
                {
                    self.next_in_leaf[point_idx] = self.nodes[cur].leaf_head;
                    self.nodes[cur].leaf_head = point_idx as u32;
                    self.leaf_of_point[point_idx] = cur as u32;
                    return;
                }

                let head = self.nodes[cur].leaf_head;
                self.nodes[cur].leaf_head = u32::MAX;
                let base_child = self.subdivide_node(cur);

                let mut existing = head;
                while existing != u32::MAX {
                    let e_idx = existing as usize;
                    let next = self.next_in_leaf[e_idx];
                    self.next_in_leaf[e_idx] = u32::MAX;
                    let ex = self.coords_x[e_idx];
                    let ey = self.coords_y[e_idx];
                    let q = self.nodes[cur].quadrant(ex, ey);
                    self.insert_into_subtree(base_child + q, depth + 1, e_idx, ex, ey);
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

    fn insert_into_subtree(
        &mut self,
        mut cur: usize,
        mut depth: usize,
        point_idx: usize,
        x: f64,
        y: f64,
    ) {
        loop {
            let node = self.nodes[cur];
            if node.count == 0 {
                let slot = &mut self.nodes[cur];
                slot.count = 1;
                slot.com_x = x;
                slot.com_y = y;
                slot.leaf_head = point_idx as u32;
                self.leaf_of_point[point_idx] = cur as u32;
                self.next_in_leaf[point_idx] = u32::MAX;
                return;
            }

            let old_c = node.count as f64;
            let new_c = (node.count + 1) as f64;
            {
                let slot = &mut self.nodes[cur];
                slot.com_x = (slot.com_x * old_c + x) / new_c;
                slot.com_y = (slot.com_y * old_c + y) / new_c;
                slot.count += 1;
            }

            if node.first_child == u32::MAX {
                let ref_idx = node.leaf_head as usize;
                let rx = self.coords_x[ref_idx];
                let ry = self.coords_y[ref_idx];
                let dist_sq = (rx - x) * (rx - x) + (ry - y) * (ry - y);
                if !dist_sq.is_finite()
                    || dist_sq < QUADTREE_COINCIDENT_DIST_SQ
                    || node.is_unsplitable_at_depth(depth)
                {
                    self.next_in_leaf[point_idx] = self.nodes[cur].leaf_head;
                    self.nodes[cur].leaf_head = point_idx as u32;
                    self.leaf_of_point[point_idx] = cur as u32;
                    return;
                }

                let head = self.nodes[cur].leaf_head;
                self.nodes[cur].leaf_head = u32::MAX;
                let base_child = self.subdivide_node(cur);
                let mut existing = head;
                while existing != u32::MAX {
                    let e_idx = existing as usize;
                    let next = self.next_in_leaf[e_idx];
                    self.next_in_leaf[e_idx] = u32::MAX;
                    let ex = self.coords_x[e_idx];
                    let ey = self.coords_y[e_idx];
                    let q = self.nodes[cur].quadrant(ex, ey);
                    let child = base_child + q;
                    let c_node = &mut self.nodes[child];
                    if c_node.count == 0 {
                        c_node.count = 1;
                        c_node.com_x = ex;
                        c_node.com_y = ey;
                        c_node.leaf_head = e_idx as u32;
                        self.leaf_of_point[e_idx] = child as u32;
                    } else {
                        let oc = c_node.count as f64;
                        let nc = (c_node.count + 1) as f64;
                        c_node.com_x = (c_node.com_x * oc + ex) / nc;
                        c_node.com_y = (c_node.com_y * oc + ey) / nc;
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

    fn subdivide_node(&mut self, parent_idx: usize) -> usize {
        let p = self.nodes[parent_idx];
        let half_w = p.width * 0.5;
        let half_h = p.height * 0.5;
        let qw = p.width * 0.25;
        let qh = p.height * 0.25;

        let base = self.nodes.len();
        self.nodes[parent_idx].first_child = base as u32;
        self.nodes.push(FieldQuadNode::new(
            p.center_x - qw,
            p.center_y - qh,
            half_w,
            half_h,
        ));
        self.nodes.push(FieldQuadNode::new(
            p.center_x + qw,
            p.center_y - qh,
            half_w,
            half_h,
        ));
        self.nodes.push(FieldQuadNode::new(
            p.center_x - qw,
            p.center_y + qh,
            half_w,
            half_h,
        ));
        self.nodes.push(FieldQuadNode::new(
            p.center_x + qw,
            p.center_y + qh,
            half_w,
            half_h,
        ));
        base
    }

    pub(super) fn compute_isotropic_repulsion<S: Scalar, F: Fn(f64) -> f64>(
        &self,
        theta: f64,
        kernel_fn: F,
        out_x: &mut [S],
        out_y: &mut [S],
    ) {
        let n = self.coords_x.len();
        if n < 2 || self.nodes.is_empty() {
            out_x.fill(S::ZERO);
            out_y.fill(S::ZERO);
            return;
        }
        let theta_sq = theta * theta;
        let mut stack = [0u32; 128];

        for i in 0..n {
            let x = self.coords_x[i];
            let y = self.coords_y[i];
            let own_leaf = self.leaf_of_point[i];
            let mut sp = 1usize;
            stack[0] = 0;

            let mut fx = 0.0_f64;
            let mut fy = 0.0_f64;

            while sp > 0 {
                sp -= 1;
                let node_idx = stack[sp];
                let node = &self.nodes[node_idx as usize];
                if node.count == 0 {
                    continue;
                }

                if node.first_child == u32::MAX {
                    if node_idx == own_leaf {
                        let mut p = node.leaf_head;
                        while p != u32::MAX {
                            let other = p as usize;
                            if other != i {
                                let dx = x - self.coords_x[other];
                                let dy = y - self.coords_y[other];
                                let k = kernel_fn(dx * dx + dy * dy);
                                fx += k * dx;
                                fy += k * dy;
                            }
                            p = self.next_in_leaf[other];
                        }
                    } else {
                        let dx = x - node.com_x;
                        let dy = y - node.com_y;
                        let k = (node.count as f64) * kernel_fn(dx * dx + dy * dy);
                        fx += k * dx;
                        fy += k * dy;
                    }
                    continue;
                }

                let dx = x - node.com_x;
                let dy = y - node.com_y;
                let dist_sq = dx * dx + dy * dy;
                let max_dim = node.width.max(node.height);
                if (max_dim * max_dim) < theta_sq * dist_sq {
                    let k = (node.count as f64) * kernel_fn(dist_sq);
                    fx += k * dx;
                    fy += k * dy;
                } else {
                    let base = node.first_child;
                    for c in 0..4 {
                        let child = base + c;
                        if self.nodes[child as usize].count > 0 && sp < stack.len() {
                            stack[sp] = child;
                            sp += 1;
                        }
                    }
                }
            }

            out_x[i] = S::from_f64(fx);
            out_y[i] = S::from_f64(fy);
        }
    }
}
