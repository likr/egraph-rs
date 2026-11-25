//! Flat `Vec`-backed arena Barnes-Hut 2D quadtree for KL and entropy repulsion forces.
//!
//! Eliminates pointer-chasing boxed-node allocations by storing tree nodes and
//! coincident-point chains inside contiguous reusable `Vec` arenas. All geometry and moments
//! are accumulated in `f64` for numerical stability when `S = f32`.

use std::marker::PhantomData;

use egraph_core::{CoreError, CoreResult, Layout2d, Scalar};

const MAX_DEPTH: usize = 32;
const COINCIDENT_DIST_SQ: f64 = 1e-12;
const ENTROPY_MIN_DIST: f64 = 1e-6;

#[derive(Debug, Clone, Copy)]
struct ArenaNode {
    center_x: f64,
    center_y: f64,
    width: f64,
    height: f64,
    count: usize,
    total_mass: f64,
    com_x: f64,
    com_y: f64,
    /// Head of intrusive `next_in_leaf` point list when leaf (`u32::MAX` if empty or internal).
    leaf_head: u32,
    /// Base index of the 4 child quadrants (`NW, NE, SW, SE`) or `u32::MAX` if leaf.
    first_child: u32,
}

impl ArenaNode {
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
        if depth >= MAX_DEPTH {
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

/// Flat arena-allocated Barnes-Hut quadtree in 2D Euclidean space.
///
/// Repeated calls to [`ArenaQuadtree2d::clear`] and [`ArenaQuadtree2d::build`] retain backing
/// `Vec` capacities so iterative layouts perform zero per-step heap allocations.
#[derive(Debug, Clone)]
pub struct ArenaQuadtree2d<S: Scalar = f64> {
    nodes: Vec<ArenaNode>,
    coords_x: Vec<f64>,
    coords_y: Vec<f64>,
    masses: Vec<f64>,
    leaf_of_point: Vec<u32>,
    next_in_leaf: Vec<u32>,
    _marker: PhantomData<S>,
}

impl<S: Scalar> Default for ArenaQuadtree2d<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Scalar> ArenaQuadtree2d<S> {
    /// Creates an empty reusable quadtree arena.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            coords_x: Vec::new(),
            coords_y: Vec::new(),
            masses: Vec::new(),
            leaf_of_point: Vec::new(),
            next_in_leaf: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// Clears all nodes and points while retaining allocated buffer capacities.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.coords_x.clear();
        self.coords_y.clear();
        self.masses.clear();
        self.leaf_of_point.clear();
        self.next_in_leaf.clear();
    }

    /// Returns the number of points indexed in the tree.
    #[inline]
    pub fn point_count(&self) -> usize {
        self.coords_x.len()
    }

    /// Returns the number of allocated arena nodes (including internal and empty leaf slots).
    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Rebuilds the quadtree from `layout`, validating that every coordinate is finite.
    pub fn build(&mut self, layout: &Layout2d<S>) -> CoreResult<()> {
        self.build_xy(layout.x(), layout.y())
    }

    /// Rebuilds the quadtree from separate `x` and `y` slices with unit masses (`1.0`).
    pub fn build_xy(&mut self, xs: &[S], ys: &[S]) -> CoreResult<()> {
        self.build_inner(xs, ys, None, "ArenaQuadtree2d::build_xy")
    }

    /// Rebuilds the weighted quadtree from `layout` and per-point `masses`.
    pub fn build_weighted(&mut self, layout: &Layout2d<S>, masses: &[f64]) -> CoreResult<()> {
        self.build_weighted_xy(layout.x(), layout.y(), masses)
    }

    /// Rebuilds the weighted quadtree from separate `x`, `y`, and `masses` slices.
    pub fn build_weighted_xy(&mut self, xs: &[S], ys: &[S], masses: &[f64]) -> CoreResult<()> {
        self.build_inner(xs, ys, Some(masses), "ArenaQuadtree2d::build_weighted_xy")
    }

    fn build_inner(
        &mut self,
        xs: &[S],
        ys: &[S],
        masses: Option<&[f64]>,
        context: &'static str,
    ) -> CoreResult<()> {
        if xs.len() != ys.len() {
            return Err(CoreError::dimension_mismatch(xs.len(), ys.len(), context));
        }
        let n = xs.len();
        if let Some(m) = masses {
            if m.len() != n {
                return Err(CoreError::dimension_mismatch(
                    n,
                    m.len(),
                    "ArenaQuadtree2d::build_weighted_xy::masses",
                ));
            }
            for &w in m {
                if !w.is_finite() {
                    return Err(CoreError::non_finite(
                        "ArenaQuadtree2d::build_weighted_xy::masses",
                    ));
                }
                if w < 0.0 {
                    return Err(CoreError::invalid_input("masses must be finite and >= 0.0"));
                }
            }
        }
        for i in 0..n {
            if !xs[i].is_finite() || !ys[i].is_finite() {
                return Err(CoreError::non_finite(context));
            }
        }

        self.clear();
        if n == 0 {
            return Ok(());
        }

        self.coords_x.reserve(n);
        self.coords_y.reserve(n);
        self.masses.reserve(n);
        let mut min_x = xs[0].to_f64();
        let mut max_x = min_x;
        let mut min_y = ys[0].to_f64();
        let mut max_y = min_y;

        for i in 0..n {
            let x = xs[i].to_f64();
            let y = ys[i].to_f64();
            let m = masses.map_or(1.0, |ms| ms[i]);
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
            self.coords_x.push(x);
            self.coords_y.push(y);
            self.masses.push(m);
        }

        self.leaf_of_point.resize(n, u32::MAX);
        self.next_in_leaf.resize(n, u32::MAX);

        let center_x = 0.5 * (min_x + max_x);
        let center_y = 0.5 * (min_y + max_y);
        let width = ((max_x - min_x) * 1.01).max(1e-4);
        let height = ((max_y - min_y) * 1.01).max(1e-4);

        self.nodes
            .push(ArenaNode::new(center_x, center_y, width, height));

        for i in 0..n {
            self.insert_iterative(i, self.coords_x[i], self.coords_y[i], self.masses[i]);
        }
        Ok(())
    }

    fn insert_iterative(&mut self, point_idx: usize, x: f64, y: f64, mass: f64) {
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

            // Update center of mass incrementally before descending or remaining at leaf.
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
                let rx = self.coords_x[ref_idx];
                let ry = self.coords_y[ref_idx];
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

                // Subdivide leaf and re-insert its existing point chain into children.
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
                    let em = self.masses[e_idx];
                    let q = self.nodes[cur].quadrant(ex, ey);
                    self.insert_into_subtree(base_child + q, depth + 1, e_idx, ex, ey, em);
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
        mass: f64,
    ) {
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
                let rx = self.coords_x[ref_idx];
                let ry = self.coords_y[ref_idx];
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

                // Re-insert existing single/coincident chain when separating distinct points.
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
                    let em = self.masses[e_idx];
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

    fn subdivide_node(&mut self, parent_idx: usize) -> usize {
        let p = self.nodes[parent_idx];
        let half_w = p.width * 0.5;
        let half_h = p.height * 0.5;
        let qw = p.width * 0.25;
        let qh = p.height * 0.25;

        let base = self.nodes.len();
        self.nodes[parent_idx].first_child = base as u32;
        self.nodes.push(ArenaNode::new(
            p.center_x - qw,
            p.center_y - qh,
            half_w,
            half_h,
        ));
        self.nodes.push(ArenaNode::new(
            p.center_x + qw,
            p.center_y - qh,
            half_w,
            half_h,
        ));
        self.nodes.push(ArenaNode::new(
            p.center_x - qw,
            p.center_y + qh,
            half_w,
            half_h,
        ));
        self.nodes.push(ArenaNode::new(
            p.center_x + qw,
            p.center_y + qh,
            half_w,
            half_h,
        ));
        base
    }

    /// Evaluates unnormalized Barnes-Hut KL sum `(z_i, raw_fx_i, raw_fy_i)` for vertex `point_idx`.
    ///
    /// Uses an explicit stack array bounded by `MAX_DEPTH` (`3 * 32 + 1 = 97 <= 128`) so
    /// traversal is strictly non-recursive and allocation-free.
    pub fn eval_point_kl_raw(
        &self,
        point_idx: usize,
        theta_sq: f64,
    ) -> CoreResult<(f64, f64, f64)> {
        let n = self.point_count();
        if point_idx >= n {
            return Err(CoreError::index_out_of_bounds(
                point_idx,
                n,
                "ArenaQuadtree2d::eval_point_kl_raw",
            ));
        }
        if !theta_sq.is_finite() || theta_sq < 0.0 {
            return Err(CoreError::invalid_input(
                "theta_sq must be finite and >= 0.0",
            ));
        }
        if self.nodes.is_empty() {
            return Ok((0.0, 0.0, 0.0));
        }

        let x = self.coords_x[point_idx];
        let y = self.coords_y[point_idx];
        let own_leaf = self.leaf_of_point[point_idx];

        let mut stack = [0u32; 128];
        let mut sp = 1usize;
        stack[0] = 0;

        let mut z_sum = 0.0_f64;
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
                    // Exact per-point leaf summation excludes self while handling sub-ULP neighbors.
                    let mut p = node.leaf_head;
                    while p != u32::MAX {
                        let other = p as usize;
                        if other != point_idx {
                            let dx = x - self.coords_x[other];
                            let dy = y - self.coords_y[other];
                            let w = 1.0 / (1.0 + dx * dx + dy * dy);
                            z_sum += w;
                            let mult = w * w;
                            fx += mult * dx;
                            fy += mult * dy;
                        }
                        p = self.next_in_leaf[other];
                    }
                } else {
                    let dx = x - node.com_x;
                    let dy = y - node.com_y;
                    let w = 1.0 / (1.0 + dx * dx + dy * dy);
                    let count_f = node.count as f64;
                    z_sum += count_f * w;
                    let mult = count_f * w * w;
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
                let w = 1.0 / (1.0 + dist_sq);
                let count_f = node.count as f64;
                z_sum += count_f * w;
                let mult = count_f * w * w;
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

        Ok((z_sum, fx, fy))
    }

    /// Evaluates normalized Barnes-Hut KL repulsive forces across all points and returns `total_z`.
    pub fn compute_kl_repulsion(
        &self,
        theta: S,
        out_x: &mut [S],
        out_y: &mut [S],
    ) -> CoreResult<S> {
        let n = self.point_count();
        if out_x.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                out_x.len(),
                "ArenaQuadtree2d::compute_kl_repulsion::out_x",
            ));
        }
        if out_y.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                out_y.len(),
                "ArenaQuadtree2d::compute_kl_repulsion::out_y",
            ));
        }
        let theta_f = theta.to_f64();
        if !theta_f.is_finite() || theta_f < 0.0 {
            return Err(CoreError::invalid_input("theta must be finite and >= 0.0"));
        }
        if n < 2 {
            out_x.fill(S::ZERO);
            out_y.fill(S::ZERO);
            return Ok(S::ONE);
        }

        let theta_sq = theta_f * theta_f;
        let mut total_z = 0.0_f64;
        for i in 0..n {
            let (zi, fx, fy) = self.eval_point_kl_raw(i, theta_sq)?;
            total_z += zi;
            out_x[i] = S::from_f64(fx);
            out_y[i] = S::from_f64(fy);
        }

        let denom = total_z.max(1e-12);
        for i in 0..n {
            out_x[i] = S::from_f64(out_x[i].to_f64() / denom);
            out_y[i] = S::from_f64(out_y[i].to_f64() / denom);
        }
        Ok(S::from_f64(denom))
    }

    /// Evaluates unnormalized Barnes-Hut entropy repulsion `(fx, fy)` for vertex `point_idx`:
    /// $\sum_{j \neq i} \frac{y_i - y_j}{\|y_i - y_j\|(\|y_i - y_j\| + \varepsilon_r)}$.
    pub fn eval_point_entropy_raw(
        &self,
        point_idx: usize,
        theta_sq: f64,
        epsilon_r: f64,
    ) -> CoreResult<(f64, f64)> {
        let n = self.point_count();
        if point_idx >= n {
            return Err(CoreError::index_out_of_bounds(
                point_idx,
                n,
                "ArenaQuadtree2d::eval_point_entropy_raw",
            ));
        }
        if !theta_sq.is_finite() || theta_sq < 0.0 {
            return Err(CoreError::invalid_input(
                "theta_sq must be finite and >= 0.0",
            ));
        }
        if !epsilon_r.is_finite() || epsilon_r <= 0.0 {
            return Err(CoreError::invalid_input(
                "epsilon_r must be finite and > 0.0",
            ));
        }
        if self.nodes.is_empty() {
            return Ok((0.0, 0.0));
        }

        let x = self.coords_x[point_idx];
        let y = self.coords_y[point_idx];
        let own_leaf = self.leaf_of_point[point_idx];

        let mut stack = [0u32; 128];
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
                        if other != point_idx {
                            let dx = x - self.coords_x[other];
                            let dy = y - self.coords_y[other];
                            let dist = dx.hypot(dy);
                            if dist > ENTROPY_MIN_DIST {
                                let mult = 1.0 / (dist * (dist + epsilon_r));
                                fx += mult * dx;
                                fy += mult * dy;
                            }
                        }
                        p = self.next_in_leaf[other];
                    }
                } else {
                    let dx = x - node.com_x;
                    let dy = y - node.com_y;
                    let dist = dx.hypot(dy);
                    if dist > ENTROPY_MIN_DIST {
                        let mult = (node.count as f64) / (dist * (dist + epsilon_r));
                        fx += mult * dx;
                        fy += mult * dy;
                    }
                }
                continue;
            }

            let dx = x - node.com_x;
            let dy = y - node.com_y;
            let dist_sq = dx * dx + dy * dy;
            let max_dim = node.width.max(node.height);
            if (max_dim * max_dim) < theta_sq * dist_sq {
                let dist = dist_sq.sqrt();
                if dist > ENTROPY_MIN_DIST {
                    let mult = (node.count as f64) / (dist * (dist + epsilon_r));
                    fx += mult * dx;
                    fy += mult * dy;
                }
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

        Ok((fx, fy))
    }

    /// Evaluates Barnes-Hut entropy repulsion forces across all points.
    pub fn compute_entropy_repulsion(
        &self,
        theta: S,
        epsilon_r: S,
        out_x: &mut [S],
        out_y: &mut [S],
    ) -> CoreResult<()> {
        let n = self.point_count();
        if out_x.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                out_x.len(),
                "ArenaQuadtree2d::compute_entropy_repulsion::out_x",
            ));
        }
        if out_y.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                out_y.len(),
                "ArenaQuadtree2d::compute_entropy_repulsion::out_y",
            ));
        }
        let theta_f = theta.to_f64();
        let eps_f = epsilon_r.to_f64();
        if !theta_f.is_finite() || theta_f < 0.0 {
            return Err(CoreError::invalid_input("theta must be finite and >= 0.0"));
        }
        if !eps_f.is_finite() || eps_f <= 0.0 {
            return Err(CoreError::invalid_input(
                "epsilon_r must be finite and > 0.0",
            ));
        }
        if n < 2 {
            out_x.fill(S::ZERO);
            out_y.fill(S::ZERO);
            return Ok(());
        }

        let theta_sq = theta_f * theta_f;
        for i in 0..n {
            let (fx, fy) = self.eval_point_entropy_raw(i, theta_sq, eps_f)?;
            out_x[i] = S::from_f64(fx);
            out_y[i] = S::from_f64(fy);
        }
        Ok(())
    }

    /// Evaluates unnormalized Barnes-Hut weighted Coulomb repulsion `(fx, fy)` for vertex `point_idx`:
    /// $\sum_{j \neq i} m_j \frac{x_i - x_j}{d_{ij}^2 + \varepsilon^2}$.
    pub fn eval_point_coulomb_raw(
        &self,
        point_idx: usize,
        theta_sq: f64,
        eps_sq: f64,
    ) -> CoreResult<(f64, f64)> {
        let n = self.point_count();
        if point_idx >= n {
            return Err(CoreError::index_out_of_bounds(
                point_idx,
                n,
                "ArenaQuadtree2d::eval_point_coulomb_raw",
            ));
        }
        if !theta_sq.is_finite() || theta_sq < 0.0 {
            return Err(CoreError::invalid_input(
                "theta_sq must be finite and >= 0.0",
            ));
        }
        if !eps_sq.is_finite() || eps_sq <= 0.0 {
            return Err(CoreError::invalid_input("eps_sq must be finite and > 0.0"));
        }
        if self.nodes.is_empty() {
            return Ok((0.0, 0.0));
        }

        let x = self.coords_x[point_idx];
        let y = self.coords_y[point_idx];
        let own_leaf = self.leaf_of_point[point_idx];

        let mut stack = [0u32; 128];
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
                        if other != point_idx {
                            let dx = x - self.coords_x[other];
                            let dy = y - self.coords_y[other];
                            let denom = dx * dx + dy * dy + eps_sq;
                            let mult = self.masses[other] / denom;
                            fx += mult * dx;
                            fy += mult * dy;
                        }
                        p = self.next_in_leaf[other];
                    }
                } else {
                    let dx = x - node.com_x;
                    let dy = y - node.com_y;
                    let denom = dx * dx + dy * dy + eps_sq;
                    let mult = node.total_mass / denom;
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
                let denom = dist_sq + eps_sq;
                let mult = node.total_mass / denom;
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

        Ok((fx, fy))
    }

    /// Evaluates weighted Barnes-Hut Coulomb repulsion across all points:
    /// $\sum_{j \neq i} m_j \frac{x_i - x_j}{d_{ij}^2 + \varepsilon^2}$.
    pub fn compute_coulomb_repulsion(
        &self,
        theta: S,
        eps_sq: S,
        out_x: &mut [S],
        out_y: &mut [S],
    ) -> CoreResult<()> {
        let n = self.point_count();
        if out_x.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                out_x.len(),
                "ArenaQuadtree2d::compute_coulomb_repulsion::out_x",
            ));
        }
        if out_y.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                out_y.len(),
                "ArenaQuadtree2d::compute_coulomb_repulsion::out_y",
            ));
        }
        let theta_f = theta.to_f64();
        let eps_sq_f = eps_sq.to_f64();
        if !theta_f.is_finite() || theta_f < 0.0 {
            return Err(CoreError::invalid_input("theta must be finite and >= 0.0"));
        }
        if !eps_sq_f.is_finite() || eps_sq_f <= 0.0 {
            return Err(CoreError::invalid_input("eps_sq must be finite and > 0.0"));
        }
        if n < 2 {
            out_x.fill(S::ZERO);
            out_y.fill(S::ZERO);
            return Ok(());
        }

        let theta_sq = theta_f * theta_f;
        for i in 0..n {
            let (fx, fy) = self.eval_point_coulomb_raw(i, theta_sq, eps_sq_f)?;
            out_x[i] = S::from_f64(fx);
            out_y[i] = S::from_f64(fy);
        }
        Ok(())
    }

    /// Evaluates unnormalized Barnes-Hut generalized Maxent $q$-power repulsion `(fx, fy)` for
    /// vertex `point_idx` ($q > -2.0$):
    /// $\sum_{j \neq i} m_j \frac{x_i - x_j}{(d_{ij}^2 + \varepsilon^2)^{(q + 2)/2}}$.
    ///
    /// When `q == 0.0`, delegates directly to [`ArenaQuadtree2d::eval_point_coulomb_raw`] without
    /// `powf`.
    pub fn eval_point_maxent_raw(
        &self,
        point_idx: usize,
        theta_sq: f64,
        q: f64,
        eps_sq: f64,
    ) -> CoreResult<(f64, f64)> {
        let n = self.point_count();
        if point_idx >= n {
            return Err(CoreError::index_out_of_bounds(
                point_idx,
                n,
                "ArenaQuadtree2d::eval_point_maxent_raw",
            ));
        }
        if !theta_sq.is_finite() || theta_sq < 0.0 {
            return Err(CoreError::invalid_input(
                "theta_sq must be finite and >= 0.0",
            ));
        }
        if !q.is_finite() || q <= -2.0 {
            return Err(CoreError::invalid_input("q must be finite and > -2.0"));
        }
        if !eps_sq.is_finite() || eps_sq <= 0.0 {
            return Err(CoreError::invalid_input("eps_sq must be finite and > 0.0"));
        }
        if q == 0.0 {
            return self.eval_point_coulomb_raw(point_idx, theta_sq, eps_sq);
        }
        if self.nodes.is_empty() {
            return Ok((0.0, 0.0));
        }

        let exp = 0.5 * (q + 2.0);
        let x = self.coords_x[point_idx];
        let y = self.coords_y[point_idx];
        let own_leaf = self.leaf_of_point[point_idx];

        let mut stack = [0u32; 128];
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
                        if other != point_idx {
                            let dx = x - self.coords_x[other];
                            let dy = y - self.coords_y[other];
                            let denom = (dx * dx + dy * dy + eps_sq).powf(exp);
                            let mult = self.masses[other] / denom;
                            fx += mult * dx;
                            fy += mult * dy;
                        }
                        p = self.next_in_leaf[other];
                    }
                } else {
                    let dx = x - node.com_x;
                    let dy = y - node.com_y;
                    let denom = (dx * dx + dy * dy + eps_sq).powf(exp);
                    let mult = node.total_mass / denom;
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
                let denom = (dist_sq + eps_sq).powf(exp);
                let mult = node.total_mass / denom;
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

        Ok((fx, fy))
    }

    /// Evaluates weighted Barnes-Hut generalized Maxent $q$-power repulsion across all points
    /// ($q > -2.0$):
    /// $\sum_{j \neq i} m_j \frac{x_i - x_j}{(d_{ij}^2 + \varepsilon^2)^{(q + 2)/2}}$.
    pub fn compute_maxent_repulsion(
        &self,
        theta: S,
        q: f64,
        eps_sq: S,
        out_x: &mut [S],
        out_y: &mut [S],
    ) -> CoreResult<()> {
        let n = self.point_count();
        if out_x.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                out_x.len(),
                "ArenaQuadtree2d::compute_maxent_repulsion::out_x",
            ));
        }
        if out_y.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                out_y.len(),
                "ArenaQuadtree2d::compute_maxent_repulsion::out_y",
            ));
        }
        let theta_f = theta.to_f64();
        let eps_sq_f = eps_sq.to_f64();
        if !theta_f.is_finite() || theta_f < 0.0 {
            return Err(CoreError::invalid_input("theta must be finite and >= 0.0"));
        }
        if !q.is_finite() || q <= -2.0 {
            return Err(CoreError::invalid_input("q must be finite and > -2.0"));
        }
        if !eps_sq_f.is_finite() || eps_sq_f <= 0.0 {
            return Err(CoreError::invalid_input("eps_sq must be finite and > 0.0"));
        }
        if q == 0.0 {
            return self.compute_coulomb_repulsion(theta, eps_sq, out_x, out_y);
        }
        if n < 2 {
            out_x.fill(S::ZERO);
            out_y.fill(S::ZERO);
            return Ok(());
        }

        let theta_sq = theta_f * theta_f;
        for i in 0..n {
            let (fx, fy) = self.eval_point_maxent_raw(i, theta_sq, q, eps_sq_f)?;
            out_x[i] = S::from_f64(fx);
            out_y[i] = S::from_f64(fy);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_quadtree_exact_when_theta_zero_and_reusable_clear() {
        let layout = Layout2d::from_xy(
            vec![0.0_f64, 0.0, 1.5, -2.1, 0.75, -0.5],
            vec![0.0_f64, 0.0, -0.8, 1.4, 2.2, -1.7],
        )
        .unwrap();
        let n = layout.len();

        let mut tree = ArenaQuadtree2d::new();
        tree.build(&layout).unwrap();
        assert_eq!(tree.point_count(), n);

        // Exact Barnes-Hut (`theta = 0.0`) must match brute-force O(N^2) even with duplicates.
        let mut bh_kl_x = vec![0.0; n];
        let mut bh_kl_y = vec![0.0; n];
        let bh_z = tree
            .compute_kl_repulsion(0.0, &mut bh_kl_x, &mut bh_kl_y)
            .unwrap();

        let mut bh_ent_x = vec![0.0; n];
        let mut bh_ent_y = vec![0.0; n];
        tree.compute_entropy_repulsion(0.0, 0.05, &mut bh_ent_x, &mut bh_ent_y)
            .unwrap();

        let mut ex_z = 0.0;
        let mut ex_kl_x = vec![0.0; n];
        let mut ex_kl_y = vec![0.0; n];
        let mut ex_ent_x = vec![0.0; n];
        let mut ex_ent_y = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dx = layout.x()[i] - layout.x()[j];
                let dy = layout.y()[i] - layout.y()[j];
                let d_sq = dx * dx + dy * dy;
                let w = 1.0 / (1.0 + d_sq);
                ex_z += w;
                ex_kl_x[i] += w * w * dx;
                ex_kl_y[i] += w * w * dy;

                let d = d_sq.sqrt();
                if d > ENTROPY_MIN_DIST {
                    let mult = 1.0 / (d * (d + 0.05));
                    ex_ent_x[i] += mult * dx;
                    ex_ent_y[i] += mult * dy;
                }
            }
        }
        assert!((bh_z - ex_z).abs() < 1e-12);
        for i in 0..n {
            assert!((bh_kl_x[i] - ex_kl_x[i] / ex_z).abs() < 1e-12);
            assert!((bh_kl_y[i] - ex_kl_y[i] / ex_z).abs() < 1e-12);
            assert!((bh_ent_x[i] - ex_ent_x[i]).abs() < 1e-12);
            assert!((bh_ent_y[i] - ex_ent_y[i]).abs() < 1e-12);
        }

        // Approximation with theta = 0.5 stays close to exact.
        tree.compute_kl_repulsion(0.5, &mut bh_kl_x, &mut bh_kl_y)
            .unwrap();
        for i in 0..n {
            assert!((bh_kl_x[i] - ex_kl_x[i] / ex_z).abs() < 0.02);
            assert!((bh_kl_y[i] - ex_kl_y[i] / ex_z).abs() < 0.02);
        }

        // Reuse capacity across clear/build cycles.
        let cap = tree.nodes.capacity();
        let layout2 = Layout2d::from_xy(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]).unwrap();
        tree.build(&layout2).unwrap();
        assert_eq!(tree.point_count(), 3);
        assert_eq!(tree.nodes.capacity(), cap);
    }

    #[test]
    fn test_arena_quadtree_sub_ulp_f32_and_empty() {
        let mut tree = ArenaQuadtree2d::<f32>::new();
        let empty = Layout2d::<f32>::zeros(0);
        tree.build(&empty).unwrap();
        assert_eq!(tree.point_count(), 0);
        assert_eq!(tree.node_count(), 0);
        assert_eq!(
            tree.compute_kl_repulsion(0.5, &mut [], &mut []).unwrap(),
            1.0
        );

        let sub_ulp = Layout2d::from_xy(
            vec![100.0_f32, 100.00001, 0.0],
            vec![-100.0_f32, -100.0, 0.0],
        )
        .unwrap();
        tree.build(&sub_ulp).unwrap();
        assert_eq!(tree.point_count(), 3);
        let mut fx = vec![0.0_f32; 3];
        let mut fy = vec![0.0_f32; 3];
        let z = tree.compute_kl_repulsion(0.5, &mut fx, &mut fy).unwrap();
        assert!(z.is_finite() && z > 0.0);
        assert!(fx.iter().all(|v| v.is_finite()));
        assert!(fy.iter().all(|v| v.is_finite()));

        let bad = Layout2d::from_xy(vec![f32::INFINITY], vec![0.0]).unwrap();
        assert!(matches!(
            tree.build(&bad),
            Err(CoreError::NonFiniteValue { .. })
        ));
    }

    #[test]
    fn test_arena_quadtree_weighted_coulomb_exact_and_approx_f64() {
        let layout = Layout2d::from_xy(
            vec![0.0_f64, 0.0, 2.5, -3.1, 1.25, -0.75, 4.0],
            vec![0.0_f64, 0.0, -1.2, 2.4, 3.2, -2.7, 0.5],
        )
        .unwrap();
        let masses = vec![1.5, 2.5, 0.0, 4.0, 0.75, 3.25, 2.0];
        let eps_sq = 0.04_f64;
        let n = layout.len();

        let mut tree = ArenaQuadtree2d::new();
        tree.build_weighted(&layout, &masses).unwrap();
        assert_eq!(tree.point_count(), n);

        let mut bh_x = vec![0.0_f64; n];
        let mut bh_y = vec![0.0_f64; n];
        tree.compute_coulomb_repulsion(0.0, eps_sq, &mut bh_x, &mut bh_y)
            .unwrap();

        let mut ex_x = vec![0.0_f64; n];
        let mut ex_y = vec![0.0_f64; n];
        for i in 0..n {
            for (j, &mj) in masses.iter().enumerate() {
                if i == j {
                    continue;
                }
                let dx = layout.x()[i] - layout.x()[j];
                let dy = layout.y()[i] - layout.y()[j];
                let mult = mj / (dx * dx + dy * dy + eps_sq);
                ex_x[i] += mult * dx;
                ex_y[i] += mult * dy;
            }
        }

        for i in 0..n {
            assert!(
                (bh_x[i] - ex_x[i]).abs() < 1e-10,
                "x mismatch at {i}: bh={} exact={}",
                bh_x[i],
                ex_x[i]
            );
            assert!(
                (bh_y[i] - ex_y[i]).abs() < 1e-10,
                "y mismatch at {i}: bh={} exact={}",
                bh_y[i],
                ex_y[i]
            );
        }

        // Approximation with theta = 0.5 remains accurate.
        let mut approx_x = vec![0.0_f64; n];
        let mut approx_y = vec![0.0_f64; n];
        tree.compute_coulomb_repulsion(0.5, eps_sq, &mut approx_x, &mut approx_y)
            .unwrap();
        for i in 0..n {
            assert!((approx_x[i] - ex_x[i]).abs() < 0.25);
            assert!((approx_y[i] - ex_y[i]).abs() < 0.25);
        }

        // Unweighted KL/entropy behavior remains identical when masses are unit.
        let unit_masses = vec![1.0_f64; n];
        let mut unweighted = ArenaQuadtree2d::new();
        unweighted.build(&layout).unwrap();
        let mut weighted_unit = ArenaQuadtree2d::new();
        weighted_unit.build_weighted(&layout, &unit_masses).unwrap();

        let (mut kl_ux, mut kl_uy) = (vec![0.0; n], vec![0.0; n]);
        let (mut kl_wx, mut kl_wy) = (vec![0.0; n], vec![0.0; n]);
        let zu = unweighted
            .compute_kl_repulsion(0.5, &mut kl_ux, &mut kl_uy)
            .unwrap();
        let zw = weighted_unit
            .compute_kl_repulsion(0.5, &mut kl_wx, &mut kl_wy)
            .unwrap();
        assert_eq!(zu, zw);
        assert_eq!(kl_ux, kl_wx);
        assert_eq!(kl_uy, kl_wy);
    }

    #[test]
    fn test_arena_quadtree_weighted_coulomb_f32() {
        let layout = Layout2d::from_xy(
            vec![0.0_f32, 1.5, -2.0, 3.25, -1.5, 0.5],
            vec![0.0_f32, -1.0, 2.5, 0.75, -2.25, 1.8],
        )
        .unwrap();
        let masses = vec![1.0_f64, 2.5, 0.5, 4.0, 1.25, 3.0];
        let eps_sq = 0.05_f32;
        let n = layout.len();

        let mut tree = ArenaQuadtree2d::<f32>::new();
        tree.build_weighted(&layout, &masses).unwrap();

        let mut bh_x = vec![0.0_f32; n];
        let mut bh_y = vec![0.0_f32; n];
        tree.compute_coulomb_repulsion(0.0, eps_sq, &mut bh_x, &mut bh_y)
            .unwrap();

        for i in 0..n {
            let mut ex_x = 0.0_f64;
            let mut ex_y = 0.0_f64;
            for (j, &mj) in masses.iter().enumerate() {
                if i == j {
                    continue;
                }
                let dx = f64::from(layout.x()[i]) - f64::from(layout.x()[j]);
                let dy = f64::from(layout.y()[i]) - f64::from(layout.y()[j]);
                let mult = mj / (dx * dx + dy * dy + f64::from(eps_sq));
                ex_x += mult * dx;
                ex_y += mult * dy;
            }
            assert!((f64::from(bh_x[i]) - ex_x).abs() < 1e-5);
            assert!((f64::from(bh_y[i]) - ex_y).abs() < 1e-5);
        }

        tree.compute_coulomb_repulsion(0.5, eps_sq, &mut bh_x, &mut bh_y)
            .unwrap();
        assert!(bh_x.iter().all(|v| v.is_finite()));
        assert!(bh_y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_arena_quadtree_weighted_coulomb_validation_and_edge_cases() {
        let mut tree = ArenaQuadtree2d::<f64>::new();

        // Empty and single-point cases.
        tree.build_weighted_xy(&[], &[], &[]).unwrap();
        tree.compute_coulomb_repulsion(0.5, 1e-4, &mut [], &mut [])
            .unwrap();

        tree.build_weighted_xy(&[1.0], &[2.0], &[5.0]).unwrap();
        let mut ox = vec![9.0];
        let mut oy = vec![9.0];
        tree.compute_coulomb_repulsion(0.5, 1e-4, &mut ox, &mut oy)
            .unwrap();
        assert_eq!(ox, vec![0.0]);
        assert_eq!(oy, vec![0.0]);

        // Length mismatches.
        assert!(matches!(
            tree.build_weighted_xy(&[1.0, 2.0], &[1.0], &[1.0, 1.0]),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            tree.build_weighted_xy(&[1.0, 2.0], &[1.0, 2.0], &[1.0]),
            Err(CoreError::DimensionMismatch { .. })
        ));

        // Invalid masses (NaN, Inf, negative).
        assert!(matches!(
            tree.build_weighted_xy(&[1.0], &[2.0], &[f64::NAN]),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            tree.build_weighted_xy(&[1.0], &[2.0], &[f64::INFINITY]),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            tree.build_weighted_xy(&[1.0], &[2.0], &[-0.1]),
            Err(CoreError::InvalidInput(_))
        ));

        // compute_coulomb_repulsion input guards.
        tree.build_weighted_xy(&[0.0, 1.0], &[0.0, 1.0], &[1.0, 2.0])
            .unwrap();
        let mut out_x = vec![0.0; 2];
        let mut out_y = vec![0.0; 2];
        assert!(matches!(
            tree.compute_coulomb_repulsion(0.5, 1e-4, &mut out_x[..1], &mut out_y),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            tree.compute_coulomb_repulsion(0.5, 1e-4, &mut out_x, &mut out_y[..1]),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            tree.compute_coulomb_repulsion(-0.1, 1e-4, &mut out_x, &mut out_y),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            tree.compute_coulomb_repulsion(f64::NAN, 1e-4, &mut out_x, &mut out_y),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            tree.compute_coulomb_repulsion(0.5, 0.0, &mut out_x, &mut out_y),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            tree.compute_coulomb_repulsion(0.5, -1.0, &mut out_x, &mut out_y),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            tree.compute_coulomb_repulsion(0.5, f64::INFINITY, &mut out_x, &mut out_y),
            Err(CoreError::InvalidInput(_))
        ));
    }

    #[test]
    fn test_arena_quadtree_maxent_repulsion_oracle_and_validation() {
        let layout = Layout2d::from_xy(
            vec![0.0_f64, 0.0, 2.5, -3.1, 1.25, -0.75, 4.0],
            vec![0.0_f64, 0.0, -1.2, 2.4, 3.2, -2.7, 0.5],
        )
        .unwrap();
        let masses = vec![1.5, 2.5, 0.5, 4.0, 0.75, 3.25, 2.0];
        let eps_sq = 0.04_f64;
        let n = layout.len();

        let mut tree = ArenaQuadtree2d::new();
        tree.build_weighted(&layout, &masses).unwrap();

        for &q in &[0.0_f64, 0.8, -0.5] {
            let mut bh_x = vec![0.0_f64; n];
            let mut bh_y = vec![0.0_f64; n];
            tree.compute_maxent_repulsion(0.0, q, eps_sq, &mut bh_x, &mut bh_y)
                .unwrap();

            let exp = 0.5 * (q + 2.0);
            let mut ex_x = vec![0.0_f64; n];
            let mut ex_y = vec![0.0_f64; n];
            for i in 0..n {
                for (j, &mj) in masses.iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    let dx = layout.x()[i] - layout.x()[j];
                    let dy = layout.y()[i] - layout.y()[j];
                    let denom = (dx * dx + dy * dy + eps_sq).powf(exp);
                    let mult = mj / denom;
                    ex_x[i] += mult * dx;
                    ex_y[i] += mult * dy;
                }
                assert!(
                    (bh_x[i] - ex_x[i]).abs() < 1e-10,
                    "q={q} x mismatch at {i}: bh={} exact={}",
                    bh_x[i],
                    ex_x[i]
                );
                assert!(
                    (bh_y[i] - ex_y[i]).abs() < 1e-10,
                    "q={q} y mismatch at {i}: bh={} exact={}",
                    bh_y[i],
                    ex_y[i]
                );
            }

            // Barnes-Hut approximation at theta = 1.2 remains finite and tracks exact sign/magnitude.
            let mut approx_x = vec![0.0_f64; n];
            let mut approx_y = vec![0.0_f64; n];
            tree.compute_maxent_repulsion(1.2, q, eps_sq, &mut approx_x, &mut approx_y)
                .unwrap();
            for i in 0..n {
                assert!(approx_x[i].is_finite() && approx_y[i].is_finite());
                assert!((approx_x[i] - ex_x[i]).abs() < 1.5);
                assert!((approx_y[i] - ex_y[i]).abs() < 1.5);
            }
        }

        // Validation guards.
        let mut out_x = vec![0.0; n];
        let mut out_y = vec![0.0; n];
        assert!(matches!(
            tree.compute_maxent_repulsion(1.2, -2.0, eps_sq, &mut out_x, &mut out_y),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            tree.compute_maxent_repulsion(1.2, -2.5, eps_sq, &mut out_x, &mut out_y),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            tree.compute_maxent_repulsion(1.2, f64::NAN, eps_sq, &mut out_x, &mut out_y),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            tree.compute_maxent_repulsion(1.2, f64::INFINITY, eps_sq, &mut out_x, &mut out_y),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            tree.eval_point_maxent_raw(n, 0.0, 0.8, eps_sq),
            Err(CoreError::IndexOutOfBounds { .. })
        ));
        assert!(matches!(
            tree.compute_maxent_repulsion(1.2, 0.8, eps_sq, &mut out_x[..1], &mut out_y),
            Err(CoreError::DimensionMismatch { .. })
        ));
    }
}
