use egraph_core::{CoreError, CoreResult, Layout2d, Scalar};

use super::{ArenaNode, ArenaQuadtree2d, COINCIDENT_DIST_SQ};

impl<S: Scalar> ArenaQuadtree2d<S> {
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
}
