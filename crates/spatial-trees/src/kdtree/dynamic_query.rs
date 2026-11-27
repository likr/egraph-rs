use egraph_core::{CoreError, CoreResult, Scalar};

use super::DynamicKdTree;
use super::dynamic::DynamicKdNode;
use super::types::{Candidate, KdQueryScratch};

#[inline]
fn dyn_dist_sq(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    for d in 0..a.len() {
        let diff = a[d] - b[d];
        sum += diff * diff;
    }
    sum
}

#[inline]
fn dyn_box_dist_sq(q: &[f64], node: &DynamicKdNode, node_bounds: &[f64]) -> f64 {
    let dims = q.len();
    let mins = &node_bounds[node.bounds_offset..node.bounds_offset + dims];
    let maxs = &node_bounds[node.bounds_offset + dims..node.bounds_offset + 2 * dims];
    let mut sum = 0.0_f64;
    for d in 0..dims {
        let v = q[d];
        let diff = if v < mins[d] {
            mins[d] - v
        } else if v > maxs[d] {
            v - maxs[d]
        } else {
            0.0
        };
        sum += diff * diff;
    }
    sum
}

impl<S: Scalar> DynamicKdTree<S> {
    fn convert_query_into(
        &self,
        query: &[S],
        buf: &mut Vec<f64>,
        context: &'static str,
    ) -> CoreResult<()> {
        if query.len() != self.dims {
            return Err(CoreError::dimension_mismatch(
                self.dims,
                query.len(),
                context,
            ));
        }
        buf.clear();
        buf.reserve(self.dims);
        for &v in query {
            if !v.is_finite() {
                return Err(CoreError::non_finite(context));
            }
            buf.push(v.to_f64());
        }
        Ok(())
    }

    /// Returns the `k` nearest points `(index, dist_sq)` to `query`, sorted by `(dist_sq, index)`.
    pub fn k_nearest(&self, query: &[S], k: usize) -> CoreResult<Vec<(usize, f64)>> {
        let mut out = Vec::new();
        self.k_nearest_into(query, k, &mut out)?;
        Ok(out)
    }

    /// Reuses `out` to return the `k` nearest points `(index, dist_sq)` to `query`.
    pub fn k_nearest_into(
        &self,
        query: &[S],
        k: usize,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        let mut scratch = KdQueryScratch::with_capacity(k);
        self.k_nearest_with_scratch(query, k, &mut scratch, out)
    }

    /// Reuses `scratch` and `out` to return the `k` nearest points `(index, dist_sq)` to `query`.
    pub fn k_nearest_with_scratch(
        &self,
        query: &[S],
        k: usize,
        scratch: &mut KdQueryScratch,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        let mut q_buf = std::mem::take(&mut scratch.query_buf);
        let res = self.convert_query_into(query, &mut q_buf, "DynamicKdTree::k_nearest");
        if res.is_ok() {
            out.clear();
            if k > 0 && !self.nodes.is_empty() {
                self.knn_search_with_scratch(&q_buf, k, None, scratch, out);
            }
        }
        scratch.query_buf = q_buf;
        res
    }

    pub(super) fn k_nearest_f64_with_scratch(
        &self,
        q: &[f64],
        k: usize,
        scratch: &mut KdQueryScratch,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        if q.len() != self.dims {
            return Err(CoreError::dimension_mismatch(
                self.dims,
                q.len(),
                "DynamicKdTree::query_knn",
            ));
        }
        if !q.iter().all(|v| v.is_finite()) {
            return Err(CoreError::non_finite("DynamicKdTree::query_knn"));
        }
        out.clear();
        if k > 0 && !self.nodes.is_empty() {
            self.knn_search_with_scratch(q, k, None, scratch, out);
        }
        Ok(())
    }

    /// Returns the `k` nearest points `(index, dist_sq)` to point `point_idx`, excluding `point_idx` itself.
    pub fn k_nearest_excluding_self(
        &self,
        point_idx: usize,
        k: usize,
    ) -> CoreResult<Vec<(usize, f64)>> {
        let mut out = Vec::new();
        self.k_nearest_excluding_self_into(point_idx, k, &mut out)?;
        Ok(out)
    }

    /// Reuses `out` to return the `k` nearest points to `point_idx` excluding `point_idx`.
    pub fn k_nearest_excluding_self_into(
        &self,
        point_idx: usize,
        k: usize,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        let mut scratch = KdQueryScratch::with_capacity(k);
        self.k_nearest_excluding_self_with_scratch(point_idx, k, &mut scratch, out)
    }

    /// Reuses `scratch` and `out` to return the `k` nearest points to `point_idx` excluding `point_idx`.
    pub fn k_nearest_excluding_self_with_scratch(
        &self,
        point_idx: usize,
        k: usize,
        scratch: &mut KdQueryScratch,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        let q = self.point(point_idx)?;
        out.clear();
        if k == 0 || self.len() <= 1 {
            return Ok(());
        }
        self.knn_search_with_scratch(q, k, Some(point_idx as u32), scratch, out);
        Ok(())
    }

    /// Returns the single nearest point `(index, dist_sq)` to `point_idx`, excluding `point_idx` itself.
    pub fn nearest_excluding_self(&self, point_idx: usize) -> CoreResult<Option<(usize, f64)>> {
        let mut scratch = KdQueryScratch::with_capacity(1);
        self.nearest_excluding_self_with_scratch(point_idx, &mut scratch)
    }

    /// Reuses `scratch` to return the single nearest point `(index, dist_sq)` to `point_idx`,
    /// excluding `point_idx` itself.
    pub fn nearest_excluding_self_with_scratch(
        &self,
        point_idx: usize,
        scratch: &mut KdQueryScratch,
    ) -> CoreResult<Option<(usize, f64)>> {
        let q = self.point(point_idx)?;
        if self.len() <= 1 {
            return Ok(None);
        }
        let exclude = point_idx as u32;
        let dims = self.dims;
        let mut best_dist_sq = f64::INFINITY;
        let mut best_idx = u32::MAX;

        scratch.stack.clear();
        scratch.stack.push((0, 0.0));

        while let Some((node_idx, b_dist)) = scratch.stack.pop() {
            if b_dist > best_dist_sq {
                continue;
            }
            let node = &self.nodes[node_idx as usize];
            if node.left == u32::MAX {
                for pos in node.start..node.end {
                    let idx = self.entry_indices[pos];
                    if idx == exclude {
                        continue;
                    }
                    let c = &self.entry_coords[pos * dims..(pos + 1) * dims];
                    let d2 = dyn_dist_sq(q, c);
                    if d2 < best_dist_sq || (d2 == best_dist_sq && idx < best_idx) {
                        best_dist_sq = d2;
                        best_idx = idx;
                    }
                }
                continue;
            }

            let left = node.left as usize;
            let right = node.right as usize;
            let dl = dyn_box_dist_sq(q, &self.nodes[left], &self.node_bounds);
            let dr = dyn_box_dist_sq(q, &self.nodes[right], &self.node_bounds);

            if q[node.split_dim] <= node.split_val {
                if dr <= best_dist_sq {
                    scratch.stack.push((right as u32, dr));
                }
                if dl <= best_dist_sq {
                    scratch.stack.push((left as u32, dl));
                }
            } else {
                if dl <= best_dist_sq {
                    scratch.stack.push((left as u32, dl));
                }
                if dr <= best_dist_sq {
                    scratch.stack.push((right as u32, dr));
                }
            }
        }

        if best_idx == u32::MAX {
            Ok(None)
        } else {
            Ok(Some((best_idx as usize, best_dist_sq)))
        }
    }

    /// Returns all points `(index, dist_sq)` within Euclidean `radius >= 0.0` of `query`,
    /// sorted by `(dist_sq, index)`.
    pub fn within_radius(&self, query: &[S], radius: f64) -> CoreResult<Vec<(usize, f64)>> {
        let mut out = Vec::new();
        self.within_radius_into(query, radius, &mut out)?;
        Ok(out)
    }

    /// Reuses `out` to return all points within Euclidean `radius >= 0.0` of `query`.
    pub fn within_radius_into(
        &self,
        query: &[S],
        radius: f64,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        if !radius.is_finite() {
            return Err(CoreError::non_finite("DynamicKdTree::within_radius"));
        }
        if radius < 0.0 {
            return Err(CoreError::invalid_input("radius must be >= 0.0"));
        }
        self.within_radius_sq_into(query, radius * radius, out)
    }

    /// Returns all points `(index, dist_sq)` with `dist_sq <= radius_sq`, sorted by `(dist_sq, index)`.
    pub fn within_radius_sq(&self, query: &[S], radius_sq: f64) -> CoreResult<Vec<(usize, f64)>> {
        let mut out = Vec::new();
        self.within_radius_sq_into(query, radius_sq, &mut out)?;
        Ok(out)
    }

    /// Reuses `out` to return all points with `dist_sq <= radius_sq`.
    pub fn within_radius_sq_into(
        &self,
        query: &[S],
        radius_sq: f64,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        let mut scratch = KdQueryScratch::with_capacity(0);
        self.within_radius_sq_with_scratch(query, radius_sq, &mut scratch, out)
    }

    /// Reuses `scratch` and `out` to return all points with `dist_sq <= radius_sq`.
    pub fn within_radius_sq_with_scratch(
        &self,
        query: &[S],
        radius_sq: f64,
        scratch: &mut KdQueryScratch,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        if !radius_sq.is_finite() {
            return Err(CoreError::non_finite("DynamicKdTree::within_radius_sq"));
        }
        if radius_sq < 0.0 {
            return Err(CoreError::invalid_input("radius_sq must be >= 0.0"));
        }
        let mut q_buf = std::mem::take(&mut scratch.query_buf);
        let res = self.convert_query_into(query, &mut q_buf, "DynamicKdTree::within_radius_sq");
        if res.is_ok() {
            self.within_radius_sq_f64_inner(&q_buf, radius_sq, scratch, out);
        }
        scratch.query_buf = q_buf;
        res
    }

    pub(super) fn within_radius_sq_f64_with_scratch(
        &self,
        q: &[f64],
        radius_sq: f64,
        scratch: &mut KdQueryScratch,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        if q.len() != self.dims {
            return Err(CoreError::dimension_mismatch(
                self.dims,
                q.len(),
                "DynamicKdTree::query_radius_sq",
            ));
        }
        if !q.iter().all(|v| v.is_finite()) || !radius_sq.is_finite() {
            return Err(CoreError::non_finite("DynamicKdTree::query_radius_sq"));
        }
        if radius_sq < 0.0 {
            return Err(CoreError::invalid_input("radius_sq must be >= 0.0"));
        }
        self.within_radius_sq_f64_inner(q, radius_sq, scratch, out);
        Ok(())
    }

    fn within_radius_sq_f64_inner(
        &self,
        q: &[f64],
        radius_sq: f64,
        scratch: &mut KdQueryScratch,
        out: &mut Vec<(usize, f64)>,
    ) {
        out.clear();
        if !self.nodes.is_empty() {
            let dims = self.dims;
            scratch.stack.clear();
            scratch.stack.push((0, 0.0));
            while let Some((node_idx, _)) = scratch.stack.pop() {
                let node = &self.nodes[node_idx as usize];
                if dyn_box_dist_sq(q, node, &self.node_bounds) > radius_sq {
                    continue;
                }
                if node.left == u32::MAX {
                    for pos in node.start..node.end {
                        let c = &self.entry_coords[pos * dims..(pos + 1) * dims];
                        let d2 = dyn_dist_sq(q, c);
                        if d2 <= radius_sq {
                            out.push((self.entry_indices[pos] as usize, d2));
                        }
                    }
                } else {
                    scratch.stack.push((node.right, 0.0));
                    scratch.stack.push((node.left, 0.0));
                }
            }
            out.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        }
    }

    fn knn_search_with_scratch(
        &self,
        q: &[f64],
        k: usize,
        exclude: Option<u32>,
        scratch: &mut KdQueryScratch,
        out: &mut Vec<(usize, f64)>,
    ) {
        let dims = self.dims;
        scratch.heap.clear();
        scratch.stack.clear();
        scratch.stack.push((0, 0.0));

        while let Some((node_idx, b_dist)) = scratch.stack.pop() {
            if scratch.heap.len() == k
                && let Some(&worst) = scratch.heap.peek()
                && b_dist > worst.dist_sq
            {
                continue;
            }

            let node = &self.nodes[node_idx as usize];
            if node.left == u32::MAX {
                for pos in node.start..node.end {
                    let idx = self.entry_indices[pos];
                    if Some(idx) == exclude {
                        continue;
                    }
                    let c = &self.entry_coords[pos * dims..(pos + 1) * dims];
                    let d2 = dyn_dist_sq(q, c);
                    let cand = Candidate {
                        dist_sq: d2,
                        index: idx,
                    };
                    if scratch.heap.len() < k {
                        scratch.heap.push(cand);
                    } else if let Some(&worst) = scratch.heap.peek()
                        && cand < worst
                    {
                        scratch.heap.pop();
                        scratch.heap.push(cand);
                    }
                }
                continue;
            }

            let left = node.left as usize;
            let right = node.right as usize;
            let dl = dyn_box_dist_sq(q, &self.nodes[left], &self.node_bounds);
            let dr = dyn_box_dist_sq(q, &self.nodes[right], &self.node_bounds);
            let thresh = if scratch.heap.len() == k {
                scratch.heap.peek().map_or(f64::INFINITY, |w| w.dist_sq)
            } else {
                f64::INFINITY
            };

            if q[node.split_dim] <= node.split_val {
                if dr <= thresh {
                    scratch.stack.push((right as u32, dr));
                }
                if dl <= thresh {
                    scratch.stack.push((left as u32, dl));
                }
            } else {
                if dl <= thresh {
                    scratch.stack.push((left as u32, dl));
                }
                if dr <= thresh {
                    scratch.stack.push((right as u32, dr));
                }
            }
        }

        // Drain heap into `out` and reverse so `scratch.heap` retains its backing allocation.
        out.clear();
        while let Some(c) = scratch.heap.pop() {
            out.push((c.index as usize, c.dist_sq));
        }
        out.reverse();
    }
}
