use egraph_core::{CoreError, CoreResult, Scalar};

use super::StaticKdTree;
use super::types::{Candidate, KdQueryScratch, box_dist_sq, convert_query, dist_sq};

impl<const D: usize, S: Scalar> StaticKdTree<D, S> {
    /// Returns the `k` nearest points `(index, dist_sq)` to `query`, sorted by `(dist_sq, index)`.
    pub fn k_nearest(&self, query: &[S; D], k: usize) -> CoreResult<Vec<(usize, f64)>> {
        let mut out = Vec::new();
        self.k_nearest_into(query, k, &mut out)?;
        Ok(out)
    }

    /// Reuses `out` to return the `k` nearest points `(index, dist_sq)` to `query`.
    pub fn k_nearest_into(
        &self,
        query: &[S; D],
        k: usize,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        let mut scratch = KdQueryScratch::with_capacity(k);
        self.k_nearest_with_scratch(query, k, &mut scratch, out)
    }

    /// Reuses `scratch` and `out` to return the `k` nearest points `(index, dist_sq)` to `query`.
    pub fn k_nearest_with_scratch(
        &self,
        query: &[S; D],
        k: usize,
        scratch: &mut KdQueryScratch,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        let q = convert_query(query)?;
        self.k_nearest_f64_with_scratch(&q, k, scratch, out);
        Ok(())
    }

    pub(super) fn k_nearest_f64_with_scratch(
        &self,
        q: &[f64; D],
        k: usize,
        scratch: &mut KdQueryScratch,
        out: &mut Vec<(usize, f64)>,
    ) {
        out.clear();
        if k == 0 || self.nodes.is_empty() {
            return;
        }
        self.knn_search_with_scratch(q, k, None, scratch, out);
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
        self.knn_search_with_scratch(&q, k, Some(point_idx as u32), scratch, out);
        Ok(())
    }

    /// Returns the single nearest point `(index, dist_sq)` to `point_idx`, excluding `point_idx` itself.
    pub fn nearest_excluding_self(&self, point_idx: usize) -> CoreResult<Option<(usize, f64)>> {
        let mut scratch = KdQueryScratch::with_capacity(0);
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
                for entry in &self.entries[node.start..node.end] {
                    if entry.index == exclude {
                        continue;
                    }
                    let d2 = dist_sq(&q, &entry.coords);
                    if d2 < best_dist_sq || (d2 == best_dist_sq && entry.index < best_idx) {
                        best_dist_sq = d2;
                        best_idx = entry.index;
                    }
                }
                continue;
            }

            let left = node.left as usize;
            let right = node.right as usize;
            let dl = box_dist_sq(&q, &self.nodes[left]);
            let dr = box_dist_sq(&q, &self.nodes[right]);

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
    pub fn within_radius(&self, query: &[S; D], radius: f64) -> CoreResult<Vec<(usize, f64)>> {
        let mut out = Vec::new();
        self.within_radius_into(query, radius, &mut out)?;
        Ok(out)
    }

    /// Reuses `out` to return all points within Euclidean `radius >= 0.0` of `query`.
    pub fn within_radius_into(
        &self,
        query: &[S; D],
        radius: f64,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        if !radius.is_finite() {
            return Err(CoreError::non_finite("StaticKdTree::within_radius"));
        }
        if radius < 0.0 {
            return Err(CoreError::invalid_input("radius must be >= 0.0"));
        }
        self.within_radius_sq_into(query, radius * radius, out)
    }

    /// Returns all points `(index, dist_sq)` with `dist_sq <= radius_sq`, sorted by `(dist_sq, index)`.
    pub fn within_radius_sq(
        &self,
        query: &[S; D],
        radius_sq: f64,
    ) -> CoreResult<Vec<(usize, f64)>> {
        let mut out = Vec::new();
        self.within_radius_sq_into(query, radius_sq, &mut out)?;
        Ok(out)
    }

    /// Reuses `out` to return all points with `dist_sq <= radius_sq`.
    pub fn within_radius_sq_into(
        &self,
        query: &[S; D],
        radius_sq: f64,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        let mut scratch = KdQueryScratch::with_capacity(0);
        self.within_radius_sq_with_scratch(query, radius_sq, &mut scratch, out)
    }

    /// Reuses `scratch` and `out` to return all points with `dist_sq <= radius_sq`.
    pub fn within_radius_sq_with_scratch(
        &self,
        query: &[S; D],
        radius_sq: f64,
        scratch: &mut KdQueryScratch,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        let q = convert_query(query)?;
        self.within_radius_sq_f64_with_scratch(&q, radius_sq, scratch, out)
    }

    pub(super) fn within_radius_sq_f64_with_scratch(
        &self,
        q: &[f64; D],
        radius_sq: f64,
        scratch: &mut KdQueryScratch,
        out: &mut Vec<(usize, f64)>,
    ) -> CoreResult<()> {
        if !radius_sq.is_finite() {
            return Err(CoreError::non_finite("StaticKdTree::within_radius_sq"));
        }
        if radius_sq < 0.0 {
            return Err(CoreError::invalid_input("radius_sq must be >= 0.0"));
        }
        out.clear();
        if self.nodes.is_empty() {
            return Ok(());
        }

        scratch.stack.clear();
        scratch.stack.push((0, 0.0));

        while let Some((node_idx, _)) = scratch.stack.pop() {
            let node = &self.nodes[node_idx as usize];
            if box_dist_sq(q, node) > radius_sq {
                continue;
            }
            if node.left == u32::MAX {
                for entry in &self.entries[node.start..node.end] {
                    let d2 = dist_sq(q, &entry.coords);
                    if d2 <= radius_sq {
                        out.push((entry.index as usize, d2));
                    }
                }
            } else {
                scratch.stack.push((node.right, 0.0));
                scratch.stack.push((node.left, 0.0));
            }
        }

        out.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        Ok(())
    }

    fn knn_search_with_scratch(
        &self,
        q: &[f64; D],
        k: usize,
        exclude: Option<u32>,
        scratch: &mut KdQueryScratch,
        out: &mut Vec<(usize, f64)>,
    ) {
        scratch.heap.clear();
        scratch.stack.clear();
        scratch.stack.push((0, 0.0));

        while let Some((node_idx, b_dist)) = scratch.stack.pop() {
            if scratch.heap.len() == k {
                // Use strict `>` so tied distances in adjacent boxes are always explored.
                if let Some(&worst) = scratch.heap.peek()
                    && b_dist > worst.dist_sq
                {
                    continue;
                }
            }

            let node = &self.nodes[node_idx as usize];
            if node.left == u32::MAX {
                for entry in &self.entries[node.start..node.end] {
                    if Some(entry.index) == exclude {
                        continue;
                    }
                    let d2 = dist_sq(q, &entry.coords);
                    let cand = Candidate {
                        dist_sq: d2,
                        index: entry.index,
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
            let dl = box_dist_sq(q, &self.nodes[left]);
            let dr = box_dist_sq(q, &self.nodes[right]);
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
