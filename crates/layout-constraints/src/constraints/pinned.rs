use egraph_core::{CoreError, CoreResult, Scalar};

use super::projector::ConstraintProjector;

impl<S: Scalar> ConstraintProjector<S> {
    fn find_splittable_edge_on_pin_path(
        &mut self,
        pin_var: u32,
        endpoint: u32,
        want_left_shift: bool,
    ) -> CoreResult<Option<(usize, f64)>> {
        if pin_var == endpoint {
            return Ok(None);
        }

        self.stamp.clear();
        self.bfs_queue.clear();
        self.stamp.mark(pin_var as usize);
        self.parent_var[pin_var as usize] = u32::MAX;
        self.parent_cidx[pin_var as usize] = u32::MAX;
        self.bfs_queue.push(pin_var);

        let mut head = 0usize;
        let mut found = false;
        while head < self.bfs_queue.len() {
            let u = self.bfs_queue[head];
            head += 1;
            if u == endpoint {
                found = true;
                break;
            }
            let start = self.adj_indptr[u as usize] as usize;
            let end = self.adj_indptr[u as usize + 1] as usize;
            for pos in start..end {
                let c_idx = self.adj_cidx[pos] as usize;
                if self.is_active[c_idx] == 0 {
                    continue;
                }
                let v = self.adj_to[pos];
                if self.stamp.visit(v as usize) {
                    self.parent_var[v as usize] = u;
                    self.parent_cidx[v as usize] = c_idx as u32;
                    self.bfs_queue.push(v);
                }
            }
        }

        if !found {
            return Err(CoreError::numerical_breakdown(
                "active constraint tree disconnected inside pinned block",
            ));
        }

        let mut best: Option<(usize, f64)> = None;
        let mut cur = endpoint;
        while cur != pin_var {
            let p = self.parent_var[cur as usize];
            let c_idx = self.parent_cidx[cur as usize] as usize;
            let c = self.constraints[c_idx];
            // Moving `endpoint` left is allowed if an edge `cur -> p` (`cur <= p - gap`) splits;
            // moving `endpoint` right is allowed if an edge `p -> cur` (`p + gap <= cur`) splits.
            let splittable = if want_left_shift {
                c.left == cur && c.right == p
            } else {
                c.left == p && c.right == cur
            };
            if splittable {
                let lm_val = self.lm[c_idx];
                match best {
                    None => best = Some((c_idx, lm_val)),
                    Some((bc, blm)) => {
                        if lm_val < blm || (lm_val == blm && c_idx < bc) {
                            best = Some((c_idx, lm_val));
                        }
                    }
                }
            }
            cur = p;
        }

        Ok(best)
    }

    /// Resolves a violated constraint `viol_c_idx` bridging two distinct pinned blocks `bl` and `br`
    /// by splitting the minimum-multiplier splittable edge separating one endpoint from its pin.
    pub(super) fn split_two_pinned_blocks(
        &mut self,
        bl: usize,
        br: usize,
        viol_c_idx: usize,
    ) -> CoreResult<()> {
        let c = self.constraints[viol_c_idx];
        let p_l = self.block_pin_var[bl];
        let p_r = self.block_pin_var[br];

        self.compute_active_tree_lm(bl, p_l);
        let left_cand = self.find_splittable_edge_on_pin_path(p_l, c.left, true)?;

        self.compute_active_tree_lm(br, p_r);
        let right_cand = self.find_splittable_edge_on_pin_path(p_r, c.right, false)?;

        let (chosen_block, sc_idx, detached_seed) = match (left_cand, right_cand) {
            (None, None) => {
                return Err(CoreError::invalid_input(
                    "separation constraints conflict with pinned nodes",
                ));
            }
            (Some((lc, _)), None) => (bl, lc, c.left),
            (None, Some((rc, _))) => (br, rc, c.right),
            (Some((lc, llm)), Some((rc, rlm))) => {
                if llm < rlm || (llm == rlm && lc <= rc) {
                    (bl, lc, c.left)
                } else {
                    (br, rc, c.right)
                }
            }
        };

        self.is_active[sc_idx] = 0;

        // Collect all nodes in the detached sub-component containing `detached_seed`.
        self.stamp.clear();
        self.bfs_queue.clear();
        self.stamp.mark(detached_seed as usize);
        self.bfs_queue.push(detached_seed);
        let mut qh = 0usize;
        while qh < self.bfs_queue.len() {
            let u = self.bfs_queue[qh];
            qh += 1;
            let start = self.adj_indptr[u as usize] as usize;
            let end = self.adj_indptr[u as usize + 1] as usize;
            for pos in start..end {
                let c_idx = self.adj_cidx[pos] as usize;
                if self.is_active[c_idx] == 0 {
                    continue;
                }
                let v = self.adj_to[pos];
                if self.stamp.visit(v as usize) {
                    self.bfs_queue.push(v);
                }
            }
        }

        let new_b = self.alloc_block();
        self.partition_block_by_stamp(chosen_block, new_b);

        // The pin stays in `chosen_block`; `new_b` is always unpinned.
        self.block_pinned[new_b] = false;
        self.block_pin_var[new_b] = u32::MAX;
        self.block_pos[chosen_block] = self.optimal_position(chosen_block);
        self.block_pos[new_b] = self.optimal_position(new_b);
        Ok(())
    }
}
