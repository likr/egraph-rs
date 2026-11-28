use egraph_core::{CoreError, CoreResult, Scalar};

use super::projector::ConstraintProjector;

const SPLIT_LM_TOL: f64 = 1e-9;

impl<S: Scalar> ConstraintProjector<S> {
    pub(super) fn merge_block(&mut self, l: usize, r: usize, c_idx: usize) {
        let c = self.constraints[c_idx];
        let d = self.var_offset[c.left as usize] + c.gap - self.var_offset[c.right as usize];
        let n_l = self.block_len[l] as f64;
        let n_r = self.block_len[r] as f64;

        if !self.block_pinned[l] && !self.block_pinned[r] {
            self.block_pos[l] =
                (self.block_pos[l] * n_l + (self.block_pos[r] - d) * n_r) / (n_l + n_r);
        } else if !self.block_pinned[l] && self.block_pinned[r] {
            self.block_pos[l] = self.block_pos[r] - d;
            self.block_pinned[l] = true;
            self.block_pin_var[l] = self.block_pin_var[r];
        }
        self.is_active[c_idx] = 1;

        // Walk r's intrusive list to update block_of_var and var_offset.
        let r_head = self.block_head[r];
        let r_tail = self.block_tail[r];
        let l_u32 = l as u32;
        let mut cur = r_head;
        while cur != u32::MAX {
            let idx = cur as usize;
            self.block_of_var[idx] = l_u32;
            self.var_offset[idx] += d;
            cur = self.var_next[idx];
        }

        // Splice r's list onto the tail of l's list in O(1).
        if r_head != u32::MAX {
            let l_tail = self.block_tail[l];
            if l_tail != u32::MAX {
                self.var_next[l_tail as usize] = r_head;
            } else {
                self.block_head[l] = r_head;
            }
            self.block_tail[l] = r_tail;
            self.block_len[l] += self.block_len[r];
        }

        self.block_head[r] = u32::MAX;
        self.block_tail[r] = u32::MAX;
        self.block_len[r] = 0;
        self.block_pinned[r] = false;
        self.block_pin_var[r] = u32::MAX;
        self.free_blocks.push(r as u32);
    }

    /// Iteratively computes Lagrange multipliers on block `b`'s active spanning tree
    /// starting at `root`, returning the active constraint index with minimum multiplier
    /// `(c_idx, lm_val)` if the block has at least one active constraint.
    pub(super) fn compute_active_tree_lm(&mut self, b: usize, root: u32) -> Option<(usize, f64)> {
        self.block_pos[b] = self.optimal_position(b);

        self.stamp.clear();
        self.dfs_order.clear();
        self.bfs_queue.clear();

        self.stamp.mark(root as usize);
        self.dfs_order.push((root, u32::MAX, u32::MAX, true));
        let mut head = 0usize;
        self.bfs_queue.push(root);

        while head < self.bfs_queue.len() {
            let u = self.bfs_queue[head];
            head += 1;
            let start = self.adj_indptr[u as usize] as usize;
            let end = self.adj_indptr[u as usize + 1] as usize;
            for pos in start..end {
                let c_idx = self.adj_cidx[pos] as usize;
                if self.is_active[c_idx] == 0 {
                    continue;
                }
                let v = self.adj_to[pos];
                if self.stamp.visit(v as usize) {
                    let is_forward = self.constraints[c_idx].left == u;
                    self.dfs_order.push((v, u, c_idx as u32, is_forward));
                    self.bfs_queue.push(v);
                }
            }
        }

        // Initialize dfdv[v] = var_pos(v) - desired_x[v] for all visited nodes in block b.
        for &(v, _, _, _) in &self.dfs_order {
            let idx = v as usize;
            self.dfdv[idx] = self.var_pos(idx) - self.desired_x[idx];
        }

        // Reverse BFS order is guaranteed post-order (children always appear after parents).
        let mut min_edge: Option<(usize, f64)> = None;
        for idx_rev in (1..self.dfs_order.len()).rev() {
            let (v, u, c_u32, is_forward) = self.dfs_order[idx_rev];
            let c_idx = c_u32 as usize;
            let val_v = self.dfdv[v as usize];
            let lm_val = if is_forward { val_v } else { -val_v };
            self.lm[c_idx] = lm_val;
            self.dfdv[u as usize] += val_v;

            match min_edge {
                None => min_edge = Some((c_idx, lm_val)),
                Some((best_c, best_lm)) => {
                    if lm_val < best_lm || (lm_val == best_lm && c_idx < best_c) {
                        min_edge = Some((c_idx, lm_val));
                    }
                }
            }
        }

        min_edge
    }

    pub(super) fn expand_block(&mut self, b: usize, viol_c_idx: usize) -> CoreResult<()> {
        let viol_c = self.constraints[viol_c_idx];
        let viol_amount = self.violation(viol_c_idx);
        let src = viol_c.left;
        let dst = viol_c.right;

        let tree_root = if self.block_pinned[b] {
            self.block_pin_var[b]
        } else {
            src
        };
        self.compute_active_tree_lm(b, tree_root);

        // BFS from src to dst along active edges to find the path and its minimum-lm forward edge.
        self.stamp.clear();
        self.bfs_queue.clear();
        self.stamp.mark(src as usize);
        self.parent_var[src as usize] = u32::MAX;
        self.parent_cidx[src as usize] = u32::MAX;
        self.bfs_queue.push(src);

        let mut head = 0usize;
        let mut found = false;
        while head < self.bfs_queue.len() {
            let u = self.bfs_queue[head];
            head += 1;
            if u == dst {
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
                "active constraint tree disconnected inside block",
            ));
        }

        let mut best_sc: Option<(usize, f64)> = None;
        let mut cur = dst;
        while cur != src {
            let p = self.parent_var[cur as usize];
            let c_idx = self.parent_cidx[cur as usize] as usize;
            let c = self.constraints[c_idx];
            if c.left == p && c.right == cur {
                let lm_val = self.lm[c_idx];
                match best_sc {
                    None => best_sc = Some((c_idx, lm_val)),
                    Some((bc, blm)) => {
                        if lm_val < blm || (lm_val == blm && c_idx < bc) {
                            best_sc = Some((c_idx, lm_val));
                        }
                    }
                }
            }
            cur = p;
        }

        let Some((sc_idx, _)) = best_sc else {
            return Err(CoreError::invalid_input(
                "separation constraints contain an infeasible cycle",
            ));
        };

        // Deactivate split constraint `sc_idx` and shift the component reachable from `dst`.
        self.is_active[sc_idx] = 0;

        self.stamp.clear();
        self.bfs_queue.clear();
        self.stamp.mark(dst as usize);
        self.bfs_queue.push(dst);
        let mut qh = 0usize;
        while qh < self.bfs_queue.len() {
            let u = self.bfs_queue[qh];
            qh += 1;
            self.var_offset[u as usize] += viol_amount;
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

        self.is_active[viol_c_idx] = 1;
        self.block_pos[b] = self.optimal_position(b);
        Ok(())
    }

    pub(super) fn alloc_block(&mut self) -> usize {
        if let Some(free_b) = self.free_blocks.pop() {
            free_b as usize
        } else {
            let id = self.block_head.len();
            self.block_head.push(u32::MAX);
            self.block_tail.push(u32::MAX);
            self.block_len.push(0);
            self.block_pos.push(0.0);
            self.block_pinned.push(false);
            self.block_pin_var.push(u32::MAX);
            id
        }
    }

    pub(super) fn partition_block_by_stamp(&mut self, b: usize, new_b: usize) {
        let mut cur = self.block_head[b];
        let mut b_head = u32::MAX;
        let mut b_tail = u32::MAX;
        let mut b_len = 0u32;
        let mut nb_head = u32::MAX;
        let mut nb_tail = u32::MAX;
        let mut nb_len = 0u32;
        let b_u32 = b as u32;
        let nb_u32 = new_b as u32;

        while cur != u32::MAX {
            let next = self.var_next[cur as usize];
            self.var_next[cur as usize] = u32::MAX;
            if self.stamp.is_visited(cur as usize) {
                self.block_of_var[cur as usize] = nb_u32;
                if nb_head == u32::MAX {
                    nb_head = cur;
                } else {
                    self.var_next[nb_tail as usize] = cur;
                }
                nb_tail = cur;
                nb_len += 1;
            } else {
                self.block_of_var[cur as usize] = b_u32;
                if b_head == u32::MAX {
                    b_head = cur;
                } else {
                    self.var_next[b_tail as usize] = cur;
                }
                b_tail = cur;
                b_len += 1;
            }
            cur = next;
        }

        self.block_head[b] = b_head;
        self.block_tail[b] = b_tail;
        self.block_len[b] = b_len;

        self.block_head[new_b] = nb_head;
        self.block_tail[new_b] = nb_tail;
        self.block_len[new_b] = nb_len;
    }

    /// Splits blocks whose most negative active Lagrange multiplier is `< -SPLIT_LM_TOL`.
    /// Returns `true` if no blocks were split.
    pub(super) fn split_blocks(&mut self) -> bool {
        let mut nosplit = true;
        let num_blocks = self.block_head.len();

        for b in 0..num_blocks {
            if self.block_len[b] == 0 {
                continue;
            }
            let root = if self.block_pinned[b] {
                self.block_pin_var[b]
            } else {
                self.block_head[b]
            };
            let Some((sc_idx, min_lm)) = self.compute_active_tree_lm(b, root) else {
                continue;
            };
            if min_lm >= -SPLIT_LM_TOL {
                continue;
            }

            nosplit = false;
            self.is_active[sc_idx] = 0;
            let right_seed = self.constraints[sc_idx].right;

            // Mark the right sub-component reachable from `right_seed` without `sc_idx`.
            self.stamp.clear();
            self.bfs_queue.clear();
            self.stamp.mark(right_seed as usize);
            self.bfs_queue.push(right_seed);
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

            // Allocate a fresh or free block ID (never overwriting `blocks[s]`!).
            let new_b = self.alloc_block();
            self.partition_block_by_stamp(b, new_b);

            if self.block_pinned[b] {
                let pin_var = self.block_pin_var[b];
                if self.stamp.is_visited(pin_var as usize) {
                    self.block_pinned[b] = false;
                    self.block_pin_var[b] = u32::MAX;
                    self.block_pinned[new_b] = true;
                    self.block_pin_var[new_b] = pin_var;
                } else {
                    self.block_pinned[new_b] = false;
                    self.block_pin_var[new_b] = u32::MAX;
                }
            } else {
                self.block_pinned[new_b] = false;
                self.block_pin_var[new_b] = u32::MAX;
            }

            self.block_pos[b] = self.optimal_position(b);
            self.block_pos[new_b] = self.optimal_position(new_b);
        }

        nosplit
    }
}
