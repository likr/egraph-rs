//! Arena-based 1D separation constraint projection and 2D rectangle overlap removal.
//!
//! Replaces the legacy `HashMap`/`HashSet` and recursive `ConstraintGraph` with a flat
//! `Vec`-backed arena (`ConstraintProjector`), fixing the legacy `split_blocks` early `break`,
//! `blocks[s]` overwrite, infinite `project()` loop, and `2 + 1e-6` / early-break scanline bugs.

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::marker::PhantomData;

use egraph_core::{CoreError, CoreResult, CsrGraph, Layout2d, Scalar, VisitStamp};
use egraph_spatial::RectangleOverlapSweep;

#[derive(Debug, Clone, Copy)]
struct OrdF64(f64);

impl PartialEq for OrdF64 {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0) == Ordering::Equal
    }
}

impl Eq for OrdF64 {}

impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

const VIOLATION_TOL: f64 = 1e-9;
const SPLIT_LM_TOL: f64 = 1e-9;

/// One-dimensional separation constraint enforcing `coord[left] + gap <= coord[right]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeparationConstraint<S: Scalar = f64> {
    pub left: u32,
    pub right: u32,
    pub gap: S,
}

impl<S: Scalar> SeparationConstraint<S> {
    /// Creates a new separation constraint `coord[left] + gap <= coord[right]`.
    #[inline]
    pub fn new(left: u32, right: u32, gap: S) -> Self {
        Self { left, right, gap }
    }
}

#[derive(Debug, Clone, Copy)]
struct InternalConstraint {
    left: u32,
    right: u32,
    gap: f64,
}

/// Flat `Vec`-backed arena projector for 1D quadratic-programming separation constraints
/// (Dwyer, Marriott & Stuckey 2006).
///
/// Uses intrusive singly-linked lists for block membership, flat CSR neighbor lists, and
/// [`VisitStamp`] for $O(1)$ clear traversals with zero `HashMap`/`HashSet` allocations and
/// zero recursion.
#[derive(Debug, Clone)]
pub struct ConstraintProjector<S: Scalar = f64> {
    num_vars: usize,
    // Per-variable arena state.
    block_of_var: Vec<u32>,
    var_offset: Vec<f64>,
    var_next: Vec<u32>,
    desired_x: Vec<f64>,
    dfdv: Vec<f64>,
    parent_var: Vec<u32>,
    parent_cidx: Vec<u32>,
    // Per-block arena state.
    block_head: Vec<u32>,
    block_tail: Vec<u32>,
    block_len: Vec<u32>,
    block_pos: Vec<f64>,
    free_blocks: Vec<u32>,
    // Per-solve constraint state and CSR adjacency.
    constraints: Vec<InternalConstraint>,
    is_active: Vec<u8>,
    lm: Vec<f64>,
    adj_indptr: Vec<u32>,
    adj_to: Vec<u32>,
    adj_cidx: Vec<u32>,
    adj_cursor: Vec<u32>,
    // Iterative DFS/BFS scratch.
    dfs_order: Vec<(u32, u32, u32, bool)>,
    bfs_queue: Vec<u32>,
    stamp: VisitStamp,
    _marker: PhantomData<S>,
}

impl<S: Scalar> ConstraintProjector<S> {
    /// Creates a [`ConstraintProjector`] pre-sized for `num_vars` variables.
    pub fn new(num_vars: usize) -> Self {
        Self {
            num_vars,
            block_of_var: vec![0; num_vars],
            var_offset: vec![0.0; num_vars],
            var_next: vec![u32::MAX; num_vars],
            desired_x: vec![0.0; num_vars],
            dfdv: vec![0.0; num_vars],
            parent_var: vec![u32::MAX; num_vars],
            parent_cidx: vec![u32::MAX; num_vars],
            block_head: vec![u32::MAX; num_vars],
            block_tail: vec![u32::MAX; num_vars],
            block_len: vec![0; num_vars],
            block_pos: vec![0.0; num_vars],
            free_blocks: Vec::new(),
            constraints: Vec::new(),
            is_active: Vec::new(),
            lm: Vec::new(),
            adj_indptr: vec![0; num_vars + 1],
            adj_to: Vec::new(),
            adj_cidx: Vec::new(),
            adj_cursor: vec![0; num_vars],
            dfs_order: Vec::with_capacity(num_vars),
            bfs_queue: Vec::with_capacity(num_vars),
            stamp: VisitStamp::new(num_vars),
            _marker: PhantomData,
        }
    }

    /// Returns the number of variables managed by this projector.
    #[inline]
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Projects `coords` in-place onto the feasible region defined by `constraints`,
    /// minimizing $\sum_i (x'_i - x_i)^2$.
    pub fn project_1d(
        &mut self,
        coords: &mut [S],
        constraints: &[SeparationConstraint<S>],
    ) -> CoreResult<()> {
        let n = self.num_vars;
        if coords.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                coords.len(),
                "ConstraintProjector::project_1d::coords",
            ));
        }
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_input(
                "ConstraintProjector num_vars exceeds u32::MAX",
            ));
        }
        for &v in coords.iter() {
            if !v.is_finite() {
                return Err(CoreError::non_finite(
                    "ConstraintProjector::project_1d::coords",
                ));
            }
        }

        // Validate and deduplicate parallel constraints (retaining maximum gap).
        self.constraints.clear();
        for c in constraints {
            if (c.left as usize) >= n {
                return Err(CoreError::index_out_of_bounds(
                    c.left as usize,
                    n,
                    "SeparationConstraint::left",
                ));
            }
            if (c.right as usize) >= n {
                return Err(CoreError::index_out_of_bounds(
                    c.right as usize,
                    n,
                    "SeparationConstraint::right",
                ));
            }
            if !c.gap.is_finite() {
                return Err(CoreError::non_finite("SeparationConstraint::gap"));
            }
            let gap = c.gap.to_f64();
            if c.left == c.right {
                if gap > VIOLATION_TOL {
                    return Err(CoreError::invalid_input(
                        "infeasible self-constraint with positive gap",
                    ));
                }
                continue;
            }
            self.constraints.push(InternalConstraint {
                left: c.left,
                right: c.right,
                gap,
            });
        }

        if n == 0 || self.constraints.is_empty() {
            return Ok(());
        }

        self.constraints.sort_unstable_by(|a, b| {
            a.left
                .cmp(&b.left)
                .then_with(|| a.right.cmp(&b.right))
                .then_with(|| b.gap.total_cmp(&a.gap))
        });
        self.constraints
            .dedup_by(|b, a| a.left == b.left && a.right == b.right);

        let m = self.constraints.len();
        self.is_active.clear();
        self.is_active.resize(m, 0);
        self.lm.clear();
        self.lm.resize(m, 0.0);

        // Initialize singleton blocks.
        self.block_head.truncate(n);
        self.block_tail.truncate(n);
        self.block_len.truncate(n);
        self.block_pos.truncate(n);
        self.free_blocks.clear();

        for (i, &coord) in coords.iter().enumerate() {
            let x_val = coord.to_f64();
            let i_u32 = i as u32;
            self.desired_x[i] = x_val;
            self.block_of_var[i] = i_u32;
            self.var_offset[i] = 0.0;
            self.var_next[i] = u32::MAX;
            self.block_head[i] = i_u32;
            self.block_tail[i] = i_u32;
            self.block_len[i] = 1;
            self.block_pos[i] = x_val;
        }

        // Build static undirected CSR neighbor list over all constraints.
        self.build_constraint_csr()?;

        // QPSC outer loop: project violations and split negative-multiplier active edges.
        let max_outer = 16usize;
        let max_expand_per_pass = (m + 1).saturating_mul(8);

        for _outer in 0..max_outer {
            self.project_feasible(max_expand_per_pass)?;
            if self.split_blocks() {
                break;
            }
        }
        // Final projection pass to guarantee feasibility after any split.
        self.project_feasible(max_expand_per_pass)?;

        for (i, out) in coords.iter_mut().enumerate() {
            *out = S::from_f64(self.var_pos(i));
        }

        Ok(())
    }

    /// Projects axis `axis` (`0` for X, `1` for Y) of `layout` in-place using `constraints`.
    pub fn project_layout_axis(
        &mut self,
        layout: &mut Layout2d<S>,
        axis: usize,
        constraints: &[SeparationConstraint<S>],
    ) -> CoreResult<()> {
        if layout.len() != self.num_vars {
            return Err(CoreError::dimension_mismatch(
                self.num_vars,
                layout.len(),
                "ConstraintProjector::project_layout_axis",
            ));
        }
        match axis {
            0 => self.project_1d(layout.x_mut(), constraints),
            1 => self.project_1d(layout.y_mut(), constraints),
            _ => Err(CoreError::invalid_input(format!(
                "layout axis must be 0 (x) or 1 (y), got {axis}"
            ))),
        }
    }

    #[inline]
    fn var_pos(&self, v: usize) -> f64 {
        let b = self.block_of_var[v] as usize;
        self.block_pos[b] + self.var_offset[v]
    }

    #[inline]
    fn violation(&self, c_idx: usize) -> f64 {
        let c = &self.constraints[c_idx];
        self.var_pos(c.left as usize) + c.gap - self.var_pos(c.right as usize)
    }

    fn build_constraint_csr(&mut self) -> CoreResult<()> {
        let n = self.num_vars;
        self.adj_indptr.fill(0);
        for c in &self.constraints {
            self.adj_indptr[c.left as usize + 1] += 1;
            self.adj_indptr[c.right as usize + 1] += 1;
        }
        for i in 0..n {
            self.adj_indptr[i + 1] = self.adj_indptr[i + 1]
                .checked_add(self.adj_indptr[i])
                .ok_or_else(|| CoreError::invalid_csr("ConstraintProjector CSR nnz overflow"))?;
        }
        let total = self.adj_indptr[n] as usize;
        self.adj_to.resize(total, 0);
        self.adj_cidx.resize(total, 0);
        self.adj_cursor.copy_from_slice(&self.adj_indptr[..n]);

        for (c_idx, c) in self.constraints.iter().enumerate() {
            let c_u32 = c_idx as u32;
            let pl = self.adj_cursor[c.left as usize] as usize;
            self.adj_to[pl] = c.right;
            self.adj_cidx[pl] = c_u32;
            self.adj_cursor[c.left as usize] += 1;

            let pr = self.adj_cursor[c.right as usize] as usize;
            self.adj_to[pr] = c.left;
            self.adj_cidx[pr] = c_u32;
            self.adj_cursor[c.right as usize] += 1;
        }
        Ok(())
    }

    fn project_feasible(&mut self, max_expands: usize) -> CoreResult<()> {
        let m = self.constraints.len();
        let mut expands = 0usize;

        loop {
            let mut best_c = usize::MAX;
            let mut best_viol = VIOLATION_TOL;
            for c_idx in 0..m {
                let v = self.violation(c_idx);
                if v > best_viol {
                    best_viol = v;
                    best_c = c_idx;
                }
            }
            if best_c == usize::MAX {
                break;
            }

            let c = self.constraints[best_c];
            let bl = self.block_of_var[c.left as usize] as usize;
            let br = self.block_of_var[c.right as usize] as usize;
            if bl != br {
                self.merge_block(bl, br, best_c);
            } else {
                expands += 1;
                if expands > max_expands {
                    return Err(CoreError::invalid_input(
                        "separation constraints failed to converge (possible infeasible cycle)",
                    ));
                }
                self.expand_block(bl, best_c)?;
            }
        }

        Ok(())
    }

    fn optimal_position(&self, b: usize) -> f64 {
        let len = self.block_len[b];
        if len == 0 {
            return 0.0;
        }
        let mut sum = 0.0_f64;
        let mut cur = self.block_head[b];
        while cur != u32::MAX {
            let idx = cur as usize;
            sum += self.desired_x[idx] - self.var_offset[idx];
            cur = self.var_next[idx];
        }
        sum / (len as f64)
    }

    fn merge_block(&mut self, l: usize, r: usize, c_idx: usize) {
        let c = self.constraints[c_idx];
        let d = self.var_offset[c.left as usize] + c.gap - self.var_offset[c.right as usize];
        let n_l = self.block_len[l] as f64;
        let n_r = self.block_len[r] as f64;

        self.block_pos[l] = (self.block_pos[l] * n_l + (self.block_pos[r] - d) * n_r) / (n_l + n_r);
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
        self.free_blocks.push(r as u32);
    }

    /// Iteratively computes Lagrange multipliers on block `b`'s active spanning tree
    /// starting at `root`, returning the active constraint index with minimum multiplier
    /// `(c_idx, lm_val)` if the block has at least one active constraint.
    fn compute_active_tree_lm(&mut self, b: usize, root: u32) -> Option<(usize, f64)> {
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

    fn expand_block(&mut self, b: usize, viol_c_idx: usize) -> CoreResult<()> {
        let viol_c = self.constraints[viol_c_idx];
        let viol_amount = self.violation(viol_c_idx);
        let src = viol_c.left;
        let dst = viol_c.right;

        self.compute_active_tree_lm(b, src);

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

    /// Splits blocks whose most negative active Lagrange multiplier is `< -SPLIT_LM_TOL`.
    /// Returns `true` if no blocks were split.
    fn split_blocks(&mut self) -> bool {
        let mut nosplit = true;
        let num_blocks = self.block_head.len();

        for b in 0..num_blocks {
            if self.block_len[b] == 0 {
                continue;
            }
            let root = self.block_head[b];
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
            let new_b = if let Some(free_b) = self.free_blocks.pop() {
                free_b as usize
            } else {
                let id = self.block_head.len();
                self.block_head.push(u32::MAX);
                self.block_tail.push(u32::MAX);
                self.block_len.push(0);
                self.block_pos.push(0.0);
                id
            };

            // Partition old block `b`'s linked list into `b` and `new_b`.
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
            self.block_pos[b] = self.optimal_position(b);

            self.block_head[new_b] = nb_head;
            self.block_tail[new_b] = nb_tail;
            self.block_len[new_b] = nb_len;
            self.block_pos[new_b] = self.optimal_position(new_b);
        }

        nosplit
    }
}

/// Generates hierarchical vertical separation constraints `y[u] + gap <= y[v]` for edges `(u, v)`
/// spanning layers `layers[u] < layers[v]`, where `gap = min_layer_distance * (layers[v] - layers[u])`.
pub fn generate_layered_constraints<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    layers: &[usize],
    min_layer_distance: S,
) -> CoreResult<Vec<SeparationConstraint<S>>> {
    let n = graph.num_nodes();
    if layers.len() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            layers.len(),
            "generate_layered_constraints::layers",
        ));
    }
    if !min_layer_distance.is_finite() {
        return Err(CoreError::non_finite(
            "generate_layered_constraints::min_layer_distance",
        ));
    }
    if min_layer_distance < S::ZERO {
        return Err(CoreError::invalid_input("min_layer_distance must be >= 0"));
    }

    let mut constraints = Vec::new();
    for u in 0..n {
        let lu = layers[u];
        for &v in graph.neighbors(u) {
            let lv = layers[v as usize];
            if lu < lv {
                let span = lv - lu;
                let gap = min_layer_distance * S::from_usize(span);
                constraints.push(SeparationConstraint::new(u as u32, v, gap));
            }
        }
    }
    Ok(constraints)
}

/// Projects 2D rectangle positions in `layout` to eliminate all axis-aligned overlaps
/// while minimizing quadratic coordinate displacement.
///
/// Uses $O(N \log N + K)$ [`RectangleOverlapSweep`] and [`ConstraintProjector`] across
/// alternating X and Y projection passes until zero overlaps remain.
pub fn project_rectangle_no_overlap_2d<S: Scalar>(
    layout: &mut Layout2d<S>,
    half_widths: &[S],
    half_heights: &[S],
) -> CoreResult<()> {
    let n = layout.len();
    if half_widths.len() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            half_widths.len(),
            "project_rectangle_no_overlap_2d::half_widths",
        ));
    }
    if half_heights.len() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            half_heights.len(),
            "project_rectangle_no_overlap_2d::half_heights",
        ));
    }
    layout.validate_finite()?;
    for i in 0..n {
        if !half_widths[i].is_finite() || !half_heights[i].is_finite() {
            return Err(CoreError::non_finite(
                "project_rectangle_no_overlap_2d::extents",
            ));
        }
        if half_widths[i] < S::ZERO || half_heights[i] < S::ZERO {
            return Err(CoreError::invalid_input(
                "half_widths and half_heights must be >= 0",
            ));
        }
    }

    if n < 2 {
        return Ok(());
    }

    let mut projector = ConstraintProjector::<S>::new(n);
    let mut x_constraints = Vec::new();
    let mut y_constraints = Vec::new();

    // Guard against 1-ULP rounding re-overlaps in f32/f64 after projection.
    let ulp_scale = 1.0 + 8.0 * S::EPSILON.to_f64();
    let ulp_abs = 8.0 * S::EPSILON.to_f64();

    let max_passes = 32usize;
    for pass in 0..max_passes {
        let overlaps =
            RectangleOverlapSweep::find_layout_overlaps(layout, half_widths, half_heights)?;
        if overlaps.is_empty() {
            return Ok(());
        }

        x_constraints.clear();
        generate_x_pass_constraints(
            layout.x(),
            layout.y(),
            half_widths,
            half_heights,
            &overlaps,
            pass > 0,
            ulp_scale,
            ulp_abs,
            &mut x_constraints,
        );
        if !x_constraints.is_empty() {
            projector.project_1d(layout.x_mut(), &x_constraints)?;
        }

        let overlaps_after_x =
            RectangleOverlapSweep::find_layout_overlaps(layout, half_widths, half_heights)?;
        if overlaps_after_x.is_empty() {
            return Ok(());
        }

        y_constraints.clear();
        generate_y_pass_constraints(
            layout.x(),
            layout.y(),
            half_widths,
            half_heights,
            &overlaps_after_x,
            ulp_scale,
            ulp_abs,
            &mut y_constraints,
        );
        if !y_constraints.is_empty() {
            projector.project_1d(layout.y_mut(), &y_constraints)?;
        }
    }

    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct SweepEvent {
    pos: f64,
    is_open: bool,
    idx: usize,
}

#[allow(clippy::too_many_arguments)]
fn generate_x_pass_constraints<S: Scalar>(
    xs: &[S],
    ys: &[S],
    half_widths: &[S],
    half_heights: &[S],
    overlaps_2d: &[(usize, usize)],
    force_all_overlaps: bool,
    ulp_scale: f64,
    ulp_abs: f64,
    out: &mut Vec<SeparationConstraint<S>>,
) {
    let n = xs.len();
    let mut events = Vec::with_capacity(n * 2);
    for i in 0..n {
        let y = ys[i].to_f64();
        let hh = half_heights[i].to_f64();
        events.push(SweepEvent {
            pos: y - hh,
            is_open: true,
            idx: i,
        });
        events.push(SweepEvent {
            pos: y + hh,
            is_open: false,
            idx: i,
        });
    }
    events.sort_unstable_by(|a, b| {
        a.pos
            .total_cmp(&b.pos)
            .then_with(|| a.is_open.cmp(&b.is_open))
            .then_with(|| a.idx.cmp(&b.idx))
    });

    let mut scanline: BTreeSet<(OrdF64, usize)> = BTreeSet::new();
    let mut emitted_overlap = false;

    for ev in events {
        let v = ev.idx;
        let key = (OrdF64(xs[v].to_f64()), v);
        if ev.is_open {
            scanline.insert(key);
            let xv = xs[v].to_f64();
            let hw_v = half_widths[v].to_f64();

            // Immediate non-overlapping right neighbor in same Y-band (prevents crossing).
            for &(_, u) in scanline.range(key..).skip(1) {
                let xu = xs[u].to_f64();
                let req_x = hw_v + half_widths[u].to_f64();
                if xu - xv >= req_x {
                    let pair_ulp = ulp_abs * (1.0 + xu.abs().max(xv.abs()));
                    let gap = S::from_f64(req_x * ulp_scale + pair_ulp);
                    out.push(SeparationConstraint::new(v as u32, u as u32, gap));
                    break;
                }
            }
            // Immediate non-overlapping left neighbor in same Y-band.
            for &(_, u) in scanline.range(..key).rev() {
                let xu = xs[u].to_f64();
                let req_x = hw_v + half_widths[u].to_f64();
                if xv - xu >= req_x {
                    let pair_ulp = ulp_abs * (1.0 + xu.abs().max(xv.abs()));
                    let gap = S::from_f64(req_x * ulp_scale + pair_ulp);
                    out.push(SeparationConstraint::new(u as u32, v as u32, gap));
                    break;
                }
            }
        } else {
            scanline.remove(&key);
        }
    }

    for &(i, j) in overlaps_2d {
        let req_x = half_widths[i].to_f64() + half_widths[j].to_f64();
        let req_y = half_heights[i].to_f64() + half_heights[j].to_f64();
        if req_x <= 0.0 || req_y <= 0.0 {
            continue;
        }
        let ox = req_x - (xs[i].to_f64() - xs[j].to_f64()).abs();
        let oy = req_y - (ys[i].to_f64() - ys[j].to_f64()).abs();
        if ox <= oy || force_all_overlaps {
            emitted_overlap = true;
            let pair_ulp = ulp_abs * (1.0 + xs[i].to_f64().abs().max(xs[j].to_f64().abs()));
            let gap = S::from_f64(req_x * ulp_scale + pair_ulp);
            let ord = xs[i]
                .to_f64()
                .total_cmp(&xs[j].to_f64())
                .then_with(|| i.cmp(&j));
            if ord == Ordering::Less {
                out.push(SeparationConstraint::new(i as u32, j as u32, gap));
            } else {
                out.push(SeparationConstraint::new(j as u32, i as u32, gap));
            }
        }
    }

    if !emitted_overlap {
        out.clear();
    }
}

#[allow(clippy::too_many_arguments)]
fn generate_y_pass_constraints<S: Scalar>(
    xs: &[S],
    ys: &[S],
    half_widths: &[S],
    half_heights: &[S],
    overlaps_2d: &[(usize, usize)],
    ulp_scale: f64,
    ulp_abs: f64,
    out: &mut Vec<SeparationConstraint<S>>,
) {
    let n = xs.len();
    let mut events = Vec::with_capacity(n * 2);
    for i in 0..n {
        let x = xs[i].to_f64();
        let hw = half_widths[i].to_f64();
        events.push(SweepEvent {
            pos: x - hw,
            is_open: true,
            idx: i,
        });
        events.push(SweepEvent {
            pos: x + hw,
            is_open: false,
            idx: i,
        });
    }
    events.sort_unstable_by(|a, b| {
        a.pos
            .total_cmp(&b.pos)
            .then_with(|| a.is_open.cmp(&b.is_open))
            .then_with(|| a.idx.cmp(&b.idx))
    });

    let mut scanline: BTreeSet<(OrdF64, usize)> = BTreeSet::new();

    for ev in events {
        let v = ev.idx;
        let key = (OrdF64(ys[v].to_f64()), v);
        if ev.is_open {
            scanline.insert(key);
            let xv = xs[v].to_f64();
            let yv = ys[v].to_f64();
            let hw_v = half_widths[v].to_f64();
            let hh_v = half_heights[v].to_f64();

            for &(_, u) in scanline.range(key..).skip(1) {
                if (xs[u].to_f64() - xv).abs() < hw_v + half_widths[u].to_f64() {
                    let yu = ys[u].to_f64();
                    let req_y = hh_v + half_heights[u].to_f64();
                    let pair_ulp = ulp_abs * (1.0 + yu.abs().max(yv.abs()));
                    let gap = S::from_f64(req_y * ulp_scale + pair_ulp);
                    out.push(SeparationConstraint::new(v as u32, u as u32, gap));
                    if yu - yv >= req_y {
                        break;
                    }
                }
            }
            for &(_, u) in scanline.range(..key).rev() {
                if (xs[u].to_f64() - xv).abs() < hw_v + half_widths[u].to_f64() {
                    let yu = ys[u].to_f64();
                    let req_y = hh_v + half_heights[u].to_f64();
                    let pair_ulp = ulp_abs * (1.0 + yu.abs().max(yv.abs()));
                    let gap = S::from_f64(req_y * ulp_scale + pair_ulp);
                    out.push(SeparationConstraint::new(u as u32, v as u32, gap));
                    if yv - yu >= req_y {
                        break;
                    }
                }
            }
        } else {
            scanline.remove(&key);
        }
    }

    for &(i, j) in overlaps_2d {
        let req_y = half_heights[i].to_f64() + half_heights[j].to_f64();
        if req_y <= 0.0 {
            continue;
        }
        let pair_ulp = ulp_abs * (1.0 + ys[i].to_f64().abs().max(ys[j].to_f64().abs()));
        let gap = S::from_f64(req_y * ulp_scale + pair_ulp);
        let ord = ys[i]
            .to_f64()
            .total_cmp(&ys[j].to_f64())
            .then_with(|| i.cmp(&j));
        if ord == Ordering::Less {
            out.push(SeparationConstraint::new(i as u32, j as u32, gap));
        } else {
            out.push(SeparationConstraint::new(j as u32, i as u32, gap));
        }
    }
}

/// Removes overlaps between axis-aligned cluster bounding boxes while rigidly preserving
/// relative node coordinates within each cluster.
pub fn project_clustered_rectangle_no_overlap_2d<S: Scalar>(
    layout: &mut Layout2d<S>,
    cluster_ids: &[u32],
    half_widths: &[S],
    half_heights: &[S],
) -> CoreResult<()> {
    let n = layout.len();
    if cluster_ids.len() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            cluster_ids.len(),
            "project_clustered_rectangle_no_overlap_2d::cluster_ids",
        ));
    }
    if half_widths.len() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            half_widths.len(),
            "project_clustered_rectangle_no_overlap_2d::half_widths",
        ));
    }
    if half_heights.len() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            half_heights.len(),
            "project_clustered_rectangle_no_overlap_2d::half_heights",
        ));
    }
    layout.validate_finite()?;
    for i in 0..n {
        if !half_widths[i].is_finite() || !half_heights[i].is_finite() {
            return Err(CoreError::non_finite(
                "project_clustered_rectangle_no_overlap_2d::extents",
            ));
        }
        if half_widths[i] < S::ZERO || half_heights[i] < S::ZERO {
            return Err(CoreError::invalid_input(
                "half_widths and half_heights must be >= 0",
            ));
        }
    }

    if n < 2 {
        return Ok(());
    }

    let mut members: Vec<(u32, u32)> = cluster_ids
        .iter()
        .enumerate()
        .map(|(i, &cid)| (cid, i as u32))
        .collect();
    members.sort_unstable();

    let mut cluster_cx = Vec::new();
    let mut cluster_cy = Vec::new();
    let mut cluster_hw = Vec::new();
    let mut cluster_hh = Vec::new();
    let mut cluster_ranges = Vec::new();

    {
        let xs = layout.x();
        let ys = layout.y();
        let mut start = 0usize;
        while start < n {
            let cid = members[start].0;
            let mut end = start + 1;
            while end < n && members[end].0 == cid {
                end += 1;
            }
            let mut min_x = f64::INFINITY;
            let mut max_x = f64::NEG_INFINITY;
            let mut min_y = f64::INFINITY;
            let mut max_y = f64::NEG_INFINITY;
            for &(_, node_u32) in &members[start..end] {
                let idx = node_u32 as usize;
                let x = xs[idx].to_f64();
                let y = ys[idx].to_f64();
                let hw = half_widths[idx].to_f64();
                let hh = half_heights[idx].to_f64();
                min_x = min_x.min(x - hw);
                max_x = max_x.max(x + hw);
                min_y = min_y.min(y - hh);
                max_y = max_y.max(y + hh);
            }
            cluster_cx.push(S::from_f64(0.5 * (min_x + max_x)));
            cluster_cy.push(S::from_f64(0.5 * (min_y + max_y)));
            cluster_hw.push(S::from_f64(0.5 * (max_x - min_x)));
            cluster_hh.push(S::from_f64(0.5 * (max_y - min_y)));
            cluster_ranges.push((start, end));
            start = end;
        }
    }

    if cluster_ranges.len() < 2 {
        return Ok(());
    }

    let old_cx = cluster_cx.clone();
    let old_cy = cluster_cy.clone();
    let mut cluster_layout = Layout2d::from_xy(cluster_cx, cluster_cy)?;
    project_rectangle_no_overlap_2d(&mut cluster_layout, &cluster_hw, &cluster_hh)?;

    let (xs_mut, ys_mut) = layout.xy_mut();
    let new_cx = cluster_layout.x();
    let new_cy = cluster_layout.y();
    for (c, &(start, end)) in cluster_ranges.iter().enumerate() {
        let dx = new_cx[c].to_f64() - old_cx[c].to_f64();
        let dy = new_cy[c].to_f64() - old_cy[c].to_f64();
        for &(_, node_u32) in &members[start..end] {
            let idx = node_u32 as usize;
            xs_mut[idx] = S::from_f64(xs_mut[idx].to_f64() + dx);
            ys_mut[idx] = S::from_f64(ys_mut[idx].to_f64() + dy);
        }
    }

    Ok(())
}
