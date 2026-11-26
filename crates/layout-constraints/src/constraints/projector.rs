use std::marker::PhantomData;

use egraph_core::{CoreError, CoreResult, CsrGraph, Layout2d, Scalar, VisitStamp};

pub(super) const VIOLATION_TOL: f64 = 1e-9;

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
pub(super) struct InternalConstraint {
    pub(super) left: u32,
    pub(super) right: u32,
    pub(super) gap: f64,
}

/// Flat `Vec`-backed arena projector for 1D quadratic-programming separation constraints
/// (Dwyer, Marriott & Stuckey 2006).
///
/// Uses intrusive singly-linked lists for block membership, flat CSR neighbor lists, and
/// [`VisitStamp`] for $O(1)$ clear traversals with zero `HashMap`/`HashSet` allocations and
/// zero recursion.
#[derive(Debug, Clone)]
pub struct ConstraintProjector<S: Scalar = f64> {
    pub(super) num_vars: usize,
    // Per-variable arena state.
    pub(super) block_of_var: Vec<u32>,
    pub(super) var_offset: Vec<f64>,
    pub(super) var_next: Vec<u32>,
    pub(super) desired_x: Vec<f64>,
    pub(super) dfdv: Vec<f64>,
    pub(super) parent_var: Vec<u32>,
    pub(super) parent_cidx: Vec<u32>,
    // Per-block arena state.
    pub(super) block_head: Vec<u32>,
    pub(super) block_tail: Vec<u32>,
    pub(super) block_len: Vec<u32>,
    pub(super) block_pos: Vec<f64>,
    pub(super) free_blocks: Vec<u32>,
    // Per-solve constraint state and CSR adjacency.
    pub(super) constraints: Vec<InternalConstraint>,
    pub(super) is_active: Vec<u8>,
    pub(super) lm: Vec<f64>,
    pub(super) adj_indptr: Vec<u32>,
    pub(super) adj_to: Vec<u32>,
    pub(super) adj_cidx: Vec<u32>,
    pub(super) adj_cursor: Vec<u32>,
    // Iterative DFS/BFS scratch.
    pub(super) dfs_order: Vec<(u32, u32, u32, bool)>,
    pub(super) bfs_queue: Vec<u32>,
    pub(super) stamp: VisitStamp,
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
    pub(super) fn var_pos(&self, v: usize) -> f64 {
        let b = self.block_of_var[v] as usize;
        self.block_pos[b] + self.var_offset[v]
    }

    #[inline]
    pub(super) fn violation(&self, c_idx: usize) -> f64 {
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

    pub(super) fn optimal_position(&self, b: usize) -> f64 {
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
