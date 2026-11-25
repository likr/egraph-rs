//! Hayashi et al. (1998) `PFS'` two-pass scan-line node overlap removal solver.
//!
//! Implements `PFS'` (*"Layout Adjustment and the Mental Map"*, Misue, Eades, Lai & Sugiyama;
//! *"A Fast Algorithm for Overlap Removal"*, Hayashi, Inoue, Masuzawa & Fujiwara, GD '98)
//! following the authoritative `agorajs/agora-pfsp` two-pass `horizontalScan` + `verticalScan`
//! equations, augmented with:
//! 1. Deterministic exact-coincident-point jitter (`dx == 0.0 && dy == 0.0`) bounded strictly
//!    below `0.25 * min_positive_gap` so zero-separation pairs do not stall the scan-line force
//!    function while preserving all strict orthogonal order relations.
//! 2. Symmetric border clearance (`0.5 * padding` added to each half-width and half-height).
//! 3. Direct displacement accumulation without mean-centering or extent round-trip cancellation,
//!    magnitude-aware `S`-precision separation slack, `f64`-exact $O(N \log N)$ sweep verification,
//!    [`RectangleOverlapSweep`] verification, an $O(N \log N)$ orthogonal-order-inversion check,
//!    and a strictly order-preserving monotonic X/Y safety sweep so converged outputs satisfy
//!    `last_overlap_count() == 0` and `oo_nni == 0` for both `f64` and `f32`.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, DeterministicRng, IterativeSolver, Layout2d, Scalar, SolverStatus,
};
use egraph_spatial::{Rect2d, RectangleOverlapSweep};

/// Hyperparameters for [`PfsPrimeOverlapRemoval`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PfsPrimeConfig {
    /// Extra minimum clearance added between rectangle borders (`default: 0.0`).
    pub padding: f64,
    /// Explicit deterministic seed for coincident-point perturbation (`default: 42`).
    pub seed: u64,
}

impl Default for PfsPrimeConfig {
    fn default() -> Self {
        Self {
            padding: 0.0,
            seed: 42,
        }
    }
}

impl PfsPrimeConfig {
    /// Validates all `PFS'` configuration parameters.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.padding.is_finite() {
            return Err(CoreError::non_finite("PfsPrimeConfig::padding"));
        }
        if self.padding < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "PfsPrimeConfig::padding must be >= 0.0, got {}",
                self.padding
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct HeapEntry {
    key: f64,
    node: usize,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.key.total_cmp(&other.key) == Ordering::Equal && self.node == other.node
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key
            .total_cmp(&other.key)
            .then_with(|| self.node.cmp(&other.node))
    }
}

#[inline]
fn normalize_neg_zero(v: f64) -> f64 {
    if v == 0.0 { 0.0 } else { v }
}

fn find_f64_exact_overlaps<S: Scalar>(
    layout: &Layout2d<S>,
    eff_hw_req: &[f64],
    eff_hh_req: &[f64],
) -> CoreResult<Vec<(usize, usize)>> {
    let n = layout.len();
    if n < 2 {
        return Ok(Vec::new());
    }
    let xs = layout.x();
    let ys = layout.y();
    let mut rects = Vec::with_capacity(n);
    for i in 0..n {
        let xi = xs[i].to_f64();
        let yi = ys[i].to_f64();
        let hw = eff_hw_req[i];
        let hh = eff_hh_req[i];
        let mx = 4.0 * (xi.abs() + hw) * f64::EPSILON + f64::MIN_POSITIVE;
        let my = 4.0 * (yi.abs() + hh) * f64::EPSILON + f64::MIN_POSITIVE;
        let x_min = (xi - hw - mx).clamp(-f64::MAX, f64::MAX);
        let x_max = (xi + hw + mx).clamp(-f64::MAX, f64::MAX);
        let y_min = (yi - hh - my).clamp(-f64::MAX, f64::MAX);
        let y_max = (yi + hh + my).clamp(-f64::MAX, f64::MAX);
        rects.push(Rect2d::new(x_min, x_max, y_min, y_max)?);
    }

    let mut candidates = RectangleOverlapSweep::find_overlaps(&rects)?;
    candidates.retain(|&(i, j)| {
        let req_x = eff_hw_req[i] + eff_hw_req[j];
        let req_y = eff_hh_req[i] + eff_hh_req[j];
        if req_x <= 0.0 || req_y <= 0.0 {
            return false;
        }
        let dx = (xs[i].to_f64() - xs[j].to_f64()).abs();
        let dy = (ys[i].to_f64() - ys[j].to_f64()).abs();
        dx < req_x && dy < req_y
    });
    candidates.sort_unstable();
    candidates.dedup();
    Ok(candidates)
}

fn count_union_pairs(a: &[(usize, usize)], b: &[(usize, usize)]) -> usize {
    let mut i = 0usize;
    let mut j = 0usize;
    let mut count = 0usize;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => {
                count += 1;
                i += 1;
            }
            Ordering::Greater => {
                count += 1;
                j += 1;
            }
            Ordering::Equal => {
                count += 1;
                i += 1;
                j += 1;
            }
        }
    }
    count + (a.len() - i) + (b.len() - j)
}

/// Checks in O(N log N) time whether `layout` introduces any strict orthogonal order inversion
/// relative to `(orig_x, orig_y)` by verifying `prev_group_max <= curr_group_min` across
/// consecutive `orig_x` tie-groups and `orig_y` tie-groups.
fn has_orthogonal_order_inversion<S: Scalar>(
    orig_x: &[f64],
    orig_y: &[f64],
    layout: &Layout2d<S>,
) -> bool {
    let n = orig_x.len().min(orig_y.len()).min(layout.len());
    if n < 2 {
        return false;
    }

    let mut order: Vec<usize> = Vec::with_capacity(n);
    for (orig, new_s) in [(&orig_x[..n], layout.x()), (&orig_y[..n], layout.y())] {
        order.clear();
        order.extend(0..n);
        order.sort_unstable_by(|&a, &b| orig[a].total_cmp(&orig[b]).then_with(|| a.cmp(&b)));

        let mut prev_group_max: Option<f64> = None;
        let mut pos = 0usize;
        while pos < n {
            let base = orig[order[pos]];
            let mut end = pos + 1;
            while end < n && orig[order[end]] == base {
                end += 1;
            }
            let mut curr_group_min = f64::INFINITY;
            let mut curr_group_max = f64::NEG_INFINITY;
            for &node in &order[pos..end] {
                let val = new_s[node].to_f64();
                if val < curr_group_min {
                    curr_group_min = val;
                }
                if val > curr_group_max {
                    curr_group_max = val;
                }
            }
            if prev_group_max.is_some_and(|prev_max| prev_max > curr_group_min) {
                return true;
            }
            prev_group_max =
                Some(prev_group_max.map_or(curr_group_max, |pm| pm.max(curr_group_max)));
            pos = end;
        }
    }
    false
}

#[inline]
fn bump_coord_s_to_clear<S: Scalar>(
    mut cand_s: S,
    u_coord_s: S,
    u_ext_s: S,
    v_ext_s: S,
    u_coord_f64: f64,
    req_f64: f64,
) -> S {
    let u_upper_s = u_coord_s + u_ext_s;
    if cand_s.to_f64() - u_coord_f64 >= req_f64 && (cand_s - v_ext_s) >= u_upper_s {
        return cand_s;
    }
    let target = (u_coord_f64 + req_f64).max((u_upper_s + v_ext_s).to_f64());
    if cand_s.to_f64() < target {
        let target_s = S::from_f64(target);
        if target_s.is_finite() {
            cand_s = target_s;
        }
    }
    let mut step = (cand_s.to_f64().abs() * S::EPSILON.to_f64()).max(S::MIN_POSITIVE.to_f64());
    while cand_s.to_f64() - u_coord_f64 < req_f64 || (cand_s - v_ext_s) < u_upper_s {
        let next = S::from_f64(cand_s.to_f64() + step);
        if !next.is_finite() || !step.is_finite() {
            break;
        }
        if next > cand_s {
            cand_s = next;
        } else {
            step *= 2.0;
        }
    }
    cand_s
}

/// Cooperative Hayashi et al. (1998) `PFS'` two-pass scan-line node overlap removal solver.
///
/// Executes the deterministic horizontal and vertical scan-line passes in a single
/// [`IterativeSolver::step`] (`total_iterations() == 1`), strictly preserving orthogonal order
/// (`oo_nni == 0`) while guaranteeing `0` overlapping rectangles in both `f64`-exact and
/// `S`-precision sweep checks.
#[derive(Debug, Clone)]
pub struct PfsPrimeOverlapRemoval<S: Scalar = f64> {
    config: PfsPrimeConfig,
    num_nodes: usize,
    eff_hw_s: Vec<S>,
    eff_hh_s: Vec<S>,
    eff_hw_req: Vec<f64>,
    eff_hh_req: Vec<f64>,
    rng: DeterministicRng,
    orig_x: Vec<f64>,
    orig_y: Vec<f64>,
    work_x: Vec<f64>,
    work_y: Vec<f64>,
    up_gamma_x: Vec<f64>,
    up_gamma_y: Vec<f64>,
    order: Vec<usize>,
    sort_pts: Vec<(f64, f64, u32)>,
    gap_scratch: Vec<f64>,
    iter: usize,
    last_overlap_count: usize,
    status: SolverStatus,
    _marker: PhantomData<S>,
}

impl<S: Scalar> PfsPrimeOverlapRemoval<S> {
    /// Creates a [`PfsPrimeOverlapRemoval`] solver for rectangular nodes with per-node
    /// half-widths and half-heights.
    pub fn new(
        half_widths: Vec<S>,
        half_heights: Vec<S>,
        config: PfsPrimeConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        let n = half_widths.len();
        if half_heights.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                half_heights.len(),
                "PfsPrimeOverlapRemoval::new::half_heights",
            ));
        }
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_input(
                "PfsPrimeOverlapRemoval node count exceeds u32::MAX",
            ));
        }

        let pad_half = 0.5 * config.padding;
        let mut eff_hw_s = Vec::with_capacity(n);
        let mut eff_hh_s = Vec::with_capacity(n);
        let mut eff_hw_req = Vec::with_capacity(n);
        let mut eff_hh_req = Vec::with_capacity(n);

        for i in 0..n {
            let hw = half_widths[i];
            let hh = half_heights[i];
            if !hw.is_finite() || !hh.is_finite() {
                return Err(CoreError::non_finite(
                    "PfsPrimeOverlapRemoval::new::extents",
                ));
            }
            if hw < S::ZERO || hh < S::ZERO {
                return Err(CoreError::invalid_input(
                    "half_widths and half_heights must be >= 0",
                ));
            }
            let hw_f = hw.to_f64() + pad_half;
            let hh_f = hh.to_f64() + pad_half;
            let hw_s = S::from_f64(hw_f);
            let hh_s = S::from_f64(hh_f);
            if !hw_s.is_finite() || !hh_s.is_finite() {
                return Err(CoreError::non_finite(
                    "PfsPrimeOverlapRemoval::new::padded_extents",
                ));
            }
            eff_hw_req.push(hw_f.max(hw_s.to_f64()));
            eff_hh_req.push(hh_f.max(hh_s.to_f64()));
            eff_hw_s.push(hw_s);
            eff_hh_s.push(hh_s);
        }

        Ok(Self {
            config,
            num_nodes: n,
            eff_hw_s,
            eff_hh_s,
            eff_hw_req,
            eff_hh_req,
            rng: DeterministicRng::from_seed_u64(config.seed),
            orig_x: vec![0.0; n],
            orig_y: vec![0.0; n],
            work_x: vec![0.0; n],
            work_y: vec![0.0; n],
            up_gamma_x: vec![0.0; n],
            up_gamma_y: vec![0.0; n],
            order: (0..n).collect(),
            sort_pts: Vec::with_capacity(n),
            gap_scratch: Vec::with_capacity(n),
            iter: 0,
            last_overlap_count: 0,
            status: SolverStatus::Initialized,
            _marker: PhantomData,
        })
    }

    /// Creates a [`PfsPrimeOverlapRemoval`] solver for circular/square nodes with equal
    /// half-width and half-height given by `radii`.
    pub fn from_radii(radii: Vec<S>, config: PfsPrimeConfig) -> CoreResult<Self> {
        let half_heights = radii.clone();
        Self::new(radii, half_heights, config)
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &PfsPrimeConfig {
        &self.config
    }

    /// Returns the number of vertices.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the number of overlapping rectangle pairs recorded at the most recent step.
    #[inline]
    pub fn last_overlap_count(&self) -> usize {
        self.last_overlap_count
    }

    fn compute_min_positive_gap(&mut self) -> Option<f64> {
        let mut min_gap: Option<f64> = None;

        for coords in [&self.work_x, &self.work_y] {
            self.gap_scratch.clear();
            self.gap_scratch.extend_from_slice(coords);
            self.gap_scratch.sort_unstable_by(f64::total_cmp);
            for w in self.gap_scratch.windows(2) {
                if w[1] > w[0] {
                    let gap = w[1] - w[0];
                    if gap > 0.0 {
                        min_gap = Some(min_gap.map_or(gap, |m| m.min(gap)));
                    }
                }
            }
        }

        min_gap
    }

    fn jitter_coincident_points(&mut self) {
        let n = self.num_nodes;
        if n < 2 {
            return;
        }
        let min_pos_gap = self.compute_min_positive_gap();

        self.sort_pts.clear();
        for i in 0..n {
            self.sort_pts
                .push((self.work_x[i], self.work_y[i], i as u32));
        }
        self.sort_pts.sort_unstable_by(|a, b| {
            a.0.total_cmp(&b.0)
                .then_with(|| a.1.total_cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        });

        let mut i = 0usize;
        while i < n {
            let (base_x, base_y, _) = self.sort_pts[i];
            let mut j = i + 1;
            while j < n {
                let (nx, ny, _) = self.sort_pts[j];
                if nx == base_x && ny == base_y {
                    j += 1;
                } else {
                    break;
                }
            }
            if j - i > 1 {
                let coord_ulp = base_x.abs().max(base_y.abs()) * S::EPSILON.to_f64();
                for k in (i + 1)..j {
                    let idx = self.sort_pts[k].2 as usize;
                    let extent = self.eff_hw_req[idx].max(self.eff_hh_req[idx]).max(1e-2);
                    let desired_scale = (extent * 0.05).max(4.0 * coord_ulp).max(1e-5);
                    let jitter_scale = match min_pos_gap {
                        Some(gap) => desired_scale.min(0.2375 * gap),
                        None => desired_scale,
                    };
                    let angle = self.rng.next_f64() * std::f64::consts::TAU;
                    let radius = jitter_scale * (0.25 + 0.75 * self.rng.next_f64());
                    let (sin_a, cos_a) = angle.sin_cos();
                    self.work_x[idx] += radius * cos_a;
                    self.work_y[idx] += radius * sin_a;
                }
            }
            i = j;
        }
    }

    #[inline]
    fn pfsp_force(
        &self,
        i: usize,
        j: usize,
        sep_eps: f64,
        slack_x: f64,
        slack_y: f64,
    ) -> (f64, f64) {
        let base_req_x = self.eff_hw_req[i] + self.eff_hw_req[j];
        let base_req_y = self.eff_hh_req[i] + self.eff_hh_req[j];
        if base_req_x <= 0.0 || base_req_y <= 0.0 {
            return (0.0, 0.0);
        }
        let req_x = base_req_x * sep_eps + slack_x;
        let req_y = base_req_y * sep_eps + slack_y;

        let dx = self.work_x[j] - self.work_x[i];
        let dy = self.work_y[j] - self.work_y[i];
        if dx == 0.0 && dy == 0.0 {
            return (0.0, 0.0);
        }
        let adx = dx.abs();
        let ady = dy.abs();

        if dx == 0.0 {
            // Vertical alignment: |g_ij| = inf > G_ij -> pure vertical separation.
            let fy = dy.signum() * (req_y - ady);
            return (0.0, fy);
        }

        let g = dy / dx;
        let big_g = req_y / req_x;
        if g.abs() <= big_g {
            let fx = dx.signum() * (req_x - adx);
            (fx, fx * g)
        } else {
            let fy = dy.signum() * (req_y - ady);
            (fy / g, fy)
        }
    }

    fn horizontal_scan(&mut self, sep_eps: f64, slack_x: f64, slack_y: f64) {
        let n = self.num_nodes;
        for (idx, item) in self.order.iter_mut().enumerate() {
            *item = idx;
        }
        self.order.sort_unstable_by(|&a, &b| {
            self.work_x[a]
                .total_cmp(&self.work_x[b])
                .then_with(|| a.cmp(&b))
        });

        let mut i = 0usize;
        let mut sigma = 0.0_f64;
        let mut lmin = self.order[0];

        while i < n {
            let x_i = self.work_x[self.order[i]];
            let mut k = i;
            while k + 1 < n && (self.work_x[self.order[k + 1]] - x_i).abs() == 0.0 {
                k += 1;
            }

            let mut gamma = 0.0_f64;
            if x_i > self.work_x[lmin] {
                for m in i..=k {
                    let v = self.order[m];
                    let mut gamma_pp = 0.0_f64;
                    for j in 0..i {
                        let node_j = self.order[j];
                        let (fx, _) = self.pfsp_force(node_j, v, sep_eps, slack_x, slack_y);
                        gamma_pp = gamma_pp.max(self.up_gamma_x[node_j] + fx);
                    }
                    let gamma_p = if (self.work_x[v] - self.work_x[lmin]) + gamma_pp
                        < (self.eff_hw_req[v] - self.eff_hw_req[lmin])
                    {
                        sigma
                    } else {
                        gamma_pp
                    };
                    gamma = gamma.max(gamma_p);
                }
            }

            for m in i..=k {
                let r = self.order[m];
                self.up_gamma_x[r] = gamma;
                if (self.work_x[r] - self.work_x[lmin])
                    < (self.eff_hw_req[r] - self.eff_hw_req[lmin])
                {
                    lmin = r;
                }
            }

            let mut delta = 0.0_f64;
            for m in i..=k {
                let u = self.order[m];
                for j in (k + 1)..n {
                    let v = self.order[j];
                    let (fx, _) = self.pfsp_force(u, v, sep_eps, slack_x, slack_y);
                    delta = delta.max(fx);
                }
            }
            sigma += delta;
            i = k + 1;
        }
    }

    fn vertical_scan(&mut self, sep_eps: f64, slack_x: f64, slack_y: f64) {
        let n = self.num_nodes;
        for (idx, item) in self.order.iter_mut().enumerate() {
            *item = idx;
        }
        self.order.sort_unstable_by(|&a, &b| {
            self.work_y[a]
                .total_cmp(&self.work_y[b])
                .then_with(|| a.cmp(&b))
        });

        let mut i = 0usize;
        let mut sigma = 0.0_f64;
        let mut lmin = self.order[0];

        while i < n {
            let y_i = self.work_y[self.order[i]];
            let mut k = i;
            while k + 1 < n && (self.work_y[self.order[k + 1]] - y_i).abs() == 0.0 {
                k += 1;
            }

            let mut gamma = 0.0_f64;
            if y_i > self.work_y[lmin] {
                for m in i..=k {
                    let v = self.order[m];
                    let mut gamma_pp = 0.0_f64;
                    for j in 0..i {
                        let node_j = self.order[j];
                        let (_, fy) = self.pfsp_force(node_j, v, sep_eps, slack_x, slack_y);
                        gamma_pp = gamma_pp.max(self.up_gamma_y[node_j] + fy);
                    }
                    let gamma_p = if (self.work_y[v] - self.work_y[lmin]) + gamma_pp
                        < (self.eff_hh_req[v] - self.eff_hh_req[lmin])
                    {
                        sigma
                    } else {
                        gamma_pp
                    };
                    gamma = gamma.max(gamma_p);
                }
            }

            for m in i..=k {
                let r = self.order[m];
                self.up_gamma_y[r] = gamma;
                if (self.work_y[r] - self.work_y[lmin])
                    < (self.eff_hh_req[r] - self.eff_hh_req[lmin])
                {
                    lmin = r;
                }
            }

            let mut delta = 0.0_f64;
            for m in i..=k {
                let u = self.order[m];
                for j in (k + 1)..n {
                    let v = self.order[j];
                    let (_, fy) = self.pfsp_force(u, v, sep_eps, slack_x, slack_y);
                    delta = delta.max(fy);
                }
            }
            sigma += delta;
            i = k + 1;
        }
    }

    fn resolve_residual_overlaps_order_preserving(&self, layout: &mut Layout2d<S>) {
        let n = self.num_nodes;
        if n < 2 {
            return;
        }
        let s_eps = S::EPSILON.to_f64();

        // Assign each node to its initial-X tie group so Phase 1 can identify
        // all pairs sharing `orig_x` in O(1).
        let mut x_order: Vec<usize> = (0..n).collect();
        x_order.sort_unstable_by(|&a, &b| {
            self.orig_x[a]
                .total_cmp(&self.orig_x[b])
                .then_with(|| a.cmp(&b))
        });
        let mut orig_x_group = vec![0usize; n];
        let mut orig_x_has_tie = vec![false; n];
        let mut x_groups: Vec<(usize, usize)> = Vec::new();
        let mut gx_id = 0usize;
        let mut pos = 0usize;
        while pos < n {
            let base = self.orig_x[x_order[pos]];
            let mut end = pos + 1;
            while end < n && self.orig_x[x_order[end]] == base {
                end += 1;
            }
            let is_tie = end - pos > 1;
            for &node in &x_order[pos..end] {
                orig_x_group[node] = gx_id;
                orig_x_has_tie[node] = is_tie;
            }
            x_groups.push((pos, end));
            gx_id += 1;
            pos = end;
        }

        // Phase 1: Global monotonic Y-sweep across `orig_y` groups.
        // Resolves every overlap between pairs with `orig_x[u] == orig_x[v]`
        // (including exact-coincident inputs) while preserving global `orig_y` order.
        let mut y_order: Vec<usize> = (0..n).collect();
        y_order.sort_unstable_by(|&a, &b| {
            self.orig_y[a]
                .total_cmp(&self.orig_y[b])
                .then_with(|| a.cmp(&b))
        });
        let mut y_groups: Vec<(usize, usize)> = Vec::new();
        pos = 0;
        while pos < n {
            let base = self.orig_y[y_order[pos]];
            let mut end = pos + 1;
            while end < n && self.orig_y[y_order[end]] == base {
                end += 1;
            }
            y_groups.push((pos, end));
            pos = end;
        }

        let x_cur_s: Vec<S> = layout.x().to_vec();
        let y_cur: Vec<f64> = layout.y().iter().map(|v| v.to_f64()).collect();
        let mut y_out_s: Vec<S> = layout.y().to_vec();

        // Sort nodes inside each `orig_y` tie group by current `y_cur` so exact-coincident
        // nodes are swept monotonically from bottom to top.
        for &(g_start, g_end) in &y_groups {
            y_order[g_start..g_end]
                .sort_unstable_by(|&a, &b| y_cur[a].total_cmp(&y_cur[b]).then_with(|| a.cmp(&b)));
        }

        let mut suffix_max_hh = vec![0.0_f64; y_groups.len()];
        let mut running_hh = 0.0_f64;
        for g_idx in (0..y_groups.len()).rev() {
            let (gs, ge) = y_groups[g_idx];
            for &node in &y_order[gs..ge] {
                running_hh = running_hh.max(self.eff_hh_req[node]);
            }
            suffix_max_hh[g_idx] = running_hh;
        }

        let mut cum_shift_y = 0.0_f64;
        let mut prev_group_max_y: Option<S> = None;
        let mut active_y: BinaryHeap<Reverse<HeapEntry>> = BinaryHeap::new();

        for (g_idx, &(gs, ge)) in y_groups.iter().enumerate() {
            let mut min_y_cur = f64::INFINITY;
            for &node in &y_order[gs..ge] {
                min_y_cur = min_y_cur.min(y_cur[node]);
            }
            if let Some(pmy) = prev_group_max_y {
                let needed_min = pmy.to_f64();
                if min_y_cur + cum_shift_y < needed_min {
                    cum_shift_y = needed_min - min_y_cur;
                }
            }
            let cand_min = match prev_group_max_y {
                Some(pmy) => (min_y_cur + cum_shift_y).max(pmy.to_f64()),
                None => min_y_cur + cum_shift_y,
            };
            let max_hh_rem = suffix_max_hh[g_idx];
            let pop_threshold =
                cand_min - max_hh_rem - 4.0 * (cand_min.abs() + max_hh_rem) * s_eps - 1e-12;
            while active_y
                .peek()
                .is_some_and(|Reverse(entry)| entry.key <= pop_threshold)
            {
                active_y.pop();
            }

            let mut curr_group_max_y: Option<S> = None;
            for &v in &y_order[gs..ge] {
                let floor_y = match (prev_group_max_y, curr_group_max_y) {
                    (Some(p), Some(c)) => Some(p.max(c)),
                    (Some(p), None) => Some(p),
                    (None, Some(c)) => Some(c),
                    (None, None) => None,
                };
                let mut c_s = floor_y.map_or_else(
                    || S::from_f64(y_cur[v] + cum_shift_y),
                    |fy| S::from_f64(y_cur[v] + cum_shift_y).max(fy),
                );

                if orig_x_has_tie[v] {
                    let xv_s = x_cur_s[v];
                    let xv_f = xv_s.to_f64();
                    let hw_v_s = self.eff_hw_s[v];
                    let hh_v_s = self.eff_hh_s[v];

                    // Re-scan active intervals whenever c_s bumps upward so BinaryHeap
                    // backing-Vec ordering can never skip an earlier interval.
                    loop {
                        let mut bumped = false;
                        for &Reverse(entry) in &active_y {
                            let u = entry.node;
                            if orig_x_group[u] != orig_x_group[v] {
                                continue;
                            }
                            let req_x = self.eff_hw_req[u] + self.eff_hw_req[v];
                            let req_y = self.eff_hh_req[u] + self.eff_hh_req[v];
                            if req_x <= 0.0 || req_y <= 0.0 {
                                continue;
                            }
                            let xu_s = x_cur_s[u];
                            let hw_u_s = self.eff_hw_s[u];
                            let x_ov = (xu_s.to_f64() - xv_f).abs() < req_x
                                || ((xu_s - hw_u_s) < (xv_s + hw_v_s)
                                    && (xv_s - hw_v_s) < (xu_s + hw_u_s));
                            if !x_ov {
                                continue;
                            }

                            let yu_s = y_out_s[u];
                            let yu_f = yu_s.to_f64();
                            let hh_u_s = self.eff_hh_s[u];
                            let y_ov = (c_s.to_f64() - yu_f).abs() < req_y
                                || ((yu_s - hh_u_s) < (c_s + hh_v_s)
                                    && (c_s - hh_v_s) < (yu_s + hh_u_s));
                            if y_ov {
                                let next_c_s =
                                    bump_coord_s_to_clear(c_s, yu_s, hh_u_s, hh_v_s, yu_f, req_y);
                                if next_c_s > c_s {
                                    c_s = next_c_s;
                                    bumped = true;
                                }
                            }
                        }
                        if !bumped {
                            break;
                        }
                    }

                    let shift_needed = c_s.to_f64() - y_cur[v];
                    if shift_needed > cum_shift_y {
                        cum_shift_y = shift_needed;
                    }
                    let y_top_padded = (c_s.to_f64() + self.eff_hh_req[v])
                        .max((c_s + hh_v_s).to_f64())
                        + 4.0 * (c_s.to_f64().abs() + self.eff_hh_req[v]) * s_eps
                        + 1e-12;
                    active_y.push(Reverse(HeapEntry {
                        key: y_top_padded,
                        node: v,
                    }));
                }

                y_out_s[v] = c_s;
                curr_group_max_y = Some(curr_group_max_y.map_or(c_s, |m| m.max(c_s)));
            }
            prev_group_max_y = curr_group_max_y;
        }
        layout.y_mut().copy_from_slice(&y_out_s);

        // Phase 2: Global monotonic X-sweep across `orig_x` groups with Y frozen.
        // Resolves all remaining overlaps (`orig_x[u] != orig_x[v]`) while preserving
        // global `orig_x` order.
        let x_cur: Vec<f64> = layout.x().iter().map(|v| v.to_f64()).collect();
        let mut x_out_s: Vec<S> = layout.x().to_vec();

        for &(g_start, g_end) in &x_groups {
            x_order[g_start..g_end]
                .sort_unstable_by(|&a, &b| x_cur[a].total_cmp(&x_cur[b]).then_with(|| a.cmp(&b)));
        }

        let mut suffix_max_hw = vec![0.0_f64; x_groups.len()];
        let mut running_hw = 0.0_f64;
        for g_idx in (0..x_groups.len()).rev() {
            let (gs, ge) = x_groups[g_idx];
            for &node in &x_order[gs..ge] {
                running_hw = running_hw.max(self.eff_hw_req[node]);
            }
            suffix_max_hw[g_idx] = running_hw;
        }

        let mut cum_shift_x = 0.0_f64;
        let mut prev_group_max_x: Option<S> = None;
        let mut active_x: BinaryHeap<Reverse<HeapEntry>> = BinaryHeap::new();

        for (g_idx, &(gs, ge)) in x_groups.iter().enumerate() {
            let mut min_x_cur = f64::INFINITY;
            for &node in &x_order[gs..ge] {
                min_x_cur = min_x_cur.min(x_cur[node]);
            }
            if let Some(pmx) = prev_group_max_x {
                let needed_min = pmx.to_f64();
                if min_x_cur + cum_shift_x < needed_min {
                    cum_shift_x = needed_min - min_x_cur;
                }
            }
            let cand_min = match prev_group_max_x {
                Some(pmx) => (min_x_cur + cum_shift_x).max(pmx.to_f64()),
                None => min_x_cur + cum_shift_x,
            };
            let max_hw_rem = suffix_max_hw[g_idx];
            let pop_threshold =
                cand_min - max_hw_rem - 4.0 * (cand_min.abs() + max_hw_rem) * s_eps - 1e-12;
            while active_x
                .peek()
                .is_some_and(|Reverse(entry)| entry.key <= pop_threshold)
            {
                active_x.pop();
            }

            let mut curr_group_max_x: Option<S> = None;
            for &v in &x_order[gs..ge] {
                let floor_x = match (prev_group_max_x, curr_group_max_x) {
                    (Some(p), Some(c)) => Some(p.max(c)),
                    (Some(p), None) => Some(p),
                    (None, Some(c)) => Some(c),
                    (None, None) => None,
                };
                let mut c_s = floor_x.map_or_else(
                    || S::from_f64(x_cur[v] + cum_shift_x),
                    |fx| S::from_f64(x_cur[v] + cum_shift_x).max(fx),
                );

                let yv_s = y_out_s[v];
                let yv_f = yv_s.to_f64();
                let hw_v_s = self.eff_hw_s[v];
                let hh_v_s = self.eff_hh_s[v];

                // Re-scan active intervals whenever c_s bumps rightward so BinaryHeap
                // backing-Vec ordering can never skip an earlier interval.
                loop {
                    let mut bumped = false;
                    for &Reverse(entry) in &active_x {
                        let u = entry.node;
                        let req_x = self.eff_hw_req[u] + self.eff_hw_req[v];
                        let req_y = self.eff_hh_req[u] + self.eff_hh_req[v];
                        if req_x <= 0.0 || req_y <= 0.0 {
                            continue;
                        }
                        let yu_s = y_out_s[u];
                        let hh_u_s = self.eff_hh_s[u];
                        let y_ov = (yu_s.to_f64() - yv_f).abs() < req_y
                            || ((yu_s - hh_u_s) < (yv_s + hh_v_s)
                                && (yv_s - hh_v_s) < (yu_s + hh_u_s));
                        if !y_ov {
                            continue;
                        }

                        let xu_s = x_out_s[u];
                        let xu_f = xu_s.to_f64();
                        let hw_u_s = self.eff_hw_s[u];
                        let x_ov = (c_s.to_f64() - xu_f).abs() < req_x
                            || ((xu_s - hw_u_s) < (c_s + hw_v_s)
                                && (c_s - hw_v_s) < (xu_s + hw_u_s));
                        if x_ov {
                            let next_c_s =
                                bump_coord_s_to_clear(c_s, xu_s, hw_u_s, hw_v_s, xu_f, req_x);
                            if next_c_s > c_s {
                                c_s = next_c_s;
                                bumped = true;
                            }
                        }
                    }
                    if !bumped {
                        break;
                    }
                }

                let shift_needed = c_s.to_f64() - x_cur[v];
                if shift_needed > cum_shift_x {
                    cum_shift_x = shift_needed;
                }
                x_out_s[v] = c_s;
                curr_group_max_x = Some(curr_group_max_x.map_or(c_s, |m| m.max(c_s)));

                let x_right_padded = (c_s.to_f64() + self.eff_hw_req[v])
                    .max((c_s + hw_v_s).to_f64())
                    + 4.0 * (c_s.to_f64().abs() + self.eff_hw_req[v]) * s_eps
                    + 1e-12;
                active_x.push(Reverse(HeapEntry {
                    key: x_right_padded,
                    node: v,
                }));
            }
            prev_group_max_x = curr_group_max_x;
        }
        layout.x_mut().copy_from_slice(&x_out_s);
    }
}

impl<S: Scalar> IterativeSolver<S> for PfsPrimeOverlapRemoval<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "PfsPrimeOverlapRemoval::step",
            ));
        }
        layout.validate_finite()?;

        if self.status.is_finished() {
            return Ok(self.status);
        }

        let n = self.num_nodes;
        if n <= 1 {
            self.last_overlap_count = 0;
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        let initial_f64_ov = find_f64_exact_overlaps(layout, &self.eff_hw_req, &self.eff_hh_req)?;
        let initial_s_ov =
            RectangleOverlapSweep::find_layout_overlaps(layout, &self.eff_hw_s, &self.eff_hh_s)?;
        if initial_f64_ov.is_empty() && initial_s_ov.is_empty() {
            self.last_overlap_count = 0;
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        {
            let xs = layout.x();
            let ys = layout.y();
            for i in 0..n {
                let xi = normalize_neg_zero(xs[i].to_f64());
                let yi = normalize_neg_zero(ys[i].to_f64());
                self.orig_x[i] = xi;
                self.orig_y[i] = yi;
                self.work_x[i] = xi;
                self.work_y[i] = yi;
            }
        }
        self.jitter_coincident_points();
        self.up_gamma_x.fill(0.0);
        self.up_gamma_y.fill(0.0);

        let max_abs_x = self.orig_x.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let max_abs_y = self.orig_y.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let sum_hw: f64 = self.eff_hw_req.iter().sum();
        let sum_hh: f64 = self.eff_hh_req.iter().sum();
        let s_eps = S::EPSILON.to_f64();
        let sep_eps = 1.0 + 1e-6_f64.max(16.0 * s_eps);
        let slack_x = 8.0 * (max_abs_x + sum_hw) * s_eps;
        let slack_y = 8.0 * (max_abs_y + sum_hh) * s_eps;

        self.horizontal_scan(sep_eps, slack_x, slack_y);
        // Per agorajs/agora-pfsp, verticalScan evaluates force() on original X coordinates
        // before committing up_gamma_x / up_gamma_y back to node centers.
        self.vertical_scan(sep_eps, slack_x, slack_y);

        {
            let (xs_mut, ys_mut) = layout.xy_mut();
            for i in 0..n {
                let nx = S::from_f64(self.work_x[i] + self.up_gamma_x[i]);
                let ny = S::from_f64(self.work_y[i] + self.up_gamma_y[i]);
                if nx.is_finite() {
                    xs_mut[i] = nx;
                }
                if ny.is_finite() {
                    ys_mut[i] = ny;
                }
            }
        }

        let mut f64_overlaps = find_f64_exact_overlaps(layout, &self.eff_hw_req, &self.eff_hh_req)?;
        let mut s_overlaps =
            RectangleOverlapSweep::find_layout_overlaps(layout, &self.eff_hw_s, &self.eff_hh_s)?;
        if !f64_overlaps.is_empty()
            || !s_overlaps.is_empty()
            || has_orthogonal_order_inversion(&self.orig_x, &self.orig_y, layout)
        {
            self.resolve_residual_overlaps_order_preserving(layout);
            f64_overlaps = find_f64_exact_overlaps(layout, &self.eff_hw_req, &self.eff_hh_req)?;
            s_overlaps = RectangleOverlapSweep::find_layout_overlaps(
                layout,
                &self.eff_hw_s,
                &self.eff_hh_s,
            )?;
        }
        let ov_count = count_union_pairs(&f64_overlaps, &s_overlaps);

        self.iter = 1;
        self.last_overlap_count = ov_count;
        self.status = if ov_count == 0 {
            SolverStatus::Converged
        } else {
            SolverStatus::Completed
        };
        Ok(self.status)
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        self.status
    }

    #[inline]
    fn iteration(&self) -> usize {
        self.iter
    }

    #[inline]
    fn total_iterations(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute_f64_exact_overlaps<S: Scalar>(
        layout: &Layout2d<S>,
        hw: &[S],
        hh: &[S],
        padding: f64,
    ) -> Vec<(usize, usize)> {
        let n = layout.len();
        let pad_half = 0.5 * padding;
        let xs = layout.x();
        let ys = layout.y();
        let mut out = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let req_x = (hw[i].to_f64() + pad_half) + (hw[j].to_f64() + pad_half);
                let req_y = (hh[i].to_f64() + pad_half) + (hh[j].to_f64() + pad_half);
                if req_x > 0.0
                    && req_y > 0.0
                    && (xs[i].to_f64() - xs[j].to_f64()).abs() < req_x
                    && (ys[i].to_f64() - ys[j].to_f64()).abs() < req_y
                {
                    out.push((i, j));
                }
            }
        }
        out
    }

    fn assert_no_overlaps_all<S: Scalar>(layout: &Layout2d<S>, hw: &[S], hh: &[S], padding: f64) {
        let pad_half = 0.5 * padding;
        let eff_hw_s: Vec<S> = hw
            .iter()
            .map(|&v| S::from_f64(v.to_f64() + pad_half))
            .collect();
        let eff_hh_s: Vec<S> = hh
            .iter()
            .map(|&v| S::from_f64(v.to_f64() + pad_half))
            .collect();
        let overlaps =
            RectangleOverlapSweep::find_layout_overlaps(layout, &eff_hw_s, &eff_hh_s).unwrap();
        assert!(
            overlaps.is_empty(),
            "expected 0 RectangleOverlapSweep overlaps, found {}: {:?}",
            overlaps.len(),
            &overlaps[..overlaps.len().min(5)]
        );
        let brute_f64 = brute_f64_exact_overlaps(layout, hw, hh, padding);
        assert!(
            brute_f64.is_empty(),
            "expected 0 f64-exact brute overlaps, found {}: {:?}",
            brute_f64.len(),
            &brute_f64[..brute_f64.len().min(5)]
        );
        // Also verify f64-exact against S-casted half-extents.
        let brute_s_f64 = brute_f64_exact_overlaps(layout, &eff_hw_s, &eff_hh_s, 0.0);
        assert!(
            brute_s_f64.is_empty(),
            "expected 0 S-extent f64-exact overlaps, found {}: {:?}",
            brute_s_f64.len(),
            &brute_s_f64[..brute_s_f64.len().min(5)]
        );
    }

    fn assert_no_overlaps<S: Scalar>(layout: &Layout2d<S>, hw: &[S], hh: &[S]) {
        assert_no_overlaps_all(layout, hw, hh, 0.0);
    }

    fn count_orthogonal_inversions(
        orig_x: &[f64],
        orig_y: &[f64],
        new_x: &[f64],
        new_y: &[f64],
    ) -> usize {
        let n = orig_x.len();
        let mut inversions = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                if (orig_x[i] < orig_x[j] && new_x[i] > new_x[j])
                    || (orig_x[i] > orig_x[j] && new_x[i] < new_x[j])
                {
                    inversions += 1;
                }
                if (orig_y[i] < orig_y[j] && new_y[i] > new_y[j])
                    || (orig_y[i] > orig_y[j] && new_y[i] < new_y[j])
                {
                    inversions += 1;
                }
            }
        }
        inversions
    }

    fn count_orthogonal_inversions_scalar<S: Scalar>(
        orig_x: &[S],
        orig_y: &[S],
        new_x: &[S],
        new_y: &[S],
    ) -> usize {
        let ox: Vec<f64> = orig_x.iter().map(|v| v.to_f64()).collect();
        let oy: Vec<f64> = orig_y.iter().map(|v| v.to_f64()).collect();
        let nx: Vec<f64> = new_x.iter().map(|v| v.to_f64()).collect();
        let ny: Vec<f64> = new_y.iter().map(|v| v.to_f64()).collect();
        count_orthogonal_inversions(&ox, &oy, &nx, &ny)
    }

    #[test]
    fn test_pfs_prime_validation_and_degenerate_inputs() {
        assert!(matches!(
            PfsPrimeConfig {
                padding: -0.1,
                ..Default::default()
            }
            .validate(),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            PfsPrimeConfig {
                padding: f64::NAN,
                ..Default::default()
            }
            .validate(),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            PfsPrimeOverlapRemoval::new(vec![1.0, 2.0], vec![1.0], PfsPrimeConfig::default()),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            PfsPrimeOverlapRemoval::new(vec![-1.0], vec![1.0], PfsPrimeConfig::default()),
            Err(CoreError::InvalidInput(_))
        ));

        // N = 0
        let mut s0 =
            PfsPrimeOverlapRemoval::<f64>::from_radii(vec![], PfsPrimeConfig::default()).unwrap();
        let mut l0 = Layout2d::zeros(0);
        assert_eq!(s0.step(&mut l0).unwrap(), SolverStatus::Converged);
        assert_eq!(s0.last_overlap_count(), 0);

        // N = 1
        let mut s1 =
            PfsPrimeOverlapRemoval::<f64>::from_radii(vec![2.0], PfsPrimeConfig::default())
                .unwrap();
        let mut l1 = Layout2d::from_xy(vec![3.0], vec![4.0]).unwrap();
        assert_eq!(s1.step(&mut l1).unwrap(), SolverStatus::Converged);
        assert_eq!(s1.last_overlap_count(), 0);
        assert_eq!(l1.x(), &[3.0]);
        assert_eq!(l1.y(), &[4.0]);
    }

    #[test]
    fn test_pfs_prime_zero_overlap_and_oo_nni_zero_f64_f32() {
        // Non-coincident overlapping boxes.
        let orig_x = vec![0.0_f64, 0.4, 0.9, 1.3, 0.2, 0.7, 1.5, 2.0];
        let orig_y = vec![0.0_f64, 0.2, 0.5, 0.1, 0.8, 0.6, 0.9, 0.3];
        let hw = vec![0.8_f64; 8];
        let hh = vec![0.6_f64; 8];

        let mut solver =
            PfsPrimeOverlapRemoval::<f64>::new(hw.clone(), hh.clone(), PfsPrimeConfig::default())
                .unwrap();
        let mut layout = Layout2d::from_xy(orig_x.clone(), orig_y.clone()).unwrap();
        let status = solver.run(&mut layout).unwrap();
        assert_eq!(status, SolverStatus::Converged);
        assert_eq!(solver.last_overlap_count(), 0);
        assert_no_overlaps(&layout, &hw, &hh);
        assert_eq!(
            count_orthogonal_inversions(&orig_x, &orig_y, layout.x(), layout.y()),
            0
        );

        // With positive padding.
        let pad = 0.5_f64;
        let cfg_pad = PfsPrimeConfig {
            padding: pad,
            seed: 7,
        };
        let mut solver_pad =
            PfsPrimeOverlapRemoval::<f64>::new(hw.clone(), hh.clone(), cfg_pad).unwrap();
        let mut layout_pad = Layout2d::from_xy(orig_x.clone(), orig_y.clone()).unwrap();
        assert_eq!(
            solver_pad.run(&mut layout_pad).unwrap(),
            SolverStatus::Converged
        );
        assert_eq!(solver_pad.last_overlap_count(), 0);
        assert_no_overlaps_all(&layout_pad, &hw, &hh, pad);
        assert_eq!(
            count_orthogonal_inversions(&orig_x, &orig_y, layout_pad.x(), layout_pad.y()),
            0
        );

        // f32 precision at large coordinate offset.
        let hw32 = vec![1.0_f32; 6];
        let hh32 = vec![0.75_f32; 6];
        let mut s32_a = PfsPrimeOverlapRemoval::<f32>::new(
            hw32.clone(),
            hh32.clone(),
            PfsPrimeConfig::default(),
        )
        .unwrap();
        let mut s32_b = PfsPrimeOverlapRemoval::<f32>::new(
            hw32.clone(),
            hh32.clone(),
            PfsPrimeConfig::default(),
        )
        .unwrap();
        let orig_x32 = vec![100.0_f32, 100.2, 100.5, 100.8, 100.1, 100.6];
        let orig_y32 = vec![-50.0_f32, -50.1, -49.8, -50.2, -49.9, -50.0];
        let mut l32_a = Layout2d::from_xy(orig_x32.clone(), orig_y32.clone()).unwrap();
        let mut l32_b = l32_a.clone();
        assert_eq!(s32_a.step(&mut l32_a).unwrap(), SolverStatus::Converged);
        assert_eq!(s32_b.run(&mut l32_b).unwrap(), SolverStatus::Converged);
        assert_no_overlaps(&l32_a, &hw32, &hh32);
        assert_eq!(
            count_orthogonal_inversions_scalar(&orig_x32, &orig_y32, l32_a.x(), l32_a.y()),
            0
        );
        assert_eq!(
            l32_a.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            l32_b.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            l32_a.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            l32_b.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_pfs_prime_collinear_and_coincident() {
        // Collinear nodes along X and Y.
        let radii = vec![1.0_f64; 5];
        let mut s_col =
            PfsPrimeOverlapRemoval::<f64>::from_radii(radii.clone(), PfsPrimeConfig::default())
                .unwrap();
        let orig_x_col = vec![0.0, 0.2, 0.4, 0.6, 0.8];
        let orig_y_col = vec![0.0; 5];
        let mut l_col = Layout2d::from_xy(orig_x_col.clone(), orig_y_col.clone()).unwrap();
        assert_eq!(s_col.run(&mut l_col).unwrap(), SolverStatus::Converged);
        assert_eq!(s_col.last_overlap_count(), 0);
        assert_no_overlaps(&l_col, &radii, &radii);
        assert_eq!(
            count_orthogonal_inversions(&orig_x_col, &orig_y_col, l_col.x(), l_col.y()),
            0
        );

        // All coincident nodes at (0, 0).
        let mut s_coin =
            PfsPrimeOverlapRemoval::<f64>::from_radii(radii.clone(), PfsPrimeConfig::default())
                .unwrap();
        let mut l_coin = Layout2d::zeros(5);
        assert_eq!(s_coin.run(&mut l_coin).unwrap(), SolverStatus::Converged);
        assert_eq!(s_coin.last_overlap_count(), 0);
        assert_no_overlaps(&l_coin, &radii, &radii);
    }

    #[test]
    fn test_pfs_prime_f32_large_offsets_zero_overlap_and_oo_nni() {
        let offsets = [0.0_f32, 100.0, 1000.0, 10000.0, -50000.0];
        let paddings = [0.0_f64, 0.25, 0.5, 1.0];

        for (offset_idx, &offset) in offsets.iter().enumerate() {
            let mut rng = DeterministicRng::from_seed_u64(9001 + offset_idx as u64);
            for trial in 0..30 {
                let n = 15 + rng.gen_range_usize(15);
                let use_equal_radii = trial % 3 == 0;
                let padding = paddings[trial % paddings.len()];
                let mut hw = Vec::with_capacity(n);
                let mut hh = Vec::with_capacity(n);
                let mut xs = Vec::with_capacity(n);
                let mut ys = Vec::with_capacity(n);

                for i in 0..n {
                    let r1 = (0.3 + 1.5 * rng.next_f64()) as f32;
                    let r2 = if use_equal_radii {
                        r1
                    } else {
                        (0.2 + 1.8 * rng.next_f64()) as f32
                    };
                    hw.push(r1);
                    hh.push(r2);

                    if trial % 7 == 0 && i < 3 {
                        // Include exact-coincident points alongside spread points.
                        xs.push(offset);
                        ys.push(offset);
                    } else {
                        xs.push(offset + (rng.next_f64() * 6.0 - 3.0) as f32);
                        ys.push(offset + (rng.next_f64() * 6.0 - 3.0) as f32);
                    }
                }

                let cfg = PfsPrimeConfig {
                    padding,
                    seed: 42 + trial as u64,
                };
                let mut solver_step = if use_equal_radii {
                    PfsPrimeOverlapRemoval::<f32>::from_radii(hw.clone(), cfg).unwrap()
                } else {
                    PfsPrimeOverlapRemoval::<f32>::new(hw.clone(), hh.clone(), cfg).unwrap()
                };
                let mut solver_run = if use_equal_radii {
                    PfsPrimeOverlapRemoval::<f32>::from_radii(hw.clone(), cfg).unwrap()
                } else {
                    PfsPrimeOverlapRemoval::<f32>::new(hw.clone(), hh.clone(), cfg).unwrap()
                };

                let mut layout_step = Layout2d::from_xy(xs.clone(), ys.clone()).unwrap();
                let mut layout_run = layout_step.clone();

                let status_step = solver_step.step(&mut layout_step).unwrap();
                let status_run = solver_run.run(&mut layout_run).unwrap();
                assert_eq!(status_step, SolverStatus::Converged);
                assert_eq!(status_run, SolverStatus::Converged);
                assert_eq!(solver_step.last_overlap_count(), 0);
                assert_eq!(solver_run.last_overlap_count(), 0);

                assert_no_overlaps_all(&layout_step, &hw, &hh, padding);
                assert_eq!(
                    count_orthogonal_inversions_scalar(&xs, &ys, layout_step.x(), layout_step.y()),
                    0,
                    "oo_nni > 0 for f32 offset={offset} trial={trial}"
                );
                assert_eq!(
                    layout_step
                        .x()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    layout_run
                        .x()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
                assert_eq!(
                    layout_step
                        .y()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    layout_run
                        .y()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn test_pfs_prime_f64_large_offset_1e10_zero_overlap_and_oo_nni() {
        let offset = 1e10_f64;
        let paddings = [0.0_f64, 0.25, 0.75];
        let mut rng = DeterministicRng::from_seed_u64(77777);

        for trial in 0..30 {
            let n = 18 + rng.gen_range_usize(12);
            let use_equal_radii = trial % 2 == 0;
            let padding = paddings[trial % paddings.len()];
            let mut hw = Vec::with_capacity(n);
            let mut hh = Vec::with_capacity(n);
            let mut xs = Vec::with_capacity(n);
            let mut ys = Vec::with_capacity(n);

            for i in 0..n {
                let r1 = 0.4 + 1.6 * rng.next_f64();
                let r2 = if use_equal_radii {
                    r1
                } else {
                    0.2 + 2.0 * rng.next_f64()
                };
                hw.push(r1);
                hh.push(r2);
                if trial % 6 == 0 && i < 2 {
                    xs.push(offset);
                    ys.push(offset);
                } else {
                    xs.push(offset + rng.next_f64() * 8.0 - 4.0);
                    ys.push(offset + rng.next_f64() * 8.0 - 4.0);
                }
            }

            let cfg = PfsPrimeConfig {
                padding,
                seed: 100 + trial as u64,
            };
            let mut solver_step =
                PfsPrimeOverlapRemoval::<f64>::new(hw.clone(), hh.clone(), cfg).unwrap();
            let mut solver_run =
                PfsPrimeOverlapRemoval::<f64>::new(hw.clone(), hh.clone(), cfg).unwrap();

            let mut layout_step = Layout2d::from_xy(xs.clone(), ys.clone()).unwrap();
            let mut layout_run = layout_step.clone();

            assert_eq!(
                solver_step.step(&mut layout_step).unwrap(),
                SolverStatus::Converged
            );
            assert_eq!(
                solver_run.run(&mut layout_run).unwrap(),
                SolverStatus::Converged
            );
            assert_eq!(solver_step.last_overlap_count(), 0);
            assert_no_overlaps_all(&layout_step, &hw, &hh, padding);
            assert_eq!(
                count_orthogonal_inversions(&xs, &ys, layout_step.x(), layout_step.y()),
                0,
                "oo_nni > 0 for f64 offset=1e10 trial={trial}"
            );
            assert_eq!(
                layout_step
                    .x()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                layout_run
                    .x()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                layout_step
                    .y()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                layout_run
                    .y()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn test_pfs_prime_near_coincident_distinct_points_oo_nni_zero() {
        let xs = vec![0.0_f64, 5e-10, 1e-9, 1.5e-9];
        let ys = vec![0.0_f64, 5e-10, 1e-9, 1.5e-9];
        let hw = vec![0.8_f64, 0.6, 1.0, 0.5];
        let hh = vec![0.5_f64, 0.9, 0.4, 0.7];

        for &padding in &[0.0_f64, 0.25] {
            let cfg = PfsPrimeConfig { padding, seed: 42 };
            let mut solver =
                PfsPrimeOverlapRemoval::<f64>::new(hw.clone(), hh.clone(), cfg).unwrap();
            let mut layout = Layout2d::from_xy(xs.clone(), ys.clone()).unwrap();
            assert_eq!(solver.run(&mut layout).unwrap(), SolverStatus::Converged);
            assert_eq!(solver.last_overlap_count(), 0);
            assert_no_overlaps_all(&layout, &hw, &hh, padding);
            assert_eq!(
                count_orthogonal_inversions(&xs, &ys, layout.x(), layout.y()),
                0
            );
        }

        // Mixed exact-coincident pair alongside 5e-10-spaced distinct points.
        let xs_mixed = vec![0.0_f64, 0.0, 5e-10, 1e-9, 1.5e-9];
        let ys_mixed = vec![0.0_f64, 0.0, 5e-10, 1e-9, 1.5e-9];
        let radii_mixed = vec![0.75_f64; 5];
        let mut solver_mixed = PfsPrimeOverlapRemoval::<f64>::from_radii(
            radii_mixed.clone(),
            PfsPrimeConfig::default(),
        )
        .unwrap();
        let mut layout_mixed = Layout2d::from_xy(xs_mixed.clone(), ys_mixed.clone()).unwrap();
        assert_eq!(
            solver_mixed.run(&mut layout_mixed).unwrap(),
            SolverStatus::Converged
        );
        assert_eq!(solver_mixed.last_overlap_count(), 0);
        assert_no_overlaps(&layout_mixed, &radii_mixed, &radii_mixed);
        assert_eq!(
            count_orthogonal_inversions(&xs_mixed, &ys_mixed, layout_mixed.x(), layout_mixed.y()),
            0
        );
    }

    #[test]
    fn test_pfs_prime_step_run_bit_exact_f64_and_f32() {
        let xs64 = vec![0.0_f64, 0.0, 0.2, 0.5, 1.1];
        let ys64 = vec![0.0_f64, 0.0, -0.1, 0.3, 0.0];
        let hw64 = vec![0.9_f64, 0.7, 0.8, 0.6, 1.0];
        let hh64 = vec![0.5_f64, 0.8, 0.6, 0.7, 0.4];
        let cfg = PfsPrimeConfig {
            padding: 0.25,
            seed: 1234,
        };
        let mut s64_step =
            PfsPrimeOverlapRemoval::<f64>::new(hw64.clone(), hh64.clone(), cfg).unwrap();
        let mut s64_run =
            PfsPrimeOverlapRemoval::<f64>::new(hw64.clone(), hh64.clone(), cfg).unwrap();
        let mut l64_step = Layout2d::from_xy(xs64.clone(), ys64.clone()).unwrap();
        let mut l64_run = l64_step.clone();

        assert_eq!(
            s64_step.step(&mut l64_step).unwrap(),
            SolverStatus::Converged
        );
        assert_eq!(s64_run.run(&mut l64_run).unwrap(), SolverStatus::Converged);
        assert_eq!(
            l64_step.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            l64_run.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            l64_step.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            l64_run.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );

        let xs32 = vec![-1000.0_f32, -1000.0, -999.8, -999.5, -1000.3];
        let ys32 = vec![500.0_f32, 500.0, 500.2, 499.9, 500.1];
        let hw32 = vec![0.9_f32, 0.7, 0.8, 0.6, 1.0];
        let hh32 = vec![0.5_f32, 0.8, 0.6, 0.7, 0.4];
        let mut s32_step =
            PfsPrimeOverlapRemoval::<f32>::new(hw32.clone(), hh32.clone(), cfg).unwrap();
        let mut s32_run =
            PfsPrimeOverlapRemoval::<f32>::new(hw32.clone(), hh32.clone(), cfg).unwrap();
        let mut l32_step = Layout2d::from_xy(xs32.clone(), ys32.clone()).unwrap();
        let mut l32_run = l32_step.clone();

        assert_eq!(
            s32_step.step(&mut l32_step).unwrap(),
            SolverStatus::Converged
        );
        assert_eq!(s32_run.run(&mut l32_run).unwrap(), SolverStatus::Converged);
        assert_no_overlaps_all(&l32_step, &hw32, &hh32, cfg.padding);
        assert_eq!(
            count_orthogonal_inversions_scalar(&xs32, &ys32, l32_step.x(), l32_step.y()),
            0
        );
        assert_eq!(
            l32_step.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            l32_run.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            l32_step.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            l32_run.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_pfs_prime_adversarial_ties_dynamic_range_and_sub_ulp_f32() {
        // 1. Same-X tie with interleaved Y node on a distinct X column:
        //    u0=(0,0), u1=(0,0.5), w=(10,1.0). Must not invert u1 past w.
        let xs_tie = vec![0.0_f64, 0.0, 10.0];
        let ys_tie = vec![0.0_f64, 0.5, 1.0];
        let hw_tie = vec![1.0_f64, 1.0, 0.5];
        let hh_tie = vec![2.0_f64, 2.0, 0.5];
        let mut s_tie = PfsPrimeOverlapRemoval::<f64>::new(
            hw_tie.clone(),
            hh_tie.clone(),
            PfsPrimeConfig::default(),
        )
        .unwrap();
        let mut l_tie = Layout2d::from_xy(xs_tie.clone(), ys_tie.clone()).unwrap();
        assert_eq!(s_tie.run(&mut l_tie).unwrap(), SolverStatus::Converged);
        assert_eq!(s_tie.last_overlap_count(), 0);
        assert_no_overlaps(&l_tie, &hw_tie, &hh_tie);
        assert_eq!(
            count_orthogonal_inversions(&xs_tie, &ys_tie, l_tie.x(), l_tie.y()),
            0
        );

        // 2. Extreme dynamic range with coincident groups at both 0 and 1e10.
        let xs_dyn = vec![0.0_f64, 0.0, 1e-10, 1e10, 1e10];
        let ys_dyn = vec![0.0_f64, 0.0, 1e-10, 1e10, 1e10];
        let hw_dyn = vec![1.0_f64; 5];
        let hh_dyn = vec![1.0_f64; 5];
        let mut s_dyn = PfsPrimeOverlapRemoval::<f64>::new(
            hw_dyn.clone(),
            hh_dyn.clone(),
            PfsPrimeConfig::default(),
        )
        .unwrap();
        let mut l_dyn = Layout2d::from_xy(xs_dyn.clone(), ys_dyn.clone()).unwrap();
        assert_eq!(s_dyn.run(&mut l_dyn).unwrap(), SolverStatus::Converged);
        assert_eq!(s_dyn.last_overlap_count(), 0);
        assert_no_overlaps(&l_dyn, &hw_dyn, &hh_dyn);
        assert_eq!(
            count_orthogonal_inversions(&xs_dyn, &ys_dyn, l_dyn.x(), l_dyn.y()),
            0
        );

        // 3. Sub-ULP f32 overlap at 10000.0 (initial S-sweep misses it, f64-exact catches it).
        //    10000.6_f32 == 10000.599609375, so dx = 0.599609375 < 2 * 0.3_f32 = 0.6000000238,
        //    whereas in f32 both 10000.0_f32 + 0.3_f32 and 10000.6_f32 - 0.3_f32 round to
        //    10000.2998046875, hiding the overlap from RectangleOverlapSweep.
        let hw_sub = vec![0.3_f32, 0.3_f32];
        let hh_sub = vec![0.5_f32, 0.5_f32];
        let xs_sub = vec![10000.0_f32, 10000.6_f32];
        let ys_sub = vec![10000.0_f32, 10000.0_f32];
        let mut l_sub = Layout2d::from_xy(xs_sub.clone(), ys_sub.clone()).unwrap();
        assert!(
            RectangleOverlapSweep::find_layout_overlaps(&l_sub, &hw_sub, &hh_sub)
                .unwrap()
                .is_empty()
        );
        assert!(!brute_f64_exact_overlaps(&l_sub, &hw_sub, &hh_sub, 0.0).is_empty());
        let mut s_sub = PfsPrimeOverlapRemoval::<f32>::new(
            hw_sub.clone(),
            hh_sub.clone(),
            PfsPrimeConfig::default(),
        )
        .unwrap();
        assert_eq!(s_sub.run(&mut l_sub).unwrap(), SolverStatus::Converged);
        assert_eq!(s_sub.last_overlap_count(), 0);
        assert_no_overlaps(&l_sub, &hw_sub, &hh_sub);
        assert_eq!(
            count_orthogonal_inversions_scalar(&xs_sub, &ys_sub, l_sub.x(), l_sub.y()),
            0
        );

        // 4. Zero-extent boxes (hw=0, hh=0) with padding=0 remain untouched.
        let zero_ext = vec![0.0_f64; 3];
        let mut s_zero = PfsPrimeOverlapRemoval::<f64>::new(
            zero_ext.clone(),
            zero_ext.clone(),
            PfsPrimeConfig::default(),
        )
        .unwrap();
        let mut l_zero = Layout2d::zeros(3);
        assert_eq!(s_zero.run(&mut l_zero).unwrap(), SolverStatus::Converged);
        assert_eq!(s_zero.last_overlap_count(), 0);
        assert_eq!(l_zero.x(), &[0.0, 0.0, 0.0]);
        assert_eq!(l_zero.y(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_pfs_prime_disjoint_column_heap_order_and_cancellation_f64_f32() {
        let htops = [-5.0_f64, -3.0, 20.0, -2.0];
        let f64_cases = [(1e10_f64, 50usize), (1e8_f64, 20usize), (1e6_f64, 20usize)];

        for &(scale, trials) in &f64_cases {
            for k in 0..trials {
                let target_y = 0.05 + (k as f64) * 0.037;
                let mut xs = Vec::with_capacity(20);
                let mut ys = Vec::with_capacity(20);
                let mut hw = Vec::with_capacity(20);
                let mut hh = Vec::with_capacity(20);

                // Tiny-gap pair keeping min_pos_gap = 1e-12 so the cluster at scale is not jittered.
                xs.extend_from_slice(&[0.0, 1e-12]);
                ys.extend_from_slice(&[0.0, 1e-12]);
                hw.extend_from_slice(&[0.1, 0.1]);
                hh.extend_from_slice(&[0.1, 0.1]);

                // 4 disjoint-column helper pairs bracketing target_y to scramble active_y heap order.
                for (idx, &htop) in htops.iter().enumerate() {
                    let hx = scale + 100.0 * ((idx + 1) as f64);
                    xs.push(hx);
                    ys.push(-scale - 10.0 - (idx as f64));
                    hw.push(1.0);
                    hh.push(scale + 10.0 + (idx as f64) + htop);

                    xs.push(hx);
                    ys.push(2.0 * scale + (idx as f64));
                    hw.push(1.0);
                    hh.push(1.0);
                }

                // Coincident cluster at (scale, -scale): u0, u1, and u2..u9 with sub-ULP hh.
                xs.push(scale);
                ys.push(-scale);
                hw.push(10.0);
                hh.push(scale + target_y);

                xs.push(scale);
                ys.push(-scale);
                hw.push(10.0);
                hh.push(1.0);

                for _ in 0..8 {
                    xs.push(scale);
                    ys.push(-scale);
                    hw.push(10.0);
                    hh.push(1e-7);
                }

                let cfg = PfsPrimeConfig {
                    padding: 0.0,
                    seed: 42 + k as u64,
                };
                let mut solver_step =
                    PfsPrimeOverlapRemoval::<f64>::new(hw.clone(), hh.clone(), cfg).unwrap();
                let mut solver_run =
                    PfsPrimeOverlapRemoval::<f64>::new(hw.clone(), hh.clone(), cfg).unwrap();
                let mut layout_step = Layout2d::from_xy(xs.clone(), ys.clone()).unwrap();
                let mut layout_run = layout_step.clone();

                assert_eq!(
                    solver_step.step(&mut layout_step).unwrap(),
                    SolverStatus::Converged,
                    "f64 Phase-1 step did not converge for scale={scale}, k={k}"
                );
                assert_eq!(
                    solver_run.run(&mut layout_run).unwrap(),
                    SolverStatus::Converged
                );
                assert_eq!(solver_step.last_overlap_count(), 0);
                assert_eq!(solver_run.last_overlap_count(), 0);
                assert_no_overlaps(&layout_step, &hw, &hh);
                assert_eq!(
                    count_orthogonal_inversions(&xs, &ys, layout_step.x(), layout_step.y()),
                    0
                );
                assert_eq!(
                    layout_step
                        .x()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    layout_run
                        .x()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
                assert_eq!(
                    layout_step
                        .y()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    layout_run
                        .y()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );

                // Transposed orientation exercising Phase 2 (X-sweep) heap-order + cancellation.
                let mut solver_t =
                    PfsPrimeOverlapRemoval::<f64>::new(hh.clone(), hw.clone(), cfg).unwrap();
                let mut layout_t = Layout2d::from_xy(ys.clone(), xs.clone()).unwrap();
                assert_eq!(
                    solver_t.step(&mut layout_t).unwrap(),
                    SolverStatus::Converged,
                    "f64 Phase-2 step did not converge for scale={scale}, k={k}"
                );
                assert_eq!(solver_t.last_overlap_count(), 0);
                assert_no_overlaps(&layout_t, &hh, &hw);
                assert_eq!(
                    count_orthogonal_inversions(&ys, &xs, layout_t.x(), layout_t.y()),
                    0
                );
            }
        }

        // f32 equivalent across scales where float32 ULP exceeds tiny cluster extents.
        let f32_cases = [(1e6_f32, 50usize), (1e5_f32, 20usize), (1e4_f32, 20usize)];
        for &(scale, trials) in &f32_cases {
            for k in 0..trials {
                let target_y = 0.05_f32 + (k as f32) * 0.037_f32;
                let mut xs = Vec::with_capacity(20);
                let mut ys = Vec::with_capacity(20);
                let mut hw = Vec::with_capacity(20);
                let mut hh = Vec::with_capacity(20);

                xs.extend_from_slice(&[0.0_f32, 1e-4_f32]);
                ys.extend_from_slice(&[0.0_f32, 1e-4_f32]);
                hw.extend_from_slice(&[0.1_f32, 0.1_f32]);
                hh.extend_from_slice(&[0.1_f32, 0.1_f32]);

                for (idx, &htop_f64) in htops.iter().enumerate() {
                    let htop = htop_f64 as f32;
                    let hx = scale + 100.0_f32 * ((idx + 1) as f32);
                    xs.push(hx);
                    ys.push(-scale - 10.0_f32 - (idx as f32));
                    hw.push(1.0_f32);
                    hh.push(scale + 10.0_f32 + (idx as f32) + htop);

                    xs.push(hx);
                    ys.push(2.0_f32 * scale + (idx as f32));
                    hw.push(1.0_f32);
                    hh.push(1.0_f32);
                }

                xs.push(scale);
                ys.push(-scale);
                hw.push(10.0_f32);
                hh.push(scale + target_y);

                xs.push(scale);
                ys.push(-scale);
                hw.push(10.0_f32);
                hh.push(1.0_f32);

                for _ in 0..8 {
                    xs.push(scale);
                    ys.push(-scale);
                    hw.push(10.0_f32);
                    hh.push(1e-4_f32);
                }

                let cfg = PfsPrimeConfig {
                    padding: 0.0,
                    seed: 42 + k as u64,
                };
                let mut solver_step =
                    PfsPrimeOverlapRemoval::<f32>::new(hw.clone(), hh.clone(), cfg).unwrap();
                let mut layout_step = Layout2d::from_xy(xs.clone(), ys.clone()).unwrap();
                assert_eq!(
                    solver_step.step(&mut layout_step).unwrap(),
                    SolverStatus::Converged,
                    "f32 Phase-1 step did not converge for scale={scale}, k={k}"
                );
                assert_eq!(solver_step.last_overlap_count(), 0);
                assert_no_overlaps(&layout_step, &hw, &hh);
                assert_eq!(
                    count_orthogonal_inversions_scalar(&xs, &ys, layout_step.x(), layout_step.y()),
                    0
                );

                // Transposed f32 case exercising Phase 2.
                let mut solver_t =
                    PfsPrimeOverlapRemoval::<f32>::new(hh.clone(), hw.clone(), cfg).unwrap();
                let mut layout_t = Layout2d::from_xy(ys.clone(), xs.clone()).unwrap();
                assert_eq!(
                    solver_t.step(&mut layout_t).unwrap(),
                    SolverStatus::Converged,
                    "f32 Phase-2 step did not converge for scale={scale}, k={k}"
                );
                assert_eq!(solver_t.last_overlap_count(), 0);
                assert_no_overlaps(&layout_t, &hh, &hw);
                assert_eq!(
                    count_orthogonal_inversions_scalar(&ys, &xs, layout_t.x(), layout_t.y()),
                    0
                );
            }
        }
    }

    #[test]
    fn test_pfs_prime_has_orthogonal_order_inversion_helper() {
        let orig_x = [0.0, 0.0, 1.0, 2.0];
        let orig_y = [-0.0, 0.0, 5.0, 10.0];
        let l_ok = Layout2d::from_xy(vec![1.0, 0.5, 1.0, 3.0], vec![0.0, 0.0, 5.0, 10.0]).unwrap();
        assert!(!has_orthogonal_order_inversion(&orig_x, &orig_y, &l_ok));

        let l_x_inv = Layout2d::from_xy(
            vec![1.0, 0.5, 1.0, 0.999999999999],
            vec![0.0, 0.0, 5.0, 10.0],
        )
        .unwrap();
        assert!(has_orthogonal_order_inversion(&orig_x, &orig_y, &l_x_inv));

        let l_y_inv =
            Layout2d::from_xy(vec![0.5, 1.0, 1.0, 3.0], vec![0.0, 0.0, 10.0, 9.999]).unwrap();
        assert!(has_orthogonal_order_inversion(&orig_x, &orig_y, &l_y_inv));

        // 1-ULP inversion at 50096.94 matching Challenger R3 3-node X repro.
        let orig_3x = [10000.0 - 1e-6, 10000.0, 10000.0 + 8e-10];
        let orig_3y = [0.0, 0.0, -1e8];
        let l_ulp_inv = Layout2d::from_xy(
            vec![9999.999998999996, 50096.94009590009, 50096.94009590008],
            vec![0.0, 0.0, -1e8],
        )
        .unwrap();
        assert!(has_orthogonal_order_inversion(
            &orig_3x, &orig_3y, &l_ulp_inv
        ));
    }

    #[test]
    fn test_pfs_prime_3node_pfsp_force_ulp_x_and_y() {
        let xs_base = vec![10000.0 - 1e-6, 10000.0, 10000.0 + 8e-10];
        let ys_base = vec![0.0, 0.0, -1e8];
        let hh_base = vec![1.0, 1.0, 1.0];

        for k in 0..50 {
            let hw_base = vec![40000.0 + (k as f64) * 13.7, 1.0, 476.0 + (k as f64) * 0.31];
            let cfg = PfsPrimeConfig::default();

            // X-axis scan-line orientation (50 trials).
            let mut sx_step =
                PfsPrimeOverlapRemoval::<f64>::new(hw_base.clone(), hh_base.clone(), cfg).unwrap();
            let mut sx_run =
                PfsPrimeOverlapRemoval::<f64>::new(hw_base.clone(), hh_base.clone(), cfg).unwrap();
            let mut lx_step = Layout2d::from_xy(xs_base.clone(), ys_base.clone()).unwrap();
            let mut lx_run = lx_step.clone();

            assert_eq!(sx_step.step(&mut lx_step).unwrap(), SolverStatus::Converged);
            assert_eq!(sx_run.run(&mut lx_run).unwrap(), SolverStatus::Converged);
            assert_eq!(sx_step.last_overlap_count(), 0);
            assert_eq!(sx_run.last_overlap_count(), 0);
            assert_no_overlaps_all(&lx_step, &hw_base, &hh_base, 0.0);
            assert_eq!(
                count_orthogonal_inversions(&xs_base, &ys_base, lx_step.x(), lx_step.y()),
                0,
                "3-node X oo_nni > 0 at k={k}"
            );
            assert_eq!(
                lx_step.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                lx_run.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(
                lx_step.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                lx_run.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );

            // Y-axis transposed orientation (50 trials).
            let mut sy_step =
                PfsPrimeOverlapRemoval::<f64>::new(hh_base.clone(), hw_base.clone(), cfg).unwrap();
            let mut sy_run =
                PfsPrimeOverlapRemoval::<f64>::new(hh_base.clone(), hw_base.clone(), cfg).unwrap();
            let mut ly_step = Layout2d::from_xy(ys_base.clone(), xs_base.clone()).unwrap();
            let mut ly_run = ly_step.clone();

            assert_eq!(sy_step.step(&mut ly_step).unwrap(), SolverStatus::Converged);
            assert_eq!(sy_run.run(&mut ly_run).unwrap(), SolverStatus::Converged);
            assert_eq!(sy_step.last_overlap_count(), 0);
            assert_eq!(sy_run.last_overlap_count(), 0);
            assert_no_overlaps_all(&ly_step, &hh_base, &hw_base, 0.0);
            assert_eq!(
                count_orthogonal_inversions(&ys_base, &xs_base, ly_step.x(), ly_step.y()),
                0,
                "3-node Y oo_nni > 0 at k={k}"
            );
            assert_eq!(
                ly_step.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                ly_run.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(
                ly_step.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                ly_run.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn test_pfs_prime_r3_fuzz_f64_9_and_multiscale_fuzz() {
        // Explicit multi-scale tight-cluster + distant-outlier layout reproducing r3_fuzz_f64_9.
        let r3_xs = vec![
            10000.0,
            10000.0 - 1e-6,
            10000.0 - 2.621e-9,
            10000.0 + 1.874e-10,
            10000.0 - 5e-9,
            10000.0 + 4e-9,
            10000.0 + 1.2,
            10000.0 - 0.8,
            -1e8,
            1e8,
            -1e10,
            1e10,
            10000.0,
            10000.0 + 3.0,
            10000.0 - 2.0,
            9999.999999997379,
            10000.000000000187,
            10000.0 + 5e-9,
            0.0,
        ];
        let r3_ys = vec![
            0.0, 0.0, -1e8, 1e8, 0.2, -0.3, 0.0, 0.1, -1e8, 1e8, 0.0, 0.0, 0.5, -0.5, 0.0, -1e8,
            1e8, 0.0, 0.0,
        ];
        let r3_hw = vec![
            40000.0, 1.0, 41.11e-6, 5.874e-9, 0.5, 2.0, 100.0, 50.0, 1.0, 1.0, 10.0, 10.0, 476.0,
            25.0, 12.0, 41.11e-6, 5.874e-9, 1e-7, 1.0,
        ];
        let r3_hh = vec![
            1.0, 1.0, 1.0, 1e-7, 0.5, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1e-8,
            1.0, 1.0,
        ];
        let cfg_r3 = PfsPrimeConfig {
            padding: 0.25,
            seed: 42,
        };
        let mut s_r3_step =
            PfsPrimeOverlapRemoval::<f64>::new(r3_hw.clone(), r3_hh.clone(), cfg_r3).unwrap();
        let mut s_r3_run =
            PfsPrimeOverlapRemoval::<f64>::new(r3_hw.clone(), r3_hh.clone(), cfg_r3).unwrap();
        let mut l_r3_step = Layout2d::from_xy(r3_xs.clone(), r3_ys.clone()).unwrap();
        let mut l_r3_run = l_r3_step.clone();
        assert_eq!(
            s_r3_step.step(&mut l_r3_step).unwrap(),
            SolverStatus::Converged
        );
        assert_eq!(
            s_r3_run.run(&mut l_r3_run).unwrap(),
            SolverStatus::Converged
        );
        assert_eq!(s_r3_step.last_overlap_count(), 0);
        assert_eq!(s_r3_run.last_overlap_count(), 0);
        assert_no_overlaps_all(&l_r3_step, &r3_hw, &r3_hh, 0.25);
        assert_eq!(
            count_orthogonal_inversions(&r3_xs, &r3_ys, l_r3_step.x(), l_r3_step.y()),
            0
        );
        assert_eq!(
            l_r3_step
                .x()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            l_r3_run.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            l_r3_step
                .y()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            l_r3_run.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );

        // 250 f64 multi-scale fuzz trials.
        let paddings = [0.0_f64, 0.25, 0.5, 1.0];
        let f64_coord_scales = [0.0_f64, 1e4, -1e4, 1e6, 1e8, -1e8, 1e10];
        let mut rng64 = DeterministicRng::from_seed_u64(0x5233_0064);
        for trial in 0..250 {
            let n = 6 + rng64.gen_range_usize(24);
            let padding = paddings[trial % paddings.len()];
            let base = f64_coord_scales[trial % f64_coord_scales.len()];
            let mut xs = Vec::with_capacity(n);
            let mut ys = Vec::with_capacity(n);
            let mut hw = Vec::with_capacity(n);
            let mut hh = Vec::with_capacity(n);

            for i in 0..n {
                let kind = rng64.gen_range_usize(5);
                let (x, y) = match kind {
                    0 => (
                        base + (rng64.next_f64() - 0.5) * 1e-8,
                        base + (rng64.next_f64() - 0.5) * 1e-8,
                    ),
                    1 => (
                        base + (rng64.next_f64() - 0.5) * 5.0,
                        base + (rng64.next_f64() - 0.5) * 5.0,
                    ),
                    2 => {
                        let s = f64_coord_scales[(i + trial) % f64_coord_scales.len()];
                        (s, base)
                    }
                    3 => (base, -base),
                    _ => (
                        1e4 + (rng64.next_f64() - 0.5) * 1e-9,
                        (rng64.next_f64() - 0.5) * 1e8,
                    ),
                };
                xs.push(x);
                ys.push(y);
                let exp_w = -9.0 + rng64.next_f64() * 12.0;
                let exp_h = -9.0 + rng64.next_f64() * 12.0;
                hw.push(10.0_f64.powf(exp_w).min(500.0));
                hh.push(10.0_f64.powf(exp_h).min(500.0));
            }

            let cfg = PfsPrimeConfig {
                padding,
                seed: 1000 + trial as u64,
            };
            let mut s_step =
                PfsPrimeOverlapRemoval::<f64>::new(hw.clone(), hh.clone(), cfg).unwrap();
            let mut s_run =
                PfsPrimeOverlapRemoval::<f64>::new(hw.clone(), hh.clone(), cfg).unwrap();
            let mut l_step = Layout2d::from_xy(xs.clone(), ys.clone()).unwrap();
            let mut l_run = l_step.clone();

            assert_eq!(
                s_step.step(&mut l_step).unwrap(),
                SolverStatus::Converged,
                "f64 multiscale fuzz step failed at trial={trial}"
            );
            assert_eq!(s_run.run(&mut l_run).unwrap(), SolverStatus::Converged);
            assert_eq!(s_step.last_overlap_count(), 0);
            assert_eq!(s_run.last_overlap_count(), 0);
            assert_no_overlaps_all(&l_step, &hw, &hh, padding);
            assert_eq!(
                count_orthogonal_inversions(&xs, &ys, l_step.x(), l_step.y()),
                0,
                "f64 multiscale fuzz oo_nni > 0 at trial={trial}"
            );
            assert_eq!(
                l_step.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                l_run.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(
                l_step.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                l_run.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }

        // 250 f32 multi-scale fuzz trials.
        let f32_coord_scales = [0.0_f32, 1e3, -1e3, 1e4, 1e5, -1e5, 1e6];
        let mut rng32 = DeterministicRng::from_seed_u64(0x5233_0032);
        for trial in 0..250 {
            let n = 6 + rng32.gen_range_usize(24);
            let padding = paddings[trial % paddings.len()];
            let base = f32_coord_scales[trial % f32_coord_scales.len()];
            let mut xs = Vec::with_capacity(n);
            let mut ys = Vec::with_capacity(n);
            let mut hw = Vec::with_capacity(n);
            let mut hh = Vec::with_capacity(n);

            for i in 0..n {
                let kind = rng32.gen_range_usize(4);
                let (x, y) = match kind {
                    0 => (
                        base + ((rng32.next_f64() - 0.5) * 1e-3) as f32,
                        base + ((rng32.next_f64() - 0.5) * 1e-3) as f32,
                    ),
                    1 => (
                        base + ((rng32.next_f64() - 0.5) * 5.0) as f32,
                        base + ((rng32.next_f64() - 0.5) * 5.0) as f32,
                    ),
                    2 => {
                        let s = f32_coord_scales[(i + trial) % f32_coord_scales.len()];
                        (s, base)
                    }
                    _ => (
                        1e4_f32 + ((rng32.next_f64() - 0.5) * 1e-2) as f32,
                        ((rng32.next_f64() - 0.5) * 1e5) as f32,
                    ),
                };
                xs.push(x);
                ys.push(y);
                let exp_w = -4.0 + rng32.next_f64() * 6.5;
                let exp_h = -4.0 + rng32.next_f64() * 6.5;
                hw.push((10.0_f64.powf(exp_w).min(300.0)) as f32);
                hh.push((10.0_f64.powf(exp_h).min(300.0)) as f32);
            }

            let cfg = PfsPrimeConfig {
                padding,
                seed: 2000 + trial as u64,
            };
            let mut s_step =
                PfsPrimeOverlapRemoval::<f32>::new(hw.clone(), hh.clone(), cfg).unwrap();
            let mut s_run =
                PfsPrimeOverlapRemoval::<f32>::new(hw.clone(), hh.clone(), cfg).unwrap();
            let mut l_step = Layout2d::from_xy(xs.clone(), ys.clone()).unwrap();
            let mut l_run = l_step.clone();

            assert_eq!(
                s_step.step(&mut l_step).unwrap(),
                SolverStatus::Converged,
                "f32 multiscale fuzz step failed at trial={trial}"
            );
            assert_eq!(s_run.run(&mut l_run).unwrap(), SolverStatus::Converged);
            assert_eq!(s_step.last_overlap_count(), 0);
            assert_eq!(s_run.last_overlap_count(), 0);
            assert_no_overlaps_all(&l_step, &hw, &hh, padding);
            assert_eq!(
                count_orthogonal_inversions_scalar(&xs, &ys, l_step.x(), l_step.y()),
                0,
                "f32 multiscale fuzz oo_nni > 0 at trial={trial}"
            );
            assert_eq!(
                l_step.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                l_run.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(
                l_step.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                l_run.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }
}
