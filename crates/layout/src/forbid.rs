//! Giovannangeli et al. (2022) FORBID node overlap removal solver.
//!
//! Implements FORBID ("FORBID: Fast Overlap Removal By stochastic gradIent Descent",
//! Giovannangeli, Lalanne, Auber, Giot & Bourqui, *Computer Graphics Forum* 2022) with
//! three selectable variants:
//! - [`ForbidVariant::Forbid`]: Cumulative layout scaling across binary-search passes.
//! - [`ForbidVariant::ForbidPrime`]: Resets each binary-search pass from uniformly scaled initial layout.
//! - [`ForbidVariant::SparseForbid`]: Uses `StaticKdTree` `sparse_k`-NN pairs plus active
//!   overlapping pairs $\mathcal{O}(X')$ for sub-quadratic scaling.

use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, DeterministicRng, IterativeSolver, Layout2d, Scalar, SolverStatus,
};
use egraph_spatial::{KdQueryScratch, RectangleOverlapSweep, StaticKdTree};

const MIN_EDGE_DIST: f64 = 1e-9;
const MIN_TARGET_DIST: f64 = 1e-6;
const SGD2_EPS: f64 = 0.01;

/// Algorithm variant for [`ForbidOverlapRemoval`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ForbidVariant {
    /// Original FORBID: scales current adjusted layout `X'` across binary-search passes.
    #[default]
    Forbid,
    /// `FORBID'`: resets each binary-search pass by scaling the initial layout `X`.
    ForbidPrime,
    /// Sparse FORBID: combines `StaticKdTree` `sparse_k`-NN pairs with all active overlaps $\mathcal{O}(X')$.
    SparseForbid,
}

/// Hyperparameters for [`ForbidOverlapRemoval`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForbidConfig {
    /// FORBID variant (`default: ForbidVariant::Forbid`).
    pub variant: ForbidVariant,
    /// Maximum number of binary-search scaling iterations (`default: 20`).
    pub max_scale_iters: usize,
    /// Maximum SGD2 iterations per scaling pass (`default: 30`).
    pub sgd_iterations: usize,
    /// Number of nearest neighbors per node in [`ForbidVariant::SparseForbid`] (`default: 30`).
    pub sparse_k: usize,
    /// Exponent multiplier $k$ in overlap pair weights $\delta_{ij}^{k \alpha}$ ($\alpha = -2$, `default: 1.0`).
    pub overlap_weight_k: f64,
    /// Binary-search stopping width `up_scale - low_scale <= scale_step` (`default: 0.1`).
    pub scale_step: f64,
    /// Extra minimum clearance added between rectangle borders (`default: 0.0`).
    pub padding: f64,
    /// Explicit deterministic seed for coincident perturbation and SGD shuffling (`default: 42`).
    pub seed: u64,
}

impl Default for ForbidConfig {
    fn default() -> Self {
        Self {
            variant: ForbidVariant::Forbid,
            max_scale_iters: 20,
            sgd_iterations: 30,
            sparse_k: 30,
            overlap_weight_k: 1.0,
            scale_step: 0.1,
            padding: 0.0,
            seed: 42,
        }
    }
}

impl ForbidConfig {
    /// Validates all FORBID configuration parameters.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.overlap_weight_k.is_finite() {
            return Err(CoreError::non_finite("ForbidConfig::overlap_weight_k"));
        }
        if !self.scale_step.is_finite() {
            return Err(CoreError::non_finite("ForbidConfig::scale_step"));
        }
        if self.scale_step <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "ForbidConfig::scale_step must be > 0.0, got {}",
                self.scale_step
            )));
        }
        if !self.padding.is_finite() {
            return Err(CoreError::non_finite("ForbidConfig::padding"));
        }
        if self.padding < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "ForbidConfig::padding must be >= 0.0, got {}",
                self.padding
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct SgdPairItem {
    u: u32,
    v: u32,
    d: f64,
    w: f64,
    is_overlap: bool,
}

/// Cooperative Giovannangeli et al. (2022) FORBID node overlap removal solver.
///
/// Each [`IterativeSolver::step`] executes one binary-search scaling + `SGD2_StressOPT`
/// pass and updates [`ForbidOverlapRemoval::last_overlap_count`]. Reaching `max_scale_iters`
/// with residual overlaps restores the best overlap-free scaled layout (with deterministic QPSC
/// fallback if needed) so converged outputs strictly have `0` overlapping rectangles.
#[derive(Debug, Clone)]
pub struct ForbidOverlapRemoval<S: Scalar = f64> {
    config: ForbidConfig,
    num_nodes: usize,
    eff_hw_f64: Vec<f64>,
    eff_hh_f64: Vec<f64>,
    eff_hw_s: Vec<S>,
    eff_hh_s: Vec<S>,
    rng: DeterministicRng,
    initialized_state: bool,
    orig_x: Vec<f64>,
    orig_y: Vec<f64>,
    orig_cx: f64,
    orig_cy: f64,
    work_layout: Layout2d<f64>,
    best_free_x: Vec<f64>,
    best_free_y: Vec<f64>,
    has_best_free: bool,
    base_pairs: Vec<(u32, u32)>,
    active_overlaps: Vec<(u32, u32)>,
    sgd_pairs: Vec<SgdPairItem>,
    sort_pts: Vec<(f64, f64, u32)>,
    low_scale: f64,
    up_scale: f64,
    cur_scale: f64,
    prev_scale: f64,
    iter: usize,
    last_overlap_count: usize,
    status: SolverStatus,
    _marker: PhantomData<S>,
}

impl<S: Scalar> ForbidOverlapRemoval<S> {
    /// Creates a [`ForbidOverlapRemoval`] solver for rectangular nodes with per-node half-widths
    /// and half-heights.
    pub fn new(
        half_widths: Vec<S>,
        half_heights: Vec<S>,
        config: ForbidConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        let n = half_widths.len();
        if half_heights.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                half_heights.len(),
                "ForbidOverlapRemoval::new::half_heights",
            ));
        }
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_input(
                "ForbidOverlapRemoval node count exceeds u32::MAX",
            ));
        }

        let pad_half = 0.5 * config.padding;
        let mut eff_hw_f64 = Vec::with_capacity(n);
        let mut eff_hh_f64 = Vec::with_capacity(n);
        let mut eff_hw_s = Vec::with_capacity(n);
        let mut eff_hh_s = Vec::with_capacity(n);

        for i in 0..n {
            let hw = half_widths[i];
            let hh = half_heights[i];
            if !hw.is_finite() || !hh.is_finite() {
                return Err(CoreError::non_finite("ForbidOverlapRemoval::new::extents"));
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
                    "ForbidOverlapRemoval::new::padded_extents",
                ));
            }
            eff_hw_f64.push(hw_f);
            eff_hh_f64.push(hh_f);
            eff_hw_s.push(hw_s);
            eff_hh_s.push(hh_s);
        }

        let status = if config.max_scale_iters == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };

        Ok(Self {
            config,
            num_nodes: n,
            eff_hw_f64,
            eff_hh_f64,
            eff_hw_s,
            eff_hh_s,
            rng: DeterministicRng::from_seed_u64(config.seed),
            initialized_state: false,
            orig_x: vec![0.0; n],
            orig_y: vec![0.0; n],
            orig_cx: 0.0,
            orig_cy: 0.0,
            work_layout: Layout2d::zeros(n),
            best_free_x: vec![0.0; n],
            best_free_y: vec![0.0; n],
            has_best_free: false,
            base_pairs: Vec::new(),
            active_overlaps: Vec::new(),
            sgd_pairs: Vec::new(),
            sort_pts: Vec::with_capacity(n),
            low_scale: 1.0,
            up_scale: 1.0,
            cur_scale: 1.0,
            prev_scale: 1.0,
            iter: 0,
            last_overlap_count: 0,
            status,
            _marker: PhantomData,
        })
    }

    /// Creates a [`ForbidOverlapRemoval`] solver for circular/square nodes with equal half-width
    /// and half-height given by `radii`.
    pub fn from_radii(radii: Vec<S>, config: ForbidConfig) -> CoreResult<Self> {
        let half_heights = radii.clone();
        Self::new(radii, half_heights, config)
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &ForbidConfig {
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

    fn jitter_coincident_orig(&mut self) {
        let n = self.num_nodes;
        if n < 2 {
            return;
        }
        self.sort_pts.clear();
        for i in 0..n {
            self.sort_pts
                .push((self.orig_x[i], self.orig_y[i], i as u32));
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
                if (nx - base_x).abs() <= MIN_EDGE_DIST && (ny - base_y).abs() <= MIN_EDGE_DIST {
                    j += 1;
                } else {
                    break;
                }
            }
            if j - i > 1 {
                let coord_ulp = base_x.abs().max(base_y.abs()) * S::EPSILON.to_f64();
                for k in (i + 1)..j {
                    let idx = self.sort_pts[k].2 as usize;
                    let extent = self.eff_hw_f64[idx].max(self.eff_hh_f64[idx]).max(1e-2);
                    let jitter_scale = (extent * 0.05).max(4.0 * coord_ulp).max(1e-5);
                    let angle = self.rng.next_f64() * std::f64::consts::TAU;
                    let radius = jitter_scale * (0.25 + 0.75 * self.rng.next_f64());
                    let (sin_a, cos_a) = angle.sin_cos();
                    self.orig_x[idx] += radius * cos_a;
                    self.orig_y[idx] += radius * sin_a;
                }
            }
            i = j;
        }
    }

    fn initialize_from_layout(&mut self, layout: &Layout2d<S>) -> CoreResult<()> {
        let n = self.num_nodes;
        let xs = layout.x();
        let ys = layout.y();
        for i in 0..n {
            self.orig_x[i] = xs[i].to_f64();
            self.orig_y[i] = ys[i].to_f64();
        }
        self.jitter_coincident_orig();

        self.orig_cx = self.orig_x.iter().sum::<f64>() / (n as f64);
        self.orig_cy = self.orig_y.iter().sum::<f64>() / (n as f64);

        {
            let (wx, wy) = self.work_layout.xy_mut();
            wx.copy_from_slice(&self.orig_x);
            wy.copy_from_slice(&self.orig_y);
        }

        // Compute `up_scale = getScalingRatio(X, S)` across all overlapping pairs in perturbed `orig`.
        let overlaps = RectangleOverlapSweep::find_layout_overlaps(
            &self.work_layout,
            &self.eff_hw_f64,
            &self.eff_hh_f64,
        )?;
        let sep_eps = 1.0 + 1e-5_f64.max(16.0 * S::EPSILON.to_f64());
        let mut up_scale = 1.0_f64;
        for (i, j) in overlaps {
            let dx = (self.orig_x[i] - self.orig_x[j]).abs();
            let dy = (self.orig_y[i] - self.orig_y[j]).abs();
            let req_x = self.eff_hw_f64[i] + self.eff_hw_f64[j];
            let req_y = self.eff_hh_f64[i] + self.eff_hh_f64[j];
            if req_x <= 0.0 || req_y <= 0.0 {
                continue;
            }
            let ulp_x = (self.orig_x[i].abs().max(self.orig_x[j].abs()) + req_x)
                * (8.0 * S::EPSILON.to_f64());
            let ulp_y = (self.orig_y[i].abs().max(self.orig_y[j].abs()) + req_y)
                * (8.0 * S::EPSILON.to_f64());
            let tx = if dx > 1e-15 {
                (req_x * sep_eps + ulp_x) / dx
            } else {
                f64::INFINITY
            };
            let ty = if dy > 1e-15 {
                (req_y * sep_eps + ulp_y) / dy
            } else {
                f64::INFINITY
            };
            let ratio = tx.min(ty);
            if ratio.is_finite() && ratio > up_scale {
                up_scale = ratio;
            }
        }

        // Verify `up_scale` produces strictly 0 overlaps in `S` precision and record `best_free`.
        for _ in 0..8 {
            for i in 0..n {
                self.best_free_x[i] = self.orig_cx + up_scale * (self.orig_x[i] - self.orig_cx);
                self.best_free_y[i] = self.orig_cy + up_scale * (self.orig_y[i] - self.orig_cy);
            }
            let check_layout = Layout2d::from_xy(
                self.best_free_x.iter().map(|&v| S::from_f64(v)).collect(),
                self.best_free_y.iter().map(|&v| S::from_f64(v)).collect(),
            )?;
            if !RectangleOverlapSweep::has_layout_overlaps(
                &check_layout,
                &self.eff_hw_s,
                &self.eff_hh_s,
            )? {
                self.has_best_free = true;
                break;
            }
            up_scale *= 1.01;
        }

        self.low_scale = 1.0;
        self.up_scale = up_scale;
        self.cur_scale = 1.0;
        self.prev_scale = 1.0;

        // Precompute static base pair list (`AllPairs` or `SparseForbid` k-NN).
        self.base_pairs.clear();
        match self.config.variant {
            ForbidVariant::Forbid | ForbidVariant::ForbidPrime => {
                self.base_pairs.reserve(n.saturating_mul(n - 1) / 2);
                for i in 0..n {
                    for j in (i + 1)..n {
                        self.base_pairs.push((i as u32, j as u32));
                    }
                }
            }
            ForbidVariant::SparseForbid => {
                let k = self.config.sparse_k.min(n - 1);
                if k > 0 {
                    let tree = StaticKdTree::<2, f64>::from_layout(&self.work_layout)?;
                    let mut scratch = KdQueryScratch::with_capacity(k);
                    let mut knn_out = Vec::with_capacity(k);
                    self.base_pairs.reserve(n * k);
                    for u in 0..n {
                        tree.k_nearest_excluding_self_with_scratch(
                            u,
                            k,
                            &mut scratch,
                            &mut knn_out,
                        )?;
                        let u32_u = u as u32;
                        for &(v, _) in &knn_out {
                            let v32 = v as u32;
                            self.base_pairs.push(if u32_u < v32 {
                                (u32_u, v32)
                            } else {
                                (v32, u32_u)
                            });
                        }
                    }
                    self.base_pairs.sort_unstable();
                    self.base_pairs.dedup();
                }
            }
        }

        self.initialized_state = true;
        Ok(())
    }

    fn apply_scale(&mut self) {
        let n = self.num_nodes;
        match self.config.variant {
            ForbidVariant::Forbid => {
                if self.iter == 0 {
                    let (wx, wy) = self.work_layout.xy_mut();
                    wx.copy_from_slice(&self.orig_x);
                    wy.copy_from_slice(&self.orig_y);
                    self.prev_scale = 1.0;
                } else {
                    let factor = if self.prev_scale > 0.0 {
                        self.cur_scale / self.prev_scale
                    } else {
                        self.cur_scale
                    };
                    let (wx, wy) = self.work_layout.xy_mut();
                    let cx = wx.iter().sum::<f64>() / (n as f64);
                    let cy = wy.iter().sum::<f64>() / (n as f64);
                    for i in 0..n {
                        wx[i] = cx + factor * (wx[i] - cx);
                        wy[i] = cy + factor * (wy[i] - cy);
                    }
                    self.prev_scale = self.cur_scale;
                }
            }
            ForbidVariant::ForbidPrime | ForbidVariant::SparseForbid => {
                let s = self.cur_scale;
                let (wx, wy) = self.work_layout.xy_mut();
                for i in 0..n {
                    wx[i] = self.orig_cx + s * (self.orig_x[i] - self.orig_cx);
                    wy[i] = self.orig_cy + s * (self.orig_y[i] - self.orig_cy);
                }
            }
        }
    }

    fn sgd2_stress_opt(&mut self) -> CoreResult<()> {
        let n = self.num_nodes;
        let t_max = self.config.sgd_iterations;
        if n <= 1 || t_max == 0 {
            return Ok(());
        }

        let alpha = -2.0_f64;
        let overlap_exp = alpha * self.config.overlap_weight_k;
        let sep_eps = 1.0 + 1e-6_f64.max(16.0 * S::EPSILON.to_f64());

        for t in 0..t_max {
            // Recompute active overlapping pairs O(X') at the start of each SGD2 iteration.
            let raw_overlaps = RectangleOverlapSweep::find_layout_overlaps(
                &self.work_layout,
                &self.eff_hw_f64,
                &self.eff_hh_f64,
            )?;
            self.active_overlaps.clear();
            self.active_overlaps
                .extend(raw_overlaps.into_iter().map(|(u, v)| (u as u32, v as u32)));

            self.sgd_pairs.clear();
            let mut w_min = f64::INFINITY;
            let mut w_max = 0.0_f64;

            if matches!(self.config.variant, ForbidVariant::SparseForbid) {
                // Merge sorted `base_pairs` and sorted `active_overlaps` cleanly.
                let mut bi = 0usize;
                let mut oi = 0usize;
                while bi < self.base_pairs.len() || oi < self.active_overlaps.len() {
                    let (u, v, is_ov) = if oi >= self.active_overlaps.len() {
                        let p = self.base_pairs[bi];
                        bi += 1;
                        (p.0, p.1, false)
                    } else if bi >= self.base_pairs.len() {
                        let p = self.active_overlaps[oi];
                        oi += 1;
                        (p.0, p.1, true)
                    } else {
                        let pb = self.base_pairs[bi];
                        let po = self.active_overlaps[oi];
                        match pb.cmp(&po) {
                            std::cmp::Ordering::Less => {
                                bi += 1;
                                (pb.0, pb.1, false)
                            }
                            std::cmp::Ordering::Greater => {
                                oi += 1;
                                (po.0, po.1, true)
                            }
                            std::cmp::Ordering::Equal => {
                                bi += 1;
                                oi += 1;
                                (po.0, po.1, true)
                            }
                        }
                    };
                    let ui = u as usize;
                    let vi = v as usize;
                    let (d, w) = if is_ov {
                        let req_x = self.eff_hw_f64[ui] + self.eff_hw_f64[vi];
                        let req_y = self.eff_hh_f64[ui] + self.eff_hh_f64[vi];
                        let d_ov = (req_x * sep_eps)
                            .hypot(req_y * sep_eps)
                            .max(MIN_TARGET_DIST);
                        let w_ov = d_ov.powf(overlap_exp);
                        (d_ov, w_ov)
                    } else {
                        let dx = self.orig_x[ui] - self.orig_x[vi];
                        let dy = self.orig_y[ui] - self.orig_y[vi];
                        let d_orig = dx.hypot(dy).max(MIN_EDGE_DIST);
                        let w_orig = 1.0 / (d_orig * d_orig);
                        (d_orig, w_orig)
                    };
                    if w.is_finite() && w > 0.0 {
                        w_min = w_min.min(w);
                        w_max = w_max.max(w);
                        self.sgd_pairs.push(SgdPairItem {
                            u,
                            v,
                            d,
                            w,
                            is_overlap: is_ov,
                        });
                    }
                }
            } else {
                // Full all-pairs (`Forbid` / `ForbidPrime`): linear merge with sorted `active_overlaps`.
                let mut oi = 0usize;
                for &(u, v) in &self.base_pairs {
                    while oi < self.active_overlaps.len() && self.active_overlaps[oi] < (u, v) {
                        oi += 1;
                    }
                    let is_ov =
                        oi < self.active_overlaps.len() && self.active_overlaps[oi] == (u, v);
                    if is_ov {
                        oi += 1;
                    }
                    let ui = u as usize;
                    let vi = v as usize;
                    let (d, w) = if is_ov {
                        let req_x = self.eff_hw_f64[ui] + self.eff_hw_f64[vi];
                        let req_y = self.eff_hh_f64[ui] + self.eff_hh_f64[vi];
                        let d_ov = (req_x * sep_eps)
                            .hypot(req_y * sep_eps)
                            .max(MIN_TARGET_DIST);
                        let w_ov = d_ov.powf(overlap_exp);
                        (d_ov, w_ov)
                    } else {
                        let dx = self.orig_x[ui] - self.orig_x[vi];
                        let dy = self.orig_y[ui] - self.orig_y[vi];
                        let d_orig = dx.hypot(dy).max(MIN_EDGE_DIST);
                        let w_orig = 1.0 / (d_orig * d_orig);
                        (d_orig, w_orig)
                    };
                    if w.is_finite() && w > 0.0 {
                        w_min = w_min.min(w);
                        w_max = w_max.max(w);
                        self.sgd_pairs.push(SgdPairItem {
                            u,
                            v,
                            d,
                            w,
                            is_overlap: is_ov,
                        });
                    }
                }
            }

            if self.sgd_pairs.is_empty() || !w_min.is_finite() || w_max <= 0.0 {
                break;
            }

            let eta_max = 1.0 / w_min;
            let eta_min = (SGD2_EPS / w_max).min(eta_max);
            let eta = if t_max <= 1 || eta_max <= eta_min {
                eta_max
            } else {
                let b = (eta_max / eta_min).ln() / ((t_max - 1) as f64);
                eta_max * (-b * (t as f64)).exp()
            };

            self.rng.shuffle(&mut self.sgd_pairs);

            let (wx, wy) = self.work_layout.xy_mut();
            let mut max_move = 0.0_f64;
            for item in &self.sgd_pairs {
                let mu = (eta * item.w).min(1.0);
                if !mu.is_finite() || mu <= 0.0 {
                    continue;
                }
                let i = item.u as usize;
                let j = item.v as usize;
                let dx = wx[i] - wx[j];
                let dy = wy[i] - wy[j];
                let norm = dx.hypot(dy);
                if norm <= 1e-12 || !norm.is_finite() {
                    if item.is_overlap {
                        let angle = ((item.u.wrapping_mul(2_654_435_761)
                            ^ item.v.wrapping_mul(2_246_822_519))
                            as f64)
                            * 1e-6;
                        let (sin_a, cos_a) = angle.sin_cos();
                        let step = 0.5 * item.d * mu;
                        let mx = step * cos_a;
                        let my = step * sin_a;
                        wx[i] -= mx;
                        wy[i] -= my;
                        wx[j] += mx;
                        wy[j] += my;
                        max_move = max_move.max(step);
                    }
                    continue;
                }
                let coeff = 0.5 * (norm - item.d) / norm * mu;
                let mx = dx * coeff;
                let my = dy * coeff;
                if mx.is_finite() && my.is_finite() {
                    wx[i] -= mx;
                    wy[i] -= my;
                    wx[j] += mx;
                    wy[j] += my;
                    max_move = max_move.max(mx.abs().max(my.abs()));
                }
            }

            if max_move <= 1e-12 {
                break;
            }
        }

        Ok(())
    }

    fn copy_work_to_layout(&self, layout: &mut Layout2d<S>) {
        let wx = self.work_layout.x();
        let wy = self.work_layout.y();
        let (xs_mut, ys_mut) = layout.xy_mut();
        for i in 0..self.num_nodes {
            let nx = S::from_f64(wx[i]);
            let ny = S::from_f64(wy[i]);
            if nx.is_finite() {
                xs_mut[i] = nx;
            }
            if ny.is_finite() {
                ys_mut[i] = ny;
            }
        }
    }
}

impl<S: Scalar> IterativeSolver<S> for ForbidOverlapRemoval<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "ForbidOverlapRemoval::step",
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

        if !self.initialized_state {
            let initial_overlaps = RectangleOverlapSweep::find_layout_overlaps(
                layout,
                &self.eff_hw_s,
                &self.eff_hh_s,
            )?;
            if initial_overlaps.is_empty() {
                self.last_overlap_count = 0;
                self.status = SolverStatus::Converged;
                return Ok(self.status);
            }
            self.initialize_from_layout(layout)?;
        }

        self.apply_scale();
        self.sgd2_stress_opt()?;
        self.copy_work_to_layout(layout);

        let mut ov_count =
            RectangleOverlapSweep::find_layout_overlaps(layout, &self.eff_hw_s, &self.eff_hh_s)?
                .len();

        let there_is_overlap = ov_count > 0;
        if !there_is_overlap {
            let xs = layout.x();
            let ys = layout.y();
            for i in 0..n {
                self.best_free_x[i] = xs[i].to_f64();
                self.best_free_y[i] = ys[i].to_f64();
            }
            self.has_best_free = true;
            self.up_scale = self.cur_scale;
        } else {
            self.low_scale = self.cur_scale;
        }
        self.cur_scale = 0.5 * (self.low_scale + self.up_scale);
        self.iter += 1;

        if !there_is_overlap && (self.up_scale - self.low_scale <= self.config.scale_step) {
            self.last_overlap_count = 0;
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        if self.iter >= self.config.max_scale_iters {
            if there_is_overlap {
                if self.has_best_free {
                    let (xs_mut, ys_mut) = layout.xy_mut();
                    for i in 0..n {
                        xs_mut[i] = S::from_f64(self.best_free_x[i]);
                        ys_mut[i] = S::from_f64(self.best_free_y[i]);
                    }
                    ov_count = RectangleOverlapSweep::find_layout_overlaps(
                        layout,
                        &self.eff_hw_s,
                        &self.eff_hh_s,
                    )?
                    .len();
                }
                if ov_count > 0 {
                    crate::constraints::project_rectangle_no_overlap_2d(
                        layout,
                        &self.eff_hw_s,
                        &self.eff_hh_s,
                    )?;
                    ov_count = RectangleOverlapSweep::find_layout_overlaps(
                        layout,
                        &self.eff_hw_s,
                        &self.eff_hh_s,
                    )?
                    .len();
                }
            }
            self.last_overlap_count = ov_count;
            self.status = if ov_count == 0 {
                SolverStatus::Converged
            } else {
                SolverStatus::Completed
            };
        } else {
            self.last_overlap_count = ov_count;
            self.status = SolverStatus::Running;
        }

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
        self.config.max_scale_iters
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egraph_core::DeterministicRng;

    fn assert_no_overlaps<S: Scalar>(layout: &Layout2d<S>, hw: &[S], hh: &[S]) {
        let overlaps = RectangleOverlapSweep::find_layout_overlaps(layout, hw, hh).unwrap();
        assert!(
            overlaps.is_empty(),
            "expected 0 overlaps, found {}: {:?}",
            overlaps.len(),
            &overlaps[..overlaps.len().min(5)]
        );
    }

    #[test]
    fn test_forbid_validation_and_degenerate_inputs() {
        assert!(matches!(
            ForbidConfig {
                scale_step: 0.0,
                ..Default::default()
            }
            .validate(),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            ForbidConfig {
                scale_step: f64::NAN,
                ..Default::default()
            }
            .validate(),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            ForbidConfig {
                padding: -1.0,
                ..Default::default()
            }
            .validate(),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            ForbidConfig {
                overlap_weight_k: f64::INFINITY,
                ..Default::default()
            }
            .validate(),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            ForbidOverlapRemoval::new(vec![1.0], vec![1.0, 2.0], ForbidConfig::default()),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            ForbidOverlapRemoval::new(vec![-0.5], vec![1.0], ForbidConfig::default()),
            Err(CoreError::InvalidInput(_))
        ));

        for n in [0usize, 1] {
            let mut solver =
                ForbidOverlapRemoval::<f64>::from_radii(vec![1.0; n], ForbidConfig::default())
                    .unwrap();
            let mut layout = Layout2d::zeros(n);
            assert_eq!(solver.run(&mut layout).unwrap(), SolverStatus::Converged);
            assert_eq!(solver.last_overlap_count(), 0);
        }

        let mut solver =
            ForbidOverlapRemoval::<f64>::from_radii(vec![1.0; 2], ForbidConfig::default()).unwrap();
        let mut wrong = Layout2d::zeros(3);
        assert!(matches!(
            solver.step(&mut wrong),
            Err(CoreError::DimensionMismatch { .. })
        ));
        let mut bad = Layout2d::from_xy(vec![0.0, f64::NAN], vec![0.0, 0.0]).unwrap();
        assert!(matches!(
            solver.step(&mut bad),
            Err(CoreError::NonFiniteValue { .. })
        ));
    }

    #[test]
    fn test_forbid_all_variants_zero_overlaps_f64_f32_and_step_eq_run() {
        let n = 30usize;
        let mut rng = DeterministicRng::from_seed_u64(9876);
        let mut xs64 = Vec::with_capacity(n);
        let mut ys64 = Vec::with_capacity(n);
        let mut hw64 = Vec::with_capacity(n);
        let mut hh64 = Vec::with_capacity(n);
        for i in 0..n {
            xs64.push((rng.next_f64() - 0.5) * 4.0);
            ys64.push((rng.next_f64() - 0.5) * 4.0);
            hw64.push(0.75 + (i % 4) as f64 * 0.35);
            hh64.push(0.65 + ((i + 1) % 3) as f64 * 0.3);
        }

        for variant in [
            ForbidVariant::Forbid,
            ForbidVariant::ForbidPrime,
            ForbidVariant::SparseForbid,
        ] {
            let cfg = ForbidConfig {
                variant,
                max_scale_iters: 20,
                sgd_iterations: 30,
                sparse_k: 10,
                overlap_weight_k: 1.0,
                scale_step: 0.1,
                padding: 0.2,
                seed: 42,
            };
            let pad_half = 0.5 * cfg.padding;
            let eff_hw64: Vec<f64> = hw64.iter().map(|&w| w + pad_half).collect();
            let eff_hh64: Vec<f64> = hh64.iter().map(|&h| h + pad_half).collect();

            let mut l64_run = Layout2d::from_xy(xs64.clone(), ys64.clone()).unwrap();
            let mut l64_step = l64_run.clone();
            let mut s64_run =
                ForbidOverlapRemoval::<f64>::new(hw64.clone(), hh64.clone(), cfg).unwrap();
            let mut s64_step =
                ForbidOverlapRemoval::<f64>::new(hw64.clone(), hh64.clone(), cfg).unwrap();

            assert_eq!(s64_run.run(&mut l64_run).unwrap(), SolverStatus::Converged);
            assert_eq!(s64_run.last_overlap_count(), 0);
            assert_no_overlaps(&l64_run, &eff_hw64, &eff_hh64);

            s64_step.run_steps(&mut l64_step, 1).unwrap();
            while !s64_step.status().is_finished() {
                s64_step.step(&mut l64_step).unwrap();
            }
            assert_eq!(s64_step.status(), SolverStatus::Converged);
            for i in 0..n {
                assert_eq!(l64_run.x()[i].to_bits(), l64_step.x()[i].to_bits());
                assert_eq!(l64_run.y()[i].to_bits(), l64_step.y()[i].to_bits());
            }

            // Verify f32 precision at offset coordinates.
            let xs32: Vec<f32> = xs64.iter().map(|&v| v as f32 + 250.0).collect();
            let ys32: Vec<f32> = ys64.iter().map(|&v| v as f32 - 250.0).collect();
            let hw32: Vec<f32> = hw64.iter().map(|&v| v as f32).collect();
            let hh32: Vec<f32> = hh64.iter().map(|&v| v as f32).collect();
            let eff_hw32: Vec<f32> = hw32
                .iter()
                .map(|&v| (v as f64 + 0.5 * cfg.padding) as f32)
                .collect();
            let eff_hh32: Vec<f32> = hh32
                .iter()
                .map(|&v| (v as f64 + 0.5 * cfg.padding) as f32)
                .collect();

            let mut l32 = Layout2d::from_xy(xs32, ys32).unwrap();
            let mut s32 = ForbidOverlapRemoval::<f32>::new(hw32, hh32, cfg).unwrap();
            assert_eq!(s32.run(&mut l32).unwrap(), SolverStatus::Converged);
            assert_eq!(s32.last_overlap_count(), 0);
            assert_no_overlaps(&l32, &eff_hw32, &eff_hh32);
        }
    }

    #[test]
    fn test_forbid_coincident_collinear_and_low_iter_fallback() {
        let n = 14usize;
        let hw = vec![1.0; n];
        let hh = vec![0.5; n];

        // 1. All coincident points at origin.
        let mut coincident = Layout2d::zeros(n);
        let mut s_coin =
            ForbidOverlapRemoval::new(hw.clone(), hh.clone(), ForbidConfig::default()).unwrap();
        assert_eq!(
            s_coin.run(&mut coincident).unwrap(),
            SolverStatus::Converged
        );
        assert_eq!(s_coin.last_overlap_count(), 0);
        assert_no_overlaps(&coincident, &hw, &hh);

        // 2. Strictly collinear layout.
        let mut collinear =
            Layout2d::from_xy((0..n).map(|i| (i as f64) * 0.3).collect(), vec![0.0; n]).unwrap();
        let mut s_coll =
            ForbidOverlapRemoval::new(hw.clone(), hh.clone(), ForbidConfig::default()).unwrap();
        assert_eq!(s_coll.run(&mut collinear).unwrap(), SolverStatus::Converged);
        assert_eq!(s_coll.last_overlap_count(), 0);
        assert_no_overlaps(&collinear, &hw, &hh);

        // 3. Tight max_scale_iters = 1 fallback guarantees 0 overlaps.
        let mut tight = Layout2d::zeros(n);
        let cfg_tight = ForbidConfig {
            max_scale_iters: 1,
            sgd_iterations: 5,
            ..ForbidConfig::default()
        };
        let mut s_tight = ForbidOverlapRemoval::new(hw.clone(), hh.clone(), cfg_tight).unwrap();
        assert_eq!(s_tight.run(&mut tight).unwrap(), SolverStatus::Converged);
        assert_eq!(s_tight.last_overlap_count(), 0);
        assert_no_overlaps(&tight, &hw, &hh);
    }
}
