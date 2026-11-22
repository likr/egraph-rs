//! Pure-Rust $O(N \log N)$ `StaticKdTree` active-set spatial intruder refiner.
//!
//! Native port of the Track A / Track B 2D spatial intruder refiner:
//! detects strictly $>2$-hop spatial false positives inside each vertex's $k_u$-NN ball
//! (plus optional 2-hop false negatives), protects 1-hop anchors, conditions pair weights,
//! and executes bounded-scale `SgdSolver` passes without distorting already clean layouts.

use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, Scalar,
    SolverStatus, VisitStamp,
};
use egraph_neighborhood::{extract_1hop_2hop_csr, extract_sparse_1hop_2hop_pairs};
use egraph_spatial::StaticKdTree;

use crate::sgd::{SgdScheduleConfig, SgdScheduleKind, SgdSolver};

const MAX_WEIGHT_RATIO: f64 = 8100.0;
const EXACT_2HOP_FLOOR_WORK: u128 = 100_000;
const EXACT_2HOP_LINEAR_FACTOR: u128 = 64;

/// Hyperparameters for [`SpatialIntruderRefiner`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatialRefinerConfig {
    /// Maximum number of outer active-set refinement rounds (`default: 4`).
    pub rounds: usize,
    /// Per-node spatial $k$-NN cap (`default: 32`).
    pub k_cap: usize,
    /// Repulsive hinge target multiplier relative to local empirical $k_u$-NN radius (`default: 1.2`).
    pub margin_scale: f64,
    /// Tier 1 (1-hop anchor) spring weight multiplier (`default: 1.0`).
    pub w_anchor: f64,
    /// Tier 2 (2-hop / elongated 1-hop false negative) spring weight multiplier (`default: 0.0`).
    pub w_fn: f64,
    /// Tier 3 (>2-hop spatial intruder false positive) weight multiplier (`default: 0.2`).
    pub w_fp: f64,
    /// Target shrinkage factor applied to active false negatives (`default: 0.88`).
    pub fn_shrink: f64,
    /// Multiplicative step-scale decay per outer round (`default: 0.75`).
    pub round_decay: f64,
    /// Inner `SgdSolver` iterations executed per active round (`default: 20`).
    pub t_max: usize,
    /// Exponential scheduler minimum learning rate ratio (`default: 0.05`).
    pub eps: f64,
    /// Whether Tier 3 FPs use one-sided hinge repulsion instead of two-sided springs (`default: true`).
    pub hinge_fp: bool,
    /// Explicit deterministic seed for jittering and SGD shuffling (`default: 42`).
    pub seed: u64,
}

impl Default for SpatialRefinerConfig {
    fn default() -> Self {
        Self {
            rounds: 4,
            k_cap: 32,
            margin_scale: 1.2,
            w_anchor: 1.0,
            w_fn: 0.0,
            w_fp: 0.2,
            fn_shrink: 0.88,
            round_decay: 0.75,
            t_max: 20,
            eps: 0.05,
            hinge_fp: true,
            seed: 42,
        }
    }
}

impl SpatialRefinerConfig {
    /// Validates all refinement hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if self.k_cap == 0 {
            return Err(CoreError::invalid_input("k_cap must be >= 1"));
        }
        if !self.margin_scale.is_finite() || self.margin_scale <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "margin_scale must be finite and > 0.0, got {}",
                self.margin_scale
            )));
        }
        for (name, val) in [
            ("w_anchor", self.w_anchor),
            ("w_fn", self.w_fn),
            ("w_fp", self.w_fp),
        ] {
            if !val.is_finite() || val < 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "{name} must be finite and >= 0.0, got {val}"
                )));
            }
        }
        if !self.fn_shrink.is_finite() || self.fn_shrink <= 0.0 || self.fn_shrink > 1.0 {
            return Err(CoreError::invalid_input(format!(
                "fn_shrink must be finite and in (0.0, 1.0], got {}",
                self.fn_shrink
            )));
        }
        if !self.round_decay.is_finite() || self.round_decay <= 0.0 || self.round_decay > 1.0 {
            return Err(CoreError::invalid_input(format!(
                "round_decay must be finite and in (0.0, 1.0], got {}",
                self.round_decay
            )));
        }
        if !self.eps.is_finite() || self.eps <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "eps must be finite and > 0.0, got {}",
                self.eps
            )));
        }
        Ok(())
    }
}

/// Per-round diagnostic counters emitted by [`SpatialIntruderRefiner`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpatialRefinerRoundStats {
    /// 1-indexed refinement round number.
    pub round: usize,
    /// Number of active Tier 1 1-hop anchor pairs.
    pub n_anchors: usize,
    /// Number of active Tier 2 false-negative pairs (including missing 1-hop boosts).
    pub n_fn: usize,
    /// Number of active Tier 3 spatial false-positive intruder pairs.
    pub n_fp: usize,
    /// Total active pairs supplied to the inner `SgdSolver`.
    pub n_pairs_total: usize,
    /// `true` when zero active intruders/stragglers were found and SGD was skipped.
    pub skipped: bool,
}

/// Cumulative diagnostic summary across rounds of [`SpatialIntruderRefiner`].
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialRefinerStats {
    pub rounds_executed: usize,
    pub n_anchors_total: usize,
    pub n_fn_total: usize,
    pub n_fp_total: usize,
    pub n_pairs_total: usize,
    pub initial_scale: f64,
    pub final_scale: f64,
    pub d_floor: f64,
    pub scale_ratio: f64,
    pub used_hub_fallback: bool,
    pub hinge_fp: bool,
    pub round_stats: Vec<SpatialRefinerRoundStats>,
}

/// Cooperative $O(N \log N)$ `StaticKdTree` active-set spatial intruder refiner implementing
/// [`IterativeSolver`].
///
/// Each call to [`IterativeSolver::step`] executes one refinement round.
#[derive(Debug, Clone)]
pub struct SpatialIntruderRefiner<S: Scalar = f64> {
    config: SpatialRefinerConfig,
    num_nodes: usize,
    u1: Vec<u32>,
    v1: Vec<u32>,
    adj1: CsrGraph<f64>,
    adj2: Option<CsrGraph<f64>>,
    u2_hub: Vec<u32>,
    v2_hub: Vec<u32>,
    k_eff: usize,
    k_arr: Vec<usize>,
    scale_initialized: bool,
    initial_scale: f64,
    initial_std: f64,
    d_floor: f64,
    round: usize,
    status: SolverStatus,
    stats: SpatialRefinerStats,
    // Reusable scratch buffers to keep per-round overhead minimal.
    coords_f64: Vec<[f64; 2]>,
    knn_scratch: Vec<(usize, f64)>,
    nbr_idx_flat: Vec<u32>,
    nbr_dist_flat: Vec<f64>,
    r_knn: Vec<f64>,
    s_indptr: Vec<usize>,
    s_indices: Vec<u32>,
    coincident: Vec<bool>,
    stamp: VisitStamp,
    _marker: std::marker::PhantomData<S>,
}

impl<S: Scalar> SpatialIntruderRefiner<S> {
    /// Precomputes canonical undirected 1-hop topology and hub-safe 2-hop structures from `graph`.
    pub fn new<W>(graph: &CsrGraph<W>, config: SpatialRefinerConfig) -> CoreResult<Self> {
        config.validate()?;
        let n = graph.num_nodes();
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_csr("num_nodes exceeds u32::MAX"));
        }

        let (adj1, u1, v1) = build_canonical_undirected_1hop(graph)?;
        let deg: Vec<usize> = (0..n).map(|u| adj1.degree(u)).collect();
        let two_hop_work: u128 = deg
            .iter()
            .map(|&d| {
                let du = d as u128;
                du * du
            })
            .sum();
        let threshold = EXACT_2HOP_FLOOR_WORK.max(EXACT_2HOP_LINEAR_FACTOR * (n as u128));
        let use_exact_2hop = two_hop_work <= threshold;

        let k_eff = if n > 1 { config.k_cap.min(n - 1) } else { 0 };
        let trivial_early = n <= 2 || u1.is_empty() || config.rounds == 0 || config.t_max == 0;

        let (adj2, u2_hub, v2_hub, k_arr, used_hub_fallback) = if trivial_early {
            (None, Vec::new(), Vec::new(), vec![0usize; n], false)
        } else if use_exact_2hop {
            let (_, exact_adj2) = extract_1hop_2hop_csr(&adj1)?;
            let mut ks = Vec::with_capacity(n);
            for u in 0..n {
                ks.push((adj1.degree(u) + exact_adj2.degree(u)).min(k_eff));
            }
            (Some(exact_adj2), Vec::new(), Vec::new(), ks, false)
        } else {
            let hub_pairs = extract_sparse_1hop_2hop_pairs(&adj1, config.k_cap, config.seed)?;
            let ks = compute_capped_2hop_sizes(&adj1, &deg, k_eff);
            (None, hub_pairs.u2, hub_pairs.v2, ks, true)
        };

        let mut s_indptr = Vec::with_capacity(n + 1);
        s_indptr.push(0usize);
        let mut s_acc = 0usize;
        for &ku in &k_arr {
            s_acc += ku;
            s_indptr.push(s_acc);
        }

        let status = if config.rounds == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };

        let stats = SpatialRefinerStats {
            rounds_executed: 0,
            n_anchors_total: 0,
            n_fn_total: 0,
            n_fp_total: 0,
            n_pairs_total: 0,
            initial_scale: 1.0,
            final_scale: 1.0,
            d_floor: 1e-3,
            scale_ratio: 1.0,
            used_hub_fallback,
            hinge_fp: config.hinge_fp,
            round_stats: Vec::with_capacity(config.rounds),
        };

        Ok(Self {
            config,
            num_nodes: n,
            u1,
            v1,
            adj1,
            adj2,
            u2_hub,
            v2_hub,
            k_eff,
            k_arr,
            scale_initialized: false,
            initial_scale: 1.0,
            initial_std: 0.0,
            d_floor: 1e-3,
            round: 0,
            status,
            stats,
            coords_f64: vec![[0.0, 0.0]; n],
            knn_scratch: Vec::with_capacity(k_eff),
            nbr_idx_flat: vec![0u32; n.saturating_mul(k_eff)],
            nbr_dist_flat: vec![0.0_f64; n.saturating_mul(k_eff)],
            r_knn: vec![0.0_f64; n],
            s_indptr,
            s_indices: vec![0u32; s_acc],
            coincident: vec![false; n],
            stamp: VisitStamp::new(n),
            _marker: std::marker::PhantomData,
        })
    }

    /// Returns the refiner configuration.
    #[inline]
    pub fn config(&self) -> &SpatialRefinerConfig {
        &self.config
    }

    /// Returns the expected number of vertices in [`Layout2d`].
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns cumulative and per-round diagnostic statistics.
    #[inline]
    pub fn stats(&self) -> &SpatialRefinerStats {
        &self.stats
    }

    #[inline]
    fn in_s_both(&self, u: usize, v: usize) -> bool {
        let u32_u = u as u32;
        let v32_v = v as u32;
        let row_u = &self.s_indices[self.s_indptr[u]..self.s_indptr[u + 1]];
        if row_u.binary_search(&v32_v).is_err() {
            return false;
        }
        let row_v = &self.s_indices[self.s_indptr[v]..self.s_indptr[v + 1]];
        row_v.binary_search(&u32_u).is_ok()
    }

    fn sync_coords_from_layout(&mut self, layout: &Layout2d<S>) {
        let xs = layout.x();
        let ys = layout.y();
        for i in 0..self.num_nodes {
            self.coords_f64[i] = [xs[i].to_f64(), ys[i].to_f64()];
        }
    }

    fn update_final_scale_stats(&mut self) {
        let final_scale = compute_layout_scale(&self.coords_f64, &self.u1, &self.v1);
        let final_std = compute_coords_std(&self.coords_f64);
        self.stats.final_scale = final_scale;
        self.stats.scale_ratio = if self.initial_std > 1e-12 {
            final_std / self.initial_std
        } else {
            1.0
        };
    }
}

impl<S: Scalar> IterativeSolver<S> for SpatialIntruderRefiner<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "SpatialIntruderRefiner::step",
            ));
        }
        layout.validate_finite()?;

        if self.status.is_finished() {
            return Ok(self.status);
        }

        let n = self.num_nodes;
        if !self.scale_initialized {
            self.sync_coords_from_layout(layout);
            self.initial_scale = compute_layout_scale(&self.coords_f64, &self.u1, &self.v1);
            self.initial_std = compute_coords_std(&self.coords_f64);
            self.d_floor = (1e-9_f64).max(1e-3 * self.initial_scale);
            self.stats.initial_scale = self.initial_scale;
            self.stats.final_scale = self.initial_scale;
            self.stats.d_floor = self.d_floor;
            self.stats.scale_ratio = 1.0;
            self.scale_initialized = true;
        }

        if n <= 2 || self.u1.is_empty() || self.config.t_max == 0 || self.k_eff == 0 {
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        self.sync_coords_from_layout(layout);

        let r = self.round;
        let round_seed =
            (self.config.seed.wrapping_add((r as u64).wrapping_mul(1009))) & 0xFFFF_FFFF;
        let mut tree = StaticKdTree::<2, f64>::build(&self.coords_f64)?;

        // Near-coincident node guard (SgdSolver skips norm == 0.0 pairs).
        let tol_coinc = 1e-12 * self.initial_scale;
        let tol_coinc_sq = tol_coinc * tol_coinc;
        self.coincident.fill(false);
        let mut any_coincident = false;
        for u in 0..n {
            tree.k_nearest_excluding_self_into(u, 1, &mut self.knn_scratch)?;
            if let Some(&(_, d2)) = self.knn_scratch.first()
                && d2 <= tol_coinc_sq
            {
                self.coincident[u] = true;
                any_coincident = true;
            }
        }

        if any_coincident {
            let mut rng_jit = DeterministicRng::from_seed_u64(round_seed);
            let jit_std = 1e-4 * self.initial_scale;
            let (xs_mut, ys_mut) = layout.xy_mut();
            for u in 0..n {
                if self.coincident[u] {
                    let (jx, jy) = sample_normal_pair(&mut rng_jit, jit_std);
                    let nx = self.coords_f64[u][0] + jx;
                    let ny = self.coords_f64[u][1] + jy;
                    if nx.is_finite() && ny.is_finite() {
                        xs_mut[u] = S::from_f64(nx);
                        ys_mut[u] = S::from_f64(ny);
                        self.coords_f64[u] = [xs_mut[u].to_f64(), ys_mut[u].to_f64()];
                    }
                }
            }
            tree = StaticKdTree::<2, f64>::build(&self.coords_f64)?;
        }

        // Query k_eff spatial neighbors excluding self and populate r_knn and S_csr.
        let k_eff = self.k_eff;
        for u in 0..n {
            tree.k_nearest_excluding_self_into(u, k_eff, &mut self.knn_scratch)?;
            let base = u * k_eff;
            let ku = self.k_arr[u];
            let mut last_d = self.d_floor;
            for idx_k in 0..k_eff {
                if let Some(&(nbr, d2)) = self.knn_scratch.get(idx_k) {
                    let dist = d2.sqrt();
                    self.nbr_idx_flat[base + idx_k] = nbr as u32;
                    self.nbr_dist_flat[base + idx_k] = dist;
                    last_d = dist;
                } else {
                    let fallback_v = ((u + 1) % n) as u32;
                    self.nbr_idx_flat[base + idx_k] = fallback_v;
                    self.nbr_dist_flat[base + idx_k] = last_d;
                }
            }
            if ku == 0 {
                self.r_knn[u] = self.d_floor;
            } else {
                self.r_knn[u] = self.nbr_dist_flat[base + ku - 1].max(self.d_floor);
                let s_start = self.s_indptr[u];
                let s_end = self.s_indptr[u + 1];
                self.s_indices[s_start..s_end].copy_from_slice(&self.nbr_idx_flat[base..base + ku]);
                self.s_indices[s_start..s_end].sort_unstable();
            }
        }

        // 1. Tier 3: False positives F+ (strictly > 2-hop, one-sided hinge d_curr < d_fp_target).
        let mut fp_src: Vec<u32> = Vec::new();
        let mut fp_tgt: Vec<u32> = Vec::new();
        let mut fp_dist: Vec<f64> = Vec::new();
        let mut fp_wt: Vec<f64> = Vec::new();
        if self.config.w_fp > 0.0 && !self.s_indices.is_empty() {
            let mut fp_candidates: Vec<(u32, u32)> = Vec::new();
            for u in 0..n {
                let s_row = &self.s_indices[self.s_indptr[u]..self.s_indptr[u + 1]];
                if s_row.is_empty() {
                    continue;
                }
                self.stamp.clear();
                self.stamp.mark(u);
                let n1_u = self.adj1.neighbors(u);
                for &w in n1_u {
                    self.stamp.mark(w as usize);
                }
                if let Some(adj2) = &self.adj2 {
                    for &w in adj2.neighbors(u) {
                        self.stamp.mark(w as usize);
                    }
                    for &v in s_row {
                        if !self.stamp.is_visited(v as usize) {
                            let u_u32 = u as u32;
                            fp_candidates.push(if u_u32 < v { (u_u32, v) } else { (v, u_u32) });
                        }
                    }
                } else {
                    for &v in s_row {
                        let vi = v as usize;
                        if self.stamp.is_visited(vi) {
                            continue;
                        }
                        let n1_v = self.adj1.neighbors(vi);
                        if !has_common_neighbor(n1_u, n1_v, &self.stamp) {
                            let u_u32 = u as u32;
                            fp_candidates.push(if u_u32 < v { (u_u32, v) } else { (v, u_u32) });
                        }
                    }
                }
            }

            fp_candidates.sort_unstable();
            fp_candidates.dedup();
            for (u_u32, v_u32) in fp_candidates {
                let ui = u_u32 as usize;
                let vi = v_u32 as usize;
                let r_pair_fp = if self.k_arr[ui] == 0 {
                    self.r_knn[vi]
                } else if self.k_arr[vi] == 0 {
                    self.r_knn[ui]
                } else {
                    0.5 * (self.r_knn[ui] + self.r_knn[vi])
                };
                let d_fp_tgt = (self.config.margin_scale * r_pair_fp).max(self.d_floor);
                let dx = self.coords_f64[ui][0] - self.coords_f64[vi][0];
                let dy = self.coords_f64[ui][1] - self.coords_f64[vi][1];
                let d_fp_curr = dx.hypot(dy);
                if d_fp_curr < d_fp_tgt {
                    fp_src.push(u_u32);
                    fp_tgt.push(v_u32);
                    fp_dist.push(d_fp_tgt);
                    fp_wt.push(self.config.w_fp / (d_fp_tgt * d_fp_tgt));
                }
            }
        }

        // 2. Tier 2: 2-hop False Negatives F- (bounded contraction when w_fn > 0.0).
        let mut fn_src: Vec<u32> = Vec::new();
        let mut fn_tgt: Vec<u32> = Vec::new();
        let mut fn_dist: Vec<f64> = Vec::new();
        let mut fn_wt: Vec<f64> = Vec::new();
        if self.config.w_fn > 0.0 {
            if let Some(adj2) = &self.adj2 {
                for u in 0..n {
                    let u_u32 = u as u32;
                    for &v_u32 in adj2.neighbors(u) {
                        if u_u32 >= v_u32 {
                            continue;
                        }
                        let vi = v_u32 as usize;
                        if self.in_s_both(u, vi) {
                            continue;
                        }
                        let r_pair_fn = 0.5 * (self.r_knn[u] + self.r_knn[vi]);
                        let dx = self.coords_f64[u][0] - self.coords_f64[vi][0];
                        let dy = self.coords_f64[u][1] - self.coords_f64[vi][1];
                        let d_fn_curr = dx.hypot(dy);
                        let d_fn_tgt = self
                            .d_floor
                            .max((self.config.fn_shrink * d_fn_curr).max(0.9 * r_pair_fn));
                        if d_fn_curr > d_fn_tgt {
                            fn_src.push(u_u32);
                            fn_tgt.push(v_u32);
                            fn_dist.push(d_fn_tgt);
                            fn_wt.push(self.config.w_fn / (d_fn_tgt * d_fn_tgt));
                        }
                    }
                }
            } else {
                for (&u_u32, &v_u32) in self.u2_hub.iter().zip(self.v2_hub.iter()) {
                    let ui = u_u32 as usize;
                    let vi = v_u32 as usize;
                    if self.in_s_both(ui, vi) {
                        continue;
                    }
                    let r_pair_fn = 0.5 * (self.r_knn[ui] + self.r_knn[vi]);
                    let dx = self.coords_f64[ui][0] - self.coords_f64[vi][0];
                    let dy = self.coords_f64[ui][1] - self.coords_f64[vi][1];
                    let d_fn_curr = dx.hypot(dy);
                    let d_fn_tgt = self
                        .d_floor
                        .max((self.config.fn_shrink * d_fn_curr).max(0.9 * r_pair_fn));
                    if d_fn_curr > d_fn_tgt {
                        fn_src.push(u_u32);
                        fn_tgt.push(v_u32);
                        fn_dist.push(d_fn_tgt);
                        fn_wt.push(self.config.w_fn / (d_fn_tgt * d_fn_tgt));
                    }
                }
            }
        }

        // 3. Tier 1: 1-hop Anchors A1 (protect 1-hop lengths; gentle w_fn boost for elongated missing 1-hop).
        let mut a1_src: Vec<u32> = Vec::with_capacity(self.u1.len());
        let mut a1_tgt: Vec<u32> = Vec::with_capacity(self.v1.len());
        let mut a1_dist: Vec<f64> = Vec::with_capacity(self.u1.len());
        let mut a1_wt: Vec<f64> = Vec::with_capacity(self.u1.len());
        let mut n_1hop_fn = 0usize;

        for (&u_u32, &v_u32) in self.u1.iter().zip(self.v1.iter()) {
            let ui = u_u32 as usize;
            let vi = v_u32 as usize;
            let dx = self.coords_f64[ui][0] - self.coords_f64[vi][0];
            let dy = self.coords_f64[ui][1] - self.coords_f64[vi][1];
            let d1_curr = dx.hypot(dy);
            let d1_safe = d1_curr.max(self.d_floor);

            let (d1, w1) = if self.config.w_fn > 0.0 {
                let in_both = self.in_s_both(ui, vi);
                let r_pair_1 = 0.5 * (self.r_knn[ui] + self.r_knn[vi]);
                let d1_fn_tgt = self
                    .d_floor
                    .max((self.config.fn_shrink * d1_curr).max(0.9 * r_pair_1));
                let is_1hop_fn = !in_both && (d1_curr > d1_fn_tgt);
                if is_1hop_fn {
                    n_1hop_fn += 1;
                    (
                        d1_fn_tgt,
                        (self.config.w_anchor + self.config.w_fn) / (d1_fn_tgt * d1_fn_tgt),
                    )
                } else {
                    (d1_safe, self.config.w_anchor / (d1_safe * d1_safe))
                }
            } else {
                (d1_safe, self.config.w_anchor / (d1_safe * d1_safe))
            };

            if w1 > 0.0 && d1 > 0.0 && w1.is_finite() && d1.is_finite() {
                a1_src.push(u_u32);
                a1_tgt.push(v_u32);
                a1_dist.push(d1);
                a1_wt.push(w1);
            }
        }

        let n_fp = fp_src.len();
        let n_fn = fn_src.len() + n_1hop_fn;
        let n_anc = a1_src.len();

        // Early stop when zero active intruders/stragglers exist so clean layouts remain untouched.
        if n_fp == 0 && n_fn == 0 {
            self.stats.round_stats.push(SpatialRefinerRoundStats {
                round: r + 1,
                n_anchors: 0,
                n_fn: 0,
                n_fp: 0,
                n_pairs_total: 0,
                skipped: true,
            });
            self.update_final_scale_stats();
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        let scale_r = self.config.round_decay.powi(r as i32);
        let sched = SgdScheduleConfig {
            t_max: self.config.t_max,
            eps: self.config.eps,
            kind: SgdScheduleKind::Exponential,
            step_scale: scale_r,
        };

        let n_tot = if self.config.hinge_fp {
            let mut sp_src = Vec::with_capacity(a1_src.len() + fn_src.len());
            let mut sp_tgt = Vec::with_capacity(a1_tgt.len() + fn_tgt.len());
            let mut sp_d = Vec::with_capacity(a1_dist.len() + fn_dist.len());
            let mut sp_w = Vec::with_capacity(a1_wt.len() + fn_wt.len());

            for ((&u, &v), (&d, &w)) in a1_src
                .iter()
                .zip(a1_tgt.iter())
                .zip(a1_dist.iter().zip(a1_wt.iter()))
                .chain(
                    fn_src
                        .iter()
                        .zip(fn_tgt.iter())
                        .zip(fn_dist.iter().zip(fn_wt.iter())),
                )
            {
                if d.is_finite() && w.is_finite() && d > 0.0 && w > 0.0 {
                    sp_src.push(u);
                    sp_tgt.push(v);
                    sp_d.push(d);
                    sp_w.push(w);
                }
            }

            let mut hg_src = Vec::with_capacity(fp_src.len());
            let mut hg_tgt = Vec::with_capacity(fp_tgt.len());
            let mut hg_d = Vec::with_capacity(fp_dist.len());
            let mut hg_w = Vec::with_capacity(fp_wt.len());
            for ((&u, &v), (&d, &w)) in fp_src
                .iter()
                .zip(fp_tgt.iter())
                .zip(fp_dist.iter().zip(fp_wt.iter()))
            {
                if d.is_finite() && w.is_finite() && d > 0.0 && w > 0.0 {
                    hg_src.push(u);
                    hg_tgt.push(v);
                    hg_d.push(d);
                    hg_w.push(w);
                }
            }

            let tot = sp_src.len() + hg_src.len();
            if tot == 0 {
                self.stats.round_stats.push(SpatialRefinerRoundStats {
                    round: r + 1,
                    n_anchors: 0,
                    n_fn: 0,
                    n_fp: 0,
                    n_pairs_total: 0,
                    skipped: true,
                });
                self.update_final_scale_stats();
                self.status = SolverStatus::Converged;
                return Ok(self.status);
            }

            let n_sp = sp_w.len();
            let mut all_w = Vec::with_capacity(tot);
            all_w.extend_from_slice(&sp_w);
            all_w.extend_from_slice(&hg_w);
            condition_weights(&mut all_w, MAX_WEIGHT_RATIO);

            let sp_d_s: Vec<S> = sp_d.iter().map(|&v| S::from_f64(v)).collect();
            let sp_w_s: Vec<S> = all_w[..n_sp].iter().map(|&v| S::from_f64(v)).collect();
            let hg_d_s: Vec<S> = hg_d.iter().map(|&v| S::from_f64(v)).collect();
            let hg_w_s: Vec<S> = all_w[n_sp..].iter().map(|&v| S::from_f64(v)).collect();

            let mut sgd = SgdSolver::<S>::from_pairs_with_hinges(
                n,
                sp_src,
                sp_tgt,
                sp_d_s.clone(),
                sp_d_s,
                sp_w_s.clone(),
                sp_w_s,
                hg_src,
                hg_tgt,
                hg_d_s.clone(),
                hg_d_s,
                hg_w_s.clone(),
                hg_w_s,
                sched,
                round_seed,
            )?;
            sgd.run(layout)?;
            tot
        } else {
            let mut src = Vec::with_capacity(a1_src.len() + fn_src.len() + fp_src.len());
            let mut tgt = Vec::with_capacity(a1_tgt.len() + fn_tgt.len() + fp_tgt.len());
            let mut dist = Vec::with_capacity(a1_dist.len() + fn_dist.len() + fp_dist.len());
            let mut wt = Vec::with_capacity(a1_wt.len() + fn_wt.len() + fp_wt.len());

            for ((&u, &v), (&d, &w)) in a1_src
                .iter()
                .zip(a1_tgt.iter())
                .zip(a1_dist.iter().zip(a1_wt.iter()))
                .chain(
                    fn_src
                        .iter()
                        .zip(fn_tgt.iter())
                        .zip(fn_dist.iter().zip(fn_wt.iter())),
                )
                .chain(
                    fp_src
                        .iter()
                        .zip(fp_tgt.iter())
                        .zip(fp_dist.iter().zip(fp_wt.iter())),
                )
            {
                if d.is_finite() && w.is_finite() && d > 0.0 && w > 0.0 {
                    src.push(u);
                    tgt.push(v);
                    dist.push(d);
                    wt.push(w);
                }
            }

            let tot = src.len();
            if tot == 0 {
                self.stats.round_stats.push(SpatialRefinerRoundStats {
                    round: r + 1,
                    n_anchors: 0,
                    n_fn: 0,
                    n_fp: 0,
                    n_pairs_total: 0,
                    skipped: true,
                });
                self.update_final_scale_stats();
                self.status = SolverStatus::Converged;
                return Ok(self.status);
            }

            condition_weights(&mut wt, MAX_WEIGHT_RATIO);
            let dist_s: Vec<S> = dist.iter().map(|&v| S::from_f64(v)).collect();
            let wt_s: Vec<S> = wt.iter().map(|&v| S::from_f64(v)).collect();

            let mut sgd = SgdSolver::<S>::from_pairs(
                n,
                src,
                tgt,
                dist_s.clone(),
                dist_s,
                wt_s.clone(),
                wt_s,
                sched,
                round_seed,
            )?;
            sgd.run(layout)?;
            tot
        };

        self.sync_coords_from_layout(layout);
        self.update_final_scale_stats();

        self.stats.rounds_executed += 1;
        self.stats.n_anchors_total += n_anc;
        self.stats.n_fn_total += n_fn;
        self.stats.n_fp_total += n_fp;
        self.stats.n_pairs_total += n_tot;
        self.stats.round_stats.push(SpatialRefinerRoundStats {
            round: r + 1,
            n_anchors: n_anc,
            n_fn,
            n_fp,
            n_pairs_total: n_tot,
            skipped: false,
        });

        self.round += 1;
        self.status = if self.round >= self.config.rounds {
            SolverStatus::Completed
        } else {
            SolverStatus::Running
        };
        Ok(self.status)
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        self.status
    }

    #[inline]
    fn iteration(&self) -> usize {
        self.round
    }

    #[inline]
    fn total_iterations(&self) -> usize {
        self.config.rounds
    }
}

fn build_canonical_undirected_1hop<W>(
    graph: &CsrGraph<W>,
) -> CoreResult<(CsrGraph<f64>, Vec<u32>, Vec<u32>)> {
    let n = graph.num_nodes();
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (u, bucket) in buckets.iter_mut().enumerate() {
        let u32_u = u as u32;
        for &v in graph.neighbors(u) {
            if v != u32_u && (v as usize) < n {
                bucket.push(v);
            }
        }
    }
    if graph.is_directed() {
        for u in 0..n {
            let u32_u = u as u32;
            for &v in graph.neighbors(u) {
                if v != u32_u && (v as usize) < n {
                    buckets[v as usize].push(u32_u);
                }
            }
        }
    }

    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0u32);
    let mut indices = Vec::new();
    let mut u1 = Vec::new();
    let mut v1 = Vec::new();
    for (u, row) in buckets.iter_mut().enumerate() {
        row.sort_unstable();
        row.dedup();
        let u_u32 = u as u32;
        for &v in row.iter() {
            if u_u32 < v {
                u1.push(u_u32);
                v1.push(v);
            }
        }
        indices.extend_from_slice(row);
        let ptr = u32::try_from(indices.len())
            .map_err(|_| CoreError::invalid_csr("canonical 1-hop nnz exceeds u32::MAX"))?;
        indptr.push(ptr);
    }

    let adj1 = CsrGraph::from_csr(&indptr, &indices, None, false, false)?;
    Ok((adj1, u1, v1))
}

fn compute_capped_2hop_sizes(adj_1hop: &CsrGraph<f64>, deg: &[usize], k_cap: usize) -> Vec<usize> {
    let n = adj_1hop.num_nodes();
    if n <= 1 || k_cap == 0 {
        return vec![0usize; n];
    }

    let mut k_arr = vec![0usize; n];
    let mut stamp = VisitStamp::new(n);

    for u in 0..n {
        let d_u = deg[u];
        if d_u == 0 {
            continue;
        }
        if d_u >= k_cap {
            k_arr[u] = k_cap;
            continue;
        }

        let n1_u = adj_1hop.neighbors(u);
        let mut max_nbr_deg = 0usize;
        for &w in n1_u {
            max_nbr_deg = max_nbr_deg.max(deg[w as usize]);
        }
        if max_nbr_deg >= k_cap {
            k_arr[u] = k_cap;
            continue;
        }

        stamp.clear();
        stamp.mark(u);
        for &w in n1_u {
            stamp.mark(w as usize);
        }
        let mut count = d_u;
        'outer: for &w in n1_u {
            for &v in adj_1hop.neighbors(w as usize) {
                if stamp.visit(v as usize) {
                    count += 1;
                    if count >= k_cap {
                        break 'outer;
                    }
                }
            }
        }
        k_arr[u] = count.min(k_cap);
    }
    k_arr
}

#[inline]
fn has_common_neighbor(n1_u: &[u32], n1_v: &[u32], stamp: &VisitStamp) -> bool {
    if n1_u.is_empty() || n1_v.is_empty() {
        return false;
    }
    if n1_v.len() <= n1_u.len() || n1_v.len() <= 64 {
        return n1_v.iter().any(|&x| stamp.is_visited(x as usize));
    }
    n1_u.iter().any(|x| n1_v.binary_search(x).is_ok())
}

fn compute_coords_std(coords: &[[f64; 2]]) -> f64 {
    if coords.is_empty() {
        return 0.0;
    }
    let count = (coords.len() * 2) as f64;
    let mut sum = 0.0_f64;
    for pt in coords {
        sum += pt[0] + pt[1];
    }
    let mean = sum / count;
    let mut sq = 0.0_f64;
    for pt in coords {
        let dx = pt[0] - mean;
        let dy = pt[1] - mean;
        sq += dx * dx + dy * dy;
    }
    (sq / count).sqrt()
}

fn compute_layout_scale(coords: &[[f64; 2]], u1: &[u32], v1: &[u32]) -> f64 {
    if !u1.is_empty() {
        let mut pos_d1: Vec<f64> = Vec::with_capacity(u1.len());
        for (&u, &v) in u1.iter().zip(v1.iter()) {
            let ui = u as usize;
            let vi = v as usize;
            let d = (coords[ui][0] - coords[vi][0]).hypot(coords[ui][1] - coords[vi][1]);
            if d.is_finite() && d > 1e-12 {
                pos_d1.push(d);
            }
        }
        if !pos_d1.is_empty() {
            pos_d1.sort_by(f64::total_cmp);
            let mid = pos_d1.len() / 2;
            return if pos_d1.len() % 2 == 1 {
                pos_d1[mid]
            } else {
                0.5 * (pos_d1[mid - 1] + pos_d1[mid])
            };
        }
    }
    let std_x = compute_coords_std(coords);
    if std_x.is_finite() && std_x > 1e-12 {
        std_x
    } else {
        1.0
    }
}

fn condition_weights(wt: &mut [f64], max_ratio: f64) {
    if wt.len() <= 1 {
        return;
    }
    let mut w_min = f64::INFINITY;
    let mut w_max = 0.0_f64;
    for &w in wt.iter() {
        if w < w_min {
            w_min = w;
        }
        if w > w_max {
            w_max = w;
        }
    }
    if w_max <= max_ratio * w_min || w_min <= 0.0 {
        return;
    }
    let half_span = max_ratio.sqrt();
    let mut sorted = wt.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    let w_med = if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        0.5 * (sorted[mid - 1] + sorted[mid])
    };
    let lo = w_med / half_span;
    let hi = w_med * half_span;
    for w in wt.iter_mut() {
        *w = w.clamp(lo, hi);
    }
}

fn sample_normal_pair(rng: &mut DeterministicRng, std_dev: f64) -> (f64, f64) {
    let u1 = rng.next_f64().max(1e-300);
    let u2 = rng.next_f64();
    let mag = std_dev * (-2.0 * u1.ln()).sqrt();
    let (s, c) = (std::f64::consts::TAU * u2).sin_cos();
    (mag * c, mag * s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid_graph(rows: usize, cols: usize) -> CsrGraph<f64> {
        let n = rows * cols;
        let mut edges = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let u = (r * cols + c) as u32;
                if c + 1 < cols {
                    edges.push((u, u + 1));
                }
                if r + 1 < rows {
                    edges.push((u, u + cols as u32));
                }
            }
        }
        CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
    }

    #[test]
    fn test_refiner_config_defaults_and_validation() {
        let cfg = SpatialRefinerConfig::default();
        assert_eq!(cfg.rounds, 4);
        assert_eq!(cfg.k_cap, 32);
        assert_eq!(cfg.margin_scale, 1.2);
        assert_eq!(cfg.w_anchor, 1.0);
        assert_eq!(cfg.w_fn, 0.0);
        assert_eq!(cfg.w_fp, 0.2);
        assert_eq!(cfg.fn_shrink, 0.88);
        assert_eq!(cfg.round_decay, 0.75);
        assert_eq!(cfg.t_max, 20);
        assert_eq!(cfg.eps, 0.05);
        assert!(cfg.hinge_fp);
        assert_eq!(cfg.seed, 42);
        cfg.validate().unwrap();

        let bad_cases = [
            SpatialRefinerConfig { k_cap: 0, ..cfg },
            SpatialRefinerConfig {
                margin_scale: 0.0,
                ..cfg
            },
            SpatialRefinerConfig { w_fp: -0.1, ..cfg },
            SpatialRefinerConfig { w_fn: -0.1, ..cfg },
            SpatialRefinerConfig {
                w_anchor: f64::NAN,
                ..cfg
            },
            SpatialRefinerConfig {
                fn_shrink: 0.0,
                ..cfg
            },
            SpatialRefinerConfig {
                fn_shrink: 1.5,
                ..cfg
            },
            SpatialRefinerConfig {
                round_decay: 0.0,
                ..cfg
            },
            SpatialRefinerConfig { eps: 0.0, ..cfg },
        ];
        for bad in bad_cases {
            assert!(bad.validate().is_err());
        }
    }

    #[test]
    fn test_refiner_leaves_clean_layout_undisturbed_and_pushes_intruders_apart() {
        // 1. Clean 3x3 grid embedded at exact lattice positions: zero >2-hop intruders -> early Converged.
        let grid = grid_graph(3, 3);
        let mut clean_x = Vec::with_capacity(9);
        let mut clean_y = Vec::with_capacity(9);
        for r in 0..3 {
            for c in 0..3 {
                clean_x.push(c as f64);
                clean_y.push(r as f64);
            }
        }
        let clean_orig = Layout2d::from_xy(clean_x, clean_y).unwrap();
        let mut clean_layout = clean_orig.clone();

        let cfg = SpatialRefinerConfig::default();
        let mut clean_refiner = SpatialIntruderRefiner::<f64>::new(&grid, cfg).unwrap();
        assert_eq!(
            clean_refiner.run(&mut clean_layout).unwrap(),
            SolverStatus::Converged
        );
        assert_eq!(clean_refiner.stats().rounds_executed, 0);
        assert_eq!(clean_refiner.stats().n_fp_total, 0);
        assert_eq!(clean_refiner.stats().round_stats.len(), 1);
        assert!(clean_refiner.stats().round_stats[0].skipped);
        for i in 0..9 {
            assert_eq!(clean_layout.x()[i].to_bits(), clean_orig.x()[i].to_bits());
            assert_eq!(clean_layout.y()[i].to_bits(), clean_orig.y()[i].to_bits());
        }

        // 2. Introduce a spatial intruder: place far corner node 8 (4 hops from 0) right next to node 0.
        let mut intruder_layout = clean_orig;
        intruder_layout.x_mut()[8] = 0.08;
        intruder_layout.y_mut()[8] = 0.08;
        let dist_before = (intruder_layout.x()[0] - intruder_layout.x()[8])
            .hypot(intruder_layout.y()[0] - intruder_layout.y()[8]);

        let mut intr_refiner = SpatialIntruderRefiner::<f64>::new(&grid, cfg).unwrap();
        let status = intr_refiner.run(&mut intruder_layout).unwrap();
        assert!(status.is_finished());
        assert!(intruder_layout.is_all_finite());
        assert!(intr_refiner.stats().n_fp_total > 0);
        assert!(intr_refiner.stats().rounds_executed >= 1);

        let dist_after = (intruder_layout.x()[0] - intruder_layout.x()[8])
            .hypot(intruder_layout.y()[0] - intruder_layout.y()[8]);
        assert!(
            dist_after > dist_before * 2.0,
            "expected spatial intruder (0, 8) pushed apart: before={dist_before}, after={dist_after}"
        );
    }

    #[test]
    fn test_refiner_step_vs_run_bit_exact_f64_and_f32_and_two_sided_fn() {
        let grid = grid_graph(3, 3);
        let mut base_x = Vec::with_capacity(9);
        let mut base_y = Vec::with_capacity(9);
        for r in 0..3 {
            for c in 0..3 {
                base_x.push(c as f64);
                base_y.push(r as f64);
            }
        }
        // Place node 8 near node 0 and stretch node 2 slightly so both FP and FN triggers work.
        base_x[8] = 0.1;
        base_y[8] = 0.1;
        base_x[2] = 4.0;

        for hinge_fp in [true, false] {
            let cfg = SpatialRefinerConfig {
                rounds: 3,
                w_fn: 0.2,
                hinge_fp,
                seed: 77,
                ..Default::default()
            };

            // f64 bit-exact parity
            let mut l_run = Layout2d::from_xy(base_x.clone(), base_y.clone()).unwrap();
            let mut l_step = l_run.clone();
            let mut r_run = SpatialIntruderRefiner::<f64>::new(&grid, cfg).unwrap();
            let mut r_step = SpatialIntruderRefiner::<f64>::new(&grid, cfg).unwrap();

            assert_eq!(r_run.total_iterations(), 3);
            let s_run = r_run.run(&mut l_run).unwrap();
            let s_step1 = r_step.step(&mut l_step).unwrap();
            assert!(!s_step1.is_finished() || r_step.iteration() == r_run.iteration());
            let s_step = r_step.run(&mut l_step).unwrap();
            assert_eq!(s_run, s_step);
            assert_eq!(r_run.stats(), r_step.stats());
            for i in 0..9 {
                assert_eq!(l_run.x()[i].to_bits(), l_step.x()[i].to_bits());
                assert_eq!(l_run.y()[i].to_bits(), l_step.y()[i].to_bits());
            }

            // f32 bit-exact parity
            let bx32: Vec<f32> = base_x.iter().map(|&v| v as f32).collect();
            let by32: Vec<f32> = base_y.iter().map(|&v| v as f32).collect();
            let mut l32_run = Layout2d::from_xy(bx32.clone(), by32.clone()).unwrap();
            let mut l32_step = l32_run.clone();
            let mut r32_run = SpatialIntruderRefiner::<f32>::new(&grid, cfg).unwrap();
            let mut r32_step = SpatialIntruderRefiner::<f32>::new(&grid, cfg).unwrap();
            r32_run.run(&mut l32_run).unwrap();
            while !r32_step.status().is_finished() {
                r32_step.step(&mut l32_step).unwrap();
            }
            for i in 0..9 {
                assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
                assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
            }
        }
    }

    #[test]
    fn test_refiner_hub_fallback_path_and_coincident_jitter() {
        // Hub star of 400 nodes (deg[0]=399 => two_hop_work > 159_000 > max(100_000, 64*403))
        // plus path 399-400-401-402 so node 402 is 4 hops from node 1.
        let n_star = 400usize;
        let mut edges: Vec<(u32, u32)> = (1..n_star as u32).map(|v| (0, v)).collect();
        edges.push((399, 400));
        edges.push((400, 401));
        edges.push((401, 402));
        let n = 403usize;
        let g = CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap();

        let mut xs = vec![0.0_f64; n];
        let mut ys = vec![0.0_f64; n];
        for i in 1..n_star {
            let ang = std::f64::consts::TAU * (i as f64) / ((n_star - 1) as f64);
            let (s, c) = ang.sin_cos();
            xs[i] = 2.0 * c;
            ys[i] = 2.0 * s;
        }
        xs[400] = xs[399] + 1.0;
        ys[400] = ys[399];
        xs[401] = xs[400] + 1.0;
        ys[401] = ys[400];
        // Place >2-hop node 402 coincident with leaf node 1!
        xs[402] = xs[1];
        ys[402] = ys[1];

        let cfg = SpatialRefinerConfig {
            rounds: 2,
            k_cap: 16,
            ..Default::default()
        };
        let mut refiner = SpatialIntruderRefiner::<f64>::new(&g, cfg).unwrap();
        assert!(refiner.stats().used_hub_fallback);

        let mut layout = Layout2d::from_xy(xs, ys).unwrap();
        let status = refiner.run(&mut layout).unwrap();
        assert!(status.is_finished());
        assert!(layout.is_all_finite());
        assert!(refiner.stats().n_fp_total > 0);
        let sep_1_402 = (layout.x()[1] - layout.x()[402]).hypot(layout.y()[1] - layout.y()[402]);
        assert!(
            sep_1_402 > 0.1,
            "expected hub fallback to separate coincident intruder (1, 402), got {sep_1_402}"
        );
    }

    #[test]
    fn test_refiner_edge_cases_and_errors() {
        let cfg = SpatialRefinerConfig::default();

        // Empty, singleton, 2-node, and edgeless graphs converge cleanly without panic.
        for n in [0usize, 1, 2, 4] {
            let g = if n == 2 {
                CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap()
            } else {
                CsrGraph::<f64>::from_edges(n, &[], None, false, false).unwrap()
            };
            let mut r = SpatialIntruderRefiner::<f64>::new(&g, cfg).unwrap();
            let mut l = Layout2d::<f64>::zeros(n);
            assert_eq!(r.run(&mut l).unwrap(), SolverStatus::Converged);
            assert_eq!(r.stats().rounds_executed, 0);
        }

        // rounds == 0 completes immediately, t_max == 0 converges immediately.
        let grid = grid_graph(3, 3);
        let r_zero_rounds =
            SpatialIntruderRefiner::<f64>::new(&grid, SpatialRefinerConfig { rounds: 0, ..cfg })
                .unwrap();
        assert_eq!(r_zero_rounds.status(), SolverStatus::Completed);
        assert_eq!(r_zero_rounds.total_iterations(), 0);

        let mut r_zero_tmax =
            SpatialIntruderRefiner::<f64>::new(&grid, SpatialRefinerConfig { t_max: 0, ..cfg })
                .unwrap();
        let mut l9 = Layout2d::<f64>::zeros(9);
        assert_eq!(r_zero_tmax.step(&mut l9).unwrap(), SolverStatus::Converged);

        // Dimension mismatch & non-finite layout rejected.
        let mut r_err = SpatialIntruderRefiner::<f64>::new(&grid, cfg).unwrap();
        let mut l_wrong = Layout2d::<f64>::zeros(4);
        assert!(matches!(
            r_err.step(&mut l_wrong),
            Err(CoreError::DimensionMismatch { .. })
        ));
        let mut l_nan = Layout2d::from_xy(vec![f64::NAN; 9], vec![0.0; 9]).unwrap();
        assert!(matches!(
            r_err.step(&mut l_nan),
            Err(CoreError::NonFiniteValue { .. })
        ));
    }
}
