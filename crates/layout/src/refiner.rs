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
use egraph_spatial::{KdQueryScratch, StaticKdTree};

use crate::sgd::{PackedSymPair, SgdScheduleConfig, SgdScheduleKind, SgdSolver};

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
    /// Tier 1/2 (2-hop anchor) spring weight multiplier (`default: 0.0`).
    pub w_anchor2: f64,
    /// Tier 2 (2-hop / elongated 1-hop false negative) spring weight multiplier (`default: 0.0`).
    pub w_fn: f64,
    /// Tier 3 (>2-hop spatial intruder false positive) weight multiplier (`default: 0.2`).
    pub w_fp: f64,
    /// Target shrinkage factor applied to active false negatives (`default: 0.88`).
    pub fn_shrink: f64,
    /// Multiplicative step-scale decay per outer round (`default: 0.75`).
    pub round_decay: f64,
    /// Per-round multiplicative decay applied to `k_cap` after each active round (`default: 1.0`).
    pub k_cap_decay: f64,
    /// Minimum lower bound for decayed `k_cap` (`default: 1`).
    pub k_cap_min: usize,
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
            w_anchor2: 0.0,
            w_fn: 0.0,
            w_fp: 0.2,
            fn_shrink: 0.88,
            round_decay: 0.75,
            k_cap_decay: 1.0,
            k_cap_min: 1,
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
            ("w_anchor2", self.w_anchor2),
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
        if !self.k_cap_decay.is_finite() || self.k_cap_decay <= 0.0 || self.k_cap_decay > 1.0 {
            return Err(CoreError::invalid_input(format!(
                "k_cap_decay must be finite and in (0.0, 1.0], got {}",
                self.k_cap_decay
            )));
        }
        if self.k_cap_min == 0 {
            return Err(CoreError::invalid_input("k_cap_min must be >= 1"));
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
    k_cap_capacity: usize,
    current_k_cap: usize,
    k_eff_round: usize,
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
    kd_scratch: KdQueryScratch,
    knn_scratch: Vec<(usize, f64)>,
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
            let (u2_hub, v2_hub) = if config.w_fn > 0.0 || config.w_anchor2 > 0.0 {
                let hub_pairs = extract_sparse_1hop_2hop_pairs(&adj1, config.k_cap, config.seed)?;
                (hub_pairs.u2, hub_pairs.v2)
            } else {
                (Vec::new(), Vec::new())
            };
            let ks = compute_capped_2hop_sizes(&adj1, &deg, k_eff);
            (None, u2_hub, v2_hub, ks, true)
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

        let k_cap_capacity = config.k_cap;
        let current_k_cap = config.k_cap;

        Ok(Self {
            config,
            num_nodes: n,
            u1,
            v1,
            adj1,
            adj2,
            u2_hub,
            v2_hub,
            k_cap_capacity,
            current_k_cap,
            k_eff_round: k_eff,
            k_arr,
            scale_initialized: false,
            initial_scale: 1.0,
            initial_std: 0.0,
            d_floor: 1e-3,
            round: 0,
            status,
            stats,
            coords_f64: vec![[0.0, 0.0]; n],
            kd_scratch: KdQueryScratch::with_capacity(k_eff),
            knn_scratch: Vec::with_capacity(k_eff),
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

    /// Returns the spatial $k$-NN cap active for the next refinement round.
    #[inline]
    pub fn current_k_cap(&self) -> usize {
        self.current_k_cap
    }

    /// Dynamically updates the spatial $k$-NN cap for subsequent refinement rounds.
    ///
    /// Shrinking or adjusting `k_cap` within the initially allocated capacity is $O(1)$ with
    /// zero heap allocations. Growing beyond initial capacity reallocates buffers on demand.
    pub fn set_k_cap(&mut self, k_cap: usize) -> CoreResult<()> {
        if k_cap == 0 {
            return Err(CoreError::invalid_input("k_cap must be >= 1"));
        }
        let n = self.num_nodes;
        let new_k_eff = if n > 1 { k_cap.min(n - 1) } else { 0 };
        let trivial_early =
            n <= 2 || self.u1.is_empty() || self.config.rounds == 0 || self.config.t_max == 0;
        if k_cap > self.k_cap_capacity && !trivial_early {
            if let Some(adj2) = &self.adj2 {
                self.k_arr = (0..n)
                    .map(|u| (self.adj1.degree(u) + adj2.degree(u)).min(new_k_eff))
                    .collect();
            } else {
                let deg: Vec<usize> = (0..n).map(|u| self.adj1.degree(u)).collect();
                self.k_arr = compute_capped_2hop_sizes(&self.adj1, &deg, new_k_eff);
                if self.config.w_fn > 0.0 || self.config.w_anchor2 > 0.0 {
                    let hub_pairs =
                        extract_sparse_1hop_2hop_pairs(&self.adj1, k_cap, self.config.seed)?;
                    self.u2_hub = hub_pairs.u2;
                    self.v2_hub = hub_pairs.v2;
                }
            }
            self.s_indptr.clear();
            self.s_indptr.push(0usize);
            let mut s_acc = 0usize;
            for &ku in &self.k_arr {
                s_acc += ku;
                self.s_indptr.push(s_acc);
            }
            self.s_indices.resize(s_acc, 0u32);
            self.kd_scratch = KdQueryScratch::with_capacity(new_k_eff);
            self.knn_scratch
                .reserve(new_k_eff.saturating_sub(self.knn_scratch.capacity()));
            self.k_cap_capacity = k_cap;
        }
        self.current_k_cap = k_cap;
        self.k_eff_round = new_k_eff;
        Ok(())
    }

    #[inline]
    fn row_k(&self, u: usize) -> usize {
        self.k_arr[u].min(self.k_eff_round)
    }

    #[inline]
    fn in_s_both(&self, u: usize, v: usize) -> bool {
        let u32_u = u as u32;
        let v32_v = v as u32;
        let ku = self.row_k(u);
        let row_u = &self.s_indices[self.s_indptr[u]..self.s_indptr[u] + ku];
        if row_u.binary_search(&v32_v).is_err() {
            return false;
        }
        let kv = self.row_k(v);
        let row_v = &self.s_indices[self.s_indptr[v]..self.s_indptr[v] + kv];
        row_v.binary_search(&u32_u).is_ok()
    }

    fn sync_coords_from_layout(&mut self, layout: &Layout2d<S>) {
        let xs = layout.x();
        let ys = layout.y();
        for i in 0..self.num_nodes {
            self.coords_f64[i] = [xs[i].to_f64(), ys[i].to_f64()];
        }
    }

    fn populate_spatial_knn(
        &mut self,
        tree: &StaticKdTree<2, f64>,
        tol_coinc_sq: f64,
        detect_coincident: bool,
    ) -> CoreResult<bool> {
        let n = self.num_nodes;
        let mut any_coincident = false;
        for u in 0..n {
            let ku = self.row_k(u);
            let k_q = if detect_coincident { ku.max(1) } else { ku };
            if k_q == 0 {
                self.r_knn[u] = self.d_floor;
                continue;
            }
            tree.k_nearest_excluding_self_with_scratch(
                u,
                k_q,
                &mut self.kd_scratch,
                &mut self.knn_scratch,
            )?;
            if detect_coincident
                && let Some(&(_, d2)) = self.knn_scratch.first()
                && d2 <= tol_coinc_sq
            {
                self.coincident[u] = true;
                any_coincident = true;
            }
            if ku == 0 {
                self.r_knn[u] = self.d_floor;
            } else {
                let mut last_d = self.d_floor;
                if let Some(&(_, d2)) = self
                    .knn_scratch
                    .get(ku - 1)
                    .or_else(|| self.knn_scratch.last())
                {
                    last_d = if d2 > 0.0 && d2.is_finite() {
                        d2.sqrt()
                    } else {
                        0.0
                    };
                }
                self.r_knn[u] = last_d.max(self.d_floor);

                let s_start = self.s_indptr[u];
                let s_end = s_start + ku;
                let s_row = &mut self.s_indices[s_start..s_end];
                let mut fallback_v = ((u + 1) % n) as u32;
                for (idx_k, slot) in s_row.iter_mut().enumerate() {
                    if let Some(&(nbr, _)) = self.knn_scratch.get(idx_k) {
                        fallback_v = nbr as u32;
                    }
                    *slot = fallback_v;
                }
                s_row.sort_unstable();
            }
        }
        Ok(any_coincident)
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

        if n <= 2 || self.u1.is_empty() || self.config.t_max == 0 || self.k_eff_round == 0 {
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        self.sync_coords_from_layout(layout);

        let r = self.round;
        let round_seed =
            (self.config.seed.wrapping_add((r as u64).wrapping_mul(1009))) & 0xFFFF_FFFF;
        let mut tree = StaticKdTree::<2, f64>::build(&self.coords_f64)?;

        // Single spatial k-NN pass (k_q = ku.max(1)) detects near-coincident nodes simultaneously.
        let tol_coinc = 1e-12 * self.initial_scale;
        let tol_coinc_sq = tol_coinc * tol_coinc;
        self.coincident.fill(false);
        let any_coincident = self.populate_spatial_knn(&tree, tol_coinc_sq, true)?;

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
            let _ = self.populate_spatial_knn(&tree, tol_coinc_sq, false)?;
        }

        // Emit active pairs directly into a single PackedSymPair buffer ([FN/A2, A1, FP])
        // without an intermediate f64 pair double buffer.
        let mut packed: Vec<PackedSymPair<S>> = Vec::with_capacity(self.u1.len());
        let mut n_springs = 0usize;
        let mut n_hinges = 0usize;

        // 1. Tier 2: 2-hop False Negatives F- and/or 2-hop Anchors A2.
        let mut n_fn_2hop = 0usize;
        let mut n_anc_2hop = 0usize;
        if self.config.w_fn > 0.0 || self.config.w_anchor2 > 0.0 {
            let mut emit_2hop = |ui: usize, vi: usize, u_u32: u32, v_u32: u32| {
                let dx = self.coords_f64[ui][0] - self.coords_f64[vi][0];
                let dy = self.coords_f64[ui][1] - self.coords_f64[vi][1];
                let d2_curr = fast_hypot_f64(dx, dy);
                let d2_safe = d2_curr.max(self.d_floor);

                let mut emitted_fn = false;
                if self.config.w_fn > 0.0 && !self.in_s_both(ui, vi) {
                    let r_pair_fn = 0.5 * (self.r_knn[ui] + self.r_knn[vi]);
                    let d2_fn_tgt = self
                        .d_floor
                        .max((self.config.fn_shrink * d2_curr).max(0.9 * r_pair_fn));
                    if d2_curr > d2_fn_tgt {
                        emitted_fn = true;
                        let w =
                            (self.config.w_anchor2 + self.config.w_fn) / (d2_fn_tgt * d2_fn_tgt);
                        if push_packed_pair(&mut packed, u_u32, v_u32, false, d2_fn_tgt, w) {
                            n_springs += 1;
                            n_fn_2hop += 1;
                            if self.config.w_anchor2 > 0.0 {
                                n_anc_2hop += 1;
                            }
                        }
                    }
                }
                if !emitted_fn && self.config.w_anchor2 > 0.0 {
                    let w = self.config.w_anchor2 / (d2_safe * d2_safe);
                    if push_packed_pair(&mut packed, u_u32, v_u32, false, d2_safe, w) {
                        n_springs += 1;
                        n_anc_2hop += 1;
                    }
                }
            };

            if let Some(adj2) = &self.adj2 {
                for u in 0..n {
                    let u_u32 = u as u32;
                    for &v_u32 in adj2.neighbors(u) {
                        if u_u32 < v_u32 {
                            emit_2hop(u, v_u32 as usize, u_u32, v_u32);
                        }
                    }
                }
            } else {
                for (&u_u32, &v_u32) in self.u2_hub.iter().zip(self.v2_hub.iter()) {
                    emit_2hop(u_u32 as usize, v_u32 as usize, u_u32, v_u32);
                }
            }
        }

        // 2. Tier 1: 1-hop Anchors A1 (protect 1-hop lengths; gentle w_fn boost for elongated missing 1-hop).
        let mut n_1hop_fn = 0usize;
        let mut n_anc_1hop = 0usize;
        for (&u_u32, &v_u32) in self.u1.iter().zip(self.v1.iter()) {
            let ui = u_u32 as usize;
            let vi = v_u32 as usize;
            let dx = self.coords_f64[ui][0] - self.coords_f64[vi][0];
            let dy = self.coords_f64[ui][1] - self.coords_f64[vi][1];
            let d1_curr = fast_hypot_f64(dx, dy);
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

            if push_packed_pair(&mut packed, u_u32, v_u32, false, d1, w1) {
                n_springs += 1;
                n_anc_1hop += 1;
            }
        }

        // 3. Tier 3: False positives F+ (strictly > 2-hop, one-sided hinge d_curr < d_fp_target).
        let mut n_fp = 0usize;
        if self.config.w_fp > 0.0 && !self.s_indices.is_empty() {
            let fp_is_hinge = self.config.hinge_fp;
            let mut fp_candidates: Vec<(u32, u32)> = Vec::new();
            for u in 0..n {
                let ku = self.row_k(u);
                if ku == 0 {
                    continue;
                }
                let s_start = self.s_indptr[u];
                let s_row = &self.s_indices[s_start..s_start + ku];
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
                let r_pair_fp = if self.row_k(ui) == 0 {
                    self.r_knn[vi]
                } else if self.row_k(vi) == 0 {
                    self.r_knn[ui]
                } else {
                    0.5 * (self.r_knn[ui] + self.r_knn[vi])
                };
                let d_fp_tgt = (self.config.margin_scale * r_pair_fp).max(self.d_floor);
                let dx = self.coords_f64[ui][0] - self.coords_f64[vi][0];
                let dy = self.coords_f64[ui][1] - self.coords_f64[vi][1];
                let d_fp_curr = fast_hypot_f64(dx, dy);
                if d_fp_curr < d_fp_tgt {
                    let w = self.config.w_fp / (d_fp_tgt * d_fp_tgt);
                    if push_packed_pair(&mut packed, u_u32, v_u32, fp_is_hinge, d_fp_tgt, w) {
                        n_fp += 1;
                        if fp_is_hinge {
                            n_hinges += 1;
                        } else {
                            n_springs += 1;
                        }
                    }
                }
            }
        }

        let n_fn = n_fn_2hop + n_1hop_fn;
        let n_anc = n_anc_1hop + n_anc_2hop;
        let n_tot_valid = packed.len();

        // Early stop when zero active intruders/stragglers exist so clean layouts remain untouched.
        if (n_fp == 0 && n_fn == 0) || n_tot_valid == 0 {
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

        condition_weights_packed(&mut packed, MAX_WEIGHT_RATIO);

        let scale_r = self.config.round_decay.powi(r as i32);
        let sched = SgdScheduleConfig {
            t_max: self.config.t_max,
            eps: self.config.eps,
            kind: SgdScheduleKind::Exponential,
            step_scale: scale_r,
        };

        let mut sgd = SgdSolver::<S>::from_packed_symmetric(
            n, packed, n_springs, n_hinges, sched, round_seed,
        )?;
        sgd.run(layout)?;

        self.sync_coords_from_layout(layout);
        self.update_final_scale_stats();

        self.stats.rounds_executed += 1;
        self.stats.n_anchors_total += n_anc;
        self.stats.n_fn_total += n_fn;
        self.stats.n_fp_total += n_fp;
        self.stats.n_pairs_total += n_tot_valid;
        self.stats.round_stats.push(SpatialRefinerRoundStats {
            round: r + 1,
            n_anchors: n_anc,
            n_fn,
            n_fp,
            n_pairs_total: n_tot_valid,
            skipped: false,
        });

        self.round += 1;
        if self.config.k_cap_decay < 1.0 {
            let decayed = ((self.current_k_cap as f64) * self.config.k_cap_decay)
                .round()
                .max(1.0) as usize;
            let min_k = self.config.k_cap_min.min(self.k_cap_capacity);
            self.current_k_cap = decayed.clamp(min_k, self.k_cap_capacity);
            self.k_eff_round = if n > 1 {
                self.current_k_cap.min(n - 1)
            } else {
                0
            };
        }
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
fn fast_hypot_f64(dx: f64, dy: f64) -> f64 {
    let d2 = dx * dx + dy * dy;
    if d2 > 0.0 && d2.is_finite() {
        d2.sqrt()
    } else {
        dx.hypot(dy)
    }
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
            let d = fast_hypot_f64(coords[ui][0] - coords[vi][0], coords[ui][1] - coords[vi][1]);
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

#[inline]
fn push_packed_pair<S: Scalar>(
    packed: &mut Vec<PackedSymPair<S>>,
    u: u32,
    v: u32,
    is_hinge: bool,
    d: f64,
    w: f64,
) -> bool {
    if !(d.is_finite() && d > 0.0 && w.is_finite() && w > 0.0) {
        return false;
    }
    let d_s = S::from_f64(d);
    let w_s = S::from_f64(w);
    if d_s.is_finite() && d_s > S::ZERO && w_s.is_finite() && w_s > S::ZERO {
        packed.push(PackedSymPair::new(u, v, is_hinge, d_s, w_s));
        true
    } else {
        false
    }
}

fn condition_weights_packed<S: Scalar>(pairs: &mut [PackedSymPair<S>], max_ratio: f64) {
    if pairs.len() <= 1 {
        return;
    }
    let mut w_min = f64::INFINITY;
    let mut w_max = 0.0_f64;
    for p in pairs.iter() {
        let w = p.w.to_f64();
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
    let mid = pairs.len() / 2;
    let (_, mid_elem, _) =
        pairs.select_nth_unstable_by(mid, |a, b| a.w.to_f64().total_cmp(&b.w.to_f64()));
    let mid_w = mid_elem.w.to_f64();
    let w_med = if pairs.len() % 2 == 1 {
        mid_w
    } else {
        let mut left_max = f64::NEG_INFINITY;
        for p in &pairs[..mid] {
            let w = p.w.to_f64();
            if w > left_max {
                left_max = w;
            }
        }
        0.5 * (left_max + mid_w)
    };
    let lo = w_med / half_span;
    let hi = w_med * half_span;
    for p in pairs.iter_mut() {
        let clamped = p.w.to_f64().clamp(lo, hi);
        let clamped_s = S::from_f64(clamped);
        if clamped_s.is_finite() && clamped_s > S::ZERO {
            p.w = clamped_s;
        }
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
        assert_eq!(cfg.w_anchor2, 0.0);
        assert_eq!(cfg.w_fn, 0.0);
        assert_eq!(cfg.w_fp, 0.2);
        assert_eq!(cfg.fn_shrink, 0.88);
        assert_eq!(cfg.round_decay, 0.75);
        assert_eq!(cfg.k_cap_decay, 1.0);
        assert_eq!(cfg.k_cap_min, 1);
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
                w_anchor2: -0.1,
                ..cfg
            },
            SpatialRefinerConfig {
                k_cap_decay: 0.0,
                ..cfg
            },
            SpatialRefinerConfig {
                k_cap_decay: 1.5,
                ..cfg
            },
            SpatialRefinerConfig {
                k_cap_min: 0,
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

    #[test]
    fn test_refiner_w_anchor2_and_hub_fallback() {
        // 1. Exact 2-hop path: 3x3 grid with intruder (0, 8).
        let grid = grid_graph(3, 3);
        let mut clean_x = Vec::with_capacity(9);
        let mut clean_y = Vec::with_capacity(9);
        for r in 0..3 {
            for c in 0..3 {
                clean_x.push(c as f64);
                clean_y.push(r as f64);
            }
        }
        let clean_layout = Layout2d::from_xy(clean_x.clone(), clean_y.clone()).unwrap();
        let cfg_a2 = SpatialRefinerConfig {
            rounds: 2,
            w_anchor2: 0.25,
            seed: 42,
            ..Default::default()
        };

        // Clean layout must remain untouched even with w_anchor2 > 0.0.
        let mut l_clean = clean_layout.clone();
        let mut r_clean = SpatialIntruderRefiner::<f64>::new(&grid, cfg_a2).unwrap();
        assert_eq!(r_clean.run(&mut l_clean).unwrap(), SolverStatus::Converged);
        assert_eq!(r_clean.stats().rounds_executed, 0);

        // Intruder layout: w_anchor2 > 0 increases n_anchors relative to w_anchor2 == 0,
        // and step == run is bit-exact in f64 and f32.
        let mut intr_x = clean_x;
        let mut intr_y = clean_y;
        intr_x[8] = 0.1;
        intr_y[8] = 0.1;
        let intr_init = Layout2d::from_xy(intr_x.clone(), intr_y.clone()).unwrap();

        let cfg_base = SpatialRefinerConfig {
            w_anchor2: 0.0,
            ..cfg_a2
        };
        let mut r_base = SpatialIntruderRefiner::<f64>::new(&grid, cfg_base).unwrap();
        let mut l_base = intr_init.clone();
        r_base.run(&mut l_base).unwrap();

        let mut r_run = SpatialIntruderRefiner::<f64>::new(&grid, cfg_a2).unwrap();
        let mut r_step = SpatialIntruderRefiner::<f64>::new(&grid, cfg_a2).unwrap();
        let mut l_run = intr_init.clone();
        let mut l_step = intr_init;
        r_run.run(&mut l_run).unwrap();
        while !r_step.status().is_finished() {
            r_step.step(&mut l_step).unwrap();
        }
        assert!(r_run.stats().n_anchors_total > r_base.stats().n_anchors_total);
        for i in 0..9 {
            assert_eq!(l_run.x()[i].to_bits(), l_step.x()[i].to_bits());
            assert_eq!(l_run.y()[i].to_bits(), l_step.y()[i].to_bits());
        }

        // f32 bit-exact parity
        let intr32 = Layout2d::from_xy(
            intr_x.iter().map(|&v| v as f32).collect(),
            intr_y.iter().map(|&v| v as f32).collect(),
        )
        .unwrap();
        let mut r32_run = SpatialIntruderRefiner::<f32>::new(&grid, cfg_a2).unwrap();
        let mut r32_step = SpatialIntruderRefiner::<f32>::new(&grid, cfg_a2).unwrap();
        let mut l32_run = intr32.clone();
        let mut l32_step = intr32;
        r32_run.run(&mut l32_run).unwrap();
        while !r32_step.status().is_finished() {
            r32_step.step(&mut l32_step).unwrap();
        }
        for i in 0..9 {
            assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
            assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
        }

        // 2. Hub fallback path with w_anchor2 > 0.0 and w_fn == 0.0 must populate u2_hub/v2_hub.
        let n_star = 400usize;
        let mut edges: Vec<(u32, u32)> = (1..n_star as u32).map(|v| (0, v)).collect();
        edges.extend([(399, 400), (400, 401), (401, 402)]);
        let n_hub = 403usize;
        let g_hub = CsrGraph::<f64>::from_edges(n_hub, &edges, None, false, false).unwrap();
        let mut hxs = vec![0.0_f64; n_hub];
        let mut hys = vec![0.0_f64; n_hub];
        for i in 1..n_star {
            let ang = std::f64::consts::TAU * (i as f64) / ((n_star - 1) as f64);
            let (s, c) = ang.sin_cos();
            hxs[i] = 2.0 * c;
            hys[i] = 2.0 * s;
        }
        hxs[400] = hxs[399] + 1.0;
        hys[400] = hys[399];
        hxs[401] = hxs[400] + 1.0;
        hys[401] = hys[400];
        hxs[402] = hxs[1];
        hys[402] = hys[1];

        let cfg_hub_a2 = SpatialRefinerConfig {
            rounds: 1,
            k_cap: 16,
            w_fn: 0.0,
            w_anchor2: 0.25,
            ..Default::default()
        };
        let mut r_hub = SpatialIntruderRefiner::<f64>::new(&g_hub, cfg_hub_a2).unwrap();
        assert!(r_hub.stats().used_hub_fallback);
        assert!(!r_hub.u2_hub.is_empty());
        let mut l_hub = Layout2d::from_xy(hxs, hys).unwrap();
        r_hub.run(&mut l_hub).unwrap();
        assert!(l_hub.is_all_finite());
        assert!(r_hub.stats().n_anchors_total > g_hub.num_edges());
    }

    #[test]
    fn test_refiner_dynamic_k_cap_decay_and_set_k_cap() {
        let grid = grid_graph(4, 4);
        let n = grid.num_nodes();
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for r in 0..4 {
            for c in 0..4 {
                xs.push(c as f64);
                ys.push(r as f64);
            }
        }
        // Place corner node 15 next to node 0 to ensure active rounds.
        xs[15] = 0.05;
        ys[15] = 0.05;
        let init = Layout2d::from_xy(xs, ys).unwrap();

        let cfg = SpatialRefinerConfig {
            rounds: 4,
            k_cap: 12,
            k_cap_decay: 0.5,
            k_cap_min: 4,
            ..Default::default()
        };
        let mut r_run = SpatialIntruderRefiner::<f64>::new(&grid, cfg).unwrap();
        let mut r_step = SpatialIntruderRefiner::<f64>::new(&grid, cfg).unwrap();
        assert_eq!(r_step.current_k_cap(), 12);
        assert!(matches!(
            r_step.set_k_cap(0),
            Err(CoreError::InvalidInput(_))
        ));

        let mut l_run = init.clone();
        let mut l_step = init.clone();
        r_run.run(&mut l_run).unwrap();

        r_step.step(&mut l_step).unwrap();
        assert_eq!(r_step.current_k_cap(), 6);
        r_step.step(&mut l_step).unwrap();
        assert_eq!(r_step.current_k_cap(), 4);
        while !r_step.status().is_finished() {
            r_step.step(&mut l_step).unwrap();
            assert!(r_step.current_k_cap() >= 4);
        }
        for i in 0..n {
            assert_eq!(l_run.x()[i].to_bits(), l_step.x()[i].to_bits());
            assert_eq!(l_run.y()[i].to_bits(), l_step.y()[i].to_bits());
        }

        // Test explicit set_k_cap shrinking and growing past initial capacity.
        let mut r_dyn = SpatialIntruderRefiner::<f64>::new(
            &grid,
            SpatialRefinerConfig {
                rounds: 3,
                k_cap: 6,
                ..Default::default()
            },
        )
        .unwrap();
        let mut l_dyn = init;
        r_dyn.set_k_cap(3).unwrap();
        assert_eq!(r_dyn.current_k_cap(), 3);
        r_dyn.step(&mut l_dyn).unwrap();
        r_dyn.set_k_cap(14).unwrap();
        assert_eq!(r_dyn.current_k_cap(), 14);
        r_dyn.run(&mut l_dyn).unwrap();
        assert!(l_dyn.is_all_finite());
    }

    #[test]
    fn test_condition_weights_packed_zero_alloc_equivalence() {
        for len in [5usize, 6] {
            let raw_weights: Vec<f64> = if len == 5 {
                vec![1e-6, 1.0, 2.0, 4.0, 1e6]
            } else {
                vec![1e-6, 1.0, 2.0, 4.0, 8.0, 1e6]
            };
            let mut pairs: Vec<PackedSymPair<f64>> = raw_weights
                .iter()
                .enumerate()
                .map(|(idx, &w)| PackedSymPair::new(idx as u32, (idx + 1) as u32, false, 1.0, w))
                .collect();

            condition_weights_packed(&mut pairs, MAX_WEIGHT_RATIO);

            let half_span = MAX_WEIGHT_RATIO.sqrt();
            let w_med = if len == 5 { 2.0 } else { 3.0 };
            let lo = w_med / half_span;
            let hi = w_med * half_span;
            let mut got: Vec<f64> = pairs.iter().map(|p| p.w).collect();
            got.sort_by(f64::total_cmp);
            let mut expected: Vec<f64> = raw_weights.iter().map(|&w| w.clamp(lo, hi)).collect();
            expected.sort_by(f64::total_cmp);
            for (g, e) in got.iter().zip(expected.iter()) {
                assert!((g - e).abs() < 1e-12);
            }
        }
    }
}
