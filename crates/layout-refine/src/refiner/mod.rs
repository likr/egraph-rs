//! Pure-Rust $O(N \log N)$ `StaticKdTree` active-set spatial intruder refiner.
//!
//! Pure-Rust 2D $O(N \log N)$ `StaticKdTree` active-set spatial intruder refiner:
//! detects strictly $>2$-hop spatial false positives inside each vertex's $k_u$-NN ball
//! (plus optional 2-hop false negatives), protects 1-hop anchors, conditions pair weights,
//! and executes bounded-scale `SgdSolver` passes via [`SgdPairBuffer`] without distorting
//! already clean layouts.

mod config;
mod helpers;
mod step;
#[cfg(test)]
mod tests_basic;
#[cfg(test)]
mod tests_hubs;
#[cfg(test)]
mod tests_options;

pub use config::{SpatialRefinerConfig, SpatialRefinerRoundStats, SpatialRefinerStats};

#[cfg(doc)]
use egraph_core::IterativeSolver;
use egraph_core::{CoreError, CoreResult, CsrGraph, Layout2d, Scalar, SolverStatus, VisitStamp};
#[cfg(doc)]
use egraph_layout_sgd::SgdPairBuffer;
use egraph_neighborhood::extract_sparse_1hop_2hop_pairs;
use egraph_spatial_trees::{KdQueryScratch, StaticKdTree};
use helpers::{
    build_canonical_undirected_1hop, build_nonhub_2hop_csr_and_k_arr, compute_capped_2hop_sizes,
    compute_coords_std, compute_layout_scale, non_hub_degree_threshold,
};

const EXACT_2HOP_FLOOR_WORK: u128 = 100_000;
const EXACT_2HOP_LINEAR_FACTOR: u128 = 64;

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
    adj1: CsrGraph<()>,
    deg: Vec<usize>,
    non_hub_thresh: usize,
    adj2: Option<CsrGraph<()>>,
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
        let non_hub_thresh = non_hub_degree_threshold(n);
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
        let trivial_early =
            n <= 2 || u1.is_empty() || config.iterations == 0 || config.sgd_iterations == 0;

        let (adj2, u2_hub, v2_hub, k_arr, used_hub_fallback) = if trivial_early {
            (None, Vec::new(), Vec::new(), vec![0usize; n], false)
        } else if use_exact_2hop {
            let (non_hub_adj2, ks) =
                build_nonhub_2hop_csr_and_k_arr(&adj1, &deg, non_hub_thresh, k_eff)?;
            (Some(non_hub_adj2), Vec::new(), Vec::new(), ks, false)
        } else {
            let (u2_hub, v2_hub) = if config.w_fn > 0.0 || config.w_anchor2 > 0.0 {
                let hub_pairs = extract_sparse_1hop_2hop_pairs(&adj1, config.k_cap, config.seed)?;
                let (_, _, u2, v2) = hub_pairs.into_parts();
                (u2, v2)
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

        let status = if config.iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };

        let stats = SpatialRefinerStats {
            rounds_executed: 0,
            n_anchors_total: 0,
            n_fn_total: 0,
            n_fp_total: 0,
            n_sep_total: 0,
            n_pairs_total: 0,
            initial_scale: 1.0,
            final_scale: 1.0,
            d_floor: 1e-3,
            scale_ratio: 1.0,
            used_hub_fallback,
            hinge_fp: config.hinge_fp,
            round_stats: Vec::with_capacity(config.iterations),
        };

        let k_cap_capacity = config.k_cap;
        let current_k_cap = config.k_cap;

        Ok(Self {
            config,
            num_nodes: n,
            u1,
            v1,
            adj1,
            deg,
            non_hub_thresh,
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
        let trivial_early = n <= 2
            || self.u1.is_empty()
            || self.config.iterations == 0
            || self.config.sgd_iterations == 0;
        if k_cap > self.k_cap_capacity && !trivial_early {
            self.k_arr = compute_capped_2hop_sizes(&self.adj1, &self.deg, new_k_eff);
            if self.adj2.is_none() && (self.config.w_fn > 0.0 || self.config.w_anchor2 > 0.0) {
                let hub_pairs =
                    extract_sparse_1hop_2hop_pairs(&self.adj1, k_cap, self.config.seed)?;
                let (_, _, u2, v2) = hub_pairs.into_parts();
                self.u2_hub = u2;
                self.v2_hub = v2;
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
