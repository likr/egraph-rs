//! Pure-Rust Structure-of-Arrays (SoA) `u32` SGD solver, 3-tier sparse barrier builder,
//! streaming non-edge negative sampling, and non-Euclidean [`ManifoldLayout2d`] updates.

use std::f64::consts::PI;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, ManifoldKind,
    ManifoldLayout2d, Scalar, SolverStatus, VisitStamp,
};
use egraph_linalg::{BfsWorkspace, DijkstraWorkspace, select_maxmin_sp_pivots};
use egraph_neighborhood::{
    DogVaradhanConfig, DogVaradhanLandmarks, DogVaradhanStats, Sparse12HopPairs,
    apply_relative_distance_floor, compute_horvitz_thompson_gamma,
    compute_regularized_dog_varadhan_landmarks, extract_sparse_1hop_2hop_pairs,
};

const HINGE_BIT: u32 = 1u32 << 31;
const INDEX_MASK: u32 = !HINGE_BIT;
const MAX_PACKED_NODES: usize = (INDEX_MASK as usize) + 1;
const BLOCK_SIZE: usize = 16_384;
const BLOCK_SHUFFLE_THRESHOLD: usize = 65_536;

/// Learning rate decay schedule family for [`SgdSolver`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SgdScheduleKind {
    /// Exponential decay $\eta(t) = \eta_{\max} \exp(-b t)$.
    #[default]
    Exponential,
    /// Linear decay $\eta(t) = \eta_{\max} - b t$.
    Linear,
    /// Quadratic decay $\eta(t) = \eta_{\max} (1 - b t)^2$.
    Quadratic,
    /// Reciprocal decay $\eta(t) = \eta_{\max} / (1 + b t)$.
    Reciprocal,
    /// Constant learning rate $\eta(t) = \mathrm{step\_scale}$.
    Constant,
}

/// Learning rate schedule parameters for [`SgdSolver`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SgdScheduleConfig {
    /// Total number of scheduled SGD iterations (`default: 60`).
    pub t_max: usize,
    /// Minimum learning rate ratio $\varepsilon > 0$ controlling $\eta_{\min} = \varepsilon / w_{\max}$ (`default: 0.01`).
    pub eps: f64,
    /// Decay curve family (`default: SgdScheduleKind::Exponential`).
    pub kind: SgdScheduleKind,
    /// Multiplicative step scale applied to $\eta(t)$ (`default: 1.0`).
    pub step_scale: f64,
}

impl Default for SgdScheduleConfig {
    fn default() -> Self {
        Self {
            t_max: 60,
            eps: 0.01,
            kind: SgdScheduleKind::Exponential,
            step_scale: 1.0,
        }
    }
}

/// Symmetrization rule used for the 2-hop barrier radius $R_{\mathrm{pair}}(u, v)$.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BarrierSymmetry {
    /// Use $\max(R_2(u), R_2(v))$.
    #[default]
    SymMax,
    /// Use $\frac{1}{2}(R_2(u) + R_2(v))$.
    SymAvg,
}

/// Configuration for Track B 3-tier Sparse Barrier SGD construction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseBarrierConfig {
    /// Hub-safe per-node cap on registered 2-hop pairs (`default: 32`).
    pub max_2hop_per_node: usize,
    /// Whether Tier 3 non-2-hop barrier pairs use one-sided hinge repulsion (`default: true`).
    pub hinge_repulsion: bool,
    /// Relative margin multiplier $\delta_{\mathrm{rel}} \ge 0$ for the 2-hop barrier (`default: 0.25`).
    pub delta_rel: f64,
    /// Symmetrization rule for $R_{\mathrm{pair}}(u, v)$ (`default: BarrierSymmetry::SymMax`).
    pub barrier_sym: BarrierSymmetry,
    /// Tier 1 (1-hop) spring weight multiplier (`default: 1.0`).
    pub omega_1: f64,
    /// Tier 2 (2-hop) spring weight multiplier (`default: 1.0`).
    pub omega_2: f64,
    /// Tier 3 (non-2-hop) weight multiplier (`default: 1.0`).
    pub gamma_non2: f64,
    /// Whether R3 Horvitz-Thompson scaling $\sqrt{N / k_{\mathrm{neg}}}$ is applied (`default: true`).
    pub ht_repulsion: bool,
    /// Relative distance floor multiplier $\mathrm{min\_ratio} \cdot \mathrm{median}(d_1 > 0)$ (`default: 0.5`).
    pub min_ratio: f64,
    /// Absolute minimum target distance clamp $> 0$ (`default: 1e-6`).
    pub eps_dist: f64,
    /// Optional explicit per-node non-2-hop sample budget (`None` uses adaptive 2-hop perplexity in `[8, 32]`).
    pub neg_per_node: Option<usize>,
    /// Number of MaxMin shortest-path pivots used for global non-2-hop coverage (`default: 50`).
    pub num_pivots: usize,
    /// Prioritize `degree >= 2` core vertices during MaxMin pivot selection (`default: true`).
    pub core_pivots: bool,
    /// SGD iteration schedule (`default: SgdScheduleConfig::default()`).
    pub schedule: SgdScheduleConfig,
    /// Deterministic RNG seed (`default: 42`).
    pub seed: u64,
}

impl Default for SparseBarrierConfig {
    fn default() -> Self {
        Self {
            max_2hop_per_node: 32,
            hinge_repulsion: true,
            delta_rel: 0.25,
            barrier_sym: BarrierSymmetry::SymMax,
            omega_1: 1.0,
            omega_2: 1.0,
            gamma_non2: 1.0,
            ht_repulsion: true,
            min_ratio: 0.5,
            eps_dist: 1e-6,
            neg_per_node: None,
            num_pivots: 50,
            core_pivots: true,
            schedule: SgdScheduleConfig::default(),
            seed: 42,
        }
    }
}

/// Summary diagnostics emitted by [`SgdSolver::sparse_barrier_from_landmarks`] and
/// [`SgdSolver::sparse_barrier_dog`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseBarrierStats {
    pub n_pairs_1hop: usize,
    pub n_pairs_2hop: usize,
    pub n_pairs_non2: usize,
    pub n_pairs_hinge: usize,
    pub n_pairs_two_sided: usize,
    pub perplexity_val: f64,
    pub per_node_neg_used: usize,
    pub perplexity_source: &'static str,
    pub d_floor: f64,
    pub gamma_eff: f64,
}

/// Configuration for minimal-RAM on-the-fly streaming non-edge negative sampling SGD.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamingNegativeConfig {
    /// Number of non-edge negatives sampled per vertex on each step (`default: 5`).
    pub negatives_per_node: usize,
    /// Target repulsive separation distance $> 0$ (`default: 2.0`).
    pub target_distance: f64,
    /// Negative pair weight $\ge 0$ (`default: 0.25`).
    pub weight: f64,
    /// Apply one-sided hinge repulsion when `norm < target_distance` (`default: true`).
    pub hinge_only: bool,
}

impl Default for StreamingNegativeConfig {
    fn default() -> Self {
        Self {
            negatives_per_node: 5,
            target_distance: 2.0,
            weight: 0.25,
            hinge_only: true,
        }
    }
}

/// Packed symmetric SGD pair encoding endpoints, hinge flag, target distance, and weight.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PackedSymPair<S: Scalar> {
    pub(crate) i: u32,
    pub(crate) j_hinge: u32,
    pub(crate) d: S,
    pub(crate) w: S,
}

impl<S: Scalar> PackedSymPair<S> {
    #[inline]
    pub(crate) fn new(i: u32, j: u32, is_hinge: bool, d: S, w: S) -> Self {
        debug_assert!(j <= INDEX_MASK);
        Self {
            i,
            j_hinge: if is_hinge { j | HINGE_BIT } else { j },
            d,
            w,
        }
    }

    #[inline]
    fn to_pair_spec(self) -> PairSpec<S> {
        PairSpec {
            i: self.i as usize,
            j: (self.j_hinge & INDEX_MASK) as usize,
            dij: self.d,
            dji: self.d,
            wij: self.w,
            wji: self.w,
            is_hinge: (self.j_hinge & HINGE_BIT) != 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct PackedGenPair<S: Scalar> {
    i: u32,
    j_hinge: u32,
    dij: S,
    dji: S,
    wij: S,
    wji: S,
}

impl<S: Scalar> PackedGenPair<S> {
    #[inline]
    fn new(i: u32, j: u32, is_hinge: bool, dij: S, dji: S, wij: S, wji: S) -> Self {
        debug_assert!(j <= INDEX_MASK);
        Self {
            i,
            j_hinge: if is_hinge { j | HINGE_BIT } else { j },
            dij,
            dji,
            wij,
            wji,
        }
    }

    #[inline]
    fn to_pair_spec(self) -> PairSpec<S> {
        PairSpec {
            i: self.i as usize,
            j: (self.j_hinge & INDEX_MASK) as usize,
            dij: self.dij,
            dji: self.dji,
            wij: self.wij,
            wji: self.wji,
            is_hinge: (self.j_hinge & HINGE_BIT) != 0,
        }
    }
}

#[derive(Debug, Clone)]
enum PairStorage<S: Scalar> {
    Symmetric(Vec<PackedSymPair<S>>),
    General(Vec<PackedGenPair<S>>),
}

impl<S: Scalar> PairStorage<S> {
    #[inline]
    fn len(&self) -> usize {
        match self {
            Self::Symmetric(v) => v.len(),
            Self::General(v) => v.len(),
        }
    }
}

struct DirectedPairAccumulator<S: Scalar> {
    pairs: Vec<PackedGenPair<S>>,
    all_symmetric: bool,
}

impl<S: Scalar> DirectedPairAccumulator<S> {
    fn with_capacity(cap: usize) -> Self {
        Self {
            pairs: Vec::with_capacity(cap),
            all_symmetric: true,
        }
    }

    fn push_f64(&mut self, u: u32, v: u32, d_uv: f64, d_vu: f64, w_uv: f64, w_vu: f64) {
        let d_uv_s = S::from_f64(d_uv);
        let d_vu_s = S::from_f64(d_vu);
        let w_uv_s = S::from_f64(w_uv);
        let w_vu_s = S::from_f64(w_vu);
        if !d_uv_s.is_finite()
            || d_uv_s <= S::ZERO
            || !d_vu_s.is_finite()
            || d_vu_s <= S::ZERO
            || !w_uv_s.is_finite()
            || w_uv_s < S::ZERO
            || !w_vu_s.is_finite()
            || w_vu_s < S::ZERO
            || (w_uv_s == S::ZERO && w_vu_s == S::ZERO)
        {
            return;
        }
        if d_uv_s != d_vu_s || w_uv_s != w_vu_s {
            self.all_symmetric = false;
        }
        self.pairs.push(PackedGenPair::new(
            u, v, false, d_uv_s, d_vu_s, w_uv_s, w_vu_s,
        ));
    }

    fn into_storage(self) -> PairStorage<S> {
        if self.all_symmetric {
            let sym: Vec<PackedSymPair<S>> = self
                .pairs
                .into_iter()
                .map(|p| PackedSymPair {
                    i: p.i,
                    j_hinge: p.j_hinge,
                    d: p.dij,
                    w: p.wij,
                })
                .collect();
            PairStorage::Symmetric(sym)
        } else {
            PairStorage::General(self.pairs)
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct PairSpec<S: Scalar> {
    i: usize,
    j: usize,
    dij: S,
    dji: S,
    wij: S,
    wji: S,
    is_hinge: bool,
}

#[derive(Debug, Clone)]
struct StreamingState<S: Scalar> {
    negatives_per_node: usize,
    target_distance: S,
    weight: S,
    hinge_only: bool,
    adj_indptr: Vec<u32>,
    adj_indices: Vec<u32>,
    node_order: Vec<u32>,
}

/// Contiguous packed Stochastic Gradient Descent (SGD) layout solver.
///
/// Supports symmetric and general two-sided spring pairs, one-sided hinge repulsion pairs,
/// Track B 3-tier sparse barrier schedules, cache-blocked L2-resident shuffling, streaming
/// negative sampling, Euclidean [`Layout2d<S>`] via [`IterativeSolver<S>`], and non-Euclidean
/// [`ManifoldLayout2d<S>`] stepping.
#[derive(Debug, Clone)]
pub struct SgdSolver<S: Scalar = f64> {
    num_nodes: usize,
    pairs: PairStorage<S>,
    num_springs: usize,
    num_hinges: usize,
    block_order: Vec<u32>,
    streaming: Option<StreamingState<S>>,
    schedule_cfg: SgdScheduleConfig,
    eta_min: f64,
    eta_max: f64,
    iter: usize,
    status: SolverStatus,
    rng: DeterministicRng,
}

fn validate_schedule_config(cfg: &SgdScheduleConfig) -> CoreResult<()> {
    if !cfg.eps.is_finite() {
        return Err(CoreError::non_finite("SgdScheduleConfig::eps"));
    }
    if cfg.eps <= 0.0 {
        return Err(CoreError::invalid_input(format!(
            "SgdScheduleConfig::eps must be > 0, got {}",
            cfg.eps
        )));
    }
    if !cfg.step_scale.is_finite() {
        return Err(CoreError::non_finite("SgdScheduleConfig::step_scale"));
    }
    if cfg.step_scale <= 0.0 {
        return Err(CoreError::invalid_input(format!(
            "SgdScheduleConfig::step_scale must be > 0, got {}",
            cfg.step_scale
        )));
    }
    Ok(())
}

fn validate_barrier_config(cfg: &SparseBarrierConfig, has_explicit_pivots: bool) -> CoreResult<()> {
    validate_schedule_config(&cfg.schedule)?;
    for (ctx, val) in [
        ("SparseBarrierConfig::delta_rel", cfg.delta_rel),
        ("SparseBarrierConfig::omega_1", cfg.omega_1),
        ("SparseBarrierConfig::omega_2", cfg.omega_2),
        ("SparseBarrierConfig::gamma_non2", cfg.gamma_non2),
        ("SparseBarrierConfig::min_ratio", cfg.min_ratio),
    ] {
        if !val.is_finite() {
            return Err(CoreError::non_finite(ctx));
        }
        if val < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "{ctx} must be >= 0, got {val}"
            )));
        }
    }
    if !cfg.eps_dist.is_finite() {
        return Err(CoreError::non_finite("SparseBarrierConfig::eps_dist"));
    }
    if cfg.eps_dist <= 0.0 {
        return Err(CoreError::invalid_input(format!(
            "SparseBarrierConfig::eps_dist must be > 0, got {}",
            cfg.eps_dist
        )));
    }
    if !has_explicit_pivots && cfg.num_pivots == 0 {
        return Err(CoreError::invalid_input("num_pivots must be >= 1"));
    }
    Ok(())
}

fn validate_graph_weights<W: Scalar>(graph: &CsrGraph<W>, ctx: &'static str) -> CoreResult<()> {
    if let Some(ws) = graph.weights() {
        for &w in ws {
            if !w.is_finite() {
                return Err(CoreError::non_finite(ctx));
            }
            if w.to_f64() < 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "{ctx}: edge weights must be non-negative"
                )));
            }
        }
    }
    Ok(())
}

fn eta_bounds_from_extrema(
    mut w_min: f64,
    mut w_max: f64,
    extra_weight: Option<f64>,
    eps: f64,
) -> (f64, f64) {
    if let Some(wf) = extra_weight
        && wf.is_finite()
        && wf > 0.0
    {
        w_min = w_min.min(wf);
        w_max = w_max.max(wf);
    }
    if !eps.is_finite() || eps <= 0.0 || w_max <= 0.0 || !w_min.is_finite() {
        return (0.0, 0.0);
    }
    let eta_max = 1.0 / w_min;
    let eta_min = eps / w_max;
    if !eta_min.is_finite() || !eta_max.is_finite() {
        return (0.0, 0.0);
    }
    (eta_min, eta_max)
}

fn compute_eta_bounds<S: Scalar>(
    pairs: &PairStorage<S>,
    extra_weight: Option<S>,
    eps: f64,
) -> (f64, f64) {
    let mut w_min = f64::INFINITY;
    let mut w_max = 0.0_f64;
    match pairs {
        PairStorage::Symmetric(vec) => {
            for p in vec {
                let wf = p.w.to_f64();
                if wf.is_finite() && wf > 0.0 {
                    w_min = w_min.min(wf);
                    w_max = w_max.max(wf);
                }
            }
        }
        PairStorage::General(vec) => {
            for p in vec {
                for w in [p.wij, p.wji] {
                    let wf = w.to_f64();
                    if wf.is_finite() && wf > 0.0 {
                        w_min = w_min.min(wf);
                        w_max = w_max.max(wf);
                    }
                }
            }
        }
    }
    eta_bounds_from_extrema(w_min, w_max, extra_weight.map(|w| w.to_f64()), eps)
}

#[inline]
fn resolve_directed_pair_params(d_uv: f64, d_vu: f64) -> Option<(f64, f64, f64, f64)> {
    let ok_uv = d_uv.is_finite() && d_uv > 0.0;
    let ok_vu = d_vu.is_finite() && d_vu > 0.0;
    match (ok_uv, ok_vu) {
        (true, true) => Some((d_uv, d_vu, 1.0 / (d_uv * d_uv), 1.0 / (d_vu * d_vu))),
        (true, false) => Some((d_uv, d_uv, 1.0 / (d_uv * d_uv), 0.0)),
        (false, true) => Some((d_vu, d_vu, 0.0, 1.0 / (d_vu * d_vu))),
        (false, false) => None,
    }
}

#[inline]
fn fast_hypot<S: Scalar>(dx: S, dy: S) -> S {
    let d2 = dx * dx + dy * dy;
    if d2 > S::ZERO && d2.is_finite() {
        d2.sqrt()
    } else {
        dx.hypot(dy)
    }
}

#[inline]
fn apply_euclidean_sym_pair<S: Scalar>(xs: &mut [S], ys: &mut [S], eta: S, p: PackedSymPair<S>) {
    let i = p.i as usize;
    let j = (p.j_hinge & INDEX_MASK) as usize;
    let is_hinge = (p.j_hinge & HINGE_BIT) != 0;
    let mu = if p.w.is_finite() && p.w > S::ZERO {
        (eta * p.w).min(S::ONE)
    } else {
        S::ZERO
    };
    if !p.d.is_finite() || p.d <= S::ZERO || !mu.is_finite() || mu <= S::ZERO {
        return;
    }
    let dx = xs[i] - xs[j];
    let dy = ys[i] - ys[j];
    let d2 = dx * dx + dy * dy;
    if is_hinge && d2.is_finite() && d2 >= p.d * p.d {
        return;
    }
    let norm = if d2 > S::ZERO && d2.is_finite() {
        d2.sqrt()
    } else {
        dx.hypot(dy)
    };
    if !norm.is_finite() || norm <= S::ZERO {
        return;
    }
    if is_hinge && norm >= p.d {
        return;
    }
    let coeff = S::from_f64(0.5) * (norm - p.d) / norm * mu;
    let step_x = dx * coeff;
    let step_y = dy * coeff;
    xs[i] -= step_x;
    ys[i] -= step_y;
    xs[j] += step_x;
    ys[j] += step_y;
}

#[inline]
fn apply_euclidean_gen_pair<S: Scalar>(xs: &mut [S], ys: &mut [S], eta: S, p: &PackedGenPair<S>) {
    if !p.dij.is_finite() || !p.dji.is_finite() || p.dij <= S::ZERO || p.dji <= S::ZERO {
        return;
    }
    let mu_i = if p.wij.is_finite() && p.wij > S::ZERO {
        (eta * p.wij).min(S::ONE)
    } else {
        S::ZERO
    };
    let mu_j = if p.wji.is_finite() && p.wji > S::ZERO {
        (eta * p.wji).min(S::ONE)
    } else {
        S::ZERO
    };
    if mu_i <= S::ZERO && mu_j <= S::ZERO {
        return;
    }
    let i = p.i as usize;
    let j = (p.j_hinge & INDEX_MASK) as usize;
    let is_hinge = (p.j_hinge & HINGE_BIT) != 0;
    let dx = xs[i] - xs[j];
    let dy = ys[i] - ys[j];
    let d2 = dx * dx + dy * dy;
    if is_hinge && d2.is_finite() && d2 >= p.dij * p.dij && d2 >= p.dji * p.dji {
        return;
    }
    let norm = fast_hypot(dx, dy);
    if !norm.is_finite() || norm <= S::ZERO {
        return;
    }
    let update_i = mu_i > S::ZERO && (!is_hinge || norm < p.dij);
    let update_j = mu_j > S::ZERO && (!is_hinge || norm < p.dji);
    if !update_i && !update_j {
        return;
    }
    let half = S::from_f64(0.5);
    if update_i {
        let coeff_i = half * (norm - p.dij) / norm * mu_i;
        xs[i] -= dx * coeff_i;
        ys[i] -= dy * coeff_i;
    }
    if update_j {
        let coeff_j = half * (norm - p.dji) / norm * mu_j;
        xs[j] += dx * coeff_j;
        ys[j] += dy * coeff_j;
    }
}

#[inline]
fn clamp_manifold_tangent_step<S: Scalar>(kind: ManifoldKind, vx: S, vy: S) -> (S, S) {
    let vxf = vx.to_f64();
    let vyf = vy.to_f64();
    if !vxf.is_finite() || !vyf.is_finite() {
        return (S::ZERO, S::ZERO);
    }
    let len = vxf.hypot(vyf);
    if !len.is_finite() || len == 0.0 {
        return (S::ZERO, S::ZERO);
    }
    let max_step = match kind {
        ManifoldKind::HyperbolicPoincare => 8.0,
        ManifoldKind::Spherical => PI,
        ManifoldKind::Torus2d => 0.5,
    };
    if len > max_step {
        let scale = max_step / len;
        (S::from_f64(vxf * scale), S::from_f64(vyf * scale))
    } else {
        (vx, vy)
    }
}

#[inline]
fn apply_clamped_manifold_vertex_step<S: Scalar>(
    layout: &mut ManifoldLayout2d<S>,
    idx: usize,
    vx: S,
    vy: S,
) -> CoreResult<()> {
    let kind = layout.kind();
    let (cvx, cvy) = clamp_manifold_tangent_step(kind, vx, vy);
    layout.apply_tangent_step(idx, cvx, cvy)?;
    // Guard f32 rounding at Spherical [0, 2*pi) x [0, pi] boundary.
    if kind == ManifoldKind::Spherical {
        let px = layout.x()[idx];
        let py = layout.y()[idx];
        if !px.is_finite() || !py.is_finite() {
            layout.set_point(idx, S::ZERO, S::ZERO)?;
        } else if px >= S::TAU || px < S::ZERO || py < S::ZERO || py > S::PI {
            let safe_lon = if px >= S::TAU || px < S::ZERO {
                S::ZERO
            } else {
                px
            };
            let safe_colat = py.clamp(S::ZERO, S::PI);
            layout.set_point(idx, safe_lon, safe_colat)?;
        }
    }
    Ok(())
}

fn apply_manifold_pair<S: Scalar>(
    layout: &mut ManifoldLayout2d<S>,
    eta: S,
    p: PairSpec<S>,
) -> CoreResult<()> {
    if !p.dij.is_finite() || !p.dji.is_finite() || p.dij <= S::ZERO || p.dji <= S::ZERO {
        return Ok(());
    }
    let mu_i = if p.wij.is_finite() && p.wij > S::ZERO {
        (eta * p.wij).min(S::ONE)
    } else {
        S::ZERO
    };
    let mu_j = if p.wji.is_finite() && p.wji > S::ZERO {
        (eta * p.wji).min(S::ONE)
    } else {
        S::ZERO
    };
    if mu_i <= S::ZERO && mu_j <= S::ZERO {
        return Ok(());
    }
    let (tix, tiy) = layout.tangent_delta(p.i, p.j)?;
    let norm = tix.hypot(tiy);
    if !norm.is_finite() || norm <= S::ZERO {
        return Ok(());
    }
    let update_i = mu_i > S::ZERO && (!p.is_hinge || norm < p.dij);
    let update_j = mu_j > S::ZERO && (!p.is_hinge || norm < p.dji);
    if !update_i && !update_j {
        return Ok(());
    }
    // Evaluate vertex j's tangent vector before mutating vertex i so both endpoints
    // compute gradients from the same pre-step configuration.
    let tj = if update_j {
        Some(layout.tangent_delta(p.j, p.i)?)
    } else {
        None
    };
    let half = S::from_f64(0.5);
    if update_i {
        let coeff_i = half * (norm - p.dij) / norm * mu_i;
        apply_clamped_manifold_vertex_step(layout, p.i, tix * coeff_i, tiy * coeff_i)?;
    }
    if let Some((tjx, tjy)) = tj {
        let norm_j = tjx.hypot(tjy);
        if norm_j.is_finite() && norm_j > S::ZERO {
            let coeff_j = half * (norm_j - p.dji) / norm_j * mu_j;
            apply_clamped_manifold_vertex_step(layout, p.j, tjx * coeff_j, tjy * coeff_j)?;
        }
    }
    Ok(())
}

/// Builds canonical sorted simple symmetric unweighted CSR `(indptr, indices)` from sorted
/// deduplicated upper-triangular pairs `(u1[k], v1[k])` (`u1[k] < v1[k]`) in $O(N + |E|)$.
fn build_symmetric_csr_from_sorted_pairs(
    num_nodes: usize,
    u1: &[u32],
    v1: &[u32],
) -> (Vec<u32>, Vec<u32>) {
    let mut deg = vec![0u32; num_nodes];
    for (&u, &v) in u1.iter().zip(v1.iter()) {
        deg[u as usize] += 1;
        deg[v as usize] += 1;
    }
    let mut indptr = Vec::with_capacity(num_nodes + 1);
    indptr.push(0u32);
    let mut acc = 0u32;
    for &d in &deg {
        acc += d;
        indptr.push(acc);
    }
    let mut cursor = indptr[..num_nodes].to_vec();
    let mut indices = vec![0u32; acc as usize];
    for (&u, &v) in u1.iter().zip(v1.iter()) {
        let pu = cursor[u as usize] as usize;
        indices[pu] = v;
        cursor[u as usize] += 1;

        let pv = cursor[v as usize] as usize;
        indices[pv] = u;
        cursor[v as usize] += 1;
    }
    for u in 0..num_nodes {
        let row = &mut indices[indptr[u] as usize..indptr[u + 1] as usize];
        if !row.windows(2).all(|w| w[0] < w[1]) {
            row.sort_unstable();
        }
    }
    (indptr, indices)
}

/// Extracts canonical deduplicated undirected 1-hop edges `(u, v, weight)` (`u < v`) from `graph`.
fn extract_canonical_undirected_edges<W: Scalar>(graph: &CsrGraph<W>) -> Vec<(u32, u32, f64)> {
    let n = graph.num_nodes();
    if n <= 1 || graph.nnz() == 0 {
        return Vec::new();
    }
    let weights_opt = graph.weights();
    if !graph.is_directed() {
        let mut edges = Vec::with_capacity(graph.num_edges());
        let mut is_sorted = true;
        for u in 0..n {
            let u32_u = u as u32;
            let start = graph.indptr()[u] as usize;
            let end = graph.indptr()[u + 1] as usize;
            let mut prev_v: Option<u32> = None;
            for pos in start..end {
                let v = graph.indices()[pos];
                if let Some(pv) = prev_v
                    && v <= pv
                {
                    is_sorted = false;
                }
                prev_v = Some(v);
                if u32_u < v {
                    let w = weights_opt.map_or(1.0, |ws| ws[pos].to_f64());
                    edges.push((u32_u, v, w));
                }
            }
        }
        if !is_sorted {
            edges.sort_unstable_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)));
        }
        return edges;
    }

    let mut raw = Vec::with_capacity(graph.nnz());
    for u in 0..n {
        let u32_u = u as u32;
        let start = graph.indptr()[u] as usize;
        let end = graph.indptr()[u + 1] as usize;
        for pos in start..end {
            let v = graph.indices()[pos];
            if u32_u == v {
                continue;
            }
            let (a, b) = if u32_u < v { (u32_u, v) } else { (v, u32_u) };
            let w = weights_opt.map_or(1.0, |ws| ws[pos].to_f64());
            raw.push((a, b, w));
        }
    }
    raw.sort_unstable_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)));
    let mut out: Vec<(u32, u32, f64)> = Vec::with_capacity(raw.len());
    for (a, b, w) in raw {
        if let Some(last) = out.last_mut()
            && last.0 == a
            && last.1 == b
        {
            last.2 = last.2.min(w);
        } else {
            out.push((a, b, w));
        }
    }
    out
}

/// Collects sorted deduplicated pairs `(u, v)` (`u < v`) combining all 1-hop edges and
/// `k` deterministically sampled random partners per node.
fn sample_edges_and_random_pairs<W: Scalar>(
    graph: &CsrGraph<W>,
    k: usize,
    seed: u64,
) -> Vec<(u32, u32)> {
    let n = graph.num_nodes();
    let base_edges = extract_canonical_undirected_edges(graph);
    let mut raw_pairs: Vec<(u32, u32)> = Vec::with_capacity(base_edges.len() + n.saturating_mul(k));
    for &(u, v, _) in &base_edges {
        raw_pairs.push((u, v));
    }
    if k > 0 && n > 1 {
        let mut sample_rng = DeterministicRng::from_seed_u64(seed);
        for i in 0..n {
            let i_u32 = i as u32;
            for _ in 0..k {
                let j = sample_rng.gen_range_usize(n) as u32;
                if j != i_u32 {
                    raw_pairs.push(if i_u32 < j { (i_u32, j) } else { (j, i_u32) });
                }
            }
        }
    }
    raw_pairs.sort_unstable();
    raw_pairs.dedup();
    raw_pairs
}

fn compute_adaptive_perplexity_from_pairs(n_nodes: usize, pairs: &Sparse12HopPairs) -> f64 {
    if n_nodes <= 3 {
        return 1.0;
    }
    let upper = 32.0_f64.min((n_nodes - 1) as f64 / 3.0);
    if upper < 1.0 {
        return 1.0;
    }
    let lower = 8.0_f64.min(upper);
    let mut deg2_counts = vec![0u32; n_nodes];
    for slice in [&pairs.u1, &pairs.v1, &pairs.u2, &pairs.v2] {
        for &node in slice {
            deg2_counts[node as usize] += 1;
        }
    }
    deg2_counts.sort_unstable();
    let mid = n_nodes / 2;
    let med = if n_nodes % 2 == 1 {
        deg2_counts[mid] as f64
    } else {
        0.5 * ((deg2_counts[mid - 1] as f64) + (deg2_counts[mid] as f64))
    };
    (1.5 * med).clamp(lower, upper)
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

fn sample_non2hop_pairs(
    n_nodes: usize,
    adj_indptr: &[u32],
    adj_indices: &[u32],
    pivots: &[u32],
    pairs: &Sparse12HopPairs,
    budget: usize,
    seed: u64,
) -> (Vec<u32>, Vec<u32>) {
    if n_nodes <= 2 || budget == 0 {
        return (Vec::new(), Vec::new());
    }
    let n_piv = pivots.len();
    let piv_quota = if n_piv > 0 {
        (budget / 2).max(1).min(n_piv)
    } else {
        0
    };

    let mut rng = DeterministicRng::from_seed_u64(seed.wrapping_add(101));
    let mut stamp = VisitStamp::new(n_nodes);
    let mut neg_stamp = VisitStamp::new(n_nodes);
    let mut raw_non2: Vec<(u32, u32)> = Vec::new();
    let max_trials = 8usize.max(3 * budget);

    for u in 0..n_nodes {
        stamp.clear();
        neg_stamp.clear();
        stamp.mark(u);
        let n1_u = &adj_indices[adj_indptr[u] as usize..adj_indptr[u + 1] as usize];
        for &w in n1_u {
            stamp.mark(w as usize);
        }

        let u_u32 = u as u32;
        let mut collected = 0usize;
        if piv_quota > 0 {
            let start_off = if n_piv > piv_quota {
                (u * 7) % n_piv
            } else {
                0
            };
            let mut checked = 0usize;
            let mut idx_p = start_off;
            while checked < n_piv && collected < piv_quota {
                let v = pivots[idx_p];
                idx_p = (idx_p + 1) % n_piv;
                checked += 1;
                let vi = v as usize;
                if stamp.is_visited(vi) || neg_stamp.is_visited(vi) {
                    continue;
                }
                let n1_v = &adj_indices[adj_indptr[vi] as usize..adj_indptr[vi + 1] as usize];
                if has_common_neighbor(n1_u, n1_v, &stamp) {
                    continue;
                }
                neg_stamp.mark(vi);
                raw_non2.push(if u_u32 < v { (u_u32, v) } else { (v, u_u32) });
                collected += 1;
            }
        }

        if collected < budget {
            for _ in 0..max_trials {
                let v = rng.gen_range_usize(n_nodes) as u32;
                let vi = v as usize;
                if stamp.is_visited(vi) || neg_stamp.is_visited(vi) {
                    continue;
                }
                let n1_v = &adj_indices[adj_indptr[vi] as usize..adj_indptr[vi + 1] as usize];
                if has_common_neighbor(n1_u, n1_v, &stamp) {
                    continue;
                }
                neg_stamp.mark(vi);
                raw_non2.push(if u_u32 < v { (u_u32, v) } else { (v, u_u32) });
                collected += 1;
                if collected >= budget {
                    break;
                }
            }
        }
    }

    raw_non2.sort_unstable();
    raw_non2.dedup();

    // Linear two-pointer disjointness check against sorted 1-hop and 2-hop pair slices.
    let mut i1 = 0usize;
    let mut i2 = 0usize;
    let mut u3 = Vec::with_capacity(raw_non2.len());
    let mut v3 = Vec::with_capacity(raw_non2.len());
    for pair in raw_non2 {
        while i1 < pairs.u1.len() && (pairs.u1[i1], pairs.v1[i1]) < pair {
            i1 += 1;
        }
        if i1 < pairs.u1.len() && (pairs.u1[i1], pairs.v1[i1]) == pair {
            continue;
        }
        while i2 < pairs.u2.len() && (pairs.u2[i2], pairs.v2[i2]) < pair {
            i2 += 1;
        }
        if i2 < pairs.u2.len() && (pairs.u2[i2], pairs.v2[i2]) == pair {
            continue;
        }
        u3.push(pair.0);
        v3.push(pair.1);
    }

    (u3, v3)
}

fn init_pair_blocking<P>(pairs: &mut [P], rng: &mut DeterministicRng) -> Vec<u32> {
    if pairs.len() > BLOCK_SHUFFLE_THRESHOLD {
        rng.shuffle(pairs);
        let num_blocks = pairs.len().div_ceil(BLOCK_SIZE);
        (0..num_blocks as u32).collect()
    } else {
        Vec::new()
    }
}

#[inline]
fn try_for_each_shuffled_pair<P>(
    pairs: &mut [P],
    block_order: &mut [u32],
    rng: &mut DeterministicRng,
    mut apply: impl FnMut(&P) -> CoreResult<()>,
) -> CoreResult<()> {
    let len = pairs.len();
    if len == 0 {
        return Ok(());
    }
    if len <= BLOCK_SHUFFLE_THRESHOLD {
        rng.shuffle(pairs);
        for p in pairs.iter() {
            apply(p)?;
        }
    } else {
        rng.shuffle(block_order);
        for &b in block_order.iter() {
            let start = (b as usize) * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(len);
            let chunk = &mut pairs[start..end];
            rng.shuffle(chunk);
            for p in chunk.iter() {
                apply(p)?;
            }
        }
    }
    Ok(())
}

fn validate_pair_endpoints(
    u: usize,
    v: usize,
    num_nodes: usize,
    idx: usize,
    ctx: &'static str,
) -> CoreResult<()> {
    if u >= num_nodes {
        return Err(CoreError::index_out_of_bounds(u, num_nodes, ctx));
    }
    if v >= num_nodes {
        return Err(CoreError::index_out_of_bounds(v, num_nodes, ctx));
    }
    if u == v {
        return Err(CoreError::invalid_input(format!(
            "{ctx}: self-loop pair ({u}, {u}) at index {idx} is not permitted"
        )));
    }
    Ok(())
}

impl<S: Scalar> SgdSolver<S> {
    pub(crate) fn from_packed_symmetric(
        num_nodes: usize,
        mut pairs: Vec<PackedSymPair<S>>,
        num_springs: usize,
        num_hinges: usize,
        schedule_cfg: SgdScheduleConfig,
        seed: u64,
    ) -> CoreResult<Self> {
        if num_nodes > MAX_PACKED_NODES {
            return Err(CoreError::invalid_input(format!(
                "num_nodes ({num_nodes}) exceeds packed u32 index limit {MAX_PACKED_NODES}"
            )));
        }
        validate_schedule_config(&schedule_cfg)?;
        if num_springs + num_hinges != pairs.len() {
            return Err(CoreError::dimension_mismatch(
                pairs.len(),
                num_springs + num_hinges,
                "SgdSolver::from_packed_symmetric",
            ));
        }

        let mut w_min = f64::INFINITY;
        let mut w_max = 0.0_f64;
        for (idx, p) in pairs.iter().enumerate() {
            let u = p.i as usize;
            let v = (p.j_hinge & INDEX_MASK) as usize;
            validate_pair_endpoints(u, v, num_nodes, idx, "SgdSolver::from_packed_symmetric")?;
            if !p.d.is_finite() {
                return Err(CoreError::non_finite("SgdSolver::from_packed_symmetric::d"));
            }
            if p.d <= S::ZERO {
                return Err(CoreError::invalid_input(format!(
                    "SgdSolver::from_packed_symmetric: target distances must be > 0 at index {idx}"
                )));
            }
            if !p.w.is_finite() {
                return Err(CoreError::non_finite("SgdSolver::from_packed_symmetric::w"));
            }
            if p.w < S::ZERO {
                return Err(CoreError::invalid_input(format!(
                    "SgdSolver::from_packed_symmetric: weights must be >= 0 at index {idx}"
                )));
            }
            let wf = p.w.to_f64();
            if wf > 0.0 {
                w_min = w_min.min(wf);
                w_max = w_max.max(wf);
            }
        }

        let (eta_min, eta_max) = eta_bounds_from_extrema(w_min, w_max, None, schedule_cfg.eps);
        let status = if schedule_cfg.t_max == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };
        let mut rng = DeterministicRng::from_seed_u64(seed);
        let block_order = init_pair_blocking(&mut pairs, &mut rng);

        Ok(Self {
            num_nodes,
            pairs: PairStorage::Symmetric(pairs),
            num_springs,
            num_hinges,
            block_order,
            streaming: None,
            schedule_cfg,
            eta_min,
            eta_max,
            iter: 0,
            status,
            rng,
        })
    }

    fn from_pair_storage(
        num_nodes: usize,
        mut pairs: PairStorage<S>,
        num_springs: usize,
        num_hinges: usize,
        schedule_cfg: SgdScheduleConfig,
        seed: u64,
    ) -> CoreResult<Self> {
        if num_nodes > MAX_PACKED_NODES {
            return Err(CoreError::invalid_input(format!(
                "num_nodes ({num_nodes}) exceeds packed u32 index limit {MAX_PACKED_NODES}"
            )));
        }
        validate_schedule_config(&schedule_cfg)?;
        let (eta_min, eta_max) = compute_eta_bounds(&pairs, None, schedule_cfg.eps);
        let status = if schedule_cfg.t_max == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };
        let mut rng = DeterministicRng::from_seed_u64(seed);
        let block_order = match &mut pairs {
            PairStorage::Symmetric(v) => init_pair_blocking(v, &mut rng),
            PairStorage::General(v) => init_pair_blocking(v, &mut rng),
        };
        Ok(Self {
            num_nodes,
            pairs,
            num_springs,
            num_hinges,
            block_order,
            streaming: None,
            schedule_cfg,
            eta_min,
            eta_max,
            iter: 0,
            status,
            rng,
        })
    }

    fn empty(num_nodes: usize, schedule: SgdScheduleConfig, seed: u64) -> CoreResult<Self> {
        Self::from_packed_symmetric(num_nodes, Vec::new(), 0, 0, schedule, seed)
    }

    /// Constructs an [`SgdSolver`] from explicit flat SoA two-sided spring arrays.
    #[allow(clippy::too_many_arguments)]
    pub fn from_pairs(
        num_nodes: usize,
        src: Vec<u32>,
        tgt: Vec<u32>,
        dij: Vec<S>,
        dji: Vec<S>,
        wij: Vec<S>,
        wji: Vec<S>,
        schedule: SgdScheduleConfig,
        seed: u64,
    ) -> CoreResult<Self> {
        Self::from_pairs_with_hinges(
            num_nodes,
            src,
            tgt,
            dij,
            dji,
            wij,
            wji,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            schedule,
            seed,
        )
    }

    /// Constructs an [`SgdSolver`] combining flat SoA two-sided spring pairs and
    /// one-sided hinge repulsion pairs.
    #[allow(clippy::too_many_arguments)]
    pub fn from_pairs_with_hinges(
        num_nodes: usize,
        spring_src: Vec<u32>,
        spring_tgt: Vec<u32>,
        spring_dij: Vec<S>,
        spring_dji: Vec<S>,
        spring_wij: Vec<S>,
        spring_wji: Vec<S>,
        hinge_src: Vec<u32>,
        hinge_tgt: Vec<u32>,
        hinge_dij: Vec<S>,
        hinge_dji: Vec<S>,
        hinge_wij: Vec<S>,
        hinge_wji: Vec<S>,
        schedule_cfg: SgdScheduleConfig,
        seed: u64,
    ) -> CoreResult<Self> {
        if num_nodes > MAX_PACKED_NODES {
            return Err(CoreError::invalid_input(format!(
                "num_nodes ({num_nodes}) exceeds packed u32 index limit {MAX_PACKED_NODES}"
            )));
        }
        validate_schedule_config(&schedule_cfg)?;

        let validate_soa_slice = |src: &[u32],
                                  tgt: &[u32],
                                  dij: &[S],
                                  dji: &[S],
                                  wij: &[S],
                                  wji: &[S],
                                  ctx: &'static str|
         -> CoreResult<bool> {
            let len = src.len();
            for actual in [tgt.len(), dij.len(), dji.len(), wij.len(), wji.len()] {
                if actual != len {
                    return Err(CoreError::dimension_mismatch(len, actual, ctx));
                }
            }
            let mut is_sym = true;
            for idx in 0..len {
                validate_pair_endpoints(src[idx] as usize, tgt[idx] as usize, num_nodes, idx, ctx)?;
                for d in [dij[idx], dji[idx]] {
                    if !d.is_finite() {
                        return Err(CoreError::non_finite(ctx));
                    }
                    if d <= S::ZERO {
                        return Err(CoreError::invalid_input(format!(
                            "{ctx}: target distances must be > 0 at index {idx}"
                        )));
                    }
                }
                for w in [wij[idx], wji[idx]] {
                    if !w.is_finite() {
                        return Err(CoreError::non_finite(ctx));
                    }
                    if w < S::ZERO {
                        return Err(CoreError::invalid_input(format!(
                            "{ctx}: weights must be >= 0 at index {idx}"
                        )));
                    }
                }
                if dij[idx] != dji[idx] || wij[idx] != wji[idx] {
                    is_sym = false;
                }
            }
            Ok(is_sym)
        };

        let sym_s = validate_soa_slice(
            &spring_src,
            &spring_tgt,
            &spring_dij,
            &spring_dji,
            &spring_wij,
            &spring_wji,
            "SgdSolver::from_pairs_with_hinges::springs",
        )?;
        let sym_h = validate_soa_slice(
            &hinge_src,
            &hinge_tgt,
            &hinge_dij,
            &hinge_dji,
            &hinge_wij,
            &hinge_wji,
            "SgdSolver::from_pairs_with_hinges::hinges",
        )?;

        let n_springs = spring_src.len();
        let n_hinges = hinge_src.len();
        let total = n_springs + n_hinges;

        if sym_s && sym_h {
            let mut pairs = Vec::with_capacity(total);
            for idx in 0..n_springs {
                pairs.push(PackedSymPair::new(
                    spring_src[idx],
                    spring_tgt[idx],
                    false,
                    spring_dij[idx],
                    spring_wij[idx],
                ));
            }
            for idx in 0..n_hinges {
                pairs.push(PackedSymPair::new(
                    hinge_src[idx],
                    hinge_tgt[idx],
                    true,
                    hinge_dij[idx],
                    hinge_wij[idx],
                ));
            }
            Self::from_pair_storage(
                num_nodes,
                PairStorage::Symmetric(pairs),
                n_springs,
                n_hinges,
                schedule_cfg,
                seed,
            )
        } else {
            let mut pairs = Vec::with_capacity(total);
            for idx in 0..n_springs {
                pairs.push(PackedGenPair::new(
                    spring_src[idx],
                    spring_tgt[idx],
                    false,
                    spring_dij[idx],
                    spring_dji[idx],
                    spring_wij[idx],
                    spring_wji[idx],
                ));
            }
            for idx in 0..n_hinges {
                pairs.push(PackedGenPair::new(
                    hinge_src[idx],
                    hinge_tgt[idx],
                    true,
                    hinge_dij[idx],
                    hinge_dji[idx],
                    hinge_wij[idx],
                    hinge_wji[idx],
                ));
            }
            Self::from_pair_storage(
                num_nodes,
                PairStorage::General(pairs),
                n_springs,
                n_hinges,
                schedule_cfg,
                seed,
            )
        }
    }

    /// Constructs a full all-pairs shortest-path SGD solver using [`BfsWorkspace`] for
    /// unweighted graphs and [`DijkstraWorkspace`] for weighted graphs.
    pub fn full<W: Scalar>(
        graph: &CsrGraph<W>,
        schedule: SgdScheduleConfig,
        seed: u64,
    ) -> CoreResult<Self> {
        validate_schedule_config(&schedule)?;
        validate_graph_weights(graph, "SgdSolver::full")?;

        let n = graph.num_nodes();
        if n <= 1 {
            return Self::empty(n, schedule, seed);
        }

        let cap = n.saturating_mul(n.saturating_sub(1)) / 2;
        if !graph.is_directed() {
            let mut pairs = Vec::with_capacity(cap);
            let mut dist_row = vec![f64::INFINITY; n];
            let mut bfs = (!graph.is_weighted()).then(|| BfsWorkspace::new(n));
            let mut dijk = graph.is_weighted().then(|| DijkstraWorkspace::new(n));
            for i in 0..(n - 1) {
                if let Some(ws) = bfs.as_mut() {
                    ws.bfs_from(graph, i, &mut dist_row)?;
                } else if let Some(ws) = dijk.as_mut() {
                    ws.dijkstra_from(graph, i, &mut dist_row)?;
                }
                for (j, &d) in dist_row.iter().enumerate().skip(i + 1) {
                    if d.is_finite() && d > 0.0 {
                        let d_s = S::from_f64(d);
                        let w_s = S::from_f64(1.0 / (d * d));
                        if d_s.is_finite() && d_s > S::ZERO && w_s.is_finite() && w_s > S::ZERO {
                            pairs.push(PackedSymPair::new(i as u32, j as u32, false, d_s, w_s));
                        }
                    }
                }
            }
            let n_springs = pairs.len();
            return Self::from_packed_symmetric(n, pairs, n_springs, 0, schedule, seed);
        }

        let total = n
            .checked_mul(n)
            .ok_or_else(|| CoreError::invalid_input("full APSP distance matrix overflow"))?;
        let mut dists = vec![f64::INFINITY; total];
        let mut bfs = (!graph.is_weighted()).then(|| BfsWorkspace::new(n));
        let mut dijk = graph.is_weighted().then(|| DijkstraWorkspace::new(n));
        for u in 0..n {
            let row = &mut dists[u * n..(u + 1) * n];
            if let Some(ws) = bfs.as_mut() {
                ws.bfs_from(graph, u, row)?;
            } else if let Some(ws) = dijk.as_mut() {
                ws.dijkstra_from(graph, u, row)?;
            }
        }
        let mut acc = DirectedPairAccumulator::with_capacity(cap);
        for i in 0..(n - 1) {
            for j in (i + 1)..n {
                if let Some((duv, dvu, wuv, wvu)) =
                    resolve_directed_pair_params(dists[i * n + j], dists[j * n + i])
                {
                    acc.push_f64(i as u32, j as u32, duv, dvu, wuv, wvu);
                }
            }
        }
        let storage = acc.into_storage();
        let n_springs = storage.len();
        Self::from_pair_storage(n, storage, n_springs, 0, schedule, seed)
    }

    /// Constructs a Track B 3-tier Sparse Barrier SGD solver directly from precomputed
    /// regularized DoG Varadhan landmark tables.
    pub fn sparse_barrier_from_landmarks<W: Scalar>(
        graph: &CsrGraph<W>,
        landmarks: &DogVaradhanLandmarks,
        pivots: Option<&[u32]>,
        config: &SparseBarrierConfig,
    ) -> CoreResult<(Self, SparseBarrierStats)> {
        validate_barrier_config(config, pivots.is_some())?;
        let n = graph.num_nodes();
        if landmarks.num_nodes() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                landmarks.num_nodes(),
                "SgdSolver::sparse_barrier_from_landmarks::landmarks",
            ));
        }
        let k_lm = landmarks.num_landmarks();
        if n > 0 && k_lm == 0 {
            return Err(CoreError::invalid_input(
                "landmarks must have at least 1 landmark when num_nodes > 0",
            ));
        }
        let expected = n
            .checked_mul(k_lm)
            .ok_or_else(|| CoreError::invalid_input("landmark table size overflow"))?;
        if landmarks.d_dog().len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                landmarks.d_dog().len(),
                "DogVaradhanLandmarks::d_dog",
            ));
        }
        if landmarks.d_dog().iter().any(|v| !v.is_finite()) {
            return Err(CoreError::non_finite("DogVaradhanLandmarks::d_dog"));
        }
        if let Some(dc) = landmarks.d_coarse() {
            if dc.len() != expected {
                return Err(CoreError::dimension_mismatch(
                    expected,
                    dc.len(),
                    "DogVaradhanLandmarks::d_coarse",
                ));
            }
            if dc.iter().any(|v| !v.is_finite()) {
                return Err(CoreError::non_finite("DogVaradhanLandmarks::d_coarse"));
            }
        }
        for &lm in landmarks.landmarks() {
            if (lm as usize) >= n {
                return Err(CoreError::index_out_of_bounds(
                    lm as usize,
                    n,
                    "DogVaradhanLandmarks::landmarks",
                ));
            }
        }

        let unique_pivots: Vec<u32> = match pivots {
            Some(supplied) => {
                let mut seen = VisitStamp::new(n);
                let mut out = Vec::with_capacity(supplied.len());
                for &p in supplied {
                    let pi = p as usize;
                    if pi >= n {
                        return Err(CoreError::index_out_of_bounds(
                            pi,
                            n,
                            "SgdSolver::sparse_barrier_from_landmarks::pivots",
                        ));
                    }
                    if seen.visit(pi) {
                        out.push(p);
                    }
                }
                out
            }
            None => {
                let min_deg = if config.core_pivots { 2 } else { 1 };
                select_maxmin_sp_pivots(graph, config.num_pivots, config.seed, min_deg)?
            }
        };

        if n <= 1 {
            let (perplexity_val, per_node_neg_used, perplexity_source) = match config.neg_per_node {
                Some(k) => (k as f64, k, "explicit"),
                None => (1.0, 1, "adaptive_2hop"),
            };
            let gamma_eff = if config.ht_repulsion {
                compute_horvitz_thompson_gamma(config.gamma_non2, n, per_node_neg_used)?
            } else {
                config.gamma_non2
            };
            let solver = Self::empty(n, config.schedule, config.seed)?;
            let stats = SparseBarrierStats {
                n_pairs_1hop: 0,
                n_pairs_2hop: 0,
                n_pairs_non2: 0,
                n_pairs_hinge: 0,
                n_pairs_two_sided: 0,
                perplexity_val,
                per_node_neg_used,
                perplexity_source,
                d_floor: config.eps_dist,
                gamma_eff,
            };
            return Ok((solver, stats));
        }

        let pairs = extract_sparse_1hop_2hop_pairs(graph, config.max_2hop_per_node, config.seed)?;
        let (perplexity_val, per_node_neg_used, perplexity_source) = match config.neg_per_node {
            Some(k) => (k as f64, k, "explicit"),
            None => {
                let p_val = compute_adaptive_perplexity_from_pairs(n, &pairs);
                let neg = if n > 3 {
                    (p_val.round_ties_even().max(0.0) as usize).max(1)
                } else {
                    1
                };
                (p_val, neg, "adaptive_2hop")
            }
        };

        let (u3, v3) = if n > 2 && per_node_neg_used > 0 {
            let (adj_indptr, adj_indices) =
                build_symmetric_csr_from_sorted_pairs(n, &pairs.u1, &pairs.v1);
            sample_non2hop_pairs(
                n,
                &adj_indptr,
                &adj_indices,
                &unique_pivots,
                &pairs,
                per_node_neg_used,
                config.seed,
            )
        } else {
            (Vec::new(), Vec::new())
        };

        let mut d1 = landmarks.alt_pair_distances(&pairs.u1, &pairs.v1)?;
        let mut d2 = landmarks.alt_pair_distances(&pairs.u2, &pairs.v2)?;
        let mut d3 = landmarks.alt_pair_distances(&u3, &v3)?;
        let d_floor = apply_relative_distance_floor(
            &mut d1,
            &mut d2,
            &mut d3,
            config.eps_dist,
            config.min_ratio,
        )?;

        let gamma_eff = if config.ht_repulsion {
            compute_horvitz_thompson_gamma(config.gamma_non2, n, per_node_neg_used)?
        } else {
            config.gamma_non2
        };

        let mut r2 = vec![0.0_f64; n];
        let mut packed_pairs = Vec::with_capacity(pairs.u1.len() + pairs.u2.len() + u3.len());
        let mut n_pairs_two_sided = 0usize;
        let mut n_pairs_hinge = 0usize;

        for (pairs_u, pairs_v, dists, omega) in [
            (&pairs.u1, &pairs.v1, &d1, config.omega_1),
            (&pairs.u2, &pairs.v2, &d2, config.omega_2),
        ] {
            for idx in 0..pairs_u.len() {
                let u = pairs_u[idx];
                let v = pairs_v[idx];
                let d = dists[idx];
                r2[u as usize] = r2[u as usize].max(d);
                r2[v as usize] = r2[v as usize].max(d);
                if omega > 0.0 {
                    let d_s = S::from_f64(d);
                    let w_s = S::from_f64(omega / (d * d));
                    if d_s.is_finite() && d_s > S::ZERO && w_s.is_finite() && w_s > S::ZERO {
                        packed_pairs.push(PackedSymPair::new(u, v, false, d_s, w_s));
                        n_pairs_two_sided += 1;
                    }
                }
            }
        }

        let barrier_mult = 1.0 + config.delta_rel;
        for idx in 0..u3.len() {
            let u = u3[idx];
            let v = v3[idx];
            let r_pair = match config.barrier_sym {
                BarrierSymmetry::SymMax => r2[u as usize].max(r2[v as usize]),
                BarrierSymmetry::SymAvg => 0.5 * (r2[u as usize] + r2[v as usize]),
            };
            let d3_bar = d3[idx].max(barrier_mult * r_pair).max(config.eps_dist);
            if gamma_eff > 0.0 {
                let d_s = S::from_f64(d3_bar);
                let w_s = S::from_f64(gamma_eff / (d3_bar * d3_bar));
                if d_s.is_finite() && d_s > S::ZERO && w_s.is_finite() && w_s > S::ZERO {
                    packed_pairs.push(PackedSymPair::new(u, v, config.hinge_repulsion, d_s, w_s));
                    if config.hinge_repulsion {
                        n_pairs_hinge += 1;
                    } else {
                        n_pairs_two_sided += 1;
                    }
                }
            }
        }

        let solver = Self::from_packed_symmetric(
            n,
            packed_pairs,
            n_pairs_two_sided,
            n_pairs_hinge,
            config.schedule,
            config.seed,
        )?;
        let stats = SparseBarrierStats {
            n_pairs_1hop: pairs.u1.len(),
            n_pairs_2hop: pairs.u2.len(),
            n_pairs_non2: u3.len(),
            n_pairs_hinge,
            n_pairs_two_sided,
            perplexity_val,
            per_node_neg_used,
            perplexity_source,
            d_floor,
            gamma_eff,
        };
        Ok((solver, stats))
    }

    /// Computes regularized Chebyshev DoG Varadhan landmarks and constructs a Track B
    /// 3-tier Sparse Barrier SGD solver in one call.
    pub fn sparse_barrier_dog<W: Scalar>(
        graph: &CsrGraph<W>,
        dog_config: &DogVaradhanConfig,
        barrier_config: &SparseBarrierConfig,
    ) -> CoreResult<(Self, SparseBarrierStats, DogVaradhanStats)> {
        validate_barrier_config(barrier_config, false)?;
        let (landmarks, dog_stats) = compute_regularized_dog_varadhan_landmarks(graph, dog_config)?;
        let (solver, barrier_stats) =
            Self::sparse_barrier_from_landmarks(graph, &landmarks, None, barrier_config)?;
        Ok((solver, barrier_stats, dog_stats))
    }

    /// Constructs a random-pair sparse SGD solver combining all 1-hop edges with up to
    /// `pairs_per_node` sampled node pairs per vertex using exact shortest-path distances.
    pub fn random_pair<W: Scalar>(
        graph: &CsrGraph<W>,
        pairs_per_node: usize,
        schedule: SgdScheduleConfig,
        seed: u64,
    ) -> CoreResult<Self> {
        validate_schedule_config(&schedule)?;
        validate_graph_weights(graph, "SgdSolver::random_pair")?;

        let n = graph.num_nodes();
        if n <= 1 {
            return Self::empty(n, schedule, seed);
        }

        let raw_pairs = sample_edges_and_random_pairs(graph, pairs_per_node, seed);
        let m = raw_pairs.len();
        let mut dist_row = vec![f64::INFINITY; n];
        let mut bfs = (!graph.is_weighted()).then(|| BfsWorkspace::new(n));
        let mut dijk = graph.is_weighted().then(|| DijkstraWorkspace::new(n));

        if !graph.is_directed() {
            let mut pairs = Vec::with_capacity(m);
            let mut idx = 0usize;
            while idx < m {
                let u = raw_pairs[idx].0;
                if let Some(ws) = bfs.as_mut() {
                    ws.bfs_from(graph, u as usize, &mut dist_row)?;
                } else if let Some(ws) = dijk.as_mut() {
                    ws.dijkstra_from(graph, u as usize, &mut dist_row)?;
                }
                while idx < m && raw_pairs[idx].0 == u {
                    let v = raw_pairs[idx].1;
                    let d = dist_row[v as usize];
                    if d.is_finite() && d > 0.0 {
                        let d_s = S::from_f64(d);
                        let w_s = S::from_f64(1.0 / (d * d));
                        if d_s.is_finite() && d_s > S::ZERO && w_s.is_finite() && w_s > S::ZERO {
                            pairs.push(PackedSymPair::new(u, v, false, d_s, w_s));
                        }
                    }
                    idx += 1;
                }
            }
            let n_springs = pairs.len();
            return Self::from_packed_symmetric(n, pairs, n_springs, 0, schedule, seed);
        }

        let mut d_uv_arr = vec![f64::INFINITY; m];
        let mut d_vu_arr = vec![f64::INFINITY; m];
        let mut queries: Vec<(u32, u32, usize, bool)> = Vec::with_capacity(m * 2);
        for (slot, &(u, v)) in raw_pairs.iter().enumerate() {
            queries.push((u, v, slot, true));
            queries.push((v, u, slot, false));
        }
        queries.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        let mut q_idx = 0usize;
        while q_idx < queries.len() {
            let s = queries[q_idx].0;
            if let Some(ws) = bfs.as_mut() {
                ws.bfs_from(graph, s as usize, &mut dist_row)?;
            } else if let Some(ws) = dijk.as_mut() {
                ws.dijkstra_from(graph, s as usize, &mut dist_row)?;
            }
            while q_idx < queries.len() && queries[q_idx].0 == s {
                let (_, t, slot, is_fwd) = queries[q_idx];
                let d = dist_row[t as usize];
                if is_fwd {
                    d_uv_arr[slot] = d;
                } else {
                    d_vu_arr[slot] = d;
                }
                q_idx += 1;
            }
        }

        let mut acc = DirectedPairAccumulator::with_capacity(m);
        for (slot, &(u, v)) in raw_pairs.iter().enumerate() {
            if let Some((duv, dvu, wuv, wvu)) =
                resolve_directed_pair_params(d_uv_arr[slot], d_vu_arr[slot])
            {
                acc.push_f64(u, v, duv, dvu, wuv, wvu);
            }
        }
        let storage = acc.into_storage();
        let n_springs = storage.len();
        Self::from_pair_storage(n, storage, n_springs, 0, schedule, seed)
    }

    /// Constructs an Omega SGD solver using Euclidean distances in a precomputed row-major
    /// `(num_nodes, dims)` spectral embedding for all 1-hop edges and `k` random partners per node.
    pub fn omega<W: Scalar>(
        graph: &CsrGraph<W>,
        embedding_flat: &[f64],
        dims: usize,
        k: usize,
        min_dist: f64,
        schedule: SgdScheduleConfig,
        seed: u64,
    ) -> CoreResult<Self> {
        validate_schedule_config(&schedule)?;
        if dims == 0 {
            return Err(CoreError::invalid_input("omega dims must be >= 1"));
        }
        if !min_dist.is_finite() {
            return Err(CoreError::non_finite("SgdSolver::omega::min_dist"));
        }
        if min_dist <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "omega min_dist must be > 0, got {min_dist}"
            )));
        }

        let n = graph.num_nodes();
        let expected = n
            .checked_mul(dims)
            .ok_or_else(|| CoreError::invalid_input("omega embedding size overflow"))?;
        if embedding_flat.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                embedding_flat.len(),
                "SgdSolver::omega::embedding_flat",
            ));
        }
        if embedding_flat.iter().any(|v| !v.is_finite()) {
            return Err(CoreError::non_finite("SgdSolver::omega::embedding_flat"));
        }
        if n <= 1 {
            return Self::empty(n, schedule, seed);
        }

        let raw_pairs = sample_edges_and_random_pairs(graph, k, seed);
        let mut pairs = Vec::with_capacity(raw_pairs.len());
        for &(u, v) in &raw_pairs {
            let row_u = &embedding_flat[(u as usize) * dims..((u as usize) + 1) * dims];
            let row_v = &embedding_flat[(v as usize) * dims..((v as usize) + 1) * dims];
            let mut sq_sum = 0.0_f64;
            for (&a, &b) in row_u.iter().zip(row_v.iter()) {
                let diff = a - b;
                sq_sum += diff * diff;
            }
            let d = sq_sum.sqrt().max(min_dist);
            let d_s = S::from_f64(d);
            let w_s = S::from_f64(1.0 / (d * d));
            if d_s.is_finite() && d_s > S::ZERO && w_s.is_finite() && w_s > S::ZERO {
                pairs.push(PackedSymPair::new(u, v, false, d_s, w_s));
            }
        }

        let n_springs = pairs.len();
        Self::from_packed_symmetric(n, pairs, n_springs, 0, schedule, seed)
    }

    /// Constructs a minimal-RAM SGD solver storing only 1-hop edges in flat packed pairs and
    /// sampling $O(|V| \cdot \mathrm{negatives\_per\_node})$ non-edge negative updates on-the-fly
    /// each step without preallocating negative pair buffers.
    pub fn streaming_negatives<W: Scalar>(
        graph: &CsrGraph<W>,
        neg_config: StreamingNegativeConfig,
        schedule: SgdScheduleConfig,
        seed: u64,
    ) -> CoreResult<Self> {
        validate_schedule_config(&schedule)?;
        if !neg_config.target_distance.is_finite() {
            return Err(CoreError::non_finite(
                "StreamingNegativeConfig::target_distance",
            ));
        }
        if neg_config.target_distance <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "target_distance must be > 0, got {}",
                neg_config.target_distance
            )));
        }
        if !neg_config.weight.is_finite() {
            return Err(CoreError::non_finite("StreamingNegativeConfig::weight"));
        }
        if neg_config.weight < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "weight must be >= 0, got {}",
                neg_config.weight
            )));
        }
        validate_graph_weights(graph, "SgdSolver::streaming_negatives")?;

        let n = graph.num_nodes();
        if n <= 1 {
            return Self::empty(n, schedule, seed);
        }

        let target_s = S::from_f64(neg_config.target_distance);
        let weight_s = S::from_f64(neg_config.weight);
        if !target_s.is_finite() || target_s <= S::ZERO || !weight_s.is_finite() {
            return Err(CoreError::numerical_breakdown(
                "StreamingNegativeConfig parameters overflowed scalar precision",
            ));
        }

        let canonical_edges = extract_canonical_undirected_edges(graph);
        let m = canonical_edges.len();
        let mut pairs = Vec::with_capacity(m);
        let mut u1 = Vec::with_capacity(m);
        let mut v1 = Vec::with_capacity(m);

        for &(u, v, edge_len) in &canonical_edges {
            u1.push(u);
            v1.push(v);
            if edge_len.is_finite() && edge_len > 0.0 {
                let d_s = S::from_f64(edge_len);
                let w_s = S::from_f64(1.0 / (edge_len * edge_len));
                if d_s.is_finite() && d_s > S::ZERO && w_s.is_finite() && w_s > S::ZERO {
                    pairs.push(PackedSymPair::new(u, v, false, d_s, w_s));
                }
            }
        }

        let n_springs = pairs.len();
        let mut solver = Self::from_packed_symmetric(n, pairs, n_springs, 0, schedule, seed)?;
        if neg_config.negatives_per_node > 0 && weight_s > S::ZERO {
            let (adj_indptr, adj_indices) = build_symmetric_csr_from_sorted_pairs(n, &u1, &v1);
            let node_order: Vec<u32> = (0..n as u32).collect();
            solver.streaming = Some(StreamingState {
                negatives_per_node: neg_config.negatives_per_node,
                target_distance: target_s,
                weight: weight_s,
                hinge_only: neg_config.hinge_only,
                adj_indptr,
                adj_indices,
                node_order,
            });
            let (eta_min, eta_max) =
                compute_eta_bounds(&solver.pairs, Some(weight_s), solver.schedule_cfg.eps);
            solver.eta_min = eta_min;
            solver.eta_max = eta_max;
        }

        Ok(solver)
    }

    /// Returns the expected number of vertices in the layout.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the number of two-sided spring pairs stored.
    #[inline]
    pub fn num_springs(&self) -> usize {
        self.num_springs
    }

    /// Returns the number of one-sided hinge repulsion pairs stored.
    #[inline]
    pub fn num_hinges(&self) -> usize {
        self.num_hinges
    }

    /// Returns the learning rate $\eta$ for the upcoming iteration (`0.0` if finished).
    pub fn current_eta(&self) -> f64 {
        self.eta_at(self.iter)
    }

    fn eta_at(&self, t: usize) -> f64 {
        let t_max = self.schedule_cfg.t_max;
        if t_max == 0 || t >= t_max {
            return 0.0;
        }
        let scale = self.schedule_cfg.step_scale;
        match self.schedule_cfg.kind {
            SgdScheduleKind::Constant => scale,
            SgdScheduleKind::Linear => {
                if self.eta_max <= 0.0 {
                    return 0.0;
                }
                let b = if t_max <= 1 {
                    0.0
                } else {
                    (self.eta_max - self.eta_min) / ((t_max - 1) as f64)
                };
                scale * (self.eta_max - b * (t as f64)).max(0.0)
            }
            SgdScheduleKind::Quadratic => {
                if self.eta_min <= 0.0 || self.eta_max <= 0.0 {
                    return 0.0;
                }
                let b = if t_max <= 1 {
                    0.0
                } else {
                    (1.0 - (self.eta_min / self.eta_max).sqrt()) / ((t_max - 1) as f64)
                };
                let factor = 1.0 - b * (t as f64);
                scale * self.eta_max * factor * factor
            }
            SgdScheduleKind::Exponential => {
                if self.eta_min <= 0.0 || self.eta_max <= 0.0 {
                    return 0.0;
                }
                let b = if t_max <= 1 {
                    0.0
                } else {
                    (self.eta_max / self.eta_min).ln() / ((t_max - 1) as f64)
                };
                scale * self.eta_max * (-b * (t as f64)).exp()
            }
            SgdScheduleKind::Reciprocal => {
                if self.eta_min <= 0.0 || self.eta_max <= 0.0 {
                    return 0.0;
                }
                let b = if t_max <= 1 {
                    0.0
                } else {
                    (self.eta_max / self.eta_min - 1.0) / ((t_max - 1) as f64)
                };
                scale * self.eta_max / (1.0 + b * (t as f64))
            }
        }
    }

    fn run_streaming_negatives<F>(&mut self, eta: S, mut apply_sym: F) -> CoreResult<()>
    where
        F: FnMut(S, PackedSymPair<S>) -> CoreResult<()>,
    {
        if let Some(stream) = &mut self.streaming
            && stream.negatives_per_node > 0
            && stream.weight > S::ZERO
        {
            self.rng.shuffle(&mut stream.node_order);
            let max_trials = stream.negatives_per_node.saturating_mul(4).max(8);
            for &u_u32 in &stream.node_order {
                let row_start = stream.adj_indptr[u_u32 as usize] as usize;
                let row_end = stream.adj_indptr[(u_u32 as usize) + 1] as usize;
                let adj_u = &stream.adj_indices[row_start..row_end];
                if adj_u.len() + 1 >= self.num_nodes {
                    continue;
                }
                let mut accepted = 0usize;
                let mut trials = 0usize;
                while accepted < stream.negatives_per_node && trials < max_trials {
                    trials += 1;
                    let v_u32 = self.rng.gen_range_usize(self.num_nodes) as u32;
                    if v_u32 == u_u32 || adj_u.binary_search(&v_u32).is_ok() {
                        continue;
                    }
                    apply_sym(
                        eta,
                        PackedSymPair::new(
                            u_u32,
                            v_u32,
                            stream.hinge_only,
                            stream.target_distance,
                            stream.weight,
                        ),
                    )?;
                    accepted += 1;
                }
            }
        }
        Ok(())
    }

    fn finish_iteration(&mut self) -> SolverStatus {
        self.iter += 1;
        self.status = if self.iter >= self.schedule_cfg.t_max {
            SolverStatus::Completed
        } else {
            SolverStatus::Running
        };
        self.status
    }

    /// Executes one SGD iteration on a non-Euclidean [`ManifoldLayout2d<S>`] across
    /// `HyperbolicPoincare`, `Spherical`, and `Torus2d` geometries.
    pub fn step_manifold(&mut self, layout: &mut ManifoldLayout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "SgdSolver::step_manifold",
            ));
        }
        layout.as_layout().validate_finite()?;
        if self.status.is_finished() {
            return Ok(self.status);
        }

        let eta_f64 = self.eta_at(self.iter);
        if self.num_nodes > 1 && eta_f64.is_finite() && eta_f64 > 0.0 {
            let eta = S::from_f64(eta_f64);
            match &mut self.pairs {
                PairStorage::Symmetric(pairs) => {
                    try_for_each_shuffled_pair(pairs, &mut self.block_order, &mut self.rng, |p| {
                        apply_manifold_pair(layout, eta, p.to_pair_spec())
                    })?;
                }
                PairStorage::General(pairs) => {
                    try_for_each_shuffled_pair(pairs, &mut self.block_order, &mut self.rng, |p| {
                        apply_manifold_pair(layout, eta, p.to_pair_spec())
                    })?;
                }
            }
            self.run_streaming_negatives(eta, |e, p| {
                apply_manifold_pair(layout, e, p.to_pair_spec())
            })?;
        }

        Ok(self.finish_iteration())
    }

    /// Executes up to `steps` manifold SGD iterations or stops when finished.
    pub fn run_steps_manifold(
        &mut self,
        layout: &mut ManifoldLayout2d<S>,
        steps: usize,
    ) -> CoreResult<SolverStatus> {
        for _ in 0..steps {
            if self.status.is_finished() {
                break;
            }
            let status = self.step_manifold(layout)?;
            if status.is_finished() {
                break;
            }
        }
        Ok(self.status)
    }

    /// Runs manifold SGD optimization to completion.
    pub fn run_manifold(&mut self, layout: &mut ManifoldLayout2d<S>) -> CoreResult<SolverStatus> {
        while !self.status.is_finished() {
            let status = self.step_manifold(layout)?;
            if status.is_finished() {
                break;
            }
        }
        Ok(self.status)
    }
}

impl<S: Scalar> IterativeSolver<S> for SgdSolver<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "SgdSolver::step",
            ));
        }
        layout.validate_finite()?;
        if self.status.is_finished() {
            return Ok(self.status);
        }

        let eta_f64 = self.eta_at(self.iter);
        if self.num_nodes > 1 && eta_f64.is_finite() && eta_f64 > 0.0 {
            let eta = S::from_f64(eta_f64);
            let (xs, ys) = layout.xy_mut();
            match &mut self.pairs {
                PairStorage::Symmetric(pairs) => {
                    try_for_each_shuffled_pair(
                        pairs,
                        &mut self.block_order,
                        &mut self.rng,
                        |&p| {
                            apply_euclidean_sym_pair(xs, ys, eta, p);
                            Ok(())
                        },
                    )?;
                }
                PairStorage::General(pairs) => {
                    try_for_each_shuffled_pair(pairs, &mut self.block_order, &mut self.rng, |p| {
                        apply_euclidean_gen_pair(xs, ys, eta, p);
                        Ok(())
                    })?;
                }
            }
            let (xs, ys) = layout.xy_mut();
            self.run_streaming_negatives(eta, |e, p| {
                apply_euclidean_sym_pair(xs, ys, e, p);
                Ok(())
            })?;
        }

        Ok(self.finish_iteration())
    }

    fn status(&self) -> SolverStatus {
        self.status
    }

    fn iteration(&self) -> usize {
        self.iter
    }

    fn total_iterations(&self) -> usize {
        self.schedule_cfg.t_max
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egraph_linalg::{PivotMdsConfig, koren_spectral_init, pivot_mds};
    use egraph_neighborhood::extract_1hop_2hop_csr;
    use std::collections::HashSet;
    use std::f64::consts::TAU;

    fn grid_graph(rows: usize, cols: usize) -> CsrGraph<f64> {
        let n = rows * cols;
        let mut edges = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let u = (r * cols + c) as u32;
                if r + 1 < rows {
                    edges.push((u, ((r + 1) * cols + c) as u32));
                }
                if c + 1 < cols {
                    edges.push((u, (r * cols + c + 1) as u32));
                }
            }
        }
        CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
    }

    fn barbell_graph(clique_size: usize) -> CsrGraph<f64> {
        let n = 2 * clique_size;
        let mut edges = Vec::new();
        for offset in [0, clique_size] {
            for i in 0..clique_size {
                for j in (i + 1)..clique_size {
                    edges.push(((offset + i) as u32, (offset + j) as u32));
                }
            }
        }
        edges.push(((clique_size - 1) as u32, clique_size as u32));
        CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
    }

    fn circle_layout<S: Scalar>(n: usize, radius: f64) -> Layout2d<S> {
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let angle = TAU * (i as f64) / (n.max(1) as f64);
            let (s, c) = angle.sin_cos();
            x.push(S::from_f64(radius * c));
            y.push(S::from_f64(radius * s));
        }
        Layout2d::from_xy(x, y).unwrap()
    }

    fn euclidean_apsp_stress<S: Scalar>(graph: &CsrGraph<f64>, layout: &Layout2d<S>) -> f64 {
        let n = graph.num_nodes();
        let mut ws = BfsWorkspace::new(n);
        let mut dist = vec![0.0_f64; n];
        let mut stress = 0.0_f64;
        for i in 0..n {
            ws.bfs_from(graph, i, &mut dist).unwrap();
            for (j, &d_g) in dist.iter().enumerate().skip(i + 1) {
                if d_g.is_finite() && d_g > 0.0 {
                    let dx = layout.x()[i].to_f64() - layout.x()[j].to_f64();
                    let dy = layout.y()[i].to_f64() - layout.y()[j].to_f64();
                    let d_e = dx.hypot(dy);
                    let diff = d_e - d_g;
                    stress += (diff * diff) / (d_g * d_g);
                }
            }
        }
        stress
    }

    fn compute_np2<S: Scalar>(graph: &CsrGraph<f64>, layout: &Layout2d<S>) -> f64 {
        let (adj1, adj2) = extract_1hop_2hop_csr(graph).unwrap();
        let n = graph.num_nodes();
        let mut total = 0.0_f64;
        let mut active = 0usize;
        for u in 0..n {
            let mut true_2hop = HashSet::new();
            for &v in adj1.neighbors(u) {
                true_2hop.insert(v);
            }
            for &v in adj2.neighbors(u) {
                true_2hop.insert(v);
            }
            let k = true_2hop.len();
            if k == 0 {
                continue;
            }
            active += 1;
            let ux = layout.x()[u].to_f64();
            let uy = layout.y()[u].to_f64();
            let mut dists: Vec<(f64, u32)> = (0..n)
                .filter(|&v| v != u)
                .map(|v| {
                    let dx = ux - layout.x()[v].to_f64();
                    let dy = uy - layout.y()[v].to_f64();
                    (dx * dx + dy * dy, v as u32)
                })
                .collect();
            dists.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            let hits = dists
                .iter()
                .take(k)
                .filter(|&&(_, v)| true_2hop.contains(&v))
                .count();
            total += (hits as f64) / (k as f64);
        }
        total / (active.max(1) as f64)
    }

    fn manifold_pair_stress<S: Scalar>(
        layout: &ManifoldLayout2d<S>,
        pairs: &[(usize, usize, f64)],
    ) -> f64 {
        let mut total = 0.0_f64;
        for &(i, j, target) in pairs {
            let d = layout.geodesic_distance(i, j).unwrap().to_f64();
            let diff = d - target;
            let w = 1.0 / (target * target);
            total += w * diff * diff;
        }
        total
    }

    #[test]
    fn test_step_vs_run_bit_exact_f64_and_f32_all_modes_and_schedules() {
        let g = grid_graph(3, 4);
        for kind in [
            SgdScheduleKind::Exponential,
            SgdScheduleKind::Linear,
            SgdScheduleKind::Quadratic,
            SgdScheduleKind::Reciprocal,
            SgdScheduleKind::Constant,
        ] {
            let sched = SgdScheduleConfig {
                t_max: 15,
                eps: 0.01,
                kind,
                step_scale: 0.85,
            };

            let mut s1_f64 = SgdSolver::<f64>::full(&g, sched, 123).unwrap();
            let mut s2_f64 = SgdSolver::<f64>::full(&g, sched, 123).unwrap();
            let mut l1_f64 = circle_layout::<f64>(g.num_nodes(), 1.0);
            let mut l2_f64 = l1_f64.clone();

            assert_eq!(s1_f64.status(), SolverStatus::Initialized);
            let initial_eta = s1_f64.current_eta();
            assert!(initial_eta > 0.0);

            assert_eq!(s1_f64.run(&mut l1_f64).unwrap(), SolverStatus::Completed);
            assert_eq!(s1_f64.current_eta(), 0.0);
            assert_eq!(
                s2_f64.run_steps(&mut l2_f64, 5).unwrap(),
                SolverStatus::Running
            );
            assert_eq!(s2_f64.iteration(), 5);
            for _ in 0..4 {
                s2_f64.step(&mut l2_f64).unwrap();
            }
            assert_eq!(s2_f64.run(&mut l2_f64).unwrap(), SolverStatus::Completed);
            assert_eq!(s2_f64.iteration(), 15);
            assert_eq!(s2_f64.total_iterations(), 15);
            assert_eq!(s2_f64.step(&mut l2_f64).unwrap(), SolverStatus::Completed);

            for (a, b) in l1_f64.x().iter().zip(l2_f64.x().iter()) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
            for (a, b) in l1_f64.y().iter().zip(l2_f64.y().iter()) {
                assert_eq!(a.to_bits(), b.to_bits());
            }

            let mut s1_f32 = SgdSolver::<f32>::full(&g, sched, 123).unwrap();
            let mut s2_f32 = SgdSolver::<f32>::full(&g, sched, 123).unwrap();
            let mut l1_f32 = circle_layout::<f32>(g.num_nodes(), 1.0);
            let mut l2_f32 = l1_f32.clone();
            s1_f32.run(&mut l1_f32).unwrap();
            s2_f32.run_steps(&mut l2_f32, 7).unwrap();
            s2_f32.run(&mut l2_f32).unwrap();
            for (a, b) in l1_f32.x().iter().zip(l2_f32.x().iter()) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
            for (a, b) in l1_f32.y().iter().zip(l2_f32.y().iter()) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }

        // Verify bit-exact step vs run for streaming negatives as well.
        let sched = SgdScheduleConfig {
            t_max: 12,
            ..Default::default()
        };
        let mut sn1 = SgdSolver::<f32>::streaming_negatives(
            &g,
            StreamingNegativeConfig::default(),
            sched,
            77,
        )
        .unwrap();
        let mut sn2 = SgdSolver::<f32>::streaming_negatives(
            &g,
            StreamingNegativeConfig::default(),
            sched,
            77,
        )
        .unwrap();
        let mut ln1 = circle_layout::<f32>(g.num_nodes(), 1.0);
        let mut ln2 = ln1.clone();
        sn1.run(&mut ln1).unwrap();
        sn2.run_steps(&mut ln2, 4).unwrap();
        sn2.step(&mut ln2).unwrap();
        sn2.run(&mut ln2).unwrap();
        for (a, b) in ln1.x().iter().zip(ln2.x().iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        for (a, b) in ln1.y().iter().zip(ln2.y().iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn test_hinge_zero_force_when_separated_and_repels_close() {
        let sched = SgdScheduleConfig {
            t_max: 10,
            eps: 0.01,
            kind: SgdScheduleKind::Constant,
            step_scale: 0.5,
        };

        let mut far_solver = SgdSolver::<f64>::from_pairs_with_hinges(
            2,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![0],
            vec![1],
            vec![2.0],
            vec![2.0],
            vec![1.0],
            vec![1.0],
            sched,
            42,
        )
        .unwrap();
        assert_eq!(far_solver.num_nodes(), 2);
        assert_eq!(far_solver.num_springs(), 0);
        assert_eq!(far_solver.num_hinges(), 1);

        let mut far_layout = Layout2d::from_xy(vec![0.0, 3.0], vec![0.0, 0.0]).unwrap();
        far_solver.step(&mut far_layout).unwrap();
        assert_eq!(far_layout.x()[0].to_bits(), 0.0_f64.to_bits());
        assert_eq!(far_layout.y()[0].to_bits(), 0.0_f64.to_bits());
        assert_eq!(far_layout.x()[1].to_bits(), 3.0_f64.to_bits());
        assert_eq!(far_layout.y()[1].to_bits(), 0.0_f64.to_bits());

        let mut boundary_layout = Layout2d::from_xy(vec![0.0, 2.0], vec![0.0, 0.0]).unwrap();
        far_solver.step(&mut boundary_layout).unwrap();
        assert_eq!(boundary_layout.x()[0].to_bits(), 0.0_f64.to_bits());
        assert_eq!(boundary_layout.x()[1].to_bits(), 2.0_f64.to_bits());

        let mut close_layout = Layout2d::from_xy(vec![0.0, 0.5], vec![0.0, 0.0]).unwrap();
        far_solver.step(&mut close_layout).unwrap();
        assert!(close_layout.x()[0] < 0.0);
        assert!(close_layout.x()[1] > 0.5);
        assert_eq!(close_layout.y()[0].to_bits(), 0.0_f64.to_bits());
        assert_eq!(close_layout.y()[1].to_bits(), 0.0_f64.to_bits());

        let mut asym_solver = SgdSolver::<f64>::from_pairs_with_hinges(
            2,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![0],
            vec![1],
            vec![2.0],
            vec![2.0],
            vec![0.0],
            vec![1.0],
            sched,
            42,
        )
        .unwrap();
        let mut asym_layout = Layout2d::from_xy(vec![0.0, 0.5], vec![0.0, 0.0]).unwrap();
        asym_solver.step(&mut asym_layout).unwrap();
        assert_eq!(asym_layout.x()[0].to_bits(), 0.0_f64.to_bits());
        assert!(asym_layout.x()[1] > 0.5);
    }

    #[test]
    fn test_sparse_barrier_from_landmarks_parity_and_star_safety() {
        let path =
            CsrGraph::<f64>::from_edges(5, &[(0, 1), (1, 2), (2, 3), (3, 4)], None, false, false)
                .unwrap();
        let landmarks = DogVaradhanLandmarks {
            num_nodes: 5,
            landmarks: vec![0],
            d_dog: vec![0.0, 1.0, 2.0, 3.0, 4.0],
            d_coarse: None,
        };
        let cfg = SparseBarrierConfig {
            max_2hop_per_node: 32,
            hinge_repulsion: true,
            delta_rel: 0.5,
            barrier_sym: BarrierSymmetry::SymMax,
            omega_1: 2.0,
            omega_2: 0.5,
            gamma_non2: 3.0,
            ht_repulsion: false,
            min_ratio: 0.5,
            eps_dist: 1e-6,
            neg_per_node: Some(8),
            num_pivots: 2,
            core_pivots: true,
            schedule: SgdScheduleConfig::default(),
            seed: 42,
        };
        let (solver, stats) =
            SgdSolver::<f64>::sparse_barrier_from_landmarks(&path, &landmarks, Some(&[0, 4]), &cfg)
                .unwrap();
        assert_eq!(stats.n_pairs_1hop, 4);
        assert_eq!(stats.n_pairs_2hop, 3);
        assert_eq!(stats.n_pairs_non2, 3);
        assert_eq!(stats.n_pairs_two_sided, 7);
        assert_eq!(stats.n_pairs_hinge, 3);
        assert!((stats.d_floor - 0.5).abs() < 1e-12);
        assert!((stats.gamma_eff - 3.0).abs() < 1e-12);
        assert_eq!(solver.num_springs(), 7);
        assert_eq!(solver.num_hinges(), 3);

        let mut cfg_ht = cfg;
        cfg_ht.ht_repulsion = true;
        let (_solver_ht, stats_ht) = SgdSolver::<f64>::sparse_barrier_from_landmarks(
            &path,
            &landmarks,
            Some(&[0, 4]),
            &cfg_ht,
        )
        .unwrap();
        let expected_gamma = 3.0 * (5.0_f64 / 8.0_f64).sqrt();
        assert!((stats_ht.gamma_eff - expected_gamma).abs() < 1e-12);

        // Star graph on 1000 nodes: hub-safe cap keeps pairs O(N * cap).
        let n_star = 1000usize;
        let star_edges: Vec<(u32, u32)> = (1..n_star as u32).map(|v| (0, v)).collect();
        let star = CsrGraph::<f64>::from_edges(n_star, &star_edges, None, false, false).unwrap();
        let star_cfg = SparseBarrierConfig {
            max_2hop_per_node: 16,
            num_pivots: 16,
            ..Default::default()
        };
        let (star_solver, star_stats, _) =
            SgdSolver::<f64>::sparse_barrier_dog(&star, &DogVaradhanConfig::default(), &star_cfg)
                .unwrap();
        assert_eq!(star_stats.n_pairs_1hop, n_star - 1);
        assert!(star_solver.num_springs() <= (n_star - 1) + n_star * 16);
        assert_eq!(star_solver.num_hinges(), 0);
    }

    #[test]
    fn test_sparse_barrier_dog_preserves_2hop_on_grid_and_barbell() {
        let grid = grid_graph(4, 4);
        let dog_cfg = DogVaradhanConfig::default();
        let barrier_cfg = SparseBarrierConfig::default();
        let (mut grid_solver, grid_stats, _) =
            SgdSolver::<f64>::sparse_barrier_dog(&grid, &dog_cfg, &barrier_cfg).unwrap();
        assert!(grid_stats.n_pairs_1hop > 0);
        assert!(grid_stats.n_pairs_2hop > 0);
        assert!(grid_stats.n_pairs_hinge > 0);

        let mut grid_layout = pivot_mds(&grid, &PivotMdsConfig::default()).unwrap();
        grid_solver.run(&mut grid_layout).unwrap();
        assert!(grid_layout.is_all_finite());
        let np2_grid = compute_np2(&grid, &grid_layout);
        assert!(
            np2_grid >= 0.85,
            "expected grid NP2 >= 0.85, got {np2_grid}"
        );

        let barbell = barbell_graph(6);
        let (mut bb_solver, _, _) =
            SgdSolver::<f64>::sparse_barrier_dog(&barbell, &dog_cfg, &barrier_cfg).unwrap();
        let mut bb_layout = pivot_mds(&barbell, &PivotMdsConfig::default()).unwrap();
        bb_solver.run(&mut bb_layout).unwrap();
        assert!(bb_layout.is_all_finite());
        let np2_bb = compute_np2(&barbell, &bb_layout);
        assert!(np2_bb >= 0.90, "expected barbell NP2 >= 0.90, got {np2_bb}");
    }

    #[test]
    fn test_streaming_negatives_separates_non_edges_and_converges() {
        let barbell = barbell_graph(5);
        let neg_cfg = StreamingNegativeConfig {
            negatives_per_node: 6,
            target_distance: 2.5,
            weight: 0.5,
            hinge_only: true,
        };
        let sched = SgdScheduleConfig {
            t_max: 60,
            eps: 0.01,
            kind: SgdScheduleKind::Exponential,
            step_scale: 1.0,
        };
        let mut solver =
            SgdSolver::<f64>::streaming_negatives(&barbell, neg_cfg, sched, 42).unwrap();
        assert_eq!(solver.num_springs(), barbell.num_edges());
        assert_eq!(solver.num_hinges(), 0);

        let mut layout = circle_layout::<f64>(barbell.num_nodes(), 0.2);
        solver.run(&mut layout).unwrap();
        assert!(layout.is_all_finite());

        let c0x = (0..5).map(|i| layout.x()[i]).sum::<f64>() / 5.0;
        let c0y = (0..5).map(|i| layout.y()[i]).sum::<f64>() / 5.0;
        let c1x = (5..10).map(|i| layout.x()[i]).sum::<f64>() / 5.0;
        let c1y = (5..10).map(|i| layout.y()[i]).sum::<f64>() / 5.0;
        let sep = (c0x - c1x).hypot(c0y - c1y);
        assert!(sep > 1.5, "expected separated clique centroids, got {sep}");

        let disc = CsrGraph::<f64>::from_edges(2, &[], None, false, false).unwrap();
        let neg_spring = StreamingNegativeConfig {
            negatives_per_node: 3,
            target_distance: 3.0,
            weight: 1.0,
            hinge_only: false,
        };
        let mut disc_solver =
            SgdSolver::<f64>::streaming_negatives(&disc, neg_spring, sched, 7).unwrap();
        let mut disc_layout = Layout2d::from_xy(vec![0.0, 0.1], vec![0.0, 0.0]).unwrap();
        disc_solver.run(&mut disc_layout).unwrap();
        let d_disc = (disc_layout.x()[0] - disc_layout.x()[1])
            .hypot(disc_layout.y()[0] - disc_layout.y()[1]);
        assert!(
            (d_disc - 3.0).abs() < 0.25,
            "expected dist ~3.0, got {d_disc}"
        );
    }

    #[test]
    fn test_streaming_negatives_unsorted_csr_never_repels_1hop_edges() {
        // Triangle 0-1-2 plus leaves (0, 3) and (2, 4) with deliberately unsorted CSR rows.
        let indptr = vec![0u32, 3, 5, 8, 9, 10];
        let indices = vec![
            3, 2, 1, // node 0 (unsorted)
            2, 0, // node 1 (unsorted)
            4, 1, 0, // node 2 (unsorted)
            0, // node 3
            2, // node 4
        ];
        let g_unsorted = CsrGraph::<f64>::from_csr(&indptr, &indices, None, false, false).unwrap();
        let g_sorted = CsrGraph::<f64>::from_csr(&indptr, &indices, None, false, true).unwrap();

        let neg_cfg = StreamingNegativeConfig {
            negatives_per_node: 8,
            target_distance: 3.0,
            weight: 0.25,
            hinge_only: true,
        };
        let sched = SgdScheduleConfig {
            t_max: 60,
            eps: 0.01,
            kind: SgdScheduleKind::Exponential,
            step_scale: 1.0,
        };

        let mut s_unsorted =
            SgdSolver::<f64>::streaming_negatives(&g_unsorted, neg_cfg, sched, 99).unwrap();
        let mut s_sorted =
            SgdSolver::<f64>::streaming_negatives(&g_sorted, neg_cfg, sched, 99).unwrap();

        let st_u = s_unsorted.streaming.as_ref().unwrap();
        let st_s = s_sorted.streaming.as_ref().unwrap();
        assert_eq!(st_u.adj_indptr, st_s.adj_indptr);
        assert_eq!(st_u.adj_indices, st_s.adj_indices);
        for u in 0..5 {
            let row =
                &st_u.adj_indices[st_u.adj_indptr[u] as usize..st_u.adj_indptr[u + 1] as usize];
            assert!(row.windows(2).all(|w| w[0] < w[1]));
        }

        let mut l_unsorted = circle_layout::<f64>(5, 0.8);
        let mut l_sorted = l_unsorted.clone();
        s_unsorted.run(&mut l_unsorted).unwrap();
        s_sorted.run(&mut l_sorted).unwrap();

        // Unsorted CSR must match sorted CSR bit-for-bit and keep 1-hop edges compact while
        // separating non-edge pair (3, 4).
        for (a, b) in l_unsorted.x().iter().zip(l_sorted.x().iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        for (a, b) in l_unsorted.y().iter().zip(l_sorted.y().iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        for (u, v) in [(0usize, 1usize), (1, 2), (0, 2), (0, 3), (2, 4)] {
            let d_uv = (l_unsorted.x()[u] - l_unsorted.x()[v])
                .hypot(l_unsorted.y()[u] - l_unsorted.y()[v]);
            assert!(
                d_uv < 2.0,
                "1-hop edge ({u}, {v}) suffered excessive repulsion: dist={d_uv}"
            );
        }
        let d_34 =
            (l_unsorted.x()[3] - l_unsorted.x()[4]).hypot(l_unsorted.y()[3] - l_unsorted.y()[4]);
        assert!(d_34 > 2.0, "expected non-edge (3, 4) separated, got {d_34}");
    }

    #[test]
    fn test_manifolds_all_three_domain_invariants_and_stress_reduction() {
        let sched = SgdScheduleConfig {
            t_max: 50,
            eps: 0.02,
            kind: SgdScheduleKind::Exponential,
            step_scale: 1.0,
        };

        // 1. Hyperbolic Poincare disk on a 7-node binary tree.
        let tree_edges = [(0u32, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)];
        let tree = CsrGraph::<f64>::from_edges(7, &tree_edges, None, false, false).unwrap();
        let mut tree_pairs = Vec::new();
        let mut bfs = BfsWorkspace::new(7);
        let mut row = vec![0.0_f64; 7];
        for i in 0..7 {
            bfs.bfs_from(&tree, i, &mut row).unwrap();
            for (j, &d_t) in row.iter().enumerate().skip(i + 1) {
                tree_pairs.push((i, j, d_t));
            }
        }
        let mut hyp_solver = SgdSolver::<f64>::full(&tree, sched, 11).unwrap();
        let mut hyp_solver_step = SgdSolver::<f64>::full(&tree, sched, 11).unwrap();
        let mut hyp_layout =
            ManifoldLayout2d::from_layout(ManifoldKind::HyperbolicPoincare, circle_layout(7, 0.35))
                .unwrap();
        let mut hyp_layout_step = hyp_layout.clone();
        let hyp_before = manifold_pair_stress(&hyp_layout, &tree_pairs);
        while !hyp_solver_step.status().is_finished() {
            hyp_solver_step.step_manifold(&mut hyp_layout_step).unwrap();
            for i in 0..7 {
                let r = hyp_layout_step.x()[i].hypot(hyp_layout_step.y()[i]);
                assert!(r.is_finite() && r < 1.0);
            }
        }
        hyp_solver.run_manifold(&mut hyp_layout).unwrap();
        assert_eq!(hyp_layout, hyp_layout_step);
        let hyp_after = manifold_pair_stress(&hyp_layout, &tree_pairs);
        assert!(
            hyp_after < hyp_before * 0.5,
            "hyperbolic stress did not drop: {hyp_before} -> {hyp_after}"
        );

        // 2. Spherical on a 6-node cycle (diameter 3 < pi).
        let cyc = CsrGraph::<f64>::from_edges(
            6,
            &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],
            None,
            false,
            false,
        )
        .unwrap();
        let mut cyc_pairs = Vec::new();
        let mut row6 = vec![0.0_f64; 6];
        for i in 0..6 {
            bfs.bfs_from(&cyc, i, &mut row6).unwrap();
            for (j, &d_c) in row6.iter().enumerate().skip(i + 1) {
                cyc_pairs.push((i, j, d_c));
            }
        }
        let mut sph_solver = SgdSolver::<f32>::full(&cyc, sched, 19).unwrap();
        let mut sph_layout = ManifoldLayout2d::<f32>::from_xy(
            ManifoldKind::Spherical,
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            vec![1.2, 1.3, 1.1, 1.4, 1.25, 1.35],
        )
        .unwrap();
        let sph_before = manifold_pair_stress(&sph_layout, &cyc_pairs);
        while !sph_solver.status().is_finished() {
            sph_solver.step_manifold(&mut sph_layout).unwrap();
            for i in 0..6 {
                let lon = sph_layout.x()[i];
                let colat = sph_layout.y()[i];
                assert!(lon.is_finite() && (0.0_f32..std::f32::consts::TAU).contains(&lon));
                assert!(colat.is_finite() && (0.0_f32..=std::f32::consts::PI).contains(&colat));
            }
        }
        let sph_after = manifold_pair_stress(&sph_layout, &cyc_pairs);
        assert!(
            sph_after < sph_before * 0.5,
            "spherical stress did not drop: {sph_before} -> {sph_after}"
        );

        // 3. Torus2d with scaled edge target lengths (0.15 per hop on 4-cycle).
        let t_src = vec![0u32, 1, 2, 3, 0, 1];
        let t_tgt = vec![1u32, 2, 3, 0, 2, 3];
        let t_dist = vec![0.15_f64, 0.15, 0.15, 0.15, 0.30, 0.30];
        let t_wt: Vec<f64> = t_dist.iter().map(|&d| 1.0 / (d * d)).collect();
        let torus_pairs: Vec<(usize, usize, f64)> = t_src
            .iter()
            .zip(t_tgt.iter())
            .zip(t_dist.iter())
            .map(|((&u, &v), &d)| (u as usize, v as usize, d))
            .collect();
        let mut tor_solver = SgdSolver::<f64>::from_pairs(
            4,
            t_src,
            t_tgt,
            t_dist.clone(),
            t_dist,
            t_wt.clone(),
            t_wt,
            sched,
            23,
        )
        .unwrap();
        let mut tor_layout = ManifoldLayout2d::from_xy(
            ManifoldKind::Torus2d,
            vec![0.05, 0.85, 0.45, 0.12],
            vec![0.92, 0.10, 0.55, 0.48],
        )
        .unwrap();
        let tor_before = manifold_pair_stress(&tor_layout, &torus_pairs);
        tor_solver.run_steps_manifold(&mut tor_layout, 25).unwrap();
        tor_solver.run_manifold(&mut tor_layout).unwrap();
        for i in 0..4 {
            let x = tor_layout.x()[i];
            let y = tor_layout.y()[i];
            assert!(x.is_finite() && (0.0..1.0).contains(&x));
            assert!(y.is_finite() && (0.0..1.0).contains(&y));
        }
        let tor_after = manifold_pair_stress(&tor_layout, &torus_pairs);
        assert!(
            tor_after < tor_before * 0.25,
            "torus stress did not drop: {tor_before} -> {tor_after}"
        );

        // Extreme distance clamp test across all 3 manifolds.
        for kind in [
            ManifoldKind::HyperbolicPoincare,
            ManifoldKind::Spherical,
            ManifoldKind::Torus2d,
        ] {
            let mut ext_solver = SgdSolver::<f64>::from_pairs(
                2,
                vec![0],
                vec![1],
                vec![1e6],
                vec![1e6],
                vec![1.0],
                vec![1.0],
                SgdScheduleConfig {
                    t_max: 5,
                    kind: SgdScheduleKind::Constant,
                    ..Default::default()
                },
                5,
            )
            .unwrap();
            let mut m = ManifoldLayout2d::from_xy(kind, vec![0.1, 0.3], vec![0.2, 0.8]).unwrap();
            ext_solver.run_manifold(&mut m).unwrap();
            assert!(m.as_layout().is_all_finite());
        }
    }

    #[test]
    fn test_full_weighted_directed_random_pair_and_omega_converge() {
        let g = grid_graph(3, 3);
        let sched = SgdScheduleConfig::default();

        // Weighted triangle: edge (0, 2) has length 10.0, path 0-1-2 has length 1 + 2 = 3.0.
        let wg = CsrGraph::<f64>::from_edges(
            3,
            &[(0, 1), (1, 2), (0, 2)],
            Some(&[1.0, 2.0, 10.0]),
            false,
            false,
        )
        .unwrap();
        let mut w_solver = SgdSolver::<f64>::full(&wg, sched, 42).unwrap();
        let mut w_layout = circle_layout::<f64>(3, 2.0);
        w_solver.run(&mut w_layout).unwrap();
        let d02 = (w_layout.x()[0] - w_layout.x()[2]).hypot(w_layout.y()[0] - w_layout.y()[2]);
        assert!(
            (d02 - 3.0).abs() < 0.2,
            "expected weighted shortest-path distance 3.0 for (0,2), got {d02}"
        );

        // Directed DAG graph does not fail or drop all pairs.
        let dag = CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], None, true, false).unwrap();
        let mut dag_solver = SgdSolver::<f64>::full(&dag, sched, 7).unwrap();
        assert_eq!(dag_solver.num_springs(), 3);
        let mut dag_layout = circle_layout::<f64>(3, 0.5);
        dag_solver.run(&mut dag_layout).unwrap();
        assert!(dag_layout.is_all_finite());

        // Full & RandomPair converge on 3x3 grid.
        let mut full_solver = SgdSolver::<f64>::full(&g, sched, 42).unwrap();
        let mut full_layout = circle_layout::<f64>(g.num_nodes(), 0.25);
        let s_before = euclidean_apsp_stress(&g, &full_layout);
        full_solver.run(&mut full_layout).unwrap();
        let s_after = euclidean_apsp_stress(&g, &full_layout);
        assert!(s_after < s_before * 0.2);

        let mut rp_solver = SgdSolver::<f64>::random_pair(&g, 4, sched, 42).unwrap();
        let mut rp_layout = circle_layout::<f64>(g.num_nodes(), 0.25);
        let rp_before = euclidean_apsp_stress(&g, &rp_layout);
        rp_solver.run(&mut rp_layout).unwrap();
        let rp_after = euclidean_apsp_stress(&g, &rp_layout);
        assert!(rp_after < rp_before * 0.3);

        // Omega converges from spectral embedding.
        let spec = koren_spectral_init::<f64, f64>(&g, Some(1.0), 1e-6, 200, 42).unwrap();
        let mut emb_flat = Vec::with_capacity(g.num_nodes() * 2);
        for i in 0..g.num_nodes() {
            emb_flat.push(spec.x()[i]);
            emb_flat.push(spec.y()[i]);
        }
        let mut om_solver = SgdSolver::<f64>::omega(&g, &emb_flat, 2, 4, 1e-3, sched, 42).unwrap();
        let mut om_layout = circle_layout::<f64>(g.num_nodes(), 0.25);
        om_solver.run(&mut om_layout).unwrap();
        assert!(om_layout.is_all_finite());
        assert!(compute_np2(&g, &om_layout) >= 0.75);
    }

    #[test]
    fn test_adversarial_empty_singleton_oob_and_non_finite_errors() {
        let sched = SgdScheduleConfig::default();
        let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();

        for g in [&empty, &single] {
            let n = g.num_nodes();
            let mut f = SgdSolver::<f64>::full(g, sched, 42).unwrap();
            let mut rp = SgdSolver::<f64>::random_pair(g, 5, sched, 42).unwrap();
            let mut sn = SgdSolver::<f64>::streaming_negatives(
                g,
                StreamingNegativeConfig::default(),
                sched,
                42,
            )
            .unwrap();
            let emb = vec![0.0_f64; n * 2];
            let mut om = SgdSolver::<f64>::omega(g, &emb, 2, 5, 1e-3, sched, 42).unwrap();
            let (mut sb, st, _) = SgdSolver::<f64>::sparse_barrier_dog(
                g,
                &DogVaradhanConfig::default(),
                &SparseBarrierConfig::default(),
            )
            .unwrap();
            assert_eq!(st.n_pairs_1hop, 0);

            for solver in [&mut f, &mut rp, &mut sn, &mut om, &mut sb] {
                let mut l = Layout2d::<f64>::zeros(n);
                assert_eq!(solver.run(&mut l).unwrap(), SolverStatus::Completed);
                let mut m = ManifoldLayout2d::<f64>::zeros(ManifoldKind::Torus2d, n);
                assert_eq!(
                    solver.run_manifold(&mut m).unwrap(),
                    SolverStatus::Completed
                );
            }
        }

        // Out-of-bounds, dimension mismatch, self-loop, and non-finite pair validation.
        assert!(matches!(
            SgdSolver::<f64>::from_pairs(
                2,
                vec![0],
                vec![2],
                vec![1.0],
                vec![1.0],
                vec![1.0],
                vec![1.0],
                sched,
                42
            ),
            Err(CoreError::IndexOutOfBounds { .. })
        ));
        assert!(matches!(
            SgdSolver::<f64>::from_pairs(
                2,
                vec![0],
                vec![0],
                vec![1.0],
                vec![1.0],
                vec![1.0],
                vec![1.0],
                sched,
                42
            ),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            SgdSolver::<f64>::from_pairs(
                2,
                vec![0],
                vec![1, 0],
                vec![1.0],
                vec![1.0],
                vec![1.0],
                vec![1.0],
                sched,
                42
            ),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            SgdSolver::<f64>::from_pairs(
                2,
                vec![0],
                vec![1],
                vec![f64::NAN],
                vec![1.0],
                vec![1.0],
                vec![1.0],
                sched,
                42
            ),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            SgdSolver::<f64>::from_pairs(
                2,
                vec![0],
                vec![1],
                vec![1.0],
                vec![1.0],
                vec![-0.5],
                vec![1.0],
                sched,
                42
            ),
            Err(CoreError::InvalidInput(_))
        ));

        // Schedule and config errors.
        assert!(matches!(
            SgdSolver::<f64>::full(
                &single,
                SgdScheduleConfig {
                    eps: f64::NAN,
                    ..sched
                },
                42
            ),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            SgdSolver::<f64>::full(
                &single,
                SgdScheduleConfig {
                    step_scale: 0.0,
                    ..sched
                },
                42
            ),
            Err(CoreError::InvalidInput(_))
        ));

        // Omega invalid dims / mismatched length / non-finite / min_dist <= 0.
        let tri = CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], None, false, false).unwrap();
        assert!(matches!(
            SgdSolver::<f64>::omega(&tri, &[], 0, 2, 1e-3, sched, 42),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            SgdSolver::<f64>::omega(&tri, &[0.0; 4], 2, 2, 1e-3, sched, 42),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            SgdSolver::<f64>::omega(&tri, &[f64::INFINITY; 6], 2, 2, 1e-3, sched, 42),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            SgdSolver::<f64>::omega(&tri, &[0.0; 6], 2, 2, 0.0, sched, 42),
            Err(CoreError::InvalidInput(_))
        ));

        // Landmark mismatch, out-of-bounds pivot, and NaN landmark table.
        let lm_ok = DogVaradhanLandmarks {
            num_nodes: 3,
            landmarks: vec![0],
            d_dog: vec![0.0, 1.0, 2.0],
            d_coarse: None,
        };
        assert!(matches!(
            SgdSolver::<f64>::sparse_barrier_from_landmarks(
                &tri,
                &lm_ok,
                Some(&[5]),
                &SparseBarrierConfig::default()
            ),
            Err(CoreError::IndexOutOfBounds { .. })
        ));
        let lm_bad = DogVaradhanLandmarks {
            num_nodes: 3,
            landmarks: vec![0],
            d_dog: vec![0.0, f64::NAN, 2.0],
            d_coarse: None,
        };
        assert!(matches!(
            SgdSolver::<f64>::sparse_barrier_from_landmarks(
                &tri,
                &lm_bad,
                None,
                &SparseBarrierConfig::default()
            ),
            Err(CoreError::NonFiniteValue { .. })
        ));
        let lm_mismatch = DogVaradhanLandmarks {
            num_nodes: 2,
            landmarks: vec![0],
            d_dog: vec![0.0, 1.0],
            d_coarse: None,
        };
        assert!(matches!(
            SgdSolver::<f64>::sparse_barrier_from_landmarks(
                &tri,
                &lm_mismatch,
                None,
                &SparseBarrierConfig::default()
            ),
            Err(CoreError::DimensionMismatch { .. })
        ));

        // Layout size mismatch and non-finite layout during step / step_manifold.
        let mut s = SgdSolver::<f64>::full(&tri, sched, 42).unwrap();
        let mut wrong_len = Layout2d::<f64>::zeros(2);
        assert!(matches!(
            s.step(&mut wrong_len),
            Err(CoreError::DimensionMismatch { .. })
        ));
        let mut nan_layout = Layout2d::from_xy(vec![0.0, f64::NAN, 1.0], vec![0.0; 3]).unwrap();
        assert!(matches!(
            s.step(&mut nan_layout),
            Err(CoreError::NonFiniteValue { .. })
        ));
        let mut wrong_m = ManifoldLayout2d::<f64>::zeros(ManifoldKind::Spherical, 2);
        assert!(matches!(
            s.step_manifold(&mut wrong_m),
            Err(CoreError::DimensionMismatch { .. })
        ));

        // Negative weights on graph must fail cleanly.
        let neg_wg =
            CsrGraph::<f64>::from_edges(3, &[(0, 1)], Some(&[-1.0]), false, false).unwrap();
        assert!(matches!(
            SgdSolver::<f64>::full(&neg_wg, sched, 42),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            SgdSolver::<f64>::random_pair(&neg_wg, 2, sched, 42),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            SgdSolver::<f64>::streaming_negatives(
                &neg_wg,
                StreamingNegativeConfig::default(),
                sched,
                42
            ),
            Err(CoreError::InvalidInput(_))
        ));
    }

    #[test]
    fn test_cache_blocked_shuffle_over_threshold_step_eq_run() {
        let n = 400usize;
        let num_pairs = 70_000usize;
        let mut pairs64 = Vec::with_capacity(num_pairs);
        let mut pairs32 = Vec::with_capacity(num_pairs);
        let mut n_springs = 0usize;
        let mut n_hinges = 0usize;
        for idx in 0..num_pairs {
            let u = (idx % n) as u32;
            let step = 1 + (idx / n);
            let v = ((u as usize + step) % n) as u32;
            let is_hinge = idx % 5 == 0;
            if is_hinge {
                n_hinges += 1;
            } else {
                n_springs += 1;
            }
            let d = 1.0 + (step % 7) as f64 * 0.25;
            let w = 1.0 / (d * d);
            pairs64.push(PackedSymPair::new(u, v, is_hinge, d, w));
            pairs32.push(PackedSymPair::new(u, v, is_hinge, d as f32, w as f32));
        }

        let sched = SgdScheduleConfig {
            t_max: 6,
            eps: 0.05,
            kind: SgdScheduleKind::Exponential,
            step_scale: 0.2,
        };

        let mut s64_run = SgdSolver::<f64>::from_packed_symmetric(
            n,
            pairs64.clone(),
            n_springs,
            n_hinges,
            sched,
            42,
        )
        .unwrap();
        let mut s64_step =
            SgdSolver::<f64>::from_packed_symmetric(n, pairs64, n_springs, n_hinges, sched, 42)
                .unwrap();
        assert!(!s64_run.block_order.is_empty());
        assert_eq!(s64_run.block_order.len(), num_pairs.div_ceil(BLOCK_SIZE));

        let mut l64_run = circle_layout::<f64>(n, 2.0);
        let mut l64_step = l64_run.clone();
        s64_run.run(&mut l64_run).unwrap();
        s64_step.run_steps(&mut l64_step, 2).unwrap();
        while !s64_step.status().is_finished() {
            s64_step.step(&mut l64_step).unwrap();
        }
        assert!(l64_run.is_all_finite());
        for i in 0..n {
            assert_eq!(l64_run.x()[i].to_bits(), l64_step.x()[i].to_bits());
            assert_eq!(l64_run.y()[i].to_bits(), l64_step.y()[i].to_bits());
        }

        let mut s32_run = SgdSolver::<f32>::from_packed_symmetric(
            n,
            pairs32.clone(),
            n_springs,
            n_hinges,
            sched,
            42,
        )
        .unwrap();
        let mut s32_step =
            SgdSolver::<f32>::from_packed_symmetric(n, pairs32, n_springs, n_hinges, sched, 42)
                .unwrap();
        let mut l32_run = circle_layout::<f32>(n, 2.0);
        let mut l32_step = l32_run.clone();
        s32_run.run(&mut l32_run).unwrap();
        s32_step.run_steps(&mut l32_step, 3).unwrap();
        while !s32_step.status().is_finished() {
            s32_step.step(&mut l32_step).unwrap();
        }
        assert!(l32_run.is_all_finite());
        for i in 0..n {
            assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
            assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
        }
    }
}
