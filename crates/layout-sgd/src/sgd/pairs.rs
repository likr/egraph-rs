use egraph_core::{CoreError, CoreResult, CsrGraph, DeterministicRng, Scalar};

use super::config::{SgdScheduleConfig, SparseBarrierConfig};
use super::pair_buffer::{SgdPairKind, SgdPairRecord};

pub(super) const HINGE_BIT: u32 = 1u32 << 31;
pub(super) const INDEX_MASK: u32 = !HINGE_BIT;
pub(super) const MAX_PACKED_NODES: usize = (INDEX_MASK as usize) + 1;
pub(super) const BLOCK_SIZE: usize = 16_384;
pub(super) const BLOCK_SHUFFLE_THRESHOLD: usize = 65_536;

/// Packed symmetric SGD pair encoding endpoints, hinge flag, target distance, and weight.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PackedSymPair<S: Scalar> {
    pub(super) i: u32,
    pub(super) j_hinge: u32,
    pub(super) d: S,
    pub(super) w: S,
}

impl<S: Scalar> PackedSymPair<S> {
    /// Creates a packed symmetric pair `(i, j)` with optional one-sided hinge mode.
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
    pub(super) fn to_record(self) -> SgdPairRecord<S> {
        let kind = if (self.j_hinge & HINGE_BIT) != 0 {
            SgdPairKind::Hinge
        } else {
            SgdPairKind::Spring
        };
        SgdPairRecord::new(
            self.i,
            self.j_hinge & INDEX_MASK,
            kind,
            self.d,
            self.d,
            self.w,
            self.w,
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct PackedGenPair<S: Scalar> {
    pub(super) i: u32,
    pub(super) j_hinge: u32,
    pub(super) dij: S,
    pub(super) dji: S,
    pub(super) wij: S,
    pub(super) wji: S,
}

impl<S: Scalar> PackedGenPair<S> {
    #[inline]
    pub(super) fn new(i: u32, j: u32, is_hinge: bool, dij: S, dji: S, wij: S, wji: S) -> Self {
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
    pub(super) fn to_record(self) -> SgdPairRecord<S> {
        let kind = if (self.j_hinge & HINGE_BIT) != 0 {
            SgdPairKind::Hinge
        } else {
            SgdPairKind::Spring
        };
        SgdPairRecord::new(
            self.i,
            self.j_hinge & INDEX_MASK,
            kind,
            self.dij,
            self.dji,
            self.wij,
            self.wji,
        )
    }
}

#[derive(Debug, Clone)]
pub(super) enum PairStorage<S: Scalar> {
    Symmetric(Vec<PackedSymPair<S>>),
    General(Vec<PackedGenPair<S>>),
}

pub(super) enum PairStorageIter<'a, S: Scalar> {
    Symmetric(std::slice::Iter<'a, PackedSymPair<S>>),
    General(std::slice::Iter<'a, PackedGenPair<S>>),
}

impl<S: Scalar> Iterator for PairStorageIter<'_, S> {
    type Item = SgdPairRecord<S>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Symmetric(iter) => iter.next().map(|&p| p.to_record()),
            Self::General(iter) => iter.next().map(|&p| p.to_record()),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Symmetric(iter) => iter.size_hint(),
            Self::General(iter) => iter.size_hint(),
        }
    }
}

impl<S: Scalar> ExactSizeIterator for PairStorageIter<'_, S> {
    #[inline]
    fn len(&self) -> usize {
        match self {
            Self::Symmetric(iter) => iter.len(),
            Self::General(iter) => iter.len(),
        }
    }
}

impl<S: Scalar> std::iter::FusedIterator for PairStorageIter<'_, S> {}

impl<S: Scalar> PairStorage<S> {
    #[inline]
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Symmetric(v) => v.len(),
            Self::General(v) => v.len(),
        }
    }

    #[inline]
    pub(super) fn iter(&self) -> PairStorageIter<'_, S> {
        match self {
            Self::Symmetric(v) => PairStorageIter::Symmetric(v.iter()),
            Self::General(v) => PairStorageIter::General(v.iter()),
        }
    }
}

pub(super) struct DirectedPairAccumulator<S: Scalar> {
    pairs: Vec<PackedGenPair<S>>,
    all_symmetric: bool,
}

impl<S: Scalar> DirectedPairAccumulator<S> {
    pub(super) fn with_capacity(cap: usize) -> Self {
        Self {
            pairs: Vec::with_capacity(cap),
            all_symmetric: true,
        }
    }

    pub(super) fn push_s(&mut self, u: u32, v: u32, d_uv_s: S, d_vu_s: S, w_uv_s: S, w_vu_s: S) {
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

    pub(super) fn push_f64(&mut self, u: u32, v: u32, d_uv: f64, d_vu: f64, w_uv: f64, w_vu: f64) {
        self.push_s(
            u,
            v,
            S::from_f64(d_uv),
            S::from_f64(d_vu),
            S::from_f64(w_uv),
            S::from_f64(w_vu),
        );
    }

    pub(super) fn into_storage(self) -> PairStorage<S> {
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

pub(super) fn validate_schedule_config(cfg: &SgdScheduleConfig) -> CoreResult<()> {
    cfg.validate()
}

pub(super) fn validate_barrier_config(
    cfg: &SparseBarrierConfig,
    has_explicit_pivots: bool,
) -> CoreResult<()> {
    validate_schedule_config(&cfg.schedule)?;
    for (ctx, val) in [
        ("SparseBarrierConfig::delta_rel", cfg.delta_rel),
        ("SparseBarrierConfig::omega_1", cfg.omega_1),
        ("SparseBarrierConfig::omega_2", cfg.omega_2),
        ("SparseBarrierConfig::gamma_non2", cfg.gamma_non2),
        ("SparseBarrierConfig::min_ratio", cfg.min_ratio),
        (
            "SparseBarrierConfig::degree_weight_beta",
            cfg.degree_weight_beta,
        ),
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

pub(super) fn validate_graph_weights<W: Scalar>(
    graph: &CsrGraph<W>,
    ctx: &'static str,
) -> CoreResult<()> {
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

pub(super) fn eta_bounds_from_extrema(
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

pub(super) fn compute_eta_bounds<S: Scalar>(
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
pub(super) fn resolve_directed_pair_params(d_uv: f64, d_vu: f64) -> Option<(f64, f64, f64, f64)> {
    let ok_uv = d_uv.is_finite() && d_uv > 0.0;
    let ok_vu = d_vu.is_finite() && d_vu > 0.0;
    match (ok_uv, ok_vu) {
        (true, true) => Some((d_uv, d_vu, 1.0 / (d_uv * d_uv), 1.0 / (d_vu * d_vu))),
        (true, false) => Some((d_uv, d_uv, 1.0 / (d_uv * d_uv), 0.0)),
        (false, true) => Some((d_vu, d_vu, 0.0, 1.0 / (d_vu * d_vu))),
        (false, false) => None,
    }
}

pub(super) fn init_pair_blocking<P>(pairs: &mut [P], rng: &mut DeterministicRng) -> Vec<u32> {
    if pairs.len() > BLOCK_SHUFFLE_THRESHOLD {
        rng.shuffle(pairs);
        let num_blocks = pairs.len().div_ceil(BLOCK_SIZE);
        (0..num_blocks as u32).collect()
    } else {
        Vec::new()
    }
}

#[inline]
pub(super) fn try_for_each_shuffled_pair<P>(
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

pub(super) fn validate_pair_endpoints(
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
