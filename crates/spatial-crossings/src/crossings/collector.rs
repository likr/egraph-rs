use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use super::cross_pred::CrossPred;
use super::geometry::{STEEP_SLOPE_THRESHOLD, Seg, seg_y_err, within_band};
use super::pair_set::{InlinePairSet, PairSet, pack_pair};

#[derive(Debug, Clone, Copy)]
struct OrderedF64(f64);

impl PartialEq for OrderedF64 {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0) == Ordering::Equal
    }
}

impl Eq for OrderedF64 {}

impl PartialOrd for OrderedF64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

pub(super) trait CrossingCollector {
    fn on_vertical_pair(&mut self, a: usize, b: usize);
    fn on_swap_pair<P: CrossPred>(
        &mut self,
        s1: &Seg,
        s2: &Seg,
        sweep_x: f64,
        eps_y: f64,
        dt_probe: f64,
        pred: &P,
    );
    #[allow(clippy::too_many_arguments)]
    fn on_band_pair<P: CrossPred>(
        &mut self,
        s_lo: &Seg,
        s_hi: &Seg,
        sweep_x: f64,
        eps_y: f64,
        dt_probe: f64,
        opt_xi: Option<f64>,
        pred: &P,
    );
    fn evict_before<P: CrossPred>(&mut self, x_evt: f64, segs: &[Seg], eps_y: f64, pred: &P);
    fn flush_at_x<P: CrossPred>(
        &mut self,
        sweep_x: f64,
        next_x: f64,
        segs: &[Seg],
        eps_y: f64,
        dt_probe: f64,
        pred: &P,
    );
}

pub(super) struct PairSetCollector {
    pub(super) reported: PairSet,
}

impl CrossingCollector for PairSetCollector {
    #[inline]
    fn on_vertical_pair(&mut self, a: usize, b: usize) {
        self.reported.insert(pack_pair(a, b));
    }

    #[inline]
    fn on_swap_pair<P: CrossPred>(
        &mut self,
        s1: &Seg,
        s2: &Seg,
        _sweep_x: f64,
        _eps_y: f64,
        _dt_probe: f64,
        pred: &P,
    ) {
        if !s1.shares_endpoint(s2) && pred.intersect(s1, s2) {
            self.reported.insert(pack_pair(s1.id, s2.id));
        }
    }

    #[inline]
    fn on_band_pair<P: CrossPred>(
        &mut self,
        s_lo: &Seg,
        s_hi: &Seg,
        _sweep_x: f64,
        _eps_y: f64,
        _dt_probe: f64,
        _opt_xi: Option<f64>,
        _pred: &P,
    ) {
        self.reported.insert(pack_pair(s_lo.id, s_hi.id));
    }

    #[inline]
    fn evict_before<P: CrossPred>(&mut self, _x_evt: f64, _segs: &[Seg], _eps_y: f64, _pred: &P) {}

    #[inline]
    fn flush_at_x<P: CrossPred>(
        &mut self,
        _sweep_x: f64,
        _next_x: f64,
        _segs: &[Seg],
        _eps_y: f64,
        _dt_probe: f64,
        _pred: &P,
    ) {
    }
}

pub(super) struct StreamingCollector<F: FnMut(usize, usize)> {
    on_crossing: F,
    at_x_inline: [(u64, u32, u32, f64); 8],
    at_x_inline_len: usize,
    at_x_overflow_set: PairSet,
    at_x_overflow_list: Vec<(u64, u32, u32, f64)>,
    active_band_reported: InlinePairSet,
    evict_heap: BinaryHeap<Reverse<(OrderedF64, u32, u32, u64)>>,
    pub(super) peak_active_band: usize,
}

impl<F: FnMut(usize, usize)> StreamingCollector<F> {
    pub(super) fn new(on_crossing: F) -> Self {
        Self {
            on_crossing,
            at_x_inline: [(0, 0, 0, f64::NAN); 8],
            at_x_inline_len: 0,
            at_x_overflow_set: PairSet::default(),
            at_x_overflow_list: Vec::new(),
            active_band_reported: InlinePairSet::new(),
            evict_heap: BinaryHeap::new(),
            peak_active_band: 0,
        }
    }

    #[inline]
    fn contains_at_x(&self, pair: u64) -> bool {
        if self.at_x_overflow_list.is_empty() {
            self.at_x_inline[..self.at_x_inline_len]
                .iter()
                .any(|&(p, _, _, _)| p == pair)
        } else {
            self.at_x_overflow_set.contains(&pair)
        }
    }

    #[inline]
    fn is_reported(&self, pair: u64) -> bool {
        self.contains_at_x(pair)
            || (!self.active_band_reported.is_empty() && self.active_band_reported.contains(pair))
    }

    #[inline]
    fn insert_at_x(&mut self, pair: u64, a_id: u32, b_id: u32, x_base: f64) {
        if self.at_x_overflow_list.is_empty() {
            if self.at_x_inline_len < self.at_x_inline.len() {
                self.at_x_inline[self.at_x_inline_len] = (pair, a_id, b_id, x_base);
                self.at_x_inline_len += 1;
                return;
            }
            for &(p, aid, bid, xb) in &self.at_x_inline {
                self.at_x_overflow_set.insert(p);
                self.at_x_overflow_list.push((p, aid, bid, xb));
            }
            self.at_x_inline_len = 0;
        }
        if self.at_x_overflow_set.insert(pair) {
            self.at_x_overflow_list.push((pair, a_id, b_id, x_base));
        }
    }

    #[inline]
    fn emit_pair_unchecked(&mut self, pair: u64, a: usize, b: usize, x_base: f64) {
        self.insert_at_x(pair, a as u32, b as u32, x_base);
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        (self.on_crossing)(lo, hi);
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn promote_band_pair<P: CrossPred>(
    pair: u64,
    a_id: u32,
    b_id: u32,
    x_base: f64,
    sweep_x: f64,
    next_x: f64,
    segs: &[Seg],
    eps_y: f64,
    dt_probe: f64,
    pred: &P,
    active_band_reported: &mut InlinePairSet,
    evict_heap: &mut BinaryHeap<Reverse<(OrderedF64, u32, u32, u64)>>,
) {
    if x_base.is_nan() {
        return;
    }
    let a = &segs[a_id as usize];
    let b = &segs[b_id as usize];
    let r = a.x2.min(b.x2);
    if next_x > r || sweep_x >= r {
        return;
    }
    let in_band_next = within_band(a, b, next_x, eps_y, pred);
    let ds = (a.slope - b.slope).abs();
    let steep = a.slope.abs() >= STEEP_SLOPE_THRESHOLD || b.slope.abs() >= STEEP_SLOPE_THRESHOLD;
    let x_evict = if steep || ds == 0.0 || !ds.is_finite() {
        r
    } else {
        let dy = (pred.y_at(a, sweep_x) - pred.y_at(b, sweep_x)).abs();
        let margin = dy + 2.0 * eps_y + 2.0 * (seg_y_err(a, r) + seg_y_err(b, r));
        let span = (margin / ds) * (1.0 + 16.0 * f64::EPSILON) + dt_probe;
        let base = x_base.max(sweep_x);
        let cand = base + span;
        if !cand.is_finite() || (cand == base && span > 0.0) {
            r
        } else {
            r.min(cand)
        }
    };
    let final_evict = if in_band_next {
        x_evict.max(next_x)
    } else {
        x_evict
    };
    if final_evict >= next_x && active_band_reported.insert(pair) {
        evict_heap.push(Reverse((OrderedF64(final_evict), a_id, b_id, pair)));
    }
}

impl<F: FnMut(usize, usize)> CrossingCollector for StreamingCollector<F> {
    #[inline]
    fn on_vertical_pair(&mut self, a: usize, b: usize) {
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        (self.on_crossing)(lo, hi);
    }

    #[inline]
    fn on_swap_pair<P: CrossPred>(
        &mut self,
        s1: &Seg,
        s2: &Seg,
        sweep_x: f64,
        _eps_y: f64,
        _dt_probe: f64,
        pred: &P,
    ) {
        if !s1.shares_endpoint(s2) && pred.intersect(s1, s2) {
            if pred.is_collinear_det_zero(s1, s2) {
                // Collinear/parallel pairs with det == 0.0 are emitted exclusively at `sweep_x == l`
                // inside `on_band_pair` without cross-x_evt tracking.
                return;
            }
            let pair = pack_pair(s1.id, s2.id);
            if !self.is_reported(pair) {
                let x_base = if pred.cmp_slope(s1, s2) == Ordering::Less {
                    pred.intersection_x(s1, s2)
                        .map_or(sweep_x, |xi| sweep_x.max(xi))
                } else {
                    sweep_x
                };
                self.emit_pair_unchecked(pair, s1.id, s2.id, x_base);
            }
        }
    }

    #[inline]
    fn on_band_pair<P: CrossPred>(
        &mut self,
        s_lo: &Seg,
        s_hi: &Seg,
        sweep_x: f64,
        eps_y: f64,
        _dt_probe: f64,
        opt_xi: Option<f64>,
        pred: &P,
    ) {
        match opt_xi {
            None => {
                // det == 0.0 (collinear / parallel, or sub-1e-160 underflow).
                let l = s_lo.x1.max(s_hi.x1);
                if sweep_x == l {
                    let pair = pack_pair(s_lo.id, s_hi.id);
                    if !self.is_reported(pair) {
                        let x_base = if pred.cmp_slope(s_lo, s_hi) == Ordering::Equal {
                            f64::NAN
                        } else {
                            sweep_x
                        };
                        self.emit_pair_unchecked(pair, s_lo.id, s_hi.id, x_base);
                    }
                }
            }
            Some(xi) => {
                // If an interior crossing falls inside the final ULP before a segment ends
                // (`x2 <= sweep_x`), `can_schedule` cannot defer a swap so emit immediately.
                let unscheduled_ending_cross = (s_lo.x2 <= sweep_x || s_hi.x2 <= sweep_x)
                    && pred.cmp_slope(s_lo, s_hi) == Ordering::Greater;
                if pred.xi_eq_sweep(s_lo, s_hi, xi, sweep_x)
                    || unscheduled_ending_cross
                    || within_band(s_lo, s_hi, sweep_x, eps_y, pred)
                {
                    let pair = pack_pair(s_lo.id, s_hi.id);
                    if !self.is_reported(pair) {
                        self.emit_pair_unchecked(pair, s_lo.id, s_hi.id, sweep_x.max(xi));
                    }
                }
            }
        }
    }

    fn evict_before<P: CrossPred>(&mut self, x_evt: f64, segs: &[Seg], eps_y: f64, pred: &P) {
        while let Some(&Reverse((OrderedF64(ex), a_id, b_id, pair))) = self.evict_heap.peek() {
            if ex < x_evt {
                let a = &segs[a_id as usize];
                let b = &segs[b_id as usize];
                // Keep pairs in `active_band_reported` while both segments remain active and
                // inside `within_band`, preventing delayed re-emission after conservative `x_evict`.
                if a.x2 >= x_evt && b.x2 >= x_evt && within_band(a, b, x_evt, eps_y, pred) {
                    self.evict_heap.pop();
                    let r = a.x2.min(b.x2);
                    self.evict_heap
                        .push(Reverse((OrderedF64(r), a_id, b_id, pair)));
                    continue;
                }
                self.evict_heap.pop();
                self.active_band_reported.remove(pair);
            } else {
                break;
            }
        }
    }

    fn flush_at_x<P: CrossPred>(
        &mut self,
        sweep_x: f64,
        next_x: f64,
        segs: &[Seg],
        eps_y: f64,
        dt_probe: f64,
        pred: &P,
    ) {
        if self.at_x_inline_len == 0 && self.at_x_overflow_list.is_empty() {
            return;
        }
        if next_x == f64::INFINITY {
            self.at_x_inline_len = 0;
            if !self.at_x_overflow_list.is_empty() {
                self.at_x_overflow_set.clear();
                self.at_x_overflow_list.clear();
            }
            return;
        }
        if self.at_x_overflow_list.is_empty() {
            for i in 0..self.at_x_inline_len {
                let (pair, a_id, b_id, x_base) = self.at_x_inline[i];
                promote_band_pair(
                    pair,
                    a_id,
                    b_id,
                    x_base,
                    sweep_x,
                    next_x,
                    segs,
                    eps_y,
                    dt_probe,
                    pred,
                    &mut self.active_band_reported,
                    &mut self.evict_heap,
                );
            }
            self.at_x_inline_len = 0;
        } else {
            for &(pair, a_id, b_id, x_base) in &self.at_x_overflow_list {
                promote_band_pair(
                    pair,
                    a_id,
                    b_id,
                    x_base,
                    sweep_x,
                    next_x,
                    segs,
                    eps_y,
                    dt_probe,
                    pred,
                    &mut self.active_band_reported,
                    &mut self.evict_heap,
                );
            }
            self.at_x_overflow_set.clear();
            self.at_x_overflow_list.clear();
        }
        self.peak_active_band = self.peak_active_band.max(self.active_band_reported.len());
    }
}
