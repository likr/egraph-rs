use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use super::active_tree::ActiveSegmentTree;
use super::band_reorder::sort_band_post_x;
use super::collector::CrossingCollector;
use super::cross_pred::CrossPred;
use super::geometry::{Seg, seg_y_err, within_band};
use super::pair_set::{InlinePairSet, pack_pair};
use super::sweep::{EventKind, SweepEvent};

pub(super) struct BandScratch {
    pub(super) dt_probe: f64,
    pub(super) cluster: Vec<(usize, usize)>,
    hub_buf: Vec<(usize, usize)>,
    pub(super) rest_buf: Vec<(usize, usize)>,
    pub(super) cur_buf: Vec<(usize, usize)>,
    pub(super) band_edges: Vec<usize>,
    pub(super) band_nodes: Vec<u32>,
    pub(super) surviving_touched: Vec<usize>,
    visited_stamp: Vec<u64>,
    visited_err: Vec<f64>,
    stamp: u64,
    pub(super) processed_int_at_x: InlinePairSet,
    pub(super) scheduled_at_x: InlinePairSet,
    pub(super) last_int_x: f64,
}

impl BandScratch {
    pub(super) fn new(num_edges: usize, dt_probe: f64) -> Self {
        Self {
            dt_probe,
            cluster: Vec::new(),
            hub_buf: Vec::new(),
            rest_buf: Vec::new(),
            cur_buf: Vec::new(),
            band_edges: Vec::new(),
            band_nodes: Vec::new(),
            surviving_touched: Vec::new(),
            visited_stamp: vec![0; num_edges],
            visited_err: vec![0.0; num_edges],
            stamp: 0,
            processed_int_at_x: InlinePairSet::new(),
            scheduled_at_x: InlinePairSet::new(),
            last_int_x: f64::NAN,
        }
    }
}

fn collect_band_around_seed<P: CrossPred>(
    seed: usize,
    x_evt: f64,
    eps_y: f64,
    segs: &[Seg],
    tree: &ActiveSegmentTree,
    out_edges: &mut Vec<usize>,
    pred: &P,
) {
    out_edges.clear();
    let s_seed = &segs[seed];
    let mut left_most = seed;
    let mut cur = seed;
    while let Some(prev) = tree.prev_edge(cur) {
        if !within_band(s_seed, &segs[prev], x_evt, eps_y, pred) {
            break;
        }
        left_most = prev;
        cur = prev;
    }

    out_edges.push(left_most);
    let mut cur_r = left_most;
    while let Some(nx) = tree.next_edge(cur_r) {
        if !within_band(s_seed, &segs[nx], x_evt, eps_y, pred) {
            break;
        }
        out_edges.push(nx);
        cur_r = nx;
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn check_eps_bands_at_x<C: CrossingCollector, P: CrossPred>(
    touched_edges: &[usize],
    x_evt: f64,
    eps_y: f64,
    segs: &[Seg],
    tree: &ActiveSegmentTree,
    events: &mut BinaryHeap<Reverse<SweepEvent>>,
    collector: &mut C,
    scratch: &mut BandScratch,
    pred: &P,
) {
    scratch.stamp = scratch.stamp.wrapping_add(1).max(1);
    let cur_stamp = scratch.stamp;

    for &seed in touched_edges {
        if !tree.is_active(seed) {
            continue;
        }
        let seed_err = seg_y_err(&segs[seed], x_evt);
        if scratch.visited_stamp[seed] == cur_stamp && seed_err <= scratch.visited_err[seed] + eps_y
        {
            continue;
        }

        collect_band_around_seed(
            seed,
            x_evt,
            eps_y,
            segs,
            tree,
            &mut scratch.band_edges,
            pred,
        );
        for &e in &scratch.band_edges {
            if scratch.visited_stamp[e] != cur_stamp || seed_err > scratch.visited_err[e] {
                scratch.visited_stamp[e] = cur_stamp;
                scratch.visited_err[e] = seed_err;
            }
        }

        let first_edge = scratch.band_edges[0];
        let last_edge = scratch.band_edges[scratch.band_edges.len() - 1];
        if let Some(left) = tree.prev_edge(first_edge) {
            check_adjacent_and_schedule(
                left,
                first_edge,
                x_evt,
                eps_y,
                segs,
                events,
                collector,
                scratch.dt_probe,
                &scratch.processed_int_at_x,
                &mut scratch.scheduled_at_x,
                pred,
            );
        }
        if let Some(right) = tree.next_edge(last_edge) {
            check_adjacent_and_schedule(
                last_edge,
                right,
                x_evt,
                eps_y,
                segs,
                events,
                collector,
                scratch.dt_probe,
                &scratch.processed_int_at_x,
                &mut scratch.scheduled_at_x,
                pred,
            );
        }

        if scratch.band_edges.len() >= 2 {
            scratch.cluster.clear();
            for (pos, &e) in scratch.band_edges.iter().enumerate() {
                scratch.cluster.push((pos, e));
            }
            check_band_pairs(x_evt, eps_y, segs, events, collector, scratch, pred);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn reorder_surviving_bands_at_x<C: CrossingCollector, P: CrossPred>(
    x_evt: f64,
    eps_y: f64,
    segs: &[Seg],
    tree: &mut ActiveSegmentTree,
    events: &mut BinaryHeap<Reverse<SweepEvent>>,
    collector: &mut C,
    scratch: &mut BandScratch,
    pred: &P,
) {
    scratch.stamp = scratch.stamp.wrapping_add(1).max(1);
    let cur_stamp = scratch.stamp;

    for idx in 0..scratch.surviving_touched.len() {
        let seed = scratch.surviving_touched[idx];
        if !tree.is_active(seed) {
            continue;
        }
        let seed_err = seg_y_err(&segs[seed], x_evt);
        if scratch.visited_stamp[seed] == cur_stamp && seed_err <= scratch.visited_err[seed] + eps_y
        {
            continue;
        }

        collect_band_around_seed(
            seed,
            x_evt,
            eps_y,
            segs,
            tree,
            &mut scratch.band_edges,
            pred,
        );
        for &e in &scratch.band_edges {
            if scratch.visited_stamp[e] != cur_stamp || seed_err > scratch.visited_err[e] {
                scratch.visited_stamp[e] = cur_stamp;
                scratch.visited_err[e] = seed_err;
            }
        }
        if scratch.band_edges.len() < 2 {
            continue;
        }

        let is_sorted = scratch
            .band_edges
            .windows(2)
            .all(|w| pred.cmp_post_x(&segs[w[0]], &segs[w[1]], x_evt) != Ordering::Greater);
        if is_sorted {
            continue;
        }

        // Check all band pairs before permuting so near-coincident crossings are never lost.
        scratch.cluster.clear();
        scratch.band_nodes.clear();
        for (pos, &e) in scratch.band_edges.iter().enumerate() {
            scratch.cluster.push((pos, e));
            scratch.band_nodes.push(tree.node_of_edge[e]);
        }
        check_band_pairs(x_evt, eps_y, segs, events, collector, scratch, pred);
        sort_band_post_x(x_evt, eps_y, segs, tree, collector, scratch, pred);

        if let (Some(&first_e), Some(&last_e)) =
            (scratch.band_edges.first(), scratch.band_edges.last())
        {
            if let Some(left) = tree.prev_edge(first_e) {
                check_adjacent_and_schedule(
                    left,
                    first_e,
                    x_evt,
                    eps_y,
                    segs,
                    events,
                    collector,
                    scratch.dt_probe,
                    &scratch.processed_int_at_x,
                    &mut scratch.scheduled_at_x,
                    pred,
                );
            }
            if let Some(right) = tree.next_edge(last_e) {
                check_adjacent_and_schedule(
                    last_e,
                    right,
                    x_evt,
                    eps_y,
                    segs,
                    events,
                    collector,
                    scratch.dt_probe,
                    &scratch.processed_int_at_x,
                    &mut scratch.scheduled_at_x,
                    pred,
                );
            }
        }
        for win in scratch.band_edges.windows(2) {
            scratch.processed_int_at_x.insert(pack_pair(win[0], win[1]));
            check_adjacent_and_schedule(
                win[0],
                win[1],
                x_evt,
                eps_y,
                segs,
                events,
                collector,
                scratch.dt_probe,
                &scratch.processed_int_at_x,
                &mut scratch.scheduled_at_x,
                pred,
            );
        }
    }
}

/// Checks all crossing pairs inside `scratch.cluster` while pruning same-vertex star fans
/// via 4-slot Misra-Gries heavy-hitter detection so pure or nested stars remain $O(|C|)$.
pub(super) fn check_band_pairs<C: CrossingCollector, P: CrossPred>(
    x_evt: f64,
    eps_y: f64,
    segs: &[Seg],
    events: &mut BinaryHeap<Reverse<SweepEvent>>,
    collector: &mut C,
    scratch: &mut BandScratch,
    pred: &P,
) {
    scratch.cur_buf.clear();
    scratch.cur_buf.extend_from_slice(&scratch.cluster);

    while scratch.cur_buf.len() >= 2 {
        let n = scratch.cur_buf.len();
        if n > 32
            && let Some(hub_v) = dominant_shared_vertex(&scratch.cur_buf, segs)
        {
            scratch.hub_buf.clear();
            scratch.rest_buf.clear();
            for &(pos, e) in &scratch.cur_buf {
                let s = &segs[e];
                if s.u == hub_v || s.v == hub_v {
                    scratch.hub_buf.push((pos, e));
                } else {
                    scratch.rest_buf.push((pos, e));
                }
            }
            // Segments in `hub_buf` share `hub_v` at the cluster point and cannot report crossings.
            for &(pos_h, edge_h) in &scratch.hub_buf {
                for &(pos_r, edge_r) in &scratch.rest_buf {
                    let (lo, hi) = if pos_h < pos_r {
                        (edge_h, edge_r)
                    } else {
                        (edge_r, edge_h)
                    };
                    check_adjacent_and_schedule(
                        lo,
                        hi,
                        x_evt,
                        eps_y,
                        segs,
                        events,
                        collector,
                        scratch.dt_probe,
                        &scratch.processed_int_at_x,
                        &mut scratch.scheduled_at_x,
                        pred,
                    );
                }
            }
            std::mem::swap(&mut scratch.cur_buf, &mut scratch.rest_buf);
            continue;
        }

        for i in 0..n {
            let (pos_a, edge_a) = scratch.cur_buf[i];
            for j in (i + 1)..n {
                let (pos_b, edge_b) = scratch.cur_buf[j];
                let (lo, hi) = if pos_a < pos_b {
                    (edge_a, edge_b)
                } else {
                    (edge_b, edge_a)
                };
                check_adjacent_and_schedule(
                    lo,
                    hi,
                    x_evt,
                    eps_y,
                    segs,
                    events,
                    collector,
                    scratch.dt_probe,
                    &scratch.processed_int_at_x,
                    &mut scratch.scheduled_at_x,
                    pred,
                );
            }
        }
        break;
    }
}

fn dominant_shared_vertex(items: &[(usize, usize)], segs: &[Seg]) -> Option<u32> {
    let mut slots: [(Option<u32>, usize); 4] = [(None, 0); 4];
    for &(_, e) in items {
        let s = &segs[e];
        for v in [s.u, s.v] {
            if let Some((_, cnt)) = slots.iter_mut().find(|(c, _)| *c == Some(v)) {
                *cnt += 1;
            } else if let Some(slot) = slots.iter_mut().find(|(_, cnt)| *cnt == 0) {
                *slot = (Some(v), 1);
            } else {
                for slot in &mut slots {
                    slot.1 -= 1;
                    if slot.1 == 0 {
                        slot.0 = None;
                    }
                }
            }
        }
    }

    let mut best_v = None;
    let mut best_cnt = 0usize;
    for (cand, _) in slots {
        let Some(v) = cand else {
            continue;
        };
        let cnt = items
            .iter()
            .filter(|&&(_, e)| segs[e].u == v || segs[e].v == v)
            .count();
        if cnt > best_cnt {
            best_cnt = cnt;
            best_v = Some(v);
        }
    }
    if best_cnt >= 4 { best_v } else { None }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn check_adjacent_and_schedule<C: CrossingCollector, P: CrossPred>(
    lower: usize,
    upper: usize,
    sweep_x: f64,
    eps_y: f64,
    segs: &[Seg],
    events: &mut BinaryHeap<Reverse<SweepEvent>>,
    collector: &mut C,
    dt_probe: f64,
    processed_int_at_x: &InlinePairSet,
    scheduled_at_x: &mut InlinePairSet,
    pred: &P,
) {
    let s_lo = &segs[lower];
    let s_hi = &segs[upper];
    let shares_ep = s_lo.shares_endpoint(s_hi);
    if shares_ep && pred.cmp_slope(s_lo, s_hi) != Ordering::Greater {
        return;
    }
    if !pred.intersect(s_lo, s_hi) {
        return;
    }
    let can_schedule = s_lo.x2 > sweep_x
        && s_hi.x2 > sweep_x
        && pred.cmp_slope(s_lo, s_hi) == Ordering::Greater
        && !pred.same_right_endpoint(s_lo, s_hi);
    if shares_ep && !can_schedule {
        return;
    }
    let opt_xi = pred.intersection_x(s_lo, s_hi);
    if !shares_ep {
        collector.on_band_pair(s_lo, s_hi, sweep_x, eps_y, dt_probe, opt_xi, pred);
    }
    // Schedule interior swaps even when segments share graph vertex IDs (e.g. wrapped torus pieces)
    // so sweep order stays valid past interior crossings, while skipping shared right tips.
    if can_schedule && let Some(x_int) = opt_xi {
        let x_push = x_int.max(sweep_x);
        let pair = pack_pair(lower, upper);
        let should_push = if x_push > sweep_x {
            scheduled_at_x.insert(pair)
        } else {
            !processed_int_at_x.contains(pair) && scheduled_at_x.insert(pair)
        };
        if should_push {
            events.push(Reverse(SweepEvent {
                x: x_push,
                kind: EventKind::Intersection {
                    e1: lower,
                    e2: upper,
                },
            }));
        }
    }
}
