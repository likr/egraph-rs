use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use super::CrossingPair;
use super::active_tree::ActiveSegmentTree;
use super::band::{BandScratch, check_adjacent_and_schedule, check_band_pairs};
use super::collector::CrossingCollector;
use super::cross_pred::CrossPred;
use super::geometry::{Seg, crossing_angle_metrics, seg_y_err};
use super::pair_set::{PairSet, pack_pair, unpack_pair};
use super::sweep::SweepEvent;

#[allow(clippy::too_many_arguments)]
fn emit_flipped_pairs<C: CrossingCollector, P: CrossPred>(
    x_evt: f64,
    eps_y: f64,
    segs: &[Seg],
    collector: &mut C,
    scratch: &mut BandScratch,
    pred: &P,
) {
    let k = scratch.cluster.len();
    if k <= 1 {
        return;
    }
    if k <= 64 {
        for i in 0..k {
            let (pos_u, edge_u) = scratch.cluster[i];
            for j in (i + 1)..k {
                let (pos_v, edge_v) = scratch.cluster[j];
                if pos_u > pos_v {
                    collector.on_swap_pair(
                        &segs[edge_v],
                        &segs[edge_u],
                        x_evt,
                        eps_y,
                        scratch.dt_probe,
                        pred,
                    );
                }
            }
        }
        return;
    }

    // Bottom-up merge sort on `orig_pos` using `cur_buf` / `rest_buf` so k > 64 remains
    // O(k log k + inversions) without per-event allocations.
    scratch.cur_buf.clear();
    scratch.cur_buf.extend_from_slice(&scratch.cluster);
    scratch.rest_buf.resize(k, (0, 0));

    let mut width = 1usize;
    while width < k {
        let mut lo = 0usize;
        while lo < k {
            let mid = (lo + width).min(k);
            let hi = (lo + 2 * width).min(k);
            let mut l = lo;
            let mut r = mid;
            let mut out = lo;
            while l < mid && r < hi {
                if scratch.cur_buf[l].0 <= scratch.cur_buf[r].0 {
                    scratch.rest_buf[out] = scratch.cur_buf[l];
                    l += 1;
                } else {
                    let (_, edge_v) = scratch.cur_buf[r];
                    for &(_, edge_u) in &scratch.cur_buf[l..mid] {
                        collector.on_swap_pair(
                            &segs[edge_v],
                            &segs[edge_u],
                            x_evt,
                            eps_y,
                            scratch.dt_probe,
                            pred,
                        );
                    }
                    scratch.rest_buf[out] = scratch.cur_buf[r];
                    r += 1;
                }
                out += 1;
            }
            while l < mid {
                scratch.rest_buf[out] = scratch.cur_buf[l];
                l += 1;
                out += 1;
            }
            while r < hi {
                scratch.rest_buf[out] = scratch.cur_buf[r];
                r += 1;
                out += 1;
            }
            lo += 2 * width;
        }
        std::mem::swap(&mut scratch.cur_buf, &mut scratch.rest_buf);
        width *= 2;
    }
}

/// Sorts `scratch.cluster` into exact post-`x_evt` order via `cmp_post_x`, resolves any local
/// cyclic adjacent inversion spans by `(slope, id)`, notifies `collector.on_swap_pair` for every
/// flipped pair, and writes the reordered edge indices back to `tree` and `scratch.band_edges`.
#[allow(clippy::too_many_arguments)]
pub(super) fn sort_band_post_x<C: CrossingCollector, P: CrossPred>(
    x_evt: f64,
    eps_y: f64,
    segs: &[Seg],
    tree: &mut ActiveSegmentTree,
    collector: &mut C,
    scratch: &mut BandScratch,
    pred: &P,
) {
    let k = scratch.cluster.len();
    if k <= 32 {
        // Insertion sort tolerates rare O(ULP) 3-cycles in `cmp_post_x` without triggering
        // `slice::sort_unstable_by` total-order assertions.
        for i in 1..k {
            let key = scratch.cluster[i];
            let mut j = i;
            while j > 0
                && pred.cmp_post_x(&segs[key.1], &segs[scratch.cluster[j - 1].1], x_evt)
                    == Ordering::Less
            {
                scratch.cluster[j] = scratch.cluster[j - 1];
                j -= 1;
            }
            scratch.cluster[j] = key;
        }
    } else {
        // Bottom-up merge sort using `cur_buf` / `rest_buf` (cycle-safe, zero allocations).
        scratch.cur_buf.clear();
        scratch.cur_buf.extend_from_slice(&scratch.cluster);
        scratch.rest_buf.resize(k, (0, 0));
        let mut width = 1usize;
        while width < k {
            let mut lo = 0usize;
            while lo < k {
                let mid = (lo + width).min(k);
                let hi = (lo + 2 * width).min(k);
                let mut l = lo;
                let mut r = mid;
                let mut out = lo;
                while l < mid && r < hi {
                    if pred.cmp_post_x(
                        &segs[scratch.cur_buf[l].1],
                        &segs[scratch.cur_buf[r].1],
                        x_evt,
                    ) != Ordering::Greater
                    {
                        scratch.rest_buf[out] = scratch.cur_buf[l];
                        l += 1;
                    } else {
                        scratch.rest_buf[out] = scratch.cur_buf[r];
                        r += 1;
                    }
                    out += 1;
                }
                while l < mid {
                    scratch.rest_buf[out] = scratch.cur_buf[l];
                    l += 1;
                    out += 1;
                }
                while r < hi {
                    scratch.rest_buf[out] = scratch.cur_buf[r];
                    r += 1;
                    out += 1;
                }
                lo += 2 * width;
            }
            std::mem::swap(&mut scratch.cur_buf, &mut scratch.rest_buf);
            width *= 2;
        }
        scratch.cluster.clear();
        scratch.cluster.extend_from_slice(&scratch.cur_buf);
    }
    let mut pass = 0usize;
    loop {
        let mut had_inversion = false;
        let mut i = 0usize;
        while i + 1 < k {
            if pred.cmp_post_x(
                &segs[scratch.cluster[i].1],
                &segs[scratch.cluster[i + 1].1],
                x_evt,
            ) == Ordering::Greater
            {
                had_inversion = true;
                let lo = i;
                let mut hi = i + 1;
                while hi + 1 < k
                    && pred.cmp_post_x(
                        &segs[scratch.cluster[hi].1],
                        &segs[scratch.cluster[hi + 1].1],
                        x_evt,
                    ) == Ordering::Greater
                {
                    hi += 1;
                }
                if hi > lo + 1 {
                    scratch.cluster[lo..=hi].sort_unstable_by(|a, b| {
                        pred.cmp_slope(&segs[a.1], &segs[b.1])
                            .then_with(|| a.1.cmp(&b.1))
                    });
                } else {
                    scratch.cluster.swap(lo, hi);
                }
                i = hi + 1;
            } else {
                i += 1;
            }
        }
        if !had_inversion {
            break;
        }
        pass += 1;
        if pass >= k {
            scratch.cluster.sort_unstable_by(|a, b| {
                pred.cmp_slope(&segs[a.1], &segs[b.1])
                    .then_with(|| a.1.cmp(&b.1))
            });
            break;
        }
    }

    emit_flipped_pairs(x_evt, eps_y, segs, collector, scratch, pred);

    scratch.band_edges.clear();
    for (&node_id, &(_, edge_id)) in scratch.band_nodes.iter().zip(scratch.cluster.iter()) {
        tree.nodes[node_id as usize].edge_idx = edge_id;
        tree.node_of_edge[edge_id] = node_id;
        scratch.band_edges.push(edge_id);
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn handle_intersection<C: CrossingCollector, P: CrossPred>(
    e1: usize,
    e2: usize,
    x_evt: f64,
    eps_y: f64,
    segs: &[Seg],
    tree: &mut ActiveSegmentTree,
    events: &mut BinaryHeap<Reverse<SweepEvent>>,
    collector: &mut C,
    touched_edges: &mut Vec<usize>,
    scratch: &mut BandScratch,
    pred: &P,
) {
    if !tree.is_active(e1) || !tree.is_active(e2) {
        return;
    }
    let Some(r1) = tree.rank_of_edge(e1) else {
        return;
    };
    let Some(r2) = tree.rank_of_edge(e2) else {
        return;
    };
    if r1 >= r2 {
        return;
    }

    if r1 + 1 == r2 {
        tree.swap_adjacent_edges(e1, e2);
        collector.on_swap_pair(&segs[e1], &segs[e2], x_evt, eps_y, scratch.dt_probe, pred);
        scratch.processed_int_at_x.insert(pack_pair(e1, e2));
        // After swap, `e2` precedes `e1`.
        let left = tree.prev_edge(e2);
        let right = tree.next_edge(e1);
        let s1 = &segs[e1];
        let s2 = &segs[e2];
        let y1 = pred.y_at(s1, x_evt);
        let y2 = pred.y_at(s2, x_evt);
        let err1 = seg_y_err(s1, x_evt);
        let err2 = seg_y_err(s2, x_evt);
        let left_isolated = match left {
            None => true,
            Some(l) => {
                let sl = &segs[l];
                let yl = pred.y_at(sl, x_evt);
                let err_l = seg_y_err(sl, x_evt);
                y2 - yl > 2.0 * eps_y + err_l + err2 && y1 - yl > 2.0 * eps_y + err_l + err1
            }
        };
        let right_isolated = match right {
            None => true,
            Some(r) => {
                let sr = &segs[r];
                let yr = pred.y_at(sr, x_evt);
                let err_r = seg_y_err(sr, x_evt);
                yr - y1 > 2.0 * eps_y + err1 + err_r && yr - y2 > 2.0 * eps_y + err2 + err_r
            }
        };
        if !left_isolated || !right_isolated {
            touched_edges.push(e1);
            touched_edges.push(e2);
        }
        if let Some(left) = left {
            check_adjacent_and_schedule(
                left,
                e2,
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
        if let Some(right) = right {
            check_adjacent_and_schedule(
                e1,
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
        return;
    }

    // Multi-way intersection or 1-ULP tie: sort [e1..=e2] by exact post-x_evt order.
    scratch.band_nodes.clear();
    scratch.band_edges.clear();
    let mut cur_edge = e1;
    let span = r2 - r1;
    for _ in 0..=span {
        scratch.band_nodes.push(tree.node_of_edge[cur_edge]);
        scratch.band_edges.push(cur_edge);
        if cur_edge == e2 {
            break;
        }
        let Some(nx) = tree.next_edge(cur_edge) else {
            break;
        };
        cur_edge = nx;
    }

    if scratch.band_edges.last() != Some(&e2) {
        return;
    }

    scratch.processed_int_at_x.insert(pack_pair(e1, e2));
    touched_edges.push(e1);
    touched_edges.push(e2);

    // Check pairwise crossings inside the multi-way band while pruning shared-hub star pairs.
    scratch.cluster.clear();
    for (pos, &edge) in scratch.band_edges.iter().enumerate() {
        scratch.cluster.push((pos, edge));
    }
    check_band_pairs(x_evt, eps_y, segs, events, collector, scratch, pred);
    sort_band_post_x(x_evt, eps_y, segs, tree, collector, scratch, pred);

    if let (Some(&first_e), Some(&last_e)) = (scratch.band_edges.first(), scratch.band_edges.last())
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

pub(super) fn finalize_pairs(reported: &PairSet, segs: &[Seg]) -> Vec<CrossingPair> {
    let mut pairs: Vec<CrossingPair> = reported
        .iter()
        .map(|&key| {
            let (a, b) = unpack_pair(key);
            let sa = &segs[a];
            let sb = &segs[b];
            let (angle_rad, cos_sq) = crossing_angle_metrics(sa, sb);
            CrossingPair {
                edge_a: a,
                edge_b: b,
                u1: sa.u,
                v1: sa.v,
                u2: sb.u,
                v2: sb.v,
                angle_rad,
                cos_sq,
            }
        })
        .collect();
    pairs.sort_unstable_by_key(|p| (p.edge_a, p.edge_b));
    pairs
}
