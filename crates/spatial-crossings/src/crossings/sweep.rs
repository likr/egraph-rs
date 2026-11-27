use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use super::active_tree::ActiveSegmentTree;
use super::band::{
    BandScratch, check_adjacent_and_schedule, check_eps_bands_at_x, reorder_surviving_bands_at_x,
};
use super::band_reorder::handle_intersection;
use super::collector::{CrossingCollector, PairSetCollector, StreamingCollector};
use super::cross_pred::{CrossPred, EuclideanPred};
use super::geometry::{Seg, seg_y_err};
use super::pair_set::PairSet;

#[derive(Debug, Clone, Copy)]
pub(super) enum EventKind {
    Intersection { e1: usize, e2: usize },
    Start { edge: usize },
    Vertical { v_idx: usize },
    End { edge: usize },
}

impl EventKind {
    #[inline]
    fn priority(self) -> u8 {
        match self {
            Self::Intersection { .. } => 0,
            Self::Start { .. } => 1,
            Self::Vertical { .. } => 2,
            Self::End { .. } => 3,
        }
    }

    #[inline]
    fn tie_key(self) -> (usize, usize) {
        match self {
            Self::Intersection { e1, e2 } => (e1, e2),
            Self::Start { edge } | Self::Vertical { v_idx: edge } | Self::End { edge } => (edge, 0),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct SweepEvent {
    pub(super) x: f64,
    pub(super) kind: EventKind,
}

impl PartialEq for SweepEvent {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for SweepEvent {}

impl PartialOrd for SweepEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SweepEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        self.x
            .total_cmp(&other.x)
            .then_with(|| self.kind.priority().cmp(&other.kind.priority()))
            .then_with(|| self.kind.tie_key().cmp(&other.kind.tie_key()))
    }
}

pub(super) fn sweep_segments(segs: &[Seg]) -> PairSet {
    sweep_segments_pred(segs, &EuclideanPred)
}

pub(super) fn sweep_segments_pred<P: CrossPred>(segs: &[Seg], pred: &P) -> PairSet {
    let mut collector = PairSetCollector {
        reported: PairSet::default(),
    };
    sweep_segments_core(segs, &mut collector, pred);
    collector.reported
}

pub(super) fn sweep_segments_with<F: FnMut(usize, usize)>(segs: &[Seg], on_crossing: F) {
    let _ = sweep_segments_with_stats(segs, on_crossing);
}

pub(super) fn sweep_segments_with_pred<P: CrossPred, F: FnMut(usize, usize)>(
    segs: &[Seg],
    pred: &P,
    on_crossing: F,
) {
    let mut collector = StreamingCollector::new(on_crossing);
    sweep_segments_core(segs, &mut collector, pred);
}

pub(super) fn sweep_segments_with_stats<F: FnMut(usize, usize)>(
    segs: &[Seg],
    on_crossing: F,
) -> usize {
    let mut collector = StreamingCollector::new(on_crossing);
    sweep_segments_core(segs, &mut collector, &EuclideanPred);
    collector.peak_active_band
}

fn sweep_segments_core<C: CrossingCollector, P: CrossPred>(
    segs: &[Seg],
    collector: &mut C,
    pred: &P,
) {
    if segs.len() < 2 {
        return;
    }

    let mut max_coord = 1.0_f64;
    let mut non_vert_count = 0usize;
    let mut vert_indices = Vec::new();

    for s in segs {
        if s.u == s.v {
            continue;
        }
        max_coord = max_coord
            .max(s.x1.abs())
            .max(s.x2.abs())
            .max(s.y1.abs())
            .max(s.y2.abs());
        if s.is_vertical {
            vert_indices.push(s.id);
        } else {
            non_vert_count += 1;
        }
    }

    let mut max_seg_y_err = 0.0_f64;
    let eps_y = 1e-10 * max_coord + 1e-12;
    for s in segs {
        if s.u != s.v && !s.is_vertical {
            max_seg_y_err =
                max_seg_y_err.max(seg_y_err(s, s.x_max).min((s.y_max - s.y_min) + eps_y));
        }
    }

    // 1. Check vertical-vs-vertical crossings via 1D sweep grouped by exact x coordinate.
    if vert_indices.len() >= 2 {
        vert_indices.sort_unstable_by(|&a, &b| {
            segs[a]
                .x1
                .total_cmp(&segs[b].x1)
                .then_with(|| segs[a].y_min.total_cmp(&segs[b].y_min))
                .then_with(|| segs[a].y_max.total_cmp(&segs[b].y_max))
                .then_with(|| a.cmp(&b))
        });

        let mut i = 0;
        let mut active_y: Vec<usize> = Vec::new();
        while i < vert_indices.len() {
            let x_cur = segs[vert_indices[i]].x1;
            let mut j = i;
            while j < vert_indices.len() && segs[vert_indices[j]].x1 == x_cur {
                j += 1;
            }
            if j - i >= 2 {
                active_y.clear();
                for &v_id in &vert_indices[i..j] {
                    let s_cur = &segs[v_id];
                    active_y.retain(|&prev_id| segs[prev_id].y_max >= s_cur.y_min);
                    for &prev_id in &active_y {
                        if pred.cross(s_cur, &segs[prev_id]) {
                            collector.on_vertical_pair(v_id, prev_id);
                        }
                    }
                    active_y.push(v_id);
                }
            }
            i = j;
        }
    }

    if non_vert_count == 0 {
        return;
    }

    // 2. X-sweep over non-vertical segments + vertical queries at exact x_v.
    let mut events: BinaryHeap<Reverse<SweepEvent>> =
        BinaryHeap::with_capacity(non_vert_count * 2 + vert_indices.len());
    for s in segs {
        if s.u == s.v || s.is_vertical {
            continue;
        }
        events.push(Reverse(SweepEvent {
            x: s.x1,
            kind: EventKind::Start { edge: s.id },
        }));
        events.push(Reverse(SweepEvent {
            x: s.x2,
            kind: EventKind::End { edge: s.id },
        }));
    }
    for &v_idx in &vert_indices {
        events.push(Reverse(SweepEvent {
            x: segs[v_idx].x1,
            kind: EventKind::Vertical { v_idx },
        }));
    }

    let mut tree = ActiveSegmentTree::new(segs.len());
    let mut started_batch: Vec<usize> = Vec::new();
    let mut end_batch: Vec<usize> = Vec::new();
    let mut vert_batch: Vec<usize> = Vec::new();
    let mut touched_edges: Vec<usize> = Vec::new();
    let mut scratch = BandScratch::new(segs.len(), (max_coord * 1e-11).max(1e-12));

    while let Some(Reverse(first)) = events.peek().copied() {
        let x_evt = first.x;
        if x_evt != scratch.last_int_x {
            scratch.processed_int_at_x.clear();
            scratch.scheduled_at_x.clear();
            scratch.last_int_x = x_evt;
            collector.evict_before(x_evt);
        }
        started_batch.clear();
        end_batch.clear();
        vert_batch.clear();
        touched_edges.clear();

        // Drain all events sharing exact x_evt, processing Intersections first so crossing
        // active pairs settle into post-crossing slope order before Starts/Verticals.
        while let Some(Reverse(ev)) = events.peek().copied() {
            if ev.x != x_evt {
                break;
            }
            events.pop();
            match ev.kind {
                EventKind::Intersection { e1, e2 } => {
                    handle_intersection(
                        e1,
                        e2,
                        x_evt,
                        eps_y,
                        segs,
                        &mut tree,
                        &mut events,
                        collector,
                        &mut touched_edges,
                        &mut scratch,
                        pred,
                    );
                }
                EventKind::Start { edge } => started_batch.push(edge),
                EventKind::Vertical { v_idx } => vert_batch.push(v_idx),
                EventKind::End { edge } => end_batch.push(edge),
            }
        }

        // Bubble any ending segments (`x2 == x_evt`) to a post-`x_evt` fixed point BEFORE
        // inserting `started_batch` or querying verticals, so co-located or adjacent ending
        // segments never block each other or leave an internal AVL node inverted.
        if !end_batch.is_empty() {
            scratch.surviving_touched.clear();
            for &edge in &end_batch {
                if !tree.is_active(edge) {
                    continue;
                }
                if let Some(p) = tree.prev_edge(edge)
                    && segs[p].x2 > x_evt
                {
                    scratch.surviving_touched.push(p);
                }
                if let Some(n) = tree.next_edge(edge)
                    && segs[n].x2 > x_evt
                {
                    scratch.surviving_touched.push(n);
                }
            }

            let max_passes = end_batch.len().max(1);
            for _drain_iter in 0..=max_passes {
                for _pass in 0..max_passes {
                    let mut pass_moved = false;
                    for &edge in &end_batch {
                        if !tree.is_active(edge) {
                            continue;
                        }
                        while let Some(next) = tree.next_edge(edge) {
                            if pred.cmp_post_x(&segs[edge], &segs[next], x_evt) != Ordering::Greater
                            {
                                break;
                            }
                            tree.swap_adjacent_edges(edge, next);
                            collector.on_swap_pair(
                                &segs[edge],
                                &segs[next],
                                x_evt,
                                eps_y,
                                scratch.dt_probe,
                                pred,
                            );
                            pass_moved = true;
                        }
                        while let Some(prev) = tree.prev_edge(edge) {
                            if pred.cmp_post_x(&segs[prev], &segs[edge], x_evt) != Ordering::Greater
                            {
                                break;
                            }
                            tree.swap_adjacent_edges(prev, edge);
                            collector.on_swap_pair(
                                &segs[prev],
                                &segs[edge],
                                x_evt,
                                eps_y,
                                scratch.dt_probe,
                                pred,
                            );
                            pass_moved = true;
                        }
                    }
                    if !pass_moved {
                        break;
                    }
                }

                for i in 0..scratch.surviving_touched.len() {
                    let s = scratch.surviving_touched[i];
                    if !tree.is_active(s) {
                        continue;
                    }
                    if let Some(p) = tree.prev_edge(s) {
                        check_adjacent_and_schedule(
                            p,
                            s,
                            x_evt,
                            eps_y,
                            segs,
                            &mut events,
                            collector,
                            scratch.dt_probe,
                            &scratch.processed_int_at_x,
                            &mut scratch.scheduled_at_x,
                            pred,
                        );
                    }
                    if let Some(n) = tree.next_edge(s) {
                        check_adjacent_and_schedule(
                            s,
                            n,
                            x_evt,
                            eps_y,
                            segs,
                            &mut events,
                            collector,
                            scratch.dt_probe,
                            &scratch.processed_int_at_x,
                            &mut scratch.scheduled_at_x,
                            pred,
                        );
                    }
                }

                let mut drained_int = false;
                while let Some(Reverse(ev)) = events.peek().copied() {
                    if ev.x != x_evt || !matches!(ev.kind, EventKind::Intersection { .. }) {
                        break;
                    }
                    events.pop();
                    if let EventKind::Intersection { e1, e2 } = ev.kind {
                        handle_intersection(
                            e1,
                            e2,
                            x_evt,
                            eps_y,
                            segs,
                            &mut tree,
                            &mut events,
                            collector,
                            &mut touched_edges,
                            &mut scratch,
                            pred,
                        );
                        drained_int = true;
                    }
                }
                if !drained_int {
                    break;
                }
            }
        }

        // Insert all segments starting at x_evt before checking clusters so co-located stars,
        // ending tips, and passing segments at (x_evt, y) are evaluated simultaneously.
        for &edge in &started_batch {
            tree.insert(edge, x_evt, segs, pred);
        }
        touched_edges.extend_from_slice(&started_batch);
        touched_edges.extend_from_slice(&end_batch);

        if !touched_edges.is_empty() {
            check_eps_bands_at_x(
                &touched_edges,
                x_evt,
                eps_y,
                segs,
                &tree,
                &mut events,
                collector,
                &mut scratch,
                pred,
            );
        }

        // Check vertical segments against active non-vertical segments, accounting for slope ULP drift.
        let vert_margin = eps_y + max_seg_y_err;
        for &v_idx in &vert_batch {
            let v_seg = &segs[v_idx];
            let lo = v_seg.y_min - vert_margin;
            let hi = v_seg.y_max + vert_margin;
            if let Some(mut cur_edge) = tree.lower_bound_y(lo, x_evt, segs, pred) {
                while let Some(prev) = tree.prev_edge(cur_edge) {
                    if pred.y_at(&segs[prev], x_evt) < lo {
                        break;
                    }
                    cur_edge = prev;
                }
                loop {
                    let s_cur = &segs[cur_edge];
                    if pred.y_at(s_cur, x_evt) > hi {
                        break;
                    }
                    if pred.cross(v_seg, s_cur) {
                        collector.on_vertical_pair(v_idx, cur_edge);
                    }
                    let Some(nx) = tree.next_edge(cur_edge) else {
                        break;
                    };
                    cur_edge = nx;
                }
            }
        }

        // Remove ending segments, collect surviving touched peers, and enforce post-x_evt
        // slope order across each local cluster so shared-coord ties never block future pairs.
        scratch.surviving_touched.clear();
        for &e in &touched_edges {
            if tree.is_active(e) {
                scratch.surviving_touched.push(e);
            }
        }
        for &edge in &end_batch {
            let (prev, next) = tree.remove_by_edge(edge);
            if let (Some(p), Some(n)) = (prev, next) {
                check_adjacent_and_schedule(
                    p,
                    n,
                    x_evt,
                    eps_y,
                    segs,
                    &mut events,
                    collector,
                    scratch.dt_probe,
                    &scratch.processed_int_at_x,
                    &mut scratch.scheduled_at_x,
                    pred,
                );
                scratch.surviving_touched.push(p);
                scratch.surviving_touched.push(n);
            }
        }
        if !scratch.surviving_touched.is_empty() {
            reorder_surviving_bands_at_x(
                x_evt,
                eps_y,
                segs,
                &mut tree,
                &mut events,
                collector,
                &mut scratch,
                pred,
            );
        }

        let next_x = events.peek().map_or(f64::INFINITY, |e| e.0.x);
        if next_x != x_evt {
            collector.flush_at_x(x_evt, next_x, segs, eps_y, scratch.dt_probe, pred);
        }
    }
}
