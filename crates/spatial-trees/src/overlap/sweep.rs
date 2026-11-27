use std::cmp::Ordering;
use std::collections::BTreeSet;

use egraph_core::{CoreError, CoreResult, Scalar};

use super::Rect2d;

#[derive(Debug, Clone, Copy)]
pub(super) struct Rect64 {
    pub(super) x_min: f64,
    pub(super) x_max: f64,
    pub(super) y_min: f64,
    pub(super) y_max: f64,
}

impl Rect64 {
    #[inline]
    fn overlaps(&self, other: &Self) -> bool {
        self.x_min < other.x_max
            && other.x_min < self.x_max
            && self.y_min < other.y_max
            && other.y_min < self.y_max
    }
}

#[derive(Debug, Clone, Copy)]
struct OrdF64(f64);

impl PartialEq for OrdF64 {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0) == Ordering::Equal
    }
}

impl Eq for OrdF64 {}

impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Edelsbrunner centered interval tree storing active y-intervals during x-sweep.
///
/// Every interval `[y_min, y_max]` is stored at the highest static node whose midpoint `y_mid`
/// satisfies `y_min <= y_mid <= y_max`, giving $O(\log N)$ insert/remove and $O(\log N + K_q)$
/// overlap reporting with zero false-positive node scans.
struct IntervalTree {
    nodes: Vec<IntervalNode>,
}

#[derive(Default)]
struct IntervalNode {
    y_mid: f64,
    left: usize,
    right: usize,
    by_ymin: BTreeSet<(OrdF64, usize)>,
    by_ymax: BTreeSet<(OrdF64, usize)>,
}

impl IntervalTree {
    fn build(rects: &[Rect64]) -> Self {
        let mut ys = Vec::with_capacity(rects.len() * 2);
        for r in rects {
            ys.push(r.y_min);
            ys.push(r.y_max);
        }
        ys.sort_unstable_by(f64::total_cmp);
        ys.dedup();

        let mut tree = Self {
            nodes: Vec::with_capacity(ys.len().saturating_mul(2) + 1),
        };
        if !ys.is_empty() {
            tree.build_range(&ys);
        }
        tree
    }

    fn build_range(&mut self, ys: &[f64]) -> usize {
        if ys.is_empty() {
            return usize::MAX;
        }
        let mid = ys.len() / 2;
        let idx = self.nodes.len();
        self.nodes.push(IntervalNode {
            y_mid: ys[mid],
            left: usize::MAX,
            right: usize::MAX,
            by_ymin: BTreeSet::new(),
            by_ymax: BTreeSet::new(),
        });
        let left = self.build_range(&ys[..mid]);
        let right = self.build_range(&ys[mid + 1..]);
        self.nodes[idx].left = left;
        self.nodes[idx].right = right;
        idx
    }

    fn insert(&mut self, rect_idx: usize, r: &Rect64) {
        let mut cur = 0usize;
        while cur != usize::MAX {
            let y_mid = self.nodes[cur].y_mid;
            if r.y_max < y_mid {
                cur = self.nodes[cur].left;
            } else if r.y_min > y_mid {
                cur = self.nodes[cur].right;
            } else {
                self.nodes[cur].by_ymin.insert((OrdF64(r.y_min), rect_idx));
                self.nodes[cur].by_ymax.insert((OrdF64(r.y_max), rect_idx));
                return;
            }
        }
    }

    fn remove(&mut self, rect_idx: usize, r: &Rect64) {
        let mut cur = 0usize;
        while cur != usize::MAX {
            let y_mid = self.nodes[cur].y_mid;
            if r.y_max < y_mid {
                cur = self.nodes[cur].left;
            } else if r.y_min > y_mid {
                cur = self.nodes[cur].right;
            } else {
                self.nodes[cur].by_ymin.remove(&(OrdF64(r.y_min), rect_idx));
                self.nodes[cur].by_ymax.remove(&(OrdF64(r.y_max), rect_idx));
                return;
            }
        }
    }

    fn query_overlaps(
        &self,
        node_idx: usize,
        query_idx: usize,
        q: &Rect64,
        rects: &[Rect64],
        stop_on_first: bool,
        out: &mut Vec<(usize, usize)>,
    ) -> bool {
        if node_idx == usize::MAX || self.nodes.is_empty() {
            return false;
        }
        let node = &self.nodes[node_idx];
        let y_mid = node.y_mid;

        if q.y_max <= y_mid {
            for &(OrdF64(ymin), other) in &node.by_ymin {
                if ymin >= q.y_max {
                    break;
                }
                if q.overlaps(&rects[other]) {
                    let pair = if query_idx < other {
                        (query_idx, other)
                    } else {
                        (other, query_idx)
                    };
                    out.push(pair);
                    if stop_on_first {
                        return true;
                    }
                }
            }
            q.y_min < y_mid
                && self.query_overlaps(node.left, query_idx, q, rects, stop_on_first, out)
        } else if q.y_min >= y_mid {
            for &(OrdF64(ymax), other) in node.by_ymax.iter().rev() {
                if ymax <= q.y_min {
                    break;
                }
                if q.overlaps(&rects[other]) {
                    let pair = if query_idx < other {
                        (query_idx, other)
                    } else {
                        (other, query_idx)
                    };
                    out.push(pair);
                    if stop_on_first {
                        return true;
                    }
                }
            }
            q.y_max > y_mid
                && self.query_overlaps(node.right, query_idx, q, rects, stop_on_first, out)
        } else {
            // y_mid is strictly inside (q.y_min, q.y_max), so every interval at this node
            // overlaps q in y.
            for &(_, other) in &node.by_ymin {
                if q.overlaps(&rects[other]) {
                    let pair = if query_idx < other {
                        (query_idx, other)
                    } else {
                        (other, query_idx)
                    };
                    out.push(pair);
                    if stop_on_first {
                        return true;
                    }
                }
            }
            if self.query_overlaps(node.left, query_idx, q, rects, stop_on_first, out) {
                return true;
            }
            self.query_overlaps(node.right, query_idx, q, rects, stop_on_first, out)
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct SweepEvent {
    x: f64,
    is_open: bool,
    rect_idx: usize,
}

pub(super) fn sweep_rects<S: Scalar>(
    rects: &[Rect2d<S>],
    stop_on_first: bool,
) -> CoreResult<Vec<(usize, usize)>> {
    let mut rects64 = Vec::with_capacity(rects.len());
    for r in rects {
        if !r.x_min.is_finite()
            || !r.x_max.is_finite()
            || !r.y_min.is_finite()
            || !r.y_max.is_finite()
        {
            return Err(CoreError::non_finite("RectangleOverlapSweep"));
        }
        if r.x_min > r.x_max || r.y_min > r.y_max {
            return Err(CoreError::invalid_input(
                "Rect2d must satisfy x_min <= x_max and y_min <= y_max",
            ));
        }
        rects64.push(Rect64 {
            x_min: r.x_min.to_f64(),
            x_max: r.x_max.to_f64(),
            y_min: r.y_min.to_f64(),
            y_max: r.y_max.to_f64(),
        });
    }
    Ok(sweep_rects64(&rects64, stop_on_first))
}

pub(super) fn sweep_rects64(rects64: &[Rect64], stop_on_first: bool) -> Vec<(usize, usize)> {
    let n = rects64.len();
    if n < 2 {
        return Vec::new();
    }

    let mut events = Vec::with_capacity(n * 2);
    for (i, r) in rects64.iter().enumerate() {
        events.push(SweepEvent {
            x: r.x_min,
            is_open: true,
            rect_idx: i,
        });
        events.push(SweepEvent {
            x: r.x_max,
            is_open: false,
            rect_idx: i,
        });
    }

    // Close events precede open events at identical x because strict boundary touching
    // (a.x_max == b.x_min) is non-overlapping.
    events.sort_unstable_by(|a, b| {
        a.x.total_cmp(&b.x)
            .then_with(|| a.is_open.cmp(&b.is_open))
            .then_with(|| a.rect_idx.cmp(&b.rect_idx))
    });

    let mut tree = IntervalTree::build(rects64);
    let mut active = vec![false; n];
    let mut out = Vec::new();

    for ev in events {
        let r = &rects64[ev.rect_idx];
        if ev.is_open {
            let found = tree.query_overlaps(0, ev.rect_idx, r, rects64, stop_on_first, &mut out);
            if stop_on_first && found {
                return out;
            }
            if r.x_min < r.x_max {
                tree.insert(ev.rect_idx, r);
                active[ev.rect_idx] = true;
            }
        } else if active[ev.rect_idx] {
            tree.remove(ev.rect_idx, r);
            active[ev.rect_idx] = false;
        }
    }

    out.sort_unstable();
    out
}
