use std::cmp::Ordering;
use std::collections::BTreeSet;

use egraph_core::{CoreError, CoreResult, Layout2d, Scalar};
use egraph_spatial_trees::RectangleOverlapSweep;

use super::projector::{ConstraintProjector, SeparationConstraint};

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

#[derive(Debug, Clone, Copy)]
struct SweepEvent {
    pos: f64,
    is_open: bool,
    idx: usize,
}

/// Projects 2D rectangle positions in `layout` to eliminate all axis-aligned overlaps
/// while minimizing quadratic coordinate displacement.
///
/// Uses $O(N \log N + K)$ [`RectangleOverlapSweep`] and [`ConstraintProjector`] across
/// alternating X and Y projection passes until zero overlaps remain.
pub fn project_rectangle_no_overlap_2d<S: Scalar>(
    layout: &mut Layout2d<S>,
    half_widths: &[S],
    half_heights: &[S],
) -> CoreResult<()> {
    let n = layout.len();
    if half_widths.len() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            half_widths.len(),
            "project_rectangle_no_overlap_2d::half_widths",
        ));
    }
    if half_heights.len() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            half_heights.len(),
            "project_rectangle_no_overlap_2d::half_heights",
        ));
    }
    layout.validate_finite()?;
    for i in 0..n {
        if !half_widths[i].is_finite() || !half_heights[i].is_finite() {
            return Err(CoreError::non_finite(
                "project_rectangle_no_overlap_2d::extents",
            ));
        }
        if half_widths[i] < S::ZERO || half_heights[i] < S::ZERO {
            return Err(CoreError::invalid_input(
                "half_widths and half_heights must be >= 0",
            ));
        }
    }

    if n < 2 {
        return Ok(());
    }

    let mut projector = ConstraintProjector::<S>::new(n);
    let mut x_constraints = Vec::new();
    let mut y_constraints = Vec::new();

    // Guard against 1-ULP rounding re-overlaps in f32/f64 after projection.
    let ulp_scale = 1.0 + 8.0 * S::EPSILON.to_f64();
    let ulp_abs = 8.0 * S::EPSILON.to_f64();

    let max_passes = 32usize;
    for pass in 0..max_passes {
        let overlaps =
            RectangleOverlapSweep::find_layout_overlaps(layout, half_widths, half_heights)?;
        if overlaps.is_empty() {
            return Ok(());
        }

        x_constraints.clear();
        generate_x_pass_constraints(
            layout.x(),
            layout.y(),
            half_widths,
            half_heights,
            &overlaps,
            pass > 0,
            ulp_scale,
            ulp_abs,
            &mut x_constraints,
        );
        if !x_constraints.is_empty() {
            projector.project_1d(layout.x_mut(), &x_constraints)?;
        }

        let overlaps_after_x =
            RectangleOverlapSweep::find_layout_overlaps(layout, half_widths, half_heights)?;
        if overlaps_after_x.is_empty() {
            return Ok(());
        }

        y_constraints.clear();
        generate_y_pass_constraints(
            layout.x(),
            layout.y(),
            half_widths,
            half_heights,
            &overlaps_after_x,
            ulp_scale,
            ulp_abs,
            &mut y_constraints,
        );
        if !y_constraints.is_empty() {
            projector.project_1d(layout.y_mut(), &y_constraints)?;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn generate_x_pass_constraints<S: Scalar>(
    xs: &[S],
    ys: &[S],
    half_widths: &[S],
    half_heights: &[S],
    overlaps_2d: &[(usize, usize)],
    force_all_overlaps: bool,
    ulp_scale: f64,
    ulp_abs: f64,
    out: &mut Vec<SeparationConstraint<S>>,
) {
    let n = xs.len();
    let mut events = Vec::with_capacity(n * 2);
    for i in 0..n {
        let y = ys[i].to_f64();
        let hh = half_heights[i].to_f64();
        events.push(SweepEvent {
            pos: y - hh,
            is_open: true,
            idx: i,
        });
        events.push(SweepEvent {
            pos: y + hh,
            is_open: false,
            idx: i,
        });
    }
    events.sort_unstable_by(|a, b| {
        a.pos
            .total_cmp(&b.pos)
            .then_with(|| a.is_open.cmp(&b.is_open))
            .then_with(|| a.idx.cmp(&b.idx))
    });

    let mut scanline: BTreeSet<(OrdF64, usize)> = BTreeSet::new();
    let mut emitted_overlap = false;

    for ev in events {
        let v = ev.idx;
        let key = (OrdF64(xs[v].to_f64()), v);
        if ev.is_open {
            scanline.insert(key);
            let xv = xs[v].to_f64();
            let hw_v = half_widths[v].to_f64();

            // Immediate non-overlapping right neighbor in same Y-band (prevents crossing).
            for &(_, u) in scanline.range(key..).skip(1) {
                let xu = xs[u].to_f64();
                let req_x = hw_v + half_widths[u].to_f64();
                if xu - xv >= req_x {
                    let pair_ulp = ulp_abs * (1.0 + xu.abs().max(xv.abs()));
                    let gap = S::from_f64(req_x * ulp_scale + pair_ulp);
                    out.push(SeparationConstraint::new(v as u32, u as u32, gap));
                    break;
                }
            }
            // Immediate non-overlapping left neighbor in same Y-band.
            for &(_, u) in scanline.range(..key).rev() {
                let xu = xs[u].to_f64();
                let req_x = hw_v + half_widths[u].to_f64();
                if xv - xu >= req_x {
                    let pair_ulp = ulp_abs * (1.0 + xu.abs().max(xv.abs()));
                    let gap = S::from_f64(req_x * ulp_scale + pair_ulp);
                    out.push(SeparationConstraint::new(u as u32, v as u32, gap));
                    break;
                }
            }
        } else {
            scanline.remove(&key);
        }
    }

    for &(i, j) in overlaps_2d {
        let req_x = half_widths[i].to_f64() + half_widths[j].to_f64();
        let req_y = half_heights[i].to_f64() + half_heights[j].to_f64();
        if req_x <= 0.0 || req_y <= 0.0 {
            continue;
        }
        let ox = req_x - (xs[i].to_f64() - xs[j].to_f64()).abs();
        let oy = req_y - (ys[i].to_f64() - ys[j].to_f64()).abs();
        if ox <= oy || force_all_overlaps {
            emitted_overlap = true;
            let pair_ulp = ulp_abs * (1.0 + xs[i].to_f64().abs().max(xs[j].to_f64().abs()));
            let gap = S::from_f64(req_x * ulp_scale + pair_ulp);
            let ord = xs[i]
                .to_f64()
                .total_cmp(&xs[j].to_f64())
                .then_with(|| i.cmp(&j));
            if ord == Ordering::Less {
                out.push(SeparationConstraint::new(i as u32, j as u32, gap));
            } else {
                out.push(SeparationConstraint::new(j as u32, i as u32, gap));
            }
        }
    }

    if !emitted_overlap {
        out.clear();
    }
}

#[allow(clippy::too_many_arguments)]
fn generate_y_pass_constraints<S: Scalar>(
    xs: &[S],
    ys: &[S],
    half_widths: &[S],
    half_heights: &[S],
    overlaps_2d: &[(usize, usize)],
    ulp_scale: f64,
    ulp_abs: f64,
    out: &mut Vec<SeparationConstraint<S>>,
) {
    let n = xs.len();
    let mut events = Vec::with_capacity(n * 2);
    for i in 0..n {
        let x = xs[i].to_f64();
        let hw = half_widths[i].to_f64();
        events.push(SweepEvent {
            pos: x - hw,
            is_open: true,
            idx: i,
        });
        events.push(SweepEvent {
            pos: x + hw,
            is_open: false,
            idx: i,
        });
    }
    events.sort_unstable_by(|a, b| {
        a.pos
            .total_cmp(&b.pos)
            .then_with(|| a.is_open.cmp(&b.is_open))
            .then_with(|| a.idx.cmp(&b.idx))
    });

    let mut scanline: BTreeSet<(OrdF64, usize)> = BTreeSet::new();

    for ev in events {
        let v = ev.idx;
        let key = (OrdF64(ys[v].to_f64()), v);
        if ev.is_open {
            scanline.insert(key);
            let xv = xs[v].to_f64();
            let yv = ys[v].to_f64();
            let hw_v = half_widths[v].to_f64();
            let hh_v = half_heights[v].to_f64();

            for &(_, u) in scanline.range(key..).skip(1) {
                if (xs[u].to_f64() - xv).abs() < hw_v + half_widths[u].to_f64() {
                    let yu = ys[u].to_f64();
                    let req_y = hh_v + half_heights[u].to_f64();
                    let pair_ulp = ulp_abs * (1.0 + yu.abs().max(yv.abs()));
                    let gap = S::from_f64(req_y * ulp_scale + pair_ulp);
                    out.push(SeparationConstraint::new(v as u32, u as u32, gap));
                    if yu - yv >= req_y {
                        break;
                    }
                }
            }
            for &(_, u) in scanline.range(..key).rev() {
                if (xs[u].to_f64() - xv).abs() < hw_v + half_widths[u].to_f64() {
                    let yu = ys[u].to_f64();
                    let req_y = hh_v + half_heights[u].to_f64();
                    let pair_ulp = ulp_abs * (1.0 + yu.abs().max(yv.abs()));
                    let gap = S::from_f64(req_y * ulp_scale + pair_ulp);
                    out.push(SeparationConstraint::new(u as u32, v as u32, gap));
                    if yv - yu >= req_y {
                        break;
                    }
                }
            }
        } else {
            scanline.remove(&key);
        }
    }

    for &(i, j) in overlaps_2d {
        let req_y = half_heights[i].to_f64() + half_heights[j].to_f64();
        if req_y <= 0.0 {
            continue;
        }
        let pair_ulp = ulp_abs * (1.0 + ys[i].to_f64().abs().max(ys[j].to_f64().abs()));
        let gap = S::from_f64(req_y * ulp_scale + pair_ulp);
        let ord = ys[i]
            .to_f64()
            .total_cmp(&ys[j].to_f64())
            .then_with(|| i.cmp(&j));
        if ord == Ordering::Less {
            out.push(SeparationConstraint::new(i as u32, j as u32, gap));
        } else {
            out.push(SeparationConstraint::new(j as u32, i as u32, gap));
        }
    }
}
