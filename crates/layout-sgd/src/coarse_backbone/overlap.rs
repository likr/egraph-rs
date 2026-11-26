use egraph_core::{CoreResult, Layout2d, Scalar};
#[cfg(feature = "delaunay")]
use egraph_spatial_trees::Delaunay2d;
use egraph_spatial_trees::RectangleOverlapSweep;
#[cfg(not(feature = "delaunay"))]
use egraph_spatial_trees::{KdQueryScratch, StaticKdTree};

#[cfg(not(feature = "delaunay"))]
pub(super) const K_PROX_FALLBACK: usize = 8;

pub(super) fn count_disk_overlaps<S: Scalar>(
    layout: &Layout2d<S>,
    radii: &[f64],
    m_gap: f64,
    half_boxes: &[S],
) -> CoreResult<usize> {
    let n = layout.len();
    if n < 2 || half_boxes.iter().all(|&h| h <= S::ZERO) {
        return Ok(0);
    }
    let boxes = RectangleOverlapSweep::find_layout_overlaps(layout, half_boxes, half_boxes)?;
    let xs = layout.x();
    let ys = layout.y();
    let mut cnt = 0usize;
    for (u, v) in boxes {
        let thr = radii[u] + radii[v] + m_gap;
        if thr > 0.0 {
            let d = (xs[u].to_f64() - xs[v].to_f64()).hypot(ys[u].to_f64() - ys[v].to_f64());
            if d < thr {
                cnt += 1;
            }
        }
    }
    Ok(cnt)
}

pub(super) fn resolve_residual_disk_overlaps<S: Scalar>(
    layout: &mut Layout2d<S>,
    radii: &[f64],
    m_gap: f64,
    half_boxes: &[S],
    coarse_sizes: &[u32],
) -> CoreResult<()> {
    let n = layout.len();
    if n < 2 || half_boxes.iter().all(|&h| h <= S::ZERO) {
        return Ok(());
    }
    let rel_margin = if std::mem::size_of::<S>() <= 4 {
        5e-4
    } else {
        1e-5
    };
    let sweep_half_boxes: Vec<S> = radii
        .iter()
        .map(|&r| {
            let h = r + 0.5 * m_gap;
            if h > 0.0 {
                S::from_f64(h * (1.0 + 1e-3) + 1e-6)
            } else {
                S::ZERO
            }
        })
        .collect();

    let max_sweeps = 512usize.max(n.saturating_mul(4));
    let mut px = vec![0.0_f64; n];
    let mut py = vec![0.0_f64; n];
    let mut moved_any = false;

    for sweep in 0..max_sweeps {
        let candidates = RectangleOverlapSweep::find_layout_overlaps(
            layout,
            &sweep_half_boxes,
            &sweep_half_boxes,
        )?;
        if candidates.is_empty() {
            break;
        }
        {
            let xs = layout.x();
            let ys = layout.y();
            for i in 0..n {
                px[i] = xs[i].to_f64();
                py[i] = ys[i].to_f64();
            }
        }

        let mut sweep_overlap = false;
        let use_symmetric = sweep >= 32;
        for (u, v) in candidates {
            let d_bar = radii[u] + radii[v] + m_gap;
            if !d_bar.is_finite() || d_bar <= 0.0 {
                continue;
            }
            let dx = px[v] - px[u];
            let dy = py[v] - py[u];
            let dist = dx.hypot(dy);
            if !dist.is_finite() || dist >= d_bar {
                continue;
            }
            sweep_overlap = true;

            let target_d = d_bar * (1.0 + rel_margin) + 1e-9;
            let push = target_d - dist;
            if !push.is_finite() || push <= 0.0 {
                continue;
            }
            let (ux, uy) = if dist > 1e-12 {
                (dx / dist, dy / dist)
            } else {
                // Deterministic non-collinear separation direction for coincident centers.
                let angle =
                    ((u as f64) * 2.399963229728653) + ((v as f64) * 0.618033988749895) + 0.37;
                (angle.cos(), angle.sin())
            };

            let (step_u, step_v) = if use_symmetric {
                (0.5 * push, 0.5 * push)
            } else {
                let sz_u = (coarse_sizes[u] as f64).max(1.0);
                let sz_v = (coarse_sizes[v] as f64).max(1.0);
                let inv_u = 1.0 / sz_u;
                let inv_v = 1.0 / sz_v;
                let inv_sum = inv_u + inv_v;
                (push * (inv_u / inv_sum), push * (inv_v / inv_sum))
            };

            let nx_u = px[u] - ux * step_u;
            let ny_u = py[u] - uy * step_u;
            let nx_v = px[v] + ux * step_v;
            let ny_v = py[v] + uy * step_v;
            if nx_u.is_finite() && ny_u.is_finite() && nx_v.is_finite() && ny_v.is_finite() {
                px[u] = nx_u;
                py[u] = ny_u;
                px[v] = nx_v;
                py[v] = ny_v;
            }
        }

        if !sweep_overlap {
            break;
        }
        moved_any = true;
        let (xs_m, ys_m) = layout.xy_mut();
        for i in 0..n {
            let sx = S::from_f64(px[i]);
            let sy = S::from_f64(py[i]);
            if sx.is_finite() && sy.is_finite() {
                xs_m[i] = sx;
                ys_m[i] = sy;
            }
        }
    }

    if moved_any {
        // Re-center without scaling so pairwise clearances are preserved, reverting if
        // floating-point rounding re-introduces any overlap.
        let (mx, my) = {
            let xs = layout.x();
            let ys = layout.y();
            let mut sx = 0.0_f64;
            let mut sy = 0.0_f64;
            for i in 0..n {
                sx += xs[i].to_f64();
                sy += ys[i].to_f64();
            }
            (sx / (n as f64), sy / (n as f64))
        };
        if mx.is_finite() && my.is_finite() && mx.hypot(my) > 1e-6 {
            let before = layout.clone();
            {
                let (xs_m, ys_m) = layout.xy_mut();
                for i in 0..n {
                    let nx = S::from_f64(xs_m[i].to_f64() - mx);
                    let ny = S::from_f64(ys_m[i].to_f64() - my);
                    if nx.is_finite() && ny.is_finite() {
                        xs_m[i] = nx;
                        ys_m[i] = ny;
                    }
                }
            }
            if count_disk_overlaps(layout, radii, m_gap, half_boxes)? > 0 {
                *layout = before;
            }
        }
    }

    Ok(())
}

pub(super) fn collect_disk_hinge_pairs<S: Scalar>(
    layout: &Layout2d<S>,
    radii: &[f64],
    m_gap: f64,
    half_boxes: &[S],
    #[cfg(not(feature = "delaunay"))] kd_scratch: &mut KdQueryScratch,
    #[cfg(not(feature = "delaunay"))] knn_out: &mut Vec<(usize, f64)>,
) -> CoreResult<Vec<(u32, u32)>> {
    let n = layout.len();
    if n < 2 {
        return Ok(Vec::new());
    }
    let mut pairs = Vec::new();
    if half_boxes.iter().any(|&h| h > S::ZERO) {
        let overlaps = RectangleOverlapSweep::find_layout_overlaps(layout, half_boxes, half_boxes)?;
        for (u, v) in overlaps {
            if radii[u] + radii[v] + m_gap > 0.0 {
                let (a, b) = if u < v { (u, v) } else { (v, u) };
                pairs.push((a as u32, b as u32));
            }
        }
    }

    #[cfg(feature = "delaunay")]
    {
        let mut dt_edges = Vec::new();
        Delaunay2d::triangulate_edges_into(layout, &mut dt_edges)?;
        for (u, v) in dt_edges {
            if radii[u as usize] + radii[v as usize] + m_gap > 0.0 {
                pairs.push((u, v));
            }
        }
    }

    #[cfg(not(feature = "delaunay"))]
    {
        let tree = StaticKdTree::<2, S>::from_layout(layout)?;
        let k = K_PROX_FALLBACK.min(n - 1);
        for u in 0..n {
            tree.k_nearest_excluding_self_with_scratch(u, k, kd_scratch, knn_out)?;
            for &(v, _) in knn_out.iter() {
                if radii[u] + radii[v] + m_gap > 0.0 {
                    let (a, b) = if u < v { (u, v) } else { (v, u) };
                    pairs.push((a as u32, b as u32));
                }
            }
        }
    }

    pairs.sort_unstable();
    pairs.dedup();
    Ok(pairs)
}
