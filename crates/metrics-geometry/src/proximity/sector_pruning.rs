use std::f64::consts::{FRAC_PI_2, FRAC_PI_3, PI, TAU};

use egraph_core::{CoreResult, Layout2d, Scalar};
use egraph_spatial_trees::StaticKdTree;

use super::exact_predicates::dist_sq_2d;

const NUM_SECTORS: usize = 16;
const SECTOR_WIDTH: f64 = TAU / NUM_SECTORS as f64;

#[derive(Debug, Clone, Copy)]
pub(super) enum ProximityMode {
    Gabriel,
    RelativeNeighborhood,
}

pub(super) fn proximity_edges_with_tree<S: Scalar>(
    layout: &Layout2d<S>,
    tree: &StaticKdTree<2, f64>,
    mode: ProximityMode,
) -> CoreResult<Vec<(u32, u32)>> {
    let n = layout.len();
    if n < 2 {
        return Ok(Vec::new());
    }

    let xs = layout.x();
    let ys = layout.y();
    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        pts.push([xs[i].to_f64(), ys[i].to_f64()]);
    }

    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for &[x, y] in &pts {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    let d_max_bound = (max_x - min_x).hypot(max_y - min_y) * (1.0 + 1e-9);

    let mut sector_bounds = vec![[f64::INFINITY; NUM_SECTORS]; n];
    let mut r_max = vec![f64::INFINITY; n];
    let mut covered = vec![false; n];
    let mut uncovered = Vec::new();
    let mut knn_buf: Vec<(usize, f64)> = Vec::new();

    // Pass 1: derive angular sector upper bounds from local k-NN witnesses so interior
    // vertices only inspect O(1) candidates within `r_max[i]`.
    for i in 0..n {
        let mut k = 32usize.min(n - 1);
        tree.k_nearest_excluding_self_into(i, k, &mut knn_buf)?;
        update_sector_bounds(i, &pts, &knn_buf, mode, d_max_bound, &mut sector_bounds[i]);
        let mut bound = sector_bounds[i].iter().copied().fold(0.0_f64, f64::max);

        if !bound.is_finite() && n - 1 > 32 {
            k = 128usize.min(n - 1);
            tree.k_nearest_excluding_self_into(i, k, &mut knn_buf)?;
            update_sector_bounds(i, &pts, &knn_buf, mode, d_max_bound, &mut sector_bounds[i]);
            bound = sector_bounds[i].iter().copied().fold(0.0_f64, f64::max);
        }

        r_max[i] = bound;
        if bound.is_finite() {
            covered[i] = true;
        } else {
            uncovered.push(i);
        }
    }

    let mut edges = Vec::new();
    let mut cand_buf: Vec<(usize, f64)> = Vec::new();
    let mut witness_buf: Vec<(usize, f64)> = Vec::new();

    // Pass 2a: enumerate all valid edges with at least one covered endpoint.
    for i in 0..n {
        if !covered[i] {
            continue;
        }
        let query_r = (r_max[i] * (1.0 + 1e-12)).min(f64::MAX);
        tree.within_radius_into(&pts[i], query_r, &mut cand_buf)?;
        for &(j, d_ij_sq) in &cand_buf {
            if j == i || d_ij_sq <= 0.0 {
                continue;
            }
            if covered[j] && j < i {
                continue;
            }
            let s_j = sector_index(pts[j][0] - pts[i][0], pts[j][1] - pts[i][1]);
            let d_ij = d_ij_sq.sqrt();
            if d_ij > sector_bounds[i][s_j] {
                continue;
            }
            if is_proximity_edge(i, j, d_ij_sq, &pts, tree, mode, &mut witness_buf)? {
                let (a, b) = if i < j { (i, j) } else { (j, i) };
                edges.push((a as u32, b as u32));
            }
        }
    }

    // Pass 2b: remaining pairs whose endpoints both lie on open/boundary wedges.
    for (idx_a, &u) in uncovered.iter().enumerate() {
        for &v in &uncovered[(idx_a + 1)..] {
            let dx = pts[v][0] - pts[u][0];
            let dy = pts[v][1] - pts[u][1];
            let d_uv_sq = dx * dx + dy * dy;
            if d_uv_sq <= 0.0 {
                continue;
            }
            let d_uv = d_uv_sq.sqrt();
            let su = sector_index(dx, dy);
            if d_uv > sector_bounds[u][su] {
                continue;
            }
            let sv = sector_index(-dx, -dy);
            if d_uv > sector_bounds[v][sv] {
                continue;
            }
            if is_proximity_edge(u, v, d_uv_sq, &pts, tree, mode, &mut witness_buf)? {
                edges.push((u as u32, v as u32));
            }
        }
    }

    edges.sort_unstable();
    edges.dedup();
    Ok(edges)
}

#[inline]
fn wrap_abs_angle(mut d: f64) -> f64 {
    while d > PI {
        d -= TAU;
    }
    while d < -PI {
        d += TAU;
    }
    d.abs()
}

#[inline]
fn sector_index(dx: f64, dy: f64) -> usize {
    let theta = dy.atan2(dx);
    let idx = ((theta + PI) / SECTOR_WIDTH).floor() as isize;
    idx.clamp(0, (NUM_SECTORS - 1) as isize) as usize
}

fn update_sector_bounds(
    center: usize,
    pts: &[[f64; 2]],
    witnesses: &[(usize, f64)],
    mode: ProximityMode,
    d_max_bound: f64,
    bounds: &mut [f64; NUM_SECTORS],
) {
    let px = pts[center][0];
    let py = pts[center][1];
    let max_allow = match mode {
        ProximityMode::Gabriel => FRAC_PI_2 - 1e-3,
        ProximityMode::RelativeNeighborhood => FRAC_PI_3 - 1e-3,
    };

    for &(r_idx, d_r_sq) in witnesses {
        if d_r_sq <= 0.0 {
            continue;
        }
        let dx = pts[r_idx][0] - px;
        let dy = pts[r_idx][1] - py;
        let d_r = d_r_sq.sqrt();
        let theta_r = dy.atan2(dx);

        for (s, bound_slot) in bounds.iter_mut().enumerate() {
            let phi0 = -PI + (s as f64) * SECTOR_WIDTH;
            let phi1 = phi0 + SECTOR_WIDTH;
            let max_delta = wrap_abs_angle(theta_r - phi0).max(wrap_abs_angle(theta_r - phi1));
            if max_delta < max_allow {
                let cos_min = max_delta.cos();
                if let ProximityMode::RelativeNeighborhood = mode {
                    // For RNG, `is_proximity_edge` requires `d_jr^2 < d_ij^2 * (1 - 1e-12)`.
                    // A tiny witness `d_r << 1e-12 * d_ij` fails to shrink `d_jr^2` below the
                    // tolerance threshold, so a witness only safely prunes up to `d_max_bound`
                    // when `2 * d_r * cos_min - d_r^2/d_ij >= 1e-11 * d_ij`.
                    if d_r < d_max_bound * (1e-11 / cos_min) {
                        continue;
                    }
                }
                let candidate = match mode {
                    ProximityMode::Gabriel => (d_r / cos_min) * (1.0 + 1e-7),
                    ProximityMode::RelativeNeighborhood => d_r * (1.0 + 1e-7),
                };
                if candidate < *bound_slot {
                    *bound_slot = candidate;
                }
            }
        }
    }
}

fn is_proximity_edge(
    i: usize,
    j: usize,
    d_ij_sq: f64,
    pts: &[[f64; 2]],
    tree: &StaticKdTree<2, f64>,
    mode: ProximityMode,
    witness_buf: &mut Vec<(usize, f64)>,
) -> CoreResult<bool> {
    // Difference-based midpoint avoids catastrophic cancellation at large coordinate offsets,
    // and ULP padding on `query_r_sq` covers the remaining midpoint representability error.
    let mid = [
        pts[i][0] + 0.5 * (pts[j][0] - pts[i][0]),
        pts[i][1] + 0.5 * (pts[j][1] - pts[i][1]),
    ];
    let base_r_sq = match mode {
        ProximityMode::Gabriel => 0.25 * d_ij_sq * (1.0 + 1e-11),
        ProximityMode::RelativeNeighborhood => 0.75 * d_ij_sq * (1.0 + 1e-11),
    };
    let max_coord_abs = pts[i][0]
        .abs()
        .max(pts[i][1].abs())
        .max(pts[j][0].abs())
        .max(pts[j][1].abs());
    let eps_mid = 4.0 * f64::EPSILON * max_coord_abs;
    let padded_r = base_r_sq.sqrt() + eps_mid;
    let query_r_sq = (padded_r * padded_r).min(f64::MAX);
    tree.within_radius_sq_into(&mid, query_r_sq, witness_buf)?;

    for &(r, _) in witness_buf.iter() {
        if r == i || r == j {
            continue;
        }
        let d_ir_sq = dist_sq_2d(pts[i], pts[r]);
        let d_jr_sq = dist_sq_2d(pts[j], pts[r]);
        let blocked = match mode {
            // Use 1e-12 relative ULP tolerance so rotated cocircular grids and sqrt(3)/trig
            // equilateral lattices behave identically to exact Euclidean geometry.
            ProximityMode::Gabriel => d_ir_sq + d_jr_sq <= d_ij_sq * (1.0 + 1e-12),
            ProximityMode::RelativeNeighborhood => {
                let thresh = d_ij_sq * (1.0 - 1e-12);
                d_ir_sq < thresh && d_jr_sq < thresh
            }
        };
        if blocked {
            return Ok(false);
        }
    }
    Ok(true)
}
