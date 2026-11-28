use egraph_core::{CoreResult, CsrGraph, EuclideanLayout, EuclideanStorageRef, Scalar};
use egraph_spatial_trees::{KdQueryScratch, NearestNeighborIndex};

use crate::extract_unique_edges;
use crate::spatial_index::{copy_euclidean_coords_f64, dist_sq_f64, euclidean_diameter};

use super::sector_pruning::{ProximityMode, proximity_edges_2d_with_index};

fn extract_2d_points<L: EuclideanLayout + ?Sized>(layout: &L) -> Option<Vec<[f64; 2]>> {
    let n = layout.len();
    match layout.as_euclidean_storage() {
        EuclideanStorageRef::Soa2d { x, y } => {
            let mut pts = Vec::with_capacity(n);
            for i in 0..n {
                pts.push([x[i].to_f64(), y[i].to_f64()]);
            }
            Some(pts)
        }
        EuclideanStorageRef::FlatNd {
            coords, dims: 2, ..
        } => {
            let mut pts = Vec::with_capacity(n);
            for i in 0..n {
                pts.push([coords[2 * i].to_f64(), coords[2 * i + 1].to_f64()]);
            }
            Some(pts)
        }
        _ => None,
    }
}

pub(super) fn gabriel_graph_property_euclidean_with_index<
    W,
    L: EuclideanLayout + ?Sized,
    I: NearestNeighborIndex + ?Sized,
>(
    graph: &CsrGraph<W>,
    layout: &L,
    index: &I,
) -> CoreResult<f64> {
    let n = layout.len();
    if n <= 2 || graph.num_edges() == 0 {
        return Ok(0.0);
    }

    let dims = layout.dims();
    let pts = copy_euclidean_coords_f64(layout);
    let edges = extract_unique_edges(graph);
    let mut mid = vec![0.0_f64; dims];
    let mut disk_buf: Vec<(usize, f64)> = Vec::new();
    let mut violation_sum = 0.0_f64;

    for (u, v) in edges {
        let ui = u as usize;
        let vi = v as usize;
        let r = 0.5 * layout.distance(ui, vi);
        if r <= 0.0 {
            continue;
        }
        let pu = &pts[ui * dims..(ui + 1) * dims];
        let pv = &pts[vi * dims..(vi + 1) * dims];
        for d in 0..dims {
            mid[d] = 0.5 * (pu[d] + pv[d]);
        }
        index.query_radius(&mid, r, &mut disk_buf)?;
        for &(idx, d_sq) in &disk_buf {
            if idx == ui || idx == vi {
                continue;
            }
            let dist = d_sq.sqrt();
            if dist < r {
                let viol = r - dist;
                violation_sum += viol * viol;
            }
        }
    }

    Ok(violation_sum)
}

pub(super) fn node_resolution_euclidean_with_index<
    L: EuclideanLayout + ?Sized,
    I: NearestNeighborIndex + ?Sized,
>(
    layout: &L,
    index: &I,
) -> CoreResult<f64> {
    let n = layout.len();
    if n <= 1 {
        return Ok(0.0);
    }

    let d_max = euclidean_diameter(layout)?;
    if d_max <= 0.0 {
        let num_pairs = (n as f64) * ((n - 1) as f64) * 0.5;
        return Ok(num_pairs);
    }

    let threshold = d_max / (n as f64).sqrt();
    let dims = layout.dims();
    let pts = copy_euclidean_coords_f64(layout);
    let mut buf: Vec<(usize, f64)> = Vec::new();
    let mut sum = 0.0_f64;

    for i in 0..n {
        let q = &pts[i * dims..(i + 1) * dims];
        index.query_radius(q, threshold, &mut buf)?;
        for &(j, d_sq) in &buf {
            if j <= i {
                continue;
            }
            let dist = d_sq.sqrt();
            if dist < threshold {
                let hinge = 1.0 - dist / threshold;
                sum += hinge * hinge;
            }
        }
    }

    Ok(sum)
}

pub(super) fn proximity_edges_euclidean_with_index<
    L: EuclideanLayout + ?Sized,
    I: NearestNeighborIndex + ?Sized,
>(
    layout: &L,
    index: &I,
    mode: ProximityMode,
) -> CoreResult<Vec<(u32, u32)>> {
    let n = layout.len();
    if n < 2 {
        return Ok(Vec::new());
    }
    if let Some(pts2) = extract_2d_points(layout) {
        return proximity_edges_2d_with_index(&pts2, index, mode);
    }

    let dims = layout.dims();
    let pts = copy_euclidean_coords_f64(layout);
    let k_witness = 32usize.min(n - 1);
    let mut knn_flat = Vec::with_capacity(n * k_witness);
    let mut gg_blocked = vec![false; n];
    let mut kd_scratch = KdQueryScratch::with_capacity(k_witness);
    let mut knn_buf: Vec<(usize, f64)> = Vec::new();

    for (i, blocked) in gg_blocked.iter_mut().enumerate() {
        index.query_knn_excluding_self_with_scratch(i, k_witness, &mut kd_scratch, &mut knn_buf)?;
        if matches!(mode, ProximityMode::Gabriel)
            && knn_buf.first().is_some_and(|&(_, d2)| d2 <= 0.0)
        {
            *blocked = true;
        }
        knn_flat.extend_from_slice(&knn_buf);
    }

    let mut edges = Vec::new();
    let mut mid = vec![0.0_f64; dims];
    let mut witness_buf: Vec<(usize, f64)> = Vec::new();

    for i in 0..n {
        if gg_blocked[i] {
            continue;
        }
        let pi = &pts[i * dims..(i + 1) * dims];
        let wi = &knn_flat[i * k_witness..(i + 1) * k_witness];

        for j in (i + 1)..n {
            if gg_blocked[j] {
                continue;
            }
            let pj = &pts[j * dims..(j + 1) * dims];
            let d_ij_sq = dist_sq_f64(pi, pj);
            if d_ij_sq <= 0.0 {
                continue;
            }
            if is_blocked_by_witnesses(j, pj, d_ij_sq, wi, &pts, dims, mode)
                || is_blocked_by_witnesses(
                    i,
                    pi,
                    d_ij_sq,
                    &knn_flat[j * k_witness..(j + 1) * k_witness],
                    &pts,
                    dims,
                    mode,
                )
            {
                continue;
            }
            if is_proximity_edge_euclidean(
                i,
                j,
                d_ij_sq,
                &pts,
                dims,
                index,
                mode,
                &mut mid,
                &mut kd_scratch,
                &mut witness_buf,
            )? {
                edges.push((i as u32, j as u32));
            }
        }
    }

    Ok(edges)
}

#[inline]
fn is_blocked_by_witnesses(
    other_endpoint: usize,
    p_other: &[f64],
    d_ij_sq: f64,
    witnesses: &[(usize, f64)],
    pts: &[f64],
    dims: usize,
    mode: ProximityMode,
) -> bool {
    match mode {
        ProximityMode::Gabriel => {
            let thresh = d_ij_sq * (1.0 + 1e-12);
            for &(r, d_cr_sq) in witnesses {
                if d_cr_sq > thresh {
                    break;
                }
                if r == other_endpoint {
                    continue;
                }
                let pr = &pts[r * dims..(r + 1) * dims];
                let d_or_sq = dist_sq_f64(p_other, pr);
                if d_cr_sq + d_or_sq <= thresh {
                    return true;
                }
            }
        }
        ProximityMode::RelativeNeighborhood => {
            let thresh = d_ij_sq * (1.0 - 1e-12);
            for &(r, d_cr_sq) in witnesses {
                if d_cr_sq >= thresh {
                    break;
                }
                if r == other_endpoint {
                    continue;
                }
                let pr = &pts[r * dims..(r + 1) * dims];
                let d_or_sq = dist_sq_f64(p_other, pr);
                if d_or_sq < thresh {
                    return true;
                }
            }
        }
    }
    false
}

#[allow(clippy::too_many_arguments)]
fn is_proximity_edge_euclidean<I: NearestNeighborIndex + ?Sized>(
    i: usize,
    j: usize,
    d_ij_sq: f64,
    pts: &[f64],
    dims: usize,
    index: &I,
    mode: ProximityMode,
    mid: &mut [f64],
    kd_scratch: &mut KdQueryScratch,
    witness_buf: &mut Vec<(usize, f64)>,
) -> CoreResult<bool> {
    let pi = &pts[i * dims..(i + 1) * dims];
    let pj = &pts[j * dims..(j + 1) * dims];
    let mut max_coord_abs = 0.0_f64;
    for d in 0..dims {
        mid[d] = pi[d] + 0.5 * (pj[d] - pi[d]);
        max_coord_abs = max_coord_abs.max(pi[d].abs()).max(pj[d].abs());
    }

    let base_r_sq = match mode {
        ProximityMode::Gabriel => 0.25 * d_ij_sq * (1.0 + 1e-11),
        ProximityMode::RelativeNeighborhood => 0.75 * d_ij_sq * (1.0 + 1e-11),
    };
    let eps_mid = 4.0 * (dims as f64).sqrt() * f64::EPSILON * max_coord_abs;
    let padded_r = base_r_sq.sqrt() + eps_mid;
    let query_r_sq = (padded_r * padded_r).min(f64::MAX);
    index.query_radius_sq_with_scratch(mid, query_r_sq, kd_scratch, witness_buf)?;

    for &(r, _) in witness_buf.iter() {
        if r == i || r == j {
            continue;
        }
        let pr = &pts[r * dims..(r + 1) * dims];
        let d_ir_sq = dist_sq_f64(pi, pr);
        let d_jr_sq = dist_sq_f64(pj, pr);
        let blocked = match mode {
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
