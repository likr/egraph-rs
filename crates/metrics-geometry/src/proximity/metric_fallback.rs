use std::any::Any;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, ManifoldKind, ManifoldLayout2d, MetricSpace, Scalar,
    lorentz_distance, lorentz_exp_map, lorentz_log_map, poincare_distance, poincare_exp_map,
    poincare_log_map, spherical_distance, spherical_exp_map, spherical_log_map, torus_delta,
    wrap_torus_point,
};

use crate::extract_unique_edges;
use crate::spatial_index::metric_diameter;

fn manifold_geodesic_interpolate_point<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    i: usize,
    j: usize,
    t: f64,
) -> CoreResult<(f64, f64)> {
    let x1 = layout.x()[i].to_f64();
    let y1 = layout.y()[i].to_f64();
    let x2 = layout.x()[j].to_f64();
    let y2 = layout.y()[j].to_f64();
    match layout.kind() {
        ManifoldKind::HyperbolicPoincare => {
            let (vx, vy) = poincare_log_map(x1, y1, x2, y2);
            Ok(poincare_exp_map(x1, y1, t * vx, t * vy))
        }
        ManifoldKind::HyperbolicLorentz => {
            let (vx, vy) = lorentz_log_map(x1, y1, x2, y2);
            Ok(lorentz_exp_map(x1, y1, t * vx, t * vy))
        }
        ManifoldKind::Spherical => {
            let (vx, vy) = spherical_log_map(x1, y1, x2, y2);
            Ok(spherical_exp_map(x1, y1, t * vx, t * vy))
        }
        ManifoldKind::Torus2d => {
            let (dx_norm, dy_norm) = torus_delta(x2, y2, x1, y1);
            Ok(wrap_torus_point(x1 + t * dx_norm, y1 + t * dy_norm))
        }
        _ => Err(CoreError::unsupported("unsupported manifold kind")),
    }
}

fn manifold_point_distance_to_node<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    pt: (f64, f64),
    q: usize,
) -> CoreResult<f64> {
    let qx = layout.x()[q].to_f64();
    let qy = layout.y()[q].to_f64();
    let s = layout.scale();
    match layout.kind() {
        ManifoldKind::HyperbolicPoincare => Ok(s * poincare_distance(pt.0, pt.1, qx, qy)),
        ManifoldKind::HyperbolicLorentz => Ok(s * lorentz_distance(pt.0, pt.1, qx, qy)),
        ManifoldKind::Spherical => Ok(s * spherical_distance(pt.0, pt.1, qx, qy)),
        ManifoldKind::Torus2d => {
            let (dx_norm, dy_norm) = torus_delta(qx, qy, pt.0, pt.1);
            let lx = s * layout.aspect();
            let ly = s;
            Ok((lx * dx_norm).hypot(ly * dy_norm))
        }
        _ => Err(CoreError::unsupported("unsupported manifold kind")),
    }
}

pub(super) fn node_resolution_metric<L: MetricSpace + ?Sized>(layout: &L) -> f64 {
    let n = layout.len();
    if n <= 1 {
        return 0.0;
    }

    let d_max = metric_diameter(layout);
    if d_max <= 0.0 {
        return (n as f64) * ((n - 1) as f64) * 0.5;
    }

    let threshold = d_max / (n as f64).sqrt();
    let mut sum = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = layout.distance(i, j);
            if dist < threshold {
                let hinge = 1.0 - dist / threshold;
                sum += hinge * hinge;
            }
        }
    }
    sum
}

pub(super) fn gabriel_graph_property_metric<W, L: MetricSpace + 'static>(
    graph: &CsrGraph<W>,
    layout: &L,
) -> CoreResult<f64> {
    let any: &dyn Any = layout;
    if let Some(ml) = any.downcast_ref::<ManifoldLayout2d<f64>>() {
        return gabriel_graph_property_manifold_impl(graph, ml);
    }
    if let Some(ml) = any.downcast_ref::<ManifoldLayout2d<f32>>() {
        return gabriel_graph_property_manifold_impl(graph, ml);
    }

    let n = layout.len();
    if n <= 2 || graph.num_edges() == 0 {
        return Ok(0.0);
    }
    let edges = extract_unique_edges(graph);
    let mut violation_sum = 0.0_f64;

    for (u, v) in edges {
        let ui = u as usize;
        let vi = v as usize;
        let d_uv = layout.distance(ui, vi);
        let r = 0.5 * d_uv;
        if r <= 0.0 {
            continue;
        }
        let d_uv_sq = d_uv * d_uv;
        for w in 0..n {
            if w == ui || w == vi {
                continue;
            }
            let d_uw = layout.distance(ui, w);
            let d_vw = layout.distance(vi, w);
            // Exact Euclidean midpoint distance squared via parallelogram law.
            let dist_sq = (0.5 * (d_uw * d_uw + d_vw * d_vw) - 0.25 * d_uv_sq).max(0.0);
            let dist = dist_sq.sqrt();
            if dist < r {
                let viol = r - dist;
                violation_sum += viol * viol;
            }
        }
    }

    Ok(violation_sum)
}

fn gabriel_graph_property_manifold_impl<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &ManifoldLayout2d<S>,
) -> CoreResult<f64> {
    let n = layout.len();
    if n <= 2 || graph.num_edges() == 0 {
        return Ok(0.0);
    }

    let edges = extract_unique_edges(graph);
    let mut violation_sum = 0.0_f64;

    for (u, v) in edges {
        let ui = u as usize;
        let vi = v as usize;
        let d_uv = layout.distance(ui, vi);
        let r = 0.5 * d_uv;
        if r <= 0.0 {
            continue;
        }
        let mid = manifold_geodesic_interpolate_point(layout, ui, vi, 0.5)?;
        for w in 0..n {
            if w == ui || w == vi {
                continue;
            }
            let dist = manifold_point_distance_to_node(layout, mid, w)?;
            if dist < r {
                let viol = r - dist;
                violation_sum += viol * viol;
            }
        }
    }

    Ok(violation_sum)
}

type DistanceAndSortedNeighbors = (Vec<f64>, Vec<Vec<(usize, f64)>>);

fn build_distance_matrix_and_sorted_neighbors<L: MetricSpace + ?Sized>(
    layout: &L,
) -> DistanceAndSortedNeighbors {
    let n = layout.len();
    let mut dists = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = layout.distance(i, j);
            dists[i * n + j] = d;
            dists[j * n + i] = d;
        }
    }
    let mut sorted = vec![Vec::with_capacity(n.saturating_sub(1)); n];
    for i in 0..n {
        let row = &dists[i * n..(i + 1) * n];
        for (j, &d) in row.iter().enumerate() {
            if j != i {
                sorted[i].push((j, d));
            }
        }
        sorted[i].sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    }
    (dists, sorted)
}

pub(super) fn relative_neighborhood_edges_metric<L: MetricSpace + ?Sized>(
    layout: &L,
) -> Vec<(u32, u32)> {
    let n = layout.len();
    if n < 2 {
        return Vec::new();
    }
    let (dists, sorted) = build_distance_matrix_and_sorted_neighbors(layout);
    let mut edges = Vec::new();

    for i in 0..n {
        let row_i = &dists[i * n..(i + 1) * n];
        let nbrs_i = &sorted[i];
        for j in (i + 1)..n {
            let d_ij = row_i[j];
            if d_ij <= 0.0 {
                continue;
            }
            let thresh = d_ij * (1.0 - 1e-12);
            let row_j = &dists[j * n..(j + 1) * n];
            let mut blocked = false;
            for &(r, d_ir) in nbrs_i {
                if d_ir >= thresh {
                    break;
                }
                if r == j {
                    continue;
                }
                if row_j[r] < thresh {
                    blocked = true;
                    break;
                }
            }
            if !blocked {
                edges.push((i as u32, j as u32));
            }
        }
    }
    edges
}

pub(super) fn gabriel_graph_edges_metric<L: MetricSpace + 'static>(
    layout: &L,
) -> CoreResult<Vec<(u32, u32)>> {
    let any: &dyn Any = layout;
    if let Some(ml) = any.downcast_ref::<ManifoldLayout2d<f64>>() {
        return gabriel_graph_edges_manifold_impl(ml);
    }
    if let Some(ml) = any.downcast_ref::<ManifoldLayout2d<f32>>() {
        return gabriel_graph_edges_manifold_impl(ml);
    }

    let n = layout.len();
    if n < 2 {
        return Ok(Vec::new());
    }
    let (dists, sorted) = build_distance_matrix_and_sorted_neighbors(layout);
    let mut gg_blocked = vec![false; n];
    for (i, blocked) in gg_blocked.iter_mut().enumerate() {
        if sorted[i].first().is_some_and(|&(_, d)| d <= 0.0) {
            *blocked = true;
        }
    }

    let mut edges = Vec::new();
    for i in 0..n {
        if gg_blocked[i] {
            continue;
        }
        let row_i = &dists[i * n..(i + 1) * n];
        let nbrs_i = &sorted[i];
        for j in (i + 1)..n {
            if gg_blocked[j] {
                continue;
            }
            let d_ij = row_i[j];
            if d_ij <= 0.0 {
                continue;
            }
            let thresh = d_ij * d_ij * (1.0 + 1e-12);
            let row_j = &dists[j * n..(j + 1) * n];
            let mut blocked = false;
            for &(r_idx, d_ir) in nbrs_i {
                if d_ir * d_ir > thresh {
                    break;
                }
                if r_idx == j {
                    continue;
                }
                let d_jr = row_j[r_idx];
                if d_ir * d_ir + d_jr * d_jr <= thresh {
                    blocked = true;
                    break;
                }
            }
            if !blocked {
                edges.push((i as u32, j as u32));
            }
        }
    }
    Ok(edges)
}

fn gabriel_graph_edges_manifold_impl<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
) -> CoreResult<Vec<(u32, u32)>> {
    let n = layout.len();
    if n < 2 {
        return Ok(Vec::new());
    }
    let (dists, sorted) = build_distance_matrix_and_sorted_neighbors(layout);
    let mut gg_blocked = vec![false; n];
    for (i, blocked) in gg_blocked.iter_mut().enumerate() {
        if sorted[i].first().is_some_and(|&(_, d)| d <= 0.0) {
            *blocked = true;
        }
    }

    let mut edges = Vec::new();
    for i in 0..n {
        if gg_blocked[i] {
            continue;
        }
        let row_i = &dists[i * n..(i + 1) * n];
        let nbrs_i = &sorted[i];
        for j in (i + 1)..n {
            if gg_blocked[j] {
                continue;
            }
            let d_ij = row_i[j];
            if d_ij <= 0.0 {
                continue;
            }
            let mid = manifold_geodesic_interpolate_point(layout, i, j, 0.5)?;
            let d_im = manifold_point_distance_to_node(layout, mid, i)?;
            let r = 0.5 * d_ij;
            let block_thresh = r * (1.0 + 1e-12);
            let prune_thresh = d_im + r * (1.0 + 1e-11);
            let mut blocked = false;
            for &(r_idx, d_ir) in nbrs_i {
                if d_ir > prune_thresh {
                    break;
                }
                if r_idx == j {
                    continue;
                }
                let d_mr = manifold_point_distance_to_node(layout, mid, r_idx)?;
                if d_mr <= block_thresh {
                    blocked = true;
                    break;
                }
            }
            if !blocked {
                edges.push((i as u32, j as u32));
            }
        }
    }
    Ok(edges)
}
