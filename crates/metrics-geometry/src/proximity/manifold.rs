use egraph_core::{
    CoreError, CoreResult, CsrGraph, ManifoldKind, ManifoldLayout2d, Scalar, lorentz_distance,
    lorentz_exp_map, lorentz_log_map, poincare_distance, poincare_exp_map, poincare_log_map,
    spherical_distance, spherical_exp_map, spherical_log_map, torus_delta, wrap_torus_point,
};

use crate::{extract_unique_edges, validate_graph_manifold_layout};

use super::mean_jaccard_graph_vs_edges;

/// Computes the exact pairwise geodesic diameter $\max_{i < j} d_{\mathcal{M}_s}(x_i, x_j)$ of
/// a [`ManifoldLayout2d`].
#[doc(hidden)]
pub fn manifold_geodesic_diameter<S: Scalar>(layout: &ManifoldLayout2d<S>) -> CoreResult<f64> {
    layout.validate_finite()?;
    let n = layout.len();
    if n <= 1 {
        return Ok(0.0);
    }
    let mut d_max = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            d_max = d_max.max(layout.geodesic_distance(i, j)?.to_f64());
        }
    }
    Ok(d_max)
}

/// Computes the geodesic interpolation point $\exp_{x_i}(t \log_{x_i}(x_j))$ on the unit
/// manifold coordinate chart in `f64`.
#[doc(hidden)]
pub fn manifold_geodesic_interpolate_point<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    i: usize,
    j: usize,
    t: f64,
) -> CoreResult<(f64, f64)> {
    let n = layout.len();
    if i >= n || j >= n {
        return Err(CoreError::index_out_of_bounds(
            i.max(j),
            n,
            "manifold_geodesic_interpolate_point",
        ));
    }
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

/// Computes the metric-scaled geodesic distance from a unit-manifold point `pt` (`f64`) to
/// vertex `q` in `layout`.
#[doc(hidden)]
pub fn manifold_point_distance_to_node<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    pt: (f64, f64),
    q: usize,
) -> CoreResult<f64> {
    let n = layout.len();
    if q >= n {
        return Err(CoreError::index_out_of_bounds(
            q,
            n,
            "manifold_point_distance_to_node",
        ));
    }
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

/// Computes the one-sided hinge node-resolution violation penalty on a [`ManifoldLayout2d`]
/// using exact geodesic distances and geodesic diameter $d_{\max}$.
pub fn node_resolution_manifold<S: Scalar>(layout: &ManifoldLayout2d<S>) -> CoreResult<f64> {
    layout.validate_finite()?;
    let n = layout.len();
    if n <= 1 {
        return Ok(0.0);
    }

    let d_max = manifold_geodesic_diameter(layout)?;
    if d_max <= 0.0 {
        let num_pairs = (n as f64) * ((n - 1) as f64) * 0.5;
        return Ok(num_pairs);
    }

    let threshold = d_max / (n as f64).sqrt();
    let mut sum = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = layout.geodesic_distance(i, j)?.to_f64();
            if dist < threshold {
                let hinge = 1.0 - dist / threshold;
                sum += hinge * hinge;
            }
        }
    }
    Ok(sum)
}

/// Evaluates geodesic Gabriel-graph violations across all edges yielded by
/// `extract_unique_edges(graph)` on a [`ManifoldLayout2d`].
///
/// For each edge `(u, v)`, computes the geodesic midpoint $m_{uv} = \exp_u(\frac{1}{2}\log_u(v))$
/// and radius $r_{uv} = \frac{1}{2} d_{\mathcal{M}_s}(u, v)$, and sums
/// $\max(0, r_{uv} - d_{\mathcal{M}_s}(m_{uv}, w))^2$ over all vertices $w \notin \{u, v\}$.
pub fn gabriel_graph_property_manifold<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &ManifoldLayout2d<S>,
) -> CoreResult<f64> {
    validate_graph_manifold_layout(graph, layout, "gabriel_graph_property_manifold")?;
    let n = layout.len();
    if n <= 2 || graph.num_edges() == 0 {
        return Ok(0.0);
    }

    let edges = extract_unique_edges(graph);
    let mut violation_sum = 0.0_f64;

    for (u, v) in edges {
        let ui = u as usize;
        let vi = v as usize;
        let d_uv = layout.geodesic_distance(ui, vi)?.to_f64();
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

fn build_distance_matrix_and_sorted_neighbors<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
) -> CoreResult<DistanceAndSortedNeighbors> {
    let n = layout.len();
    let mut dists = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = layout.geodesic_distance(i, j)?.to_f64();
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
    Ok((dists, sorted))
}

/// Extracts exact geodesic Relative Neighborhood Graph (`RNG`) edges `(u, v)` (`u < v`) over
/// a [`ManifoldLayout2d`].
#[doc(hidden)]
pub fn relative_neighborhood_edges_manifold<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
) -> CoreResult<Vec<(u32, u32)>> {
    layout.validate_finite()?;
    let n = layout.len();
    if n < 2 {
        return Ok(Vec::new());
    }
    let (dists, sorted) = build_distance_matrix_and_sorted_neighbors(layout)?;
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
    Ok(edges)
}

/// Extracts exact geodesic Gabriel Graph (`GG`) edges `(u, v)` (`u < v`) over a
/// [`ManifoldLayout2d`].
#[doc(hidden)]
pub fn gabriel_graph_edges_manifold<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
) -> CoreResult<Vec<(u32, u32)>> {
    layout.validate_finite()?;
    let n = layout.len();
    if n < 2 {
        return Ok(Vec::new());
    }
    let (dists, sorted) = build_distance_matrix_and_sorted_neighbors(layout)?;
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

/// Computes Eades et al. (2017) unweighted geodesic Relative Neighborhood Graph preservation
/// (`S_RNG`) in `[0.0, 1.0]` over a [`ManifoldLayout2d`].
pub fn relative_neighborhood_preservation_manifold<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &ManifoldLayout2d<S>,
) -> CoreResult<f64> {
    validate_graph_manifold_layout(graph, layout, "relative_neighborhood_preservation_manifold")?;
    if layout.is_empty() {
        return Ok(0.0);
    }
    let rng_edges = relative_neighborhood_edges_manifold(layout)?;
    Ok(mean_jaccard_graph_vs_edges(graph, &rng_edges))
}

/// Computes Eades et al. (2017) unweighted geodesic Gabriel Graph preservation (`S_GG`) in
/// `[0.0, 1.0]` over a [`ManifoldLayout2d`].
pub fn gabriel_graph_preservation_manifold<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &ManifoldLayout2d<S>,
) -> CoreResult<f64> {
    validate_graph_manifold_layout(graph, layout, "gabriel_graph_preservation_manifold")?;
    if layout.is_empty() {
        return Ok(0.0);
    }
    let gg_edges = gabriel_graph_edges_manifold(layout)?;
    Ok(mean_jaccard_graph_vs_edges(graph, &gg_edges))
}
