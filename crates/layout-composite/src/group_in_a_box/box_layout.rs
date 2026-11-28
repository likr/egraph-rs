use egraph_core::{
    CoreError, CoreResult, CsrGraph, IterativeSolver, Layout2d, NodeExtents2d, Scalar,
};
use egraph_layout_constraints::project_rectangle_no_overlap_2d;
use egraph_layout_force::{FrBackend, FruchtermanReingoldConfig, FruchtermanReingoldSolver};
use egraph_spatial_trees::{RectangleOverlapSweep, squarified_treemap};

use super::config::{GibBox, GroupInABoxConfig};

pub(super) fn build_treemap_boxes(
    orig_cluster_ids: &[u32],
    counts: &[usize],
    cluster_weights: &[f64],
    config: &GroupInABoxConfig,
) -> CoreResult<Vec<GibBox>> {
    let k = orig_cluster_ids.len();
    let rects = squarified_treemap(
        cluster_weights,
        0.0,
        0.0,
        config.total_width,
        config.total_height,
    )?;
    Ok((0..k)
        .map(|c| GibBox {
            cluster_id: orig_cluster_ids[c],
            x_min: rects[c][0],
            x_max: rects[c][1],
            y_min: rects[c][2],
            y_max: rects[c][3],
            node_count: counts[c],
            weight: cluster_weights[c],
        })
        .collect())
}

pub(super) fn compute_force_directed_boxes<W: Scalar>(
    graph: &CsrGraph<W>,
    canon_of_node: &[u32],
    orig_cluster_ids: &[u32],
    counts: &[usize],
    cluster_weights: &[f64],
    config: &GroupInABoxConfig,
) -> CoreResult<Vec<GibBox>> {
    let k = orig_cluster_ids.len();
    if k == 1 {
        return Ok(vec![GibBox {
            cluster_id: orig_cluster_ids[0],
            x_min: 0.0,
            x_max: config.total_width,
            y_min: 0.0,
            y_max: config.total_height,
            node_count: counts[0],
            weight: cluster_weights[0],
        }]);
    }

    let n = graph.num_nodes();
    let directed = graph.is_directed();
    let mut raw_inter: Vec<((u32, u32), f64)> = Vec::new();
    for u in 0..n {
        let cu = canon_of_node[u];
        let nbrs = graph.neighbors(u);
        let ws = graph.neighbor_weights(u);
        for (pos, &v_u32) in nbrs.iter().enumerate() {
            let v = v_u32 as usize;
            if !directed && u > v {
                continue;
            }
            let cv = canon_of_node[v];
            if cu == cv {
                continue;
            }
            let w = ws.map_or(1.0, |slice| slice[pos].to_f64());
            if !w.is_finite() {
                return Err(CoreError::non_finite("GroupInABoxSolver::edge_weight"));
            }
            if w > 0.0 {
                let pair = if cu < cv { (cu, cv) } else { (cv, cu) };
                raw_inter.push((pair, w));
            }
        }
    }
    let inter_graph = aggregate_inter_cluster_graph(k, raw_inter)?;
    layout_cluster_boxes(
        &inter_graph,
        orig_cluster_ids,
        counts,
        cluster_weights,
        config,
    )
}

pub(super) fn compute_force_directed_boxes_unweighted<W>(
    graph: &CsrGraph<W>,
    canon_of_node: &[u32],
    orig_cluster_ids: &[u32],
    counts: &[usize],
    cluster_weights: &[f64],
    config: &GroupInABoxConfig,
) -> CoreResult<Vec<GibBox>> {
    let k = orig_cluster_ids.len();
    if k == 1 {
        return Ok(vec![GibBox {
            cluster_id: orig_cluster_ids[0],
            x_min: 0.0,
            x_max: config.total_width,
            y_min: 0.0,
            y_max: config.total_height,
            node_count: counts[0],
            weight: cluster_weights[0],
        }]);
    }

    let n = graph.num_nodes();
    let directed = graph.is_directed();
    let mut raw_inter: Vec<((u32, u32), f64)> = Vec::new();
    for u in 0..n {
        let cu = canon_of_node[u];
        for &v_u32 in graph.neighbors(u) {
            let v = v_u32 as usize;
            if !directed && u > v {
                continue;
            }
            let cv = canon_of_node[v];
            if cu == cv {
                continue;
            }
            let pair = if cu < cv { (cu, cv) } else { (cv, cu) };
            raw_inter.push((pair, 1.0));
        }
    }
    let inter_graph = aggregate_inter_cluster_graph(k, raw_inter)?;
    layout_cluster_boxes(
        &inter_graph,
        orig_cluster_ids,
        counts,
        cluster_weights,
        config,
    )
}

fn aggregate_inter_cluster_graph(
    k: usize,
    mut raw_inter: Vec<((u32, u32), f64)>,
) -> CoreResult<CsrGraph<f64>> {
    raw_inter.sort_unstable_by_key(|a| a.0);

    let mut inter_edges = Vec::new();
    let mut inter_weights = Vec::new();
    let mut idx = 0usize;
    while idx < raw_inter.len() {
        let pair = raw_inter[idx].0;
        let mut w_sum = raw_inter[idx].1;
        idx += 1;
        while idx < raw_inter.len() && raw_inter[idx].0 == pair {
            w_sum += raw_inter[idx].1;
            idx += 1;
        }
        inter_edges.push(pair);
        inter_weights.push(w_sum);
    }

    CsrGraph::from_edges(k, &inter_edges, Some(&inter_weights), false, false)
}

fn layout_cluster_boxes(
    inter_graph: &CsrGraph<f64>,
    orig_cluster_ids: &[u32],
    counts: &[usize],
    cluster_weights: &[f64],
    config: &GroupInABoxConfig,
) -> CoreResult<Vec<GibBox>> {
    let k = orig_cluster_ids.len();
    let total_area = config.total_width * config.total_height;
    let total_weight: f64 = cluster_weights.iter().sum();

    // Square box extents proportional to sqrt(weight).
    let mut hw = Vec::with_capacity(k);
    let mut hh = Vec::with_capacity(k);
    for &w in cluster_weights {
        let side = (total_area * (w / total_weight)).sqrt();
        let half = 0.5 * side;
        hw.push(half);
        hh.push(half);
    }

    // Deterministic circular initialization for cluster box centers.
    let radius = 0.35 * config.total_width.min(config.total_height);
    let mut cx = Vec::with_capacity(k);
    let mut cy = Vec::with_capacity(k);
    for c in 0..k {
        let angle = (c as f64) * std::f64::consts::TAU / (k as f64);
        cx.push(0.5 * config.total_width + radius * angle.cos());
        cy.push(0.5 * config.total_height + radius * angle.sin());
    }
    let mut box_layout = Layout2d::from_xy(cx, cy)?;

    if config.fd_iterations > 0 && inter_graph.num_edges() > 0 {
        let fr_cfg = FruchtermanReingoldConfig::new()
            .with_iterations(config.fd_iterations)
            .with_area(Some(total_area))
            .with_gravity(0.05)
            .with_seed(config.seed);
        let mut fr = FruchtermanReingoldSolver::<Layout2d<f64>, FrBackend>::with_backend(
            inter_graph,
            fr_cfg,
            config.fr_backend.clone(),
        )?;
        fr.run(&mut box_layout)?;
    }

    // Strict overlap removal across all cluster boxes.
    let extents = NodeExtents2d::new(&hw, &hh)?;
    project_rectangle_no_overlap_2d(&mut box_layout, extents)?;
    enforce_strict_box_separation(&mut box_layout, &hw, &hh, &extents)?;

    // Re-center bounding box around (total_width / 2, total_height / 2).
    let bxs = box_layout.x();
    let bys = box_layout.y();
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for c in 0..k {
        min_x = min_x.min(bxs[c] - hw[c]);
        max_x = max_x.max(bxs[c] + hw[c]);
        min_y = min_y.min(bys[c] - hh[c]);
        max_y = max_y.max(bys[c] + hh[c]);
    }
    let width = (max_x - min_x).max(0.0);
    let height = (max_y - min_y).max(0.0);
    let mid_x = 0.5 * (min_x + max_x);
    let mid_y = 0.5 * (min_y + max_y);
    let scale = if config.fit_bounds
        && width > 0.0
        && height > 0.0
        && (width > config.total_width || height > config.total_height)
    {
        (config.total_width / width)
            .min(config.total_height / height)
            .min(1.0)
    } else {
        1.0
    };
    let target_cx = 0.5 * config.total_width;
    let target_cy = 0.5 * config.total_height;

    let mut boxes = Vec::with_capacity(k);
    for c in 0..k {
        let cx_c = target_cx + (bxs[c] - mid_x) * scale;
        let cy_c = target_cy + (bys[c] - mid_y) * scale;
        let hw_c = hw[c] * scale;
        let hh_c = hh[c] * scale;
        boxes.push(GibBox {
            cluster_id: orig_cluster_ids[c],
            x_min: cx_c - hw_c,
            x_max: cx_c + hw_c,
            y_min: cy_c - hh_c,
            y_max: cy_c + hh_c,
            node_count: counts[c],
            weight: cluster_weights[c],
        });
    }
    Ok(boxes)
}

fn enforce_strict_box_separation(
    layout: &mut Layout2d<f64>,
    hw: &[f64],
    hh: &[f64],
    extents: &NodeExtents2d<'_, f64>,
) -> CoreResult<()> {
    let k = layout.len();
    if k < 2 {
        return Ok(());
    }
    for _ in 0..(k * 2 + 8) {
        let overlaps = RectangleOverlapSweep::find_layout_extents_overlaps(layout, extents)?;
        if overlaps.is_empty() {
            return Ok(());
        }
        let (xs_mut, ys_mut) = layout.xy_mut();
        for &(i, j) in &overlaps {
            let dx = xs_mut[j] - xs_mut[i];
            let dy = ys_mut[j] - ys_mut[i];
            let req_x = hw[i] + hw[j];
            let req_y = hh[i] + hh[j];
            let pen_x = req_x - dx.abs();
            let pen_y = req_y - dy.abs();
            if pen_x > 0.0 && pen_y > 0.0 {
                if pen_x <= pen_y {
                    let sign = if dx > 0.0 || (dx == 0.0 && i < j) {
                        1.0
                    } else {
                        -1.0
                    };
                    let delta = 0.5 * pen_x + 1e-6 * (1.0 + req_x);
                    xs_mut[i] -= sign * delta;
                    xs_mut[j] += sign * delta;
                } else {
                    let sign = if dy > 0.0 || (dy == 0.0 && i < j) {
                        1.0
                    } else {
                        -1.0
                    };
                    let delta = 0.5 * pen_y + 1e-6 * (1.0 + req_y);
                    ys_mut[i] -= sign * delta;
                    ys_mut[j] += sign * delta;
                }
            }
        }
    }
    Ok(())
}
