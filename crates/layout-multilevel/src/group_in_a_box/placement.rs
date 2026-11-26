use egraph_core::{CoreError, CoreResult, CsrGraph, IterativeSolver, Layout2d, Scalar};
use egraph_layout_constraints::project_rectangle_no_overlap_2d;
use egraph_layout_force::{FruchtermanReingoldConfig, FruchtermanReingoldSolver, YifanHuSolver};
use egraph_layout_sgd::SgdSolver;
use egraph_spatial_trees::{Rect2d, RectangleOverlapSweep};

use super::config::{GibBox, GroupInABoxConfig};

const GEOM_EPS: f64 = 1e-12;
const GOLDEN_ANGLE: f64 = 2.399_963_229_728_653;

pub(super) enum IntraSolver {
    Trivial,
    Fr(Box<FruchtermanReingoldSolver<f64>>),
    Sgd(Box<SgdSolver<f64>>),
    YifanHu(Box<YifanHuSolver<f64>>),
}

impl IntraSolver {
    pub(super) fn is_finished(&self) -> bool {
        match self {
            Self::Trivial => true,
            Self::Fr(s) => s.status().is_finished(),
            Self::Sgd(s) => s.status().is_finished(),
            Self::YifanHu(s) => s.status().is_finished(),
        }
    }

    pub(super) fn step(&mut self, layout: &mut Layout2d<f64>) -> CoreResult<()> {
        match self {
            Self::Trivial => Ok(()),
            Self::Fr(s) => {
                s.step(layout)?;
                Ok(())
            }
            Self::Sgd(s) => {
                s.step(layout)?;
                Ok(())
            }
            Self::YifanHu(s) => {
                s.step(layout)?;
                Ok(())
            }
        }
    }
}

pub(super) struct ClusterState {
    pub(super) nodes: Vec<u32>,
    pub(super) has_edges: bool,
    pub(super) sub_layout: Layout2d<f64>,
    pub(super) solver: IntraSolver,
    pub(super) initialized: bool,
}

pub(super) fn canonical_cluster_groups(
    labels: &[u32],
) -> CoreResult<(Vec<u32>, Vec<u32>, Vec<usize>)> {
    let n = labels.len();
    if n == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }

    let max_label = labels.iter().copied().max().unwrap_or(0) as usize;
    let mut canon_of_node = vec![0u32; n];
    let mut orig_ids = Vec::new();
    let mut counts = Vec::new();

    let fast_limit = n.max(64).saturating_mul(4).min(1_000_000);
    if max_label < fast_limit {
        let mut remap = vec![u32::MAX; max_label + 1];
        for (i, &lbl) in labels.iter().enumerate() {
            let idx = lbl as usize;
            if remap[idx] == u32::MAX {
                let cid = orig_ids.len() as u32;
                remap[idx] = cid;
                orig_ids.push(lbl);
                counts.push(0usize);
            }
            let cid = remap[idx];
            canon_of_node[i] = cid;
            counts[cid as usize] += 1;
        }
        return Ok((canon_of_node, orig_ids, counts));
    }

    let mut pairs: Vec<(u32, u32)> = labels
        .iter()
        .enumerate()
        .map(|(i, &lbl)| (lbl, i as u32))
        .collect();
    pairs.sort_unstable();

    let mut groups: Vec<(u32, u32, usize, usize)> = Vec::new();
    let mut start = 0usize;
    while start < n {
        let orig = pairs[start].0;
        let mut end = start + 1;
        while end < n && pairs[end].0 == orig {
            end += 1;
        }
        groups.push((pairs[start].1, orig, start, end));
        start = end;
    }
    groups.sort_unstable_by_key(|&(first_node, _, _, _)| first_node);

    for (cid, &(_, orig, g_start, g_end)) in groups.iter().enumerate() {
        let cid_u32 = cid as u32;
        orig_ids.push(orig);
        counts.push(g_end - g_start);
        for &(_, node_idx) in &pairs[g_start..g_end] {
            canon_of_node[node_idx as usize] = cid_u32;
        }
    }
    Ok((canon_of_node, orig_ids, counts))
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
    let total_area = config.total_width * config.total_height;
    let total_weight: f64 = cluster_weights.iter().sum();

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

    // Build inter-cluster undirected weighted graph.
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

    let inter_graph = CsrGraph::from_edges(k, &inter_edges, Some(&inter_weights), false, false)?;

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
        let mut fr = FruchtermanReingoldSolver::<f64>::new(
            &inter_graph,
            FruchtermanReingoldConfig {
                iterations: config.fd_iterations,
                area: Some(total_area),
                gravity: 0.05,
                seed: config.seed,
                ..Default::default()
            },
        )?;
        fr.run(&mut box_layout)?;
    }

    // Strict overlap removal across all cluster boxes.
    project_rectangle_no_overlap_2d(&mut box_layout, &hw, &hh)?;
    enforce_strict_box_separation(&mut box_layout, &hw, &hh)?;

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
) -> CoreResult<()> {
    let k = layout.len();
    if k < 2 {
        return Ok(());
    }
    for _ in 0..(k * 2 + 8) {
        let mut rects = Vec::with_capacity(k);
        {
            let xs = layout.x();
            let ys = layout.y();
            for i in 0..k {
                rects.push(Rect2d::new(
                    xs[i] - hw[i],
                    xs[i] + hw[i],
                    ys[i] - hh[i],
                    ys[i] + hh[i],
                )?);
            }
        }
        let overlaps = RectangleOverlapSweep::find_overlaps(&rects)?;
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

pub(super) fn init_cluster_sub_layout<S: Scalar>(cluster: &mut ClusterState, layout: &Layout2d<S>) {
    let m = cluster.nodes.len();
    if m == 0 {
        return;
    }
    if m == 1 {
        cluster.sub_layout.x_mut()[0] = 0.0;
        cluster.sub_layout.y_mut()[0] = 0.0;
        return;
    }

    let xs_in = layout.x();
    let ys_in = layout.y();
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for &u in &cluster.nodes {
        let x = xs_in[u as usize].to_f64();
        let y = ys_in[u as usize].to_f64();
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }

    let collapsed = (max_x - min_x) <= GEOM_EPS && (max_y - min_y) <= GEOM_EPS;
    let (sub_x, sub_y) = cluster.sub_layout.xy_mut();
    if !cluster.has_edges || collapsed {
        // Deterministic sunflower/disk packing so edgeless or collapsed clusters space out cleanly.
        let m_f = m as f64;
        for i in 0..m {
            let r = ((i as f64 + 0.5) / m_f).sqrt();
            let theta = (i as f64) * GOLDEN_ANGLE;
            sub_x[i] = r * theta.cos();
            sub_y[i] = r * theta.sin();
        }
    } else {
        for (i, &u) in cluster.nodes.iter().enumerate() {
            sub_x[i] = xs_in[u as usize].to_f64();
            sub_y[i] = ys_in[u as usize].to_f64();
        }
    }
}

pub(super) fn map_cluster_into_box<S: Scalar>(
    cluster: &ClusterState,
    gib_box: &GibBox,
    padding: f64,
    out_layout: &mut Layout2d<S>,
) {
    let m = cluster.nodes.len();
    if m == 0 {
        return;
    }
    let bw = gib_box.x_max - gib_box.x_min;
    let bh = gib_box.y_max - gib_box.y_min;
    let pad_x = padding * bw;
    let pad_y = padding * bh;
    let ix_min = gib_box.x_min + pad_x;
    let ix_max = gib_box.x_max - pad_x;
    let iy_min = gib_box.y_min + pad_y;
    let iy_max = gib_box.y_max - pad_y;

    let cx_box = 0.5 * (ix_min + ix_max);
    let cy_box = 0.5 * (iy_min + iy_max);
    let (out_x, out_y) = out_layout.xy_mut();

    if m == 1 {
        let u = cluster.nodes[0] as usize;
        out_x[u] = S::from_f64(cx_box);
        out_y[u] = S::from_f64(cy_box);
        return;
    }

    let sx = cluster.sub_layout.x();
    let sy = cluster.sub_layout.y();
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for i in 0..m {
        min_x = min_x.min(sx[i]);
        max_x = max_x.max(sx[i]);
        min_y = min_y.min(sy[i]);
        max_y = max_y.max(sy[i]);
    }

    let span_x = max_x - min_x;
    let span_y = max_y - min_y;
    let pad_w = (ix_max - ix_min).max(0.0);
    let pad_h = (iy_max - iy_min).max(0.0);
    let cx_sub = 0.5 * (min_x + max_x);
    let cy_sub = 0.5 * (min_y + max_y);

    if span_x <= GEOM_EPS && span_y <= GEOM_EPS {
        for &u in &cluster.nodes {
            let ui = u as usize;
            out_x[ui] = S::from_f64(cx_box);
            out_y[ui] = S::from_f64(cy_box);
        }
    } else if span_x <= GEOM_EPS {
        let scale = pad_h / span_y;
        for (i, &u) in cluster.nodes.iter().enumerate() {
            let ui = u as usize;
            let my = (cy_box + (sy[i] - cy_sub) * scale).clamp(iy_min, iy_max);
            out_x[ui] = S::from_f64(cx_box);
            out_y[ui] = S::from_f64(my);
        }
    } else if span_y <= GEOM_EPS {
        let scale = pad_w / span_x;
        for (i, &u) in cluster.nodes.iter().enumerate() {
            let ui = u as usize;
            let mx = (cx_box + (sx[i] - cx_sub) * scale).clamp(ix_min, ix_max);
            out_x[ui] = S::from_f64(mx);
            out_y[ui] = S::from_f64(cy_box);
        }
    } else {
        let scale = (pad_w / span_x).min(pad_h / span_y);
        for (i, &u) in cluster.nodes.iter().enumerate() {
            let ui = u as usize;
            let mx = (cx_box + (sx[i] - cx_sub) * scale).clamp(ix_min, ix_max);
            let my = (cy_box + (sy[i] - cy_sub) * scale).clamp(iy_min, iy_max);
            out_x[ui] = S::from_f64(mx);
            out_y[ui] = S::from_f64(my);
        }
    }
}
