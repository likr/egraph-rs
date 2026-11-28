use egraph_core::{CoreResult, CsrGraph, IterativeSolver, Layout2d, Scalar};
use egraph_layout_force::{
    ForceBackend, FrBackend, FruchtermanReingoldConfig, FruchtermanReingoldSolver, YifanHuConfig,
    YifanHuSolver,
};
use egraph_layout_sgd::{SgdScheduleConfig, SgdSolver};

use super::config::{GibBox, GibIntraLayoutMode, GroupInABoxConfig};

const GEOM_EPS: f64 = 1e-12;
const GOLDEN_ANGLE: f64 = 2.399_963_229_728_653;

pub(super) enum IntraSolver {
    Trivial,
    Fr(Box<FruchtermanReingoldSolver<Layout2d<f64>, FrBackend>>),
    Sgd(Box<SgdSolver<Layout2d<f64>>>),
    YifanHu(Box<YifanHuSolver<Layout2d<f64>, ForceBackend>>),
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

pub(super) fn build_unweighted_induced_subgraph<W>(
    graph: &CsrGraph<W>,
    nodes: &[u32],
    remap: &mut [u32],
) -> CoreResult<CsrGraph<f64>> {
    for (local_idx, &old_u) in nodes.iter().enumerate() {
        remap[old_u as usize] = local_idx as u32;
    }
    let directed = graph.is_directed();
    let mut sub_edges = Vec::new();
    for (local_u, &old_u) in nodes.iter().enumerate() {
        let lu = local_u as u32;
        for &old_v in graph.neighbors(old_u as usize) {
            let lv = remap[old_v as usize];
            if lv != u32::MAX && (directed || lu < lv) {
                sub_edges.push((lu, lv));
            }
        }
    }
    for &old_u in nodes {
        remap[old_u as usize] = u32::MAX;
    }
    CsrGraph::<f64>::from_unweighted_edges(nodes.len(), &sub_edges, directed, true)
}

pub(super) fn build_intra_solver<W: Scalar>(
    sub_g: &CsrGraph<W>,
    m: usize,
    has_edges: bool,
    cluster_idx: usize,
    config: &GroupInABoxConfig,
    weighted: bool,
) -> CoreResult<IntraSolver> {
    if m <= 1 || !has_edges || config.intra_iterations == 0 {
        return Ok(IntraSolver::Trivial);
    }
    let sub_seed = config
        .seed
        .wrapping_add((cluster_idx as u64).wrapping_mul(0x9E37_79B9));
    match config.intra_mode {
        GibIntraLayoutMode::Fr => {
            let fr_cfg = FruchtermanReingoldConfig::new()
                .with_iterations(config.intra_iterations)
                .with_seed(sub_seed);
            let solver = if weighted {
                FruchtermanReingoldSolver::<Layout2d<f64>, FrBackend>::with_backend(
                    sub_g,
                    fr_cfg,
                    config.fr_backend.clone(),
                )?
            } else {
                FruchtermanReingoldSolver::<Layout2d<f64>, FrBackend>::with_backend_unweighted(
                    sub_g,
                    fr_cfg,
                    config.fr_backend.clone(),
                )?
            };
            Ok(IntraSolver::Fr(Box::new(solver)))
        }
        GibIntraLayoutMode::SgdFull => Ok(IntraSolver::Sgd(Box::new(SgdSolver::full(
            sub_g,
            SgdScheduleConfig::new(config.intra_iterations),
            sub_seed,
        )?))),
        GibIntraLayoutMode::YifanHu => {
            let yh_cfg = YifanHuConfig::new()
                .with_max_iterations_per_level(config.intra_iterations)
                .with_seed(sub_seed);
            let solver = if weighted {
                YifanHuSolver::<Layout2d<f64>, ForceBackend>::with_backend(
                    sub_g,
                    yh_cfg,
                    config.yifan_hu_backend.clone(),
                )?
            } else {
                YifanHuSolver::<Layout2d<f64>, ForceBackend>::with_backend_unweighted(
                    sub_g,
                    yh_cfg,
                    config.yifan_hu_backend.clone(),
                )?
            };
            Ok(IntraSolver::YifanHu(Box::new(solver)))
        }
    }
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
