use egraph_core::{CoreError, CoreResult, Layout2d, Scalar};
use egraph_metrics_geometry::{
    build_kdtree_f64,
    proximity::{gabriel_graph_edges_with_tree, relative_neighborhood_edges_with_tree},
    validate_tree_layout,
};
use egraph_spatial_trees::{KdQueryScratch, StaticKdTree};

use super::{compact_labels, validate_layout_labels};

pub(super) fn dir_cpt_from_degree_counts(
    comp: &[u32],
    class_sizes: &[usize],
    k_classes: usize,
    deg_out: &[usize],
    same_out: &[usize],
) -> f64 {
    let mut class_sum = vec![0.0_f64; k_classes];
    for (i, &c) in comp.iter().enumerate() {
        let p_i = if deg_out[i] == 0 {
            1.0
        } else {
            same_out[i] as f64 / deg_out[i] as f64
        };
        class_sum[c as usize] += p_i;
    }
    let macro_sum: f64 = class_sum
        .iter()
        .zip(class_sizes.iter())
        .map(|(&s, &sz)| s / sz as f64)
        .sum();
    macro_sum / k_classes as f64
}

pub(super) fn dir_cpt_from_undirected_edges(
    n: usize,
    comp: &[u32],
    class_sizes: &[usize],
    k_classes: usize,
    edges: &[(u32, u32)],
) -> f64 {
    let mut deg = vec![0usize; n];
    let mut same = vec![0usize; n];
    for &(u, v) in edges {
        let ui = u as usize;
        let vi = v as usize;
        deg[ui] += 1;
        deg[vi] += 1;
        if comp[ui] == comp[vi] {
            same[ui] += 1;
            same[vi] += 1;
        }
    }
    dir_cpt_from_degree_counts(comp, class_sizes, k_classes, &deg, &same)
}

/// Computes Aupetit & Sedlmair (2016) `SepMe` $k$-Nearest Neighbor Graph (`KNNG DIR CPT`)
/// in `[0.0, 1.0]`.
pub fn sepme_knng<S: Scalar>(layout: &Layout2d<S>, labels: &[u32], k: usize) -> CoreResult<f64> {
    if k == 0 {
        return Err(CoreError::invalid_input("sepme_knng requires k >= 1"));
    }
    validate_layout_labels(layout, labels, "sepme_knng")?;
    if layout.is_empty() {
        return Ok(0.0);
    }
    if layout.len() == 1 {
        return Ok(1.0);
    }
    let tree = build_kdtree_f64(layout)?;
    sepme_knng_with_tree(layout, labels, k, &tree)
}

#[doc(hidden)]
pub fn sepme_knng_with_tree<S: Scalar>(
    layout: &Layout2d<S>,
    labels: &[u32],
    k: usize,
    tree: &StaticKdTree<2, f64>,
) -> CoreResult<f64> {
    if k == 0 {
        return Err(CoreError::invalid_input("sepme_knng requires k >= 1"));
    }
    validate_layout_labels(layout, labels, "sepme_knng_with_tree")?;
    validate_tree_layout(tree, layout, "sepme_knng_with_tree")?;

    let n = layout.len();
    if n == 0 {
        return Ok(0.0);
    }
    if n == 1 {
        return Ok(1.0);
    }

    let (comp, class_sizes, k_classes) = compact_labels(labels);
    let k_eff = k.min(n - 1);
    let mut scratch = KdQueryScratch::with_capacity(k_eff);
    let mut nbrs = Vec::with_capacity(k_eff);
    let mut deg = vec![0usize; n];
    let mut same = vec![0usize; n];

    for i in 0..n {
        tree.k_nearest_excluding_self_with_scratch(i, k_eff, &mut scratch, &mut nbrs)?;
        deg[i] = nbrs.len();
        let ci = comp[i];
        let mut s = 0usize;
        for &(j, _) in &nbrs {
            if comp[j] == ci {
                s += 1;
            }
        }
        same[i] = s;
    }

    Ok(dir_cpt_from_degree_counts(
        &comp,
        &class_sizes,
        k_classes,
        &deg,
        &same,
    ))
}

/// Computes Aupetit & Sedlmair (2016) `SepMe` Relative Neighborhood Graph (`RNG DIR CPT`)
/// in `[0.0, 1.0]`.
pub fn sepme_rng<S: Scalar>(layout: &Layout2d<S>, labels: &[u32]) -> CoreResult<f64> {
    validate_layout_labels(layout, labels, "sepme_rng")?;
    if layout.is_empty() {
        return Ok(0.0);
    }
    if layout.len() == 1 {
        return Ok(1.0);
    }
    let tree = build_kdtree_f64(layout)?;
    sepme_rng_with_tree(layout, labels, &tree)
}

#[doc(hidden)]
pub fn sepme_rng_with_tree<S: Scalar>(
    layout: &Layout2d<S>,
    labels: &[u32],
    tree: &StaticKdTree<2, f64>,
) -> CoreResult<f64> {
    validate_layout_labels(layout, labels, "sepme_rng_with_tree")?;
    validate_tree_layout(tree, layout, "sepme_rng_with_tree")?;
    let n = layout.len();
    if n == 0 {
        return Ok(0.0);
    }
    if n == 1 {
        return Ok(1.0);
    }
    let (comp, class_sizes, k_classes) = compact_labels(labels);
    let edges = relative_neighborhood_edges_with_tree(layout, tree)?;
    Ok(dir_cpt_from_undirected_edges(
        n,
        &comp,
        &class_sizes,
        k_classes,
        &edges,
    ))
}

/// Computes Aupetit & Sedlmair (2016) `SepMe` Gabriel Graph (`GAB DIR CPT`) in `[0.0, 1.0]`.
pub fn sepme_gab<S: Scalar>(layout: &Layout2d<S>, labels: &[u32]) -> CoreResult<f64> {
    validate_layout_labels(layout, labels, "sepme_gab")?;
    if layout.is_empty() {
        return Ok(0.0);
    }
    if layout.len() == 1 {
        return Ok(1.0);
    }
    let tree = build_kdtree_f64(layout)?;
    sepme_gab_with_tree(layout, labels, &tree)
}

#[doc(hidden)]
pub fn sepme_gab_with_tree<S: Scalar>(
    layout: &Layout2d<S>,
    labels: &[u32],
    tree: &StaticKdTree<2, f64>,
) -> CoreResult<f64> {
    validate_layout_labels(layout, labels, "sepme_gab_with_tree")?;
    validate_tree_layout(tree, layout, "sepme_gab_with_tree")?;
    let n = layout.len();
    if n == 0 {
        return Ok(0.0);
    }
    if n == 1 {
        return Ok(1.0);
    }
    let (comp, class_sizes, k_classes) = compact_labels(labels);
    let edges = gabriel_graph_edges_with_tree(layout, tree)?;
    Ok(dir_cpt_from_undirected_edges(
        n,
        &comp,
        &class_sizes,
        k_classes,
        &edges,
    ))
}

#[inline]
fn is_gong_directed_edge(
    i: usize,
    j: usize,
    gamma: f64,
    pts: &[[f64; 2]],
    tree: &StaticKdTree<2, f64>,
    witness_buf: &mut Vec<(usize, f64)>,
) -> CoreResult<bool> {
    let dx = pts[j][0] - pts[i][0];
    let dy = pts[j][1] - pts[i][1];
    let d_ij_sq = dx * dx + dy * dy;
    if d_ij_sq <= 0.0 {
        return Ok(false);
    }
    let one_minus_g = 1.0 - gamma;
    let m = [
        gamma * pts[j][0] + one_minus_g * pts[i][0],
        gamma * pts[j][1] + one_minus_g * pts[i][1],
    ];
    let r_sq = one_minus_g * one_minus_g * d_ij_sq;
    let query_r_sq = r_sq * (1.0 + 1e-11);
    tree.within_radius_sq_into(&m, query_r_sq, witness_buf)?;

    let block_thresh = r_sq * (1.0 + 1e-12);
    for &(q, d_mq_sq) in witness_buf.iter() {
        if q == i || q == j {
            continue;
        }
        if d_mq_sq <= block_thresh {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Computes Aupetit & Sedlmair (2016) `SepMe` $\gamma$-Observable Neighbor Graph (`GONG DIR CPT`,
/// default `gamma = 0.35`) in `[0.0, 1.0]`.
///
/// For `0.0 <= gamma <= 0.5`, prunes candidates via `gabriel_graph_edges_with_tree` since
/// $\mathrm{GONG}_\gamma \subseteq \mathrm{GG}$ holds strictly on $[0, 0.5]$.
pub fn sepme_gong<S: Scalar>(layout: &Layout2d<S>, labels: &[u32], gamma: f64) -> CoreResult<f64> {
    if !gamma.is_finite() {
        return Err(CoreError::non_finite("sepme_gong::gamma"));
    }
    if !(0.0..=1.0).contains(&gamma) {
        return Err(CoreError::invalid_input(
            "sepme_gong requires gamma in [0.0, 1.0]",
        ));
    }
    validate_layout_labels(layout, labels, "sepme_gong")?;
    if layout.is_empty() {
        return Ok(0.0);
    }
    if layout.len() == 1 {
        return Ok(1.0);
    }
    let tree = build_kdtree_f64(layout)?;
    sepme_gong_with_tree(layout, labels, gamma, &tree)
}

#[doc(hidden)]
pub fn sepme_gong_with_tree<S: Scalar>(
    layout: &Layout2d<S>,
    labels: &[u32],
    gamma: f64,
    tree: &StaticKdTree<2, f64>,
) -> CoreResult<f64> {
    if !gamma.is_finite() {
        return Err(CoreError::non_finite("sepme_gong::gamma"));
    }
    if !(0.0..=1.0).contains(&gamma) {
        return Err(CoreError::invalid_input(
            "sepme_gong requires gamma in [0.0, 1.0]",
        ));
    }
    validate_layout_labels(layout, labels, "sepme_gong_with_tree")?;
    validate_tree_layout(tree, layout, "sepme_gong_with_tree")?;

    let n = layout.len();
    if n == 0 {
        return Ok(0.0);
    }
    if n == 1 {
        return Ok(1.0);
    }

    let (comp, class_sizes, k_classes) = compact_labels(labels);
    let xs = layout.x();
    let ys = layout.y();
    let pts: Vec<[f64; 2]> = (0..n).map(|i| [xs[i].to_f64(), ys[i].to_f64()]).collect();

    let mut deg_out = vec![0usize; n];
    let mut same_out = vec![0usize; n];
    let mut witness_buf = Vec::new();

    if gamma <= 0.5 {
        let gg_edges = gabriel_graph_edges_with_tree(layout, tree)?;
        for (u, v) in gg_edges {
            let ui = u as usize;
            let vi = v as usize;
            let same_class = comp[ui] == comp[vi];
            if is_gong_directed_edge(ui, vi, gamma, &pts, tree, &mut witness_buf)? {
                deg_out[ui] += 1;
                if same_class {
                    same_out[ui] += 1;
                }
            }
            if is_gong_directed_edge(vi, ui, gamma, &pts, tree, &mut witness_buf)? {
                deg_out[vi] += 1;
                if same_class {
                    same_out[vi] += 1;
                }
            }
        }
    } else {
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                if is_gong_directed_edge(i, j, gamma, &pts, tree, &mut witness_buf)? {
                    deg_out[i] += 1;
                    if comp[i] == comp[j] {
                        same_out[i] += 1;
                    }
                }
            }
        }
    }

    Ok(dir_cpt_from_degree_counts(
        &comp,
        &class_sizes,
        k_classes,
        &deg_out,
        &same_out,
    ))
}

/// Computes Paulovich et al. (2008) / Espadoto et al. (2021) Neighborhood Hit (`M_NH`, default
/// `k = 7`) in `[0.0, 1.0]`, micro-averaged over all `N` vertices (`N < 2` returns `0.0`).
pub fn neighborhood_hit<S: Scalar>(
    layout: &Layout2d<S>,
    labels: &[u32],
    k: usize,
) -> CoreResult<f64> {
    if k == 0 {
        return Err(CoreError::invalid_input("neighborhood_hit requires k >= 1"));
    }
    validate_layout_labels(layout, labels, "neighborhood_hit")?;
    if layout.len() < 2 {
        return Ok(0.0);
    }
    let tree = build_kdtree_f64(layout)?;
    neighborhood_hit_with_tree(layout, labels, k, &tree)
}

#[doc(hidden)]
pub fn neighborhood_hit_with_tree<S: Scalar>(
    layout: &Layout2d<S>,
    labels: &[u32],
    k: usize,
    tree: &StaticKdTree<2, f64>,
) -> CoreResult<f64> {
    if k == 0 {
        return Err(CoreError::invalid_input("neighborhood_hit requires k >= 1"));
    }
    validate_layout_labels(layout, labels, "neighborhood_hit_with_tree")?;
    validate_tree_layout(tree, layout, "neighborhood_hit_with_tree")?;

    let n = layout.len();
    if n < 2 {
        return Ok(0.0);
    }
    let k_eff = k.min(n - 1);
    let mut scratch = KdQueryScratch::with_capacity(k_eff);
    let mut nbrs = Vec::with_capacity(k_eff);
    let mut total_same = 0usize;

    for i in 0..n {
        tree.k_nearest_excluding_self_with_scratch(i, k_eff, &mut scratch, &mut nbrs)?;
        let li = labels[i];
        for &(j, _) in &nbrs {
            if labels[j] == li {
                total_same += 1;
            }
        }
    }

    Ok(total_same as f64 / (k_eff * n) as f64)
}
