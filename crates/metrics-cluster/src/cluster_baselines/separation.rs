use egraph_core::{CoreError, CoreResult, EuclideanLayout, MetricSpace};
use egraph_metrics_geometry::{
    gabriel_graph_edges, gabriel_graph_edges_with_index, relative_neighborhood_edges,
    relative_neighborhood_edges_with_index,
};
use egraph_spatial_trees::{KdQueryScratch, NearestNeighborIndex};

use super::compact_labels;
use super::space_dispatch::{
    try_build_spatial_index, validate_index_layout, validate_metric_labels,
};

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

fn k_nearest_metric_excluding_self_into<L: MetricSpace + ?Sized>(
    layout: &L,
    query: usize,
    k: usize,
    buf: &mut Vec<(usize, f64)>,
) {
    let n = layout.len();
    buf.clear();
    if k == 0 || n <= 1 {
        return;
    }
    let k_eff = k.min(n - 1);
    buf.reserve(n - 1);
    for j in 0..n {
        if j == query {
            continue;
        }
        buf.push((j, layout.distance(query, j)));
    }
    let cmp = |a: &(usize, f64), b: &(usize, f64)| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0));
    if k_eff < buf.len() {
        buf.select_nth_unstable_by(k_eff - 1, cmp);
        buf.truncate(k_eff);
    }
    buf.sort_unstable_by(cmp);
}

fn sepme_knng_with_index_inner<I: NearestNeighborIndex + ?Sized>(
    labels: &[u32],
    k: usize,
    index: &I,
) -> CoreResult<f64> {
    let n = labels.len();
    let (comp, class_sizes, k_classes) = compact_labels(labels);
    let k_eff = k.min(n - 1);
    let mut scratch = KdQueryScratch::with_capacity(k_eff);
    let mut nbrs = Vec::with_capacity(k_eff);
    let mut deg = vec![0usize; n];
    let mut same = vec![0usize; n];

    for i in 0..n {
        index.query_knn_excluding_self_with_scratch(i, k_eff, &mut scratch, &mut nbrs)?;
        deg[i] = nbrs.len();
        let ci = comp[i];
        same[i] = nbrs.iter().filter(|&&(j, _)| comp[j] == ci).count();
    }

    Ok(dir_cpt_from_degree_counts(
        &comp,
        &class_sizes,
        k_classes,
        &deg,
        &same,
    ))
}

/// Computes Aupetit & Sedlmair (2016) `SepMe` $k$-Nearest Neighbor Graph (`KNNG DIR CPT`)
/// in `[0.0, 1.0]` across any [`MetricSpace`].
///
/// Automatically accelerates [`EuclideanLayout`] inputs via [`NearestNeighborIndex`].
pub fn sepme_knng<L: MetricSpace + 'static>(
    layout: &L,
    labels: &[u32],
    k: usize,
) -> CoreResult<f64> {
    if k == 0 {
        return Err(CoreError::invalid_input("sepme_knng requires k >= 1"));
    }
    validate_metric_labels(layout, labels, "sepme_knng")?;
    let n = layout.len();
    if n == 0 {
        return Ok(0.0);
    }
    if n == 1 {
        return Ok(1.0);
    }
    if let Some(index) = try_build_spatial_index(layout)? {
        return sepme_knng_with_index_inner(labels, k, index.as_ref());
    }

    let (comp, class_sizes, k_classes) = compact_labels(labels);
    let k_eff = k.min(n - 1);
    let mut nbrs = Vec::with_capacity(k_eff);
    let mut deg = vec![0usize; n];
    let mut same = vec![0usize; n];

    for i in 0..n {
        k_nearest_metric_excluding_self_into(layout, i, k_eff, &mut nbrs);
        deg[i] = nbrs.len();
        let ci = comp[i];
        same[i] = nbrs.iter().filter(|&&(j, _)| comp[j] == ci).count();
    }

    Ok(dir_cpt_from_degree_counts(
        &comp,
        &class_sizes,
        k_classes,
        &deg,
        &same,
    ))
}

/// Computes [`sepme_knng`] reusing a prebuilt [`NearestNeighborIndex`].
pub fn sepme_knng_with_index<L: EuclideanLayout + ?Sized, I: NearestNeighborIndex + ?Sized>(
    layout: &L,
    labels: &[u32],
    k: usize,
    index: &I,
) -> CoreResult<f64> {
    if k == 0 {
        return Err(CoreError::invalid_input("sepme_knng requires k >= 1"));
    }
    validate_metric_labels(layout, labels, "sepme_knng_with_index")?;
    validate_index_layout(index, layout, "sepme_knng_with_index")?;

    let n = layout.len();
    if n == 0 {
        return Ok(0.0);
    }
    if n == 1 {
        return Ok(1.0);
    }
    sepme_knng_with_index_inner(labels, k, index)
}

/// Computes Aupetit & Sedlmair (2016) `SepMe` Relative Neighborhood Graph (`RNG DIR CPT`)
/// in `[0.0, 1.0]` across any [`MetricSpace`].
pub fn sepme_rng<L: MetricSpace + 'static>(layout: &L, labels: &[u32]) -> CoreResult<f64> {
    validate_metric_labels(layout, labels, "sepme_rng")?;
    let n = layout.len();
    if n == 0 {
        return Ok(0.0);
    }
    if n == 1 {
        return Ok(1.0);
    }
    let (comp, class_sizes, k_classes) = compact_labels(labels);
    let edges = relative_neighborhood_edges(layout)?;
    Ok(dir_cpt_from_undirected_edges(
        n,
        &comp,
        &class_sizes,
        k_classes,
        &edges,
    ))
}

/// Computes [`sepme_rng`] reusing a prebuilt [`NearestNeighborIndex`].
pub fn sepme_rng_with_index<L: EuclideanLayout + ?Sized, I: NearestNeighborIndex + ?Sized>(
    layout: &L,
    labels: &[u32],
    index: &I,
) -> CoreResult<f64> {
    validate_metric_labels(layout, labels, "sepme_rng_with_index")?;
    validate_index_layout(index, layout, "sepme_rng_with_index")?;
    let n = layout.len();
    if n == 0 {
        return Ok(0.0);
    }
    if n == 1 {
        return Ok(1.0);
    }
    let (comp, class_sizes, k_classes) = compact_labels(labels);
    let edges = relative_neighborhood_edges_with_index(layout, index)?;
    Ok(dir_cpt_from_undirected_edges(
        n,
        &comp,
        &class_sizes,
        k_classes,
        &edges,
    ))
}

/// Computes Aupetit & Sedlmair (2016) `SepMe` Gabriel Graph (`GAB DIR CPT`) in `[0.0, 1.0]`
/// across any [`MetricSpace`].
pub fn sepme_gab<L: MetricSpace + 'static>(layout: &L, labels: &[u32]) -> CoreResult<f64> {
    validate_metric_labels(layout, labels, "sepme_gab")?;
    let n = layout.len();
    if n == 0 {
        return Ok(0.0);
    }
    if n == 1 {
        return Ok(1.0);
    }
    let (comp, class_sizes, k_classes) = compact_labels(labels);
    let edges = gabriel_graph_edges(layout)?;
    Ok(dir_cpt_from_undirected_edges(
        n,
        &comp,
        &class_sizes,
        k_classes,
        &edges,
    ))
}

/// Computes [`sepme_gab`] reusing a prebuilt [`NearestNeighborIndex`].
pub fn sepme_gab_with_index<L: EuclideanLayout + ?Sized, I: NearestNeighborIndex + ?Sized>(
    layout: &L,
    labels: &[u32],
    index: &I,
) -> CoreResult<f64> {
    validate_metric_labels(layout, labels, "sepme_gab_with_index")?;
    validate_index_layout(index, layout, "sepme_gab_with_index")?;
    let n = layout.len();
    if n == 0 {
        return Ok(0.0);
    }
    if n == 1 {
        return Ok(1.0);
    }
    let (comp, class_sizes, k_classes) = compact_labels(labels);
    let edges = gabriel_graph_edges_with_index(layout, index)?;
    Ok(dir_cpt_from_undirected_edges(
        n,
        &comp,
        &class_sizes,
        k_classes,
        &edges,
    ))
}

fn neighborhood_hit_with_index_inner<I: NearestNeighborIndex + ?Sized>(
    labels: &[u32],
    k: usize,
    index: &I,
) -> CoreResult<f64> {
    let n = labels.len();
    let k_eff = k.min(n - 1);
    let mut scratch = KdQueryScratch::with_capacity(k_eff);
    let mut nbrs = Vec::with_capacity(k_eff);
    let mut total_same = 0usize;

    for (i, &li) in labels.iter().enumerate() {
        index.query_knn_excluding_self_with_scratch(i, k_eff, &mut scratch, &mut nbrs)?;
        total_same += nbrs.iter().filter(|&&(j, _)| labels[j] == li).count();
    }

    Ok(total_same as f64 / (k_eff * n) as f64)
}

/// Computes Paulovich et al. (2008) / Espadoto et al. (2021) Neighborhood Hit (`M_NH`, default
/// `k = 7`) in `[0.0, 1.0]`, micro-averaged over all `N` vertices (`N < 2` returns `0.0`) across
/// any [`MetricSpace`].
///
/// Automatically accelerates [`EuclideanLayout`] inputs via [`NearestNeighborIndex`].
pub fn neighborhood_hit<L: MetricSpace + 'static>(
    layout: &L,
    labels: &[u32],
    k: usize,
) -> CoreResult<f64> {
    if k == 0 {
        return Err(CoreError::invalid_input("neighborhood_hit requires k >= 1"));
    }
    validate_metric_labels(layout, labels, "neighborhood_hit")?;
    let n = layout.len();
    if n < 2 {
        return Ok(0.0);
    }
    if let Some(index) = try_build_spatial_index(layout)? {
        return neighborhood_hit_with_index_inner(labels, k, index.as_ref());
    }

    let k_eff = k.min(n - 1);
    let mut nbrs = Vec::with_capacity(k_eff);
    let mut total_same = 0usize;

    for (i, &li) in labels.iter().enumerate() {
        k_nearest_metric_excluding_self_into(layout, i, k_eff, &mut nbrs);
        total_same += nbrs.iter().filter(|&&(j, _)| labels[j] == li).count();
    }

    Ok(total_same as f64 / (k_eff * n) as f64)
}

/// Computes [`neighborhood_hit`] reusing a prebuilt [`NearestNeighborIndex`].
pub fn neighborhood_hit_with_index<
    L: EuclideanLayout + ?Sized,
    I: NearestNeighborIndex + ?Sized,
>(
    layout: &L,
    labels: &[u32],
    k: usize,
    index: &I,
) -> CoreResult<f64> {
    if k == 0 {
        return Err(CoreError::invalid_input("neighborhood_hit requires k >= 1"));
    }
    validate_metric_labels(layout, labels, "neighborhood_hit_with_index")?;
    validate_index_layout(index, layout, "neighborhood_hit_with_index")?;

    if layout.len() < 2 {
        return Ok(0.0);
    }
    neighborhood_hit_with_index_inner(labels, k, index)
}
