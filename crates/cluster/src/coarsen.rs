//! Graph coarsening and multilevel layout uncoarsening/projection primitives.

use egraph_core::{CoreError, CoreResult, CsrGraph, DeterministicRng, Layout2d, Scalar};

/// Result of contracting vertex clusters via [`coarsen_graph`].
#[derive(Debug, Clone, PartialEq)]
pub struct CoarsenedGraph {
    /// Weighted undirected super-graph over `0..num_clusters`.
    pub graph: CsrGraph<f64>,
    /// Compacted cluster ID in `0..num_clusters` for each original vertex.
    pub node_to_coarse: Vec<u32>,
    /// Number of original vertices belonging to each coarse cluster.
    pub coarse_sizes: Vec<u32>,
    /// Number of distinct compacted clusters.
    pub num_clusters: usize,
}

/// Compacts arbitrary `u32` labels into canonical `0..num_clusters` in first-appearance order
/// without hash maps.
fn compact_cluster_labels(labels: &[u32]) -> (Vec<u32>, Vec<u32>, usize) {
    let n = labels.len();
    if n == 0 {
        return (Vec::new(), Vec::new(), 0);
    }

    let max_label = labels.iter().copied().max().unwrap_or(0) as usize;
    let mut node_to_coarse = vec![0u32; n];

    // Use a flat lookup array when labels are bounded; otherwise sort (label, node) pairs.
    let fast_limit = n.max(64).saturating_mul(4).min(1_000_000);
    if max_label < fast_limit {
        let mut remap = vec![u32::MAX; max_label + 1];
        let mut next_id = 0u32;
        for (i, &l) in labels.iter().enumerate() {
            let idx = l as usize;
            if remap[idx] == u32::MAX {
                remap[idx] = next_id;
                next_id += 1;
            }
            node_to_coarse[i] = remap[idx];
        }
        let num_clusters = next_id as usize;
        let mut coarse_sizes = vec![0u32; num_clusters];
        for &c in &node_to_coarse {
            coarse_sizes[c as usize] += 1;
        }
        return (node_to_coarse, coarse_sizes, num_clusters);
    }

    let mut pairs: Vec<(u32, u32)> = labels
        .iter()
        .enumerate()
        .map(|(i, &l)| (l, i as u32))
        .collect();
    pairs.sort_unstable();

    let mut groups: Vec<(u32, usize, usize)> = Vec::new();
    let mut start = 0usize;
    while start < n {
        let key = pairs[start].0;
        let mut end = start + 1;
        while end < n && pairs[end].0 == key {
            end += 1;
        }
        groups.push((pairs[start].1, start, end));
        start = end;
    }

    // Order clusters by their smallest (first-appearing) original vertex index.
    groups.sort_unstable_by_key(|&(first_node, _, _)| first_node);

    let num_clusters = groups.len();
    let mut coarse_sizes = Vec::with_capacity(num_clusters);
    for (cid, &(_, g_start, g_end)) in groups.iter().enumerate() {
        let cid_u32 = cid as u32;
        coarse_sizes.push((g_end - g_start) as u32);
        for &(_, orig_node) in &pairs[g_start..g_end] {
            node_to_coarse[orig_node as usize] = cid_u32;
        }
    }
    (node_to_coarse, coarse_sizes, num_clusters)
}

/// Coarsens an undirected [`CsrGraph`] by contracting vertices sharing a cluster label.
///
/// Arbitrary `labels` values are canonically compacted to `0..num_clusters` in ascending
/// order of first appearance. Parallel inter-cluster edges are summed into a single
/// weighted undirected edge in `CoarsenedGraph::graph`, and intra-cluster edges are omitted.
pub fn coarsen_graph<W: Scalar>(graph: &CsrGraph<W>, labels: &[u32]) -> CoreResult<CoarsenedGraph> {
    if graph.is_directed() {
        return Err(CoreError::invalid_input(
            "coarsen_graph requires an undirected graph",
        ));
    }
    let n = graph.num_nodes();
    if labels.len() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            labels.len(),
            "coarsen_graph::labels",
        ));
    }
    if n == 0 {
        let empty = CsrGraph::from_csr(&[0], &[], None, false, false)?;
        return Ok(CoarsenedGraph {
            graph: empty,
            node_to_coarse: Vec::new(),
            coarse_sizes: Vec::new(),
            num_clusters: 0,
        });
    }

    let weights_opt = graph.weights();
    if let Some(ws) = weights_opt {
        for &w in ws {
            let wf = w.to_f64();
            if !wf.is_finite() {
                return Err(CoreError::non_finite("coarsen_graph::graph::weights"));
            }
            if wf < 0.0 {
                return Err(CoreError::invalid_input(
                    "coarsen_graph edge weights must be non-negative",
                ));
            }
        }
    }

    let (node_to_coarse, coarse_sizes, num_clusters) = compact_cluster_labels(labels);

    let mut inter_edges: Vec<(u32, u32, f64)> = Vec::new();
    for u in 0..n {
        let cu = node_to_coarse[u];
        let start = graph.indptr()[u] as usize;
        let end = graph.indptr()[u + 1] as usize;
        for pos in start..end {
            let v = graph.indices()[pos] as usize;
            // Visit each undirected edge once in canonical endpoint order.
            if v <= u {
                continue;
            }
            let cv = node_to_coarse[v];
            if cu == cv {
                continue;
            }
            let w = match weights_opt {
                Some(ws) => ws[pos].to_f64(),
                None => 1.0,
            };
            let (a, b) = if cu < cv { (cu, cv) } else { (cv, cu) };
            inter_edges.push((a, b, w));
        }
    }

    inter_edges.sort_unstable_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)));

    let mut coarse_pairs: Vec<(u32, u32)> = Vec::new();
    let mut coarse_weights: Vec<f64> = Vec::new();
    for &(a, b, w) in &inter_edges {
        if let Some(&last_pair) = coarse_pairs.last()
            && last_pair == (a, b)
        {
            if let Some(last_w) = coarse_weights.last_mut() {
                *last_w += w;
            }
        } else {
            coarse_pairs.push((a, b));
            coarse_weights.push(w);
        }
    }

    let coarse_graph = CsrGraph::from_edges(
        num_clusters,
        &coarse_pairs,
        Some(&coarse_weights),
        false,
        false,
    )?;

    Ok(CoarsenedGraph {
        graph: coarse_graph,
        node_to_coarse,
        coarse_sizes,
        num_clusters,
    })
}

/// Projects a 2D layout of coarse super-nodes onto the fine vertices specified by
/// `node_to_coarse`, adding deterministic disk offsets within `spread_radius` for
/// multi-member clusters to avoid exact coordinate overlap.
pub fn project_clustered_layout<S: Scalar>(
    coarse_layout: &Layout2d<S>,
    node_to_coarse: &[u32],
    spread_radius: S,
    seed: u64,
) -> CoreResult<Layout2d<S>> {
    let r = spread_radius.to_f64();
    if !r.is_finite() {
        return Err(CoreError::non_finite(
            "project_clustered_layout::spread_radius",
        ));
    }
    if r < 0.0 {
        return Err(CoreError::invalid_input(
            "spread_radius must be non-negative",
        ));
    }
    coarse_layout.validate_finite()?;

    let n = node_to_coarse.len();
    if n == 0 {
        return Ok(Layout2d::zeros(0));
    }

    let num_clusters = coarse_layout.len();
    let mut cluster_sizes = vec![0usize; num_clusters];
    for &c in node_to_coarse {
        let cid = c as usize;
        if cid >= num_clusters {
            return Err(CoreError::index_out_of_bounds(
                cid,
                num_clusters,
                "project_clustered_layout::node_to_coarse",
            ));
        }
        cluster_sizes[cid] += 1;
    }

    let mut rng = DeterministicRng::from_seed_u64(seed);
    let cx = coarse_layout.x();
    let cy = coarse_layout.y();
    let mut fine_x = Vec::with_capacity(n);
    let mut fine_y = Vec::with_capacity(n);

    for &c in node_to_coarse {
        let cid = c as usize;
        let base_x = cx[cid].to_f64();
        let base_y = cy[cid].to_f64();
        if r > 0.0 && cluster_sizes[cid] > 1 {
            let angle = std::f64::consts::TAU * rng.next_f64();
            let rad = r * rng.next_f64().sqrt();
            let (sin_a, cos_a) = angle.sin_cos();
            fine_x.push(S::from_f64(base_x + rad * cos_a));
            fine_y.push(S::from_f64(base_y + rad * sin_a));
        } else {
            fine_x.push(S::from_f64(base_x));
            fine_y.push(S::from_f64(base_y));
        }
    }

    Layout2d::from_xy(fine_x, fine_y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coarsen_and_project_roundtrip() {
        // Two triangles {0, 1, 2} and {3, 4, 5} linked by two edges (0, 3) and (2, 4).
        let g = CsrGraph::<f64>::from_edges(
            6,
            &[
                (0, 1),
                (1, 2),
                (0, 2),
                (3, 4),
                (4, 5),
                (3, 5),
                (0, 3),
                (2, 4),
            ],
            None,
            false,
            false,
        )
        .unwrap();

        // Non-contiguous large labels verify canonical compaction without HashMap.
        let labels = [100, 100, 100, u32::MAX - 5, u32::MAX - 5, u32::MAX - 5];
        let coarse = coarsen_graph(&g, &labels).unwrap();
        assert_eq!(coarse.num_clusters, 2);
        assert_eq!(coarse.node_to_coarse, vec![0, 0, 0, 1, 1, 1]);
        assert_eq!(coarse.coarse_sizes, vec![3, 3]);
        assert_eq!(coarse.graph.num_nodes(), 2);
        assert_eq!(coarse.graph.num_edges(), 1);
        assert_eq!(coarse.graph.neighbors(0), &[1]);
        assert_eq!(coarse.graph.neighbor_weights(0), Some(&[2.0][..]));

        let coarse_layout =
            Layout2d::from_xy(vec![-5.0_f64, 5.0_f64], vec![1.0_f64, -1.0_f64]).unwrap();
        let fine1 =
            project_clustered_layout(&coarse_layout, &coarse.node_to_coarse, 0.5, 42).unwrap();
        let fine2 =
            project_clustered_layout(&coarse_layout, &coarse.node_to_coarse, 0.5, 42).unwrap();
        assert_eq!(fine1, fine2);
        assert_eq!(fine1.len(), 6);

        for (i, &cid_u32) in coarse.node_to_coarse.iter().enumerate() {
            let cid = cid_u32 as usize;
            let dx = fine1.x()[i] - coarse_layout.x()[cid];
            let dy = fine1.y()[i] - coarse_layout.y()[cid];
            assert!(dx.hypot(dy) <= 0.5 + 1e-12);
        }

        // Zero spread radius recovers exact cluster centers.
        let exact =
            project_clustered_layout(&coarse_layout, &coarse.node_to_coarse, 0.0, 42).unwrap();
        assert_eq!(exact.x(), &[-5.0, -5.0, -5.0, 5.0, 5.0, 5.0]);
        assert_eq!(exact.y(), &[1.0, 1.0, 1.0, -1.0, -1.0, -1.0]);

        // Weighted f32 coarsening + singleton cluster preservation in f32 projection.
        let wg = CsrGraph::from_edges(
            4,
            &[(0, 1), (0, 2), (1, 2), (2, 3)],
            Some(&[2.0_f32, 3.0, 4.0, 1.5]),
            false,
            false,
        )
        .unwrap();
        let wc = coarsen_graph(&wg, &[7, 7, 9, 7]).unwrap();
        assert_eq!(wc.num_clusters, 2);
        assert_eq!(wc.node_to_coarse, vec![0, 0, 1, 0]);
        assert_eq!(wc.coarse_sizes, vec![3, 1]);
        assert_eq!(wc.graph.neighbor_weights(0), Some(&[8.5_f64][..]));

        let cl_f32 = Layout2d::from_xy(vec![1.0_f32, 4.0_f32], vec![2.0_f32, -3.0_f32]).unwrap();
        let fine_f32 = project_clustered_layout(&cl_f32, &wc.node_to_coarse, 0.25_f32, 99).unwrap();
        assert!(fine_f32.is_all_finite());
        // Singleton cluster (node 2 -> coarse 1) sits directly at coarse center.
        assert_eq!(fine_f32.x()[2], 4.0_f32);
        assert_eq!(fine_f32.y()[2], -3.0_f32);
    }

    #[test]
    fn test_coarsen_and_project_errors_and_empty() {
        let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        let ce = coarsen_graph(&empty, &[]).unwrap();
        assert_eq!(ce.num_clusters, 0);
        assert_eq!(ce.graph.num_nodes(), 0);
        let pe = project_clustered_layout(&Layout2d::<f32>::zeros(0), &[], 0.1_f32, 1).unwrap();
        assert!(pe.is_empty());

        let directed = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, true, false).unwrap();
        assert!(matches!(
            coarsen_graph(&directed, &[0, 1]),
            Err(CoreError::InvalidInput(_))
        ));

        let g = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
        assert!(matches!(
            coarsen_graph(&g, &[0]),
            Err(CoreError::DimensionMismatch { .. })
        ));

        let cl = Layout2d::from_xy(vec![0.0_f64, 1.0], vec![0.0, 1.0]).unwrap();
        assert!(matches!(
            project_clustered_layout(&cl, &[0, 2], 0.5, 42),
            Err(CoreError::IndexOutOfBounds { .. })
        ));
        assert!(matches!(
            project_clustered_layout(&cl, &[0, 1], -0.1, 42),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            project_clustered_layout(&cl, &[0, 1], f64::NAN, 42),
            Err(CoreError::NonFiniteValue { .. })
        ));
        let bad_cl = Layout2d::from_xy(vec![f64::NAN, 1.0], vec![0.0, 1.0]).unwrap();
        assert!(matches!(
            project_clustered_layout(&bad_cl, &[0, 1], 0.5, 42),
            Err(CoreError::NonFiniteValue { .. })
        ));
    }
}
