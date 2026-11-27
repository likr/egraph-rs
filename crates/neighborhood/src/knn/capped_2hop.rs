//! Hub-safe capped 1+2-hop pair extraction (`Sparse12HopPairs`) and capped 1+2-hop [`KnnGraph`]
//! construction in $O(|E| + N \cdot \mathrm{cap})$ time.

use egraph_core::{CoreError, CoreResult, CsrGraph, DeterministicRng, VisitStamp};

use super::graph::{KnnGraph, KnnStats, Sparse12HopPairs};
use super::two_hop::canonical_unweighted_1hop;

/// Strategy used by [`extract_sparse_1hop_2hop_pairs_with_mode`] to sample capped 2-hop pairs.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TwoHopSamplingMode {
    /// Expands bridge neighbors in ascending `(deg(w), w)` order while unconditionally
    /// suppressing super-hubs (`deg(w) > max(64, floor(sqrt(N)))`).
    #[default]
    DegreeOrderedHubSuppressed,
    /// Uniform 2-step random walk with dead-end leaf filtering (`deg(w) > 1`), exact enumeration
    /// when `sum(deg(w) - 1) <= max_2hop_per_node`, and up to `4 * max_2hop_per_node` draws
    /// otherwise.
    TwoStepRandomWalk,
}

/// Extracts canonical undirected 1-hop (`u < v`) and hub-safe capped 2-hop (`u < v`) node pairs
/// without $O(N^2)$ pair explosion on star hubs.
///
/// For each vertex `u`, 1-hop bridge candidates `w` are scanned in ascending `(deg(w), w)` order so
/// low-degree local bridges fill the `max_2hop_per_node` budget before higher-degree intermediates.
/// Degree-1 neighbors are skipped (they only point back to `u`), and super-hubs with
/// `deg(w) > max(64, floor(sqrt(N)))` are unconditionally suppressed while marking `u` as
/// truncated (preventing barrel-mediated cross-cluster contamination). Non-hub neighbors `w` with
/// `max_2hop_per_node < deg(w) <= max(64, floor(sqrt(N)))` are deterministically subsampled via
/// `DeterministicRng::from_seed_u64(seed)`, which preserves exact vertex-permutation invariance
/// whenever all expanded bridges satisfy `deg(w) <= max_2hop_per_node` and truncation does not
/// split a tied-degree tier.
pub fn extract_sparse_1hop_2hop_pairs<W>(
    graph: &CsrGraph<W>,
    max_2hop_per_node: usize,
    seed: u64,
) -> CoreResult<Sparse12HopPairs> {
    let (pairs, _) = extract_sparse_1hop_2hop_pairs_inner(graph, max_2hop_per_node, seed)?;
    Ok(pairs)
}

/// Extracts canonical undirected 1-hop (`u < v`) and capped 2-hop (`u < v`) node pairs using the
/// specified [`TwoHopSamplingMode`] and optional `max_bridge_degree` cap on intermediate vertices.
pub fn extract_sparse_1hop_2hop_pairs_with_mode<W>(
    graph: &CsrGraph<W>,
    max_2hop_per_node: usize,
    mode: TwoHopSamplingMode,
    max_bridge_degree: Option<usize>,
    seed: u64,
) -> CoreResult<Sparse12HopPairs> {
    match mode {
        TwoHopSamplingMode::DegreeOrderedHubSuppressed => {
            let (pairs, _) = extract_sparse_1hop_2hop_pairs_degree_ordered(
                graph,
                max_2hop_per_node,
                max_bridge_degree,
                seed,
            )?;
            Ok(pairs)
        }
        TwoHopSamplingMode::TwoStepRandomWalk => extract_sparse_1hop_2hop_pairs_random_walk(
            graph,
            max_2hop_per_node,
            max_bridge_degree,
            seed,
        ),
    }
}

#[inline]
fn super_hub_degree_threshold(n_nodes: usize) -> usize {
    64usize.max(n_nodes.isqrt())
}

#[inline]
fn collect_1hop_pairs(adj1: &CsrGraph<f64>) -> (Vec<u32>, Vec<u32>) {
    let n = adj1.num_nodes();
    let mut u1 = Vec::with_capacity(adj1.num_edges());
    let mut v1 = Vec::with_capacity(adj1.num_edges());
    for u in 0..n {
        let u_u32 = u as u32;
        for &v in adj1.neighbors(u) {
            if u_u32 < v {
                u1.push(u_u32);
                v1.push(v);
            }
        }
    }
    (u1, v1)
}

#[inline]
fn finish_2hop_pairs(
    u1: Vec<u32>,
    v1: Vec<u32>,
    mut raw_2hop: Vec<(u32, u32)>,
) -> Sparse12HopPairs {
    raw_2hop.sort_unstable();
    raw_2hop.dedup();
    let mut u2 = Vec::with_capacity(raw_2hop.len());
    let mut v2 = Vec::with_capacity(raw_2hop.len());
    for (a, b) in raw_2hop {
        u2.push(a);
        v2.push(b);
    }
    Sparse12HopPairs::from_validated_parts(u1, v1, u2, v2)
}

pub(crate) fn extract_sparse_1hop_2hop_pairs_inner<W>(
    graph: &CsrGraph<W>,
    max_2hop_per_node: usize,
    seed: u64,
) -> CoreResult<(Sparse12HopPairs, usize)> {
    extract_sparse_1hop_2hop_pairs_degree_ordered(graph, max_2hop_per_node, None, seed)
}

fn extract_sparse_1hop_2hop_pairs_degree_ordered<W>(
    graph: &CsrGraph<W>,
    max_2hop_per_node: usize,
    max_bridge_degree: Option<usize>,
    seed: u64,
) -> CoreResult<(Sparse12HopPairs, usize)> {
    let adj1 = canonical_unweighted_1hop(graph)?;
    let n = adj1.num_nodes();
    if n <= 1 || adj1.nnz() == 0 {
        return Ok((Sparse12HopPairs::default(), 0));
    }

    let (u1, v1) = collect_1hop_pairs(&adj1);

    if max_2hop_per_node == 0 || n <= 2 {
        let mut n_truncated_2hop = 0usize;
        if max_2hop_per_node == 0 && n > 2 {
            for u in 0..n {
                if adj1
                    .neighbors(u)
                    .iter()
                    .any(|&w| adj1.degree(w as usize) > 1)
                {
                    n_truncated_2hop += 1;
                }
            }
        }
        return Ok((
            Sparse12HopPairs::from_validated_parts(u1, v1, Vec::new(), Vec::new()),
            n_truncated_2hop,
        ));
    }

    let hub_threshold = match max_bridge_degree {
        Some(b) => super_hub_degree_threshold(n).min(b),
        None => super_hub_degree_threshold(n),
    };
    let mut w_candidates: Vec<(usize, u32)> = Vec::new();
    let mut rng = DeterministicRng::from_seed_u64(seed);
    let mut stamp = VisitStamp::new(n);
    let mut raw_2hop: Vec<(u32, u32)> = Vec::new();
    let mut n_truncated_2hop = 0usize;

    for u in 0..n {
        let n1_u = adj1.neighbors(u);
        if n1_u.is_empty() {
            continue;
        }
        stamp.clear();
        stamp.mark(u);
        w_candidates.clear();
        let mut truncated_u = false;
        for &w in n1_u {
            stamp.mark(w as usize);
            let deg_w = adj1.degree(w as usize);
            if deg_w <= 1 {
                continue;
            }
            if deg_w > hub_threshold {
                truncated_u = true;
                continue;
            }
            w_candidates.push((deg_w, w));
        }
        w_candidates.sort_unstable();

        let u_u32 = u as u32;
        let mut collected = 0usize;
        for &(deg_w, w) in &w_candidates {
            let n1_w = adj1.neighbors(w as usize);
            if deg_w <= max_2hop_per_node {
                for &v in n1_w {
                    if stamp.visit(v as usize) {
                        raw_2hop.push(if u_u32 < v { (u_u32, v) } else { (v, u_u32) });
                        collected += 1;
                        if collected >= max_2hop_per_node {
                            truncated_u = true;
                            break;
                        }
                    }
                }
            } else {
                truncated_u = true;
                let n_draws = max_2hop_per_node.saturating_mul(2).min(deg_w);
                for _ in 0..n_draws {
                    let v = n1_w[rng.gen_range_usize(deg_w)];
                    if stamp.visit(v as usize) {
                        raw_2hop.push(if u_u32 < v { (u_u32, v) } else { (v, u_u32) });
                        collected += 1;
                        if collected >= max_2hop_per_node {
                            break;
                        }
                    }
                }
            }
            if collected >= max_2hop_per_node {
                truncated_u = true;
                break;
            }
        }
        if truncated_u {
            n_truncated_2hop += 1;
        }
    }

    Ok((finish_2hop_pairs(u1, v1, raw_2hop), n_truncated_2hop))
}

fn extract_sparse_1hop_2hop_pairs_random_walk<W>(
    graph: &CsrGraph<W>,
    max_2hop_per_node: usize,
    max_bridge_degree: Option<usize>,
    seed: u64,
) -> CoreResult<Sparse12HopPairs> {
    let adj1 = canonical_unweighted_1hop(graph)?;
    let n = adj1.num_nodes();
    if n <= 1 || adj1.nnz() == 0 {
        return Ok(Sparse12HopPairs::default());
    }

    let (u1, v1) = collect_1hop_pairs(&adj1);
    if max_2hop_per_node == 0 || n <= 2 {
        return Ok(Sparse12HopPairs::from_validated_parts(
            u1,
            v1,
            Vec::new(),
            Vec::new(),
        ));
    }

    let mut non_leaf: Vec<u32> = Vec::new();
    let mut rng = DeterministicRng::from_seed_u64(seed);
    let mut stamp = VisitStamp::new(n);
    let mut raw_2hop: Vec<(u32, u32)> = Vec::new();

    for u in 0..n {
        let n1_u = adj1.neighbors(u);
        if n1_u.is_empty() {
            continue;
        }
        stamp.clear();
        stamp.mark(u);
        non_leaf.clear();
        let mut pot = 0usize;
        for &w in n1_u {
            stamp.mark(w as usize);
            let deg_w = adj1.degree(w as usize);
            // Skip dead-end leaves (deg == 1) so star+branch hubs do not waste random-walk
            // draws bouncing straight back to u.
            if deg_w > 1 && max_bridge_degree.is_none_or(|b| deg_w <= b) {
                non_leaf.push(w);
                pot = pot.saturating_add(deg_w - 1);
            }
        }
        if non_leaf.is_empty() {
            continue;
        }

        let u_u32 = u as u32;
        if pot <= max_2hop_per_node {
            for &w in &non_leaf {
                for &v in adj1.neighbors(w as usize) {
                    if stamp.visit(v as usize) {
                        raw_2hop.push(if u_u32 < v { (u_u32, v) } else { (v, u_u32) });
                    }
                }
            }
        } else {
            let max_draws = max_2hop_per_node.saturating_mul(4);
            let mut collected = 0usize;
            for _ in 0..max_draws {
                let w = non_leaf[rng.gen_range_usize(non_leaf.len())];
                let n1_w = adj1.neighbors(w as usize);
                let v = n1_w[rng.gen_range_usize(n1_w.len())];
                if stamp.visit(v as usize) {
                    raw_2hop.push(if u_u32 < v { (u_u32, v) } else { (v, u_u32) });
                    collected += 1;
                    if collected >= max_2hop_per_node {
                        break;
                    }
                }
            }
        }
    }

    Ok(finish_2hop_pairs(u1, v1, raw_2hop))
}

/// Builds a hub-safe capped 1+2-hop CSR [`KnnGraph`] with `{1.0, 2.0}` discrete distances,
/// per-node neighborhood sizes `k2_deg`, and summary [`KnnStats`] in $O(|E| + N \cdot \mathrm{cap})$
/// time via [`extract_sparse_1hop_2hop_pairs`] and a 2-way lexicographic merge scatter.
pub fn build_capped_12hop_knn<W>(
    graph: &CsrGraph<W>,
    max_2hop_per_node: usize,
    seed: u64,
) -> CoreResult<(KnnGraph, Vec<u32>, KnnStats)> {
    let (sp, n_truncated_2hop) =
        extract_sparse_1hop_2hop_pairs_inner(graph, max_2hop_per_node, seed)?;
    let n = graph.num_nodes();
    let u1 = sp.u1();
    let v1 = sp.v1();
    let u2 = sp.u2();
    let v2 = sp.v2();
    let n_1hop = sp.num_1hop().saturating_mul(2);
    let n_2hop = sp.num_2hop().saturating_mul(2);
    let total_nnz = n_1hop.saturating_add(n_2hop);

    if n == 0 || total_nnz == 0 {
        let knn = KnnGraph::empty(n)?;
        let k2_deg = vec![0u32; n];
        let stats = KnnStats {
            n_truncated_2hop,
            n_pure_2hop_rows: n,
            ..KnnStats::default()
        };
        return Ok((knn, k2_deg, stats));
    }

    let mut deg1 = vec![0usize; n];
    let mut deg2 = vec![0usize; n];
    for (&u, &v) in u1.iter().zip(v1.iter()) {
        deg1[u as usize] += 1;
        deg1[v as usize] += 1;
    }
    for (&u, &v) in u2.iter().zip(v2.iter()) {
        deg2[u as usize] += 1;
        deg2[v as usize] += 1;
    }

    if u32::try_from(total_nnz).is_err() {
        return Err(CoreError::invalid_csr(
            "capped 1+2-hop nnz exceeds u32::MAX",
        ));
    }

    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0u32);
    let mut k2_deg = Vec::with_capacity(n);
    let mut min_row_k = usize::MAX;
    let mut max_row_k = 0usize;
    let mut running = 0usize;
    for u in 0..n {
        let row_k = deg1[u] + deg2[u];
        k2_deg.push(row_k as u32);
        min_row_k = min_row_k.min(row_k);
        max_row_k = max_row_k.max(row_k);
        running += row_k;
        let ptr = u32::try_from(running)
            .map_err(|_| CoreError::invalid_csr("capped 1+2-hop nnz exceeds u32::MAX"))?;
        indptr.push(ptr);
    }

    let mut cursor: Vec<usize> = indptr[..n].iter().map(|&p| p as usize).collect();
    let mut indices = vec![0u32; total_nnz];
    let mut distances = vec![0.0_f64; total_nnz];

    let mut i1 = 0usize;
    let mut i2 = 0usize;
    let len1 = u1.len();
    let len2 = u2.len();
    while i1 < len1 || i2 < len2 {
        let take_1hop = match (i1 < len1, i2 < len2) {
            (true, true) => (u1[i1], v1[i1]) <= (u2[i2], v2[i2]),
            (true, false) => true,
            (false, true) => false,
            (false, false) => unreachable!(),
        };
        let (u, v, dist) = if take_1hop {
            let p = (u1[i1], v1[i1], 1.0_f64);
            i1 += 1;
            p
        } else {
            let p = (u2[i2], v2[i2], 2.0_f64);
            i2 += 1;
            p
        };
        let pu = cursor[u as usize];
        indices[pu] = v;
        distances[pu] = dist;
        cursor[u as usize] += 1;

        let pv = cursor[v as usize];
        indices[pv] = u;
        distances[pv] = dist;
        cursor[v as usize] += 1;
    }

    let knn = KnnGraph::new_validated(n, indptr, indices, distances)?;
    let n_f64 = n as f64;
    let avg_deg1 = (n_1hop as f64) / n_f64;
    let avg_deg2 = (total_nnz as f64) / n_f64;
    let stats = KnnStats {
        avg_deg1,
        avg_deg2,
        avg_row_k: avg_deg2,
        min_row_k,
        max_row_k,
        nnz: total_nnz,
        n_1hop,
        n_2hop,
        n_truncated_1hop: 0,
        n_truncated_2hop,
        n_filled_residual: 0,
        n_pure_2hop_rows: n,
    };
    Ok((knn, k2_deg, stats))
}
