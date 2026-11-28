use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, RiemannianLayout, Scalar, VisitStamp,
};

pub(crate) const MAX_PACKED_NODES: usize = 1usize << 31;
pub(crate) const BLOCK_SIZE: usize = 16_384;
pub(crate) const BLOCK_SHUFFLE_THRESHOLD: usize = 65_536;

#[derive(Debug, Clone)]
pub(crate) struct AliasTable {
    prob: Vec<f64>,
    alias: Vec<u32>,
}

impl AliasTable {
    pub(crate) fn from_weights(weights: &[f64]) -> Option<Self> {
        let k = weights.len();
        if k == 0 {
            return None;
        }
        let mut total = 0.0_f64;
        for &w in weights {
            if w.is_finite() && w > 0.0 {
                total += w;
            }
        }
        if !total.is_finite() || total <= 0.0 {
            return None;
        }
        let scale = (k as f64) / total;
        let mut scaled = vec![0.0_f64; k];
        let mut small = Vec::with_capacity(k);
        let mut large = Vec::with_capacity(k);
        for (i, &w) in weights.iter().enumerate() {
            let p = if w.is_finite() && w > 0.0 {
                w * scale
            } else {
                0.0
            };
            scaled[i] = p;
            if p < 1.0 {
                small.push(i as u32);
            } else {
                large.push(i as u32);
            }
        }
        let mut prob = vec![1.0_f64; k];
        let mut alias: Vec<u32> = (0..k as u32).collect();
        while let (Some(s_idx), Some(l_idx)) = (small.pop(), large.pop()) {
            let s = s_idx as usize;
            let l = l_idx as usize;
            prob[s] = scaled[s].clamp(0.0, 1.0);
            alias[s] = l_idx;
            scaled[l] = ((scaled[l] + scaled[s]) - 1.0).max(0.0);
            if scaled[l] < 1.0 {
                small.push(l_idx);
            } else {
                large.push(l_idx);
            }
        }
        while let Some(l_idx) = large.pop() {
            prob[l_idx as usize] = 1.0;
        }
        while let Some(s_idx) = small.pop() {
            prob[s_idx as usize] = 1.0;
        }
        Some(Self { prob, alias })
    }

    #[inline]
    pub(crate) fn sample(&self, rng: &mut DeterministicRng) -> usize {
        let k = self.prob.len();
        let col = rng.gen_range_usize(k);
        let coin = rng.next_f64();
        if coin < self.prob[col] {
            col
        } else {
            self.alias[col] as usize
        }
    }
}

pub(crate) fn validate_graph_weights<W: Scalar>(
    graph: &CsrGraph<W>,
    ctx: &'static str,
) -> CoreResult<()> {
    if let Some(ws) = graph.weights() {
        for &w in ws {
            if !w.is_finite() {
                return Err(CoreError::non_finite(ctx));
            }
            if w.to_f64() < 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "{ctx}: edge weights must be non-negative"
                )));
            }
        }
    }
    Ok(())
}

/// Validates target length, non-zero tangent dimension, locked dimensionality, and finiteness
/// without mutating solver or target state.
pub(crate) fn validate_target_riemannian<L: RiemannianLayout>(
    num_nodes: usize,
    locked_tangent_dim: Option<usize>,
    target: &L,
    ctx: &'static str,
) -> CoreResult<usize> {
    if target.len() != num_nodes {
        return Err(CoreError::dimension_mismatch(num_nodes, target.len(), ctx));
    }
    let dim = target.tangent_dim();
    if dim == 0 {
        return Err(CoreError::invalid_input(format!(
            "{ctx}: tangent_dim must be >= 1"
        )));
    }
    if let Some(expected) = locked_tangent_dim
        && expected != dim
    {
        return Err(CoreError::dimension_mismatch(expected, dim, ctx));
    }
    target.validate_finite()?;
    Ok(dim)
}

#[inline]
pub(crate) fn is_collapsed_riemannian<L: RiemannianLayout>(
    layout: &L,
    tol: f64,
) -> CoreResult<bool> {
    let n = layout.len();
    if n <= 1 {
        return Ok(false);
    }
    for i in 1..n {
        let d = layout.distance(0, i);
        if !d.is_finite() {
            return Err(CoreError::numerical_breakdown(
                "non-finite pairwise distance during collapsed-layout check",
            ));
        }
        if d >= tol {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(crate) fn scatter_collapsed_riemannian_if_needed<L: RiemannianLayout>(
    layout: &mut L,
    span: f64,
    rng: &mut DeterministicRng,
    delta_scratch: &mut Vec<f64>,
) -> CoreResult<()> {
    let n = layout.len();
    if n <= 1 || !is_collapsed_riemannian(layout, 1e-12)? {
        return Ok(());
    }
    let dim = layout.tangent_dim();
    delta_scratch.resize(dim, 0.0);
    for i in 0..n {
        for slot in delta_scratch.iter_mut() {
            *slot = span * (rng.next_f64() - 0.5);
        }
        layout.exp_map(i, delta_scratch)?;
    }
    Ok(())
}

pub(crate) fn init_pair_blocking<P>(pairs: &mut [P], rng: &mut DeterministicRng) -> Vec<u32> {
    if pairs.len() > BLOCK_SHUFFLE_THRESHOLD {
        rng.shuffle(pairs);
        let num_blocks = pairs.len().div_ceil(BLOCK_SIZE);
        (0..num_blocks as u32).collect()
    } else {
        Vec::new()
    }
}

/// Builds canonical sorted simple symmetric unweighted CSR `(indptr, indices)` from sorted
/// deduplicated upper-triangular pairs `(u1[k], v1[k])` (`u1[k] < v1[k]`) in $O(N + |E|)$.
pub(crate) fn build_symmetric_csr_from_sorted_pairs(
    num_nodes: usize,
    u1: &[u32],
    v1: &[u32],
) -> (Vec<u32>, Vec<u32>) {
    let mut deg = vec![0u32; num_nodes];
    for (&u, &v) in u1.iter().zip(v1.iter()) {
        deg[u as usize] += 1;
        deg[v as usize] += 1;
    }
    let mut indptr = Vec::with_capacity(num_nodes + 1);
    indptr.push(0u32);
    let mut acc = 0u32;
    for &d in &deg {
        acc += d;
        indptr.push(acc);
    }
    let mut cursor = indptr[..num_nodes].to_vec();
    let mut indices = vec![0u32; acc as usize];
    for (&u, &v) in u1.iter().zip(v1.iter()) {
        let pu = cursor[u as usize] as usize;
        indices[pu] = v;
        cursor[u as usize] += 1;

        let pv = cursor[v as usize] as usize;
        indices[pv] = u;
        cursor[v as usize] += 1;
    }
    for u in 0..num_nodes {
        let row = &mut indices[indptr[u] as usize..indptr[u + 1] as usize];
        if !row.windows(2).all(|w| w[0] < w[1]) {
            row.sort_unstable();
        }
    }
    (indptr, indices)
}

/// Extracts canonical deduplicated undirected 1-hop edges `(u, v, weight)` (`u < v`) from `graph`.
pub(crate) fn extract_canonical_undirected_edges<W: Scalar>(
    graph: &CsrGraph<W>,
) -> Vec<(u32, u32, f64)> {
    let n = graph.num_nodes();
    if n <= 1 || graph.nnz() == 0 {
        return Vec::new();
    }
    let weights_opt = graph.weights();
    if !graph.is_directed() {
        let mut edges = Vec::with_capacity(graph.num_edges());
        let mut is_sorted = true;
        for u in 0..n {
            let u32_u = u as u32;
            let start = graph.indptr()[u] as usize;
            let end = graph.indptr()[u + 1] as usize;
            let mut prev_v: Option<u32> = None;
            for pos in start..end {
                let v = graph.indices()[pos];
                if let Some(pv) = prev_v
                    && v <= pv
                {
                    is_sorted = false;
                }
                prev_v = Some(v);
                if u32_u < v {
                    let w = weights_opt.map_or(1.0, |ws| ws[pos].to_f64());
                    edges.push((u32_u, v, w));
                }
            }
        }
        if !is_sorted {
            edges.sort_unstable_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)));
        }
        return edges;
    }

    let mut raw = Vec::with_capacity(graph.nnz());
    for u in 0..n {
        let u32_u = u as u32;
        let start = graph.indptr()[u] as usize;
        let end = graph.indptr()[u + 1] as usize;
        for pos in start..end {
            let v = graph.indices()[pos];
            if u32_u == v {
                continue;
            }
            let (a, b) = if u32_u < v { (u32_u, v) } else { (v, u32_u) };
            let w = weights_opt.map_or(1.0, |ws| ws[pos].to_f64());
            raw.push((a, b, w));
        }
    }
    raw.sort_unstable_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)));
    let mut out: Vec<(u32, u32, f64)> = Vec::with_capacity(raw.len());
    for (a, b, w) in raw {
        if let Some(last) = out.last_mut()
            && last.0 == a
            && last.1 == b
        {
            last.2 = last.2.min(w);
        } else {
            out.push((a, b, w));
        }
    }
    out
}

pub(crate) type CanonicalPositiveGraph = (CsrGraph<f64>, Vec<(u32, u32, f64)>);

pub(crate) fn build_positive_canonical_csr<W: Scalar>(
    graph: &CsrGraph<W>,
) -> CoreResult<CanonicalPositiveGraph> {
    let n = graph.num_nodes();
    let raw_edges = extract_canonical_undirected_edges(graph);
    let weighted = graph.is_weighted();
    let mut positive_edges = Vec::with_capacity(raw_edges.len());
    for (u, v, w) in raw_edges {
        let w_pos = if weighted { w } else { 1.0 };
        if w_pos.is_finite() && w_pos > 0.0 {
            positive_edges.push((u, v, w_pos));
        }
    }
    let mut pair_uv = Vec::with_capacity(positive_edges.len());
    let mut pair_w = Vec::with_capacity(positive_edges.len());
    for &(u, v, w) in &positive_edges {
        pair_uv.push((u, v));
        pair_w.push(w);
    }
    let weights_opt = if weighted {
        Some(pair_w.as_slice())
    } else {
        None
    };
    let sym_graph = CsrGraph::<f64>::from_edges(n, &pair_uv, weights_opt, false, false)?;
    Ok((sym_graph, positive_edges))
}

#[inline]
pub(crate) fn has_common_neighbor_non_hub(
    n1_u: &[u32],
    n1_v: &[u32],
    stamp: &VisitStamp,
    adj_indptr: &[u32],
    non_hub_thresh: usize,
) -> bool {
    if n1_u.is_empty() || n1_v.is_empty() {
        return false;
    }
    if n1_v.len() <= n1_u.len() || n1_v.len() <= 64 {
        return n1_v.iter().any(|&x| {
            let xi = x as usize;
            stamp.is_visited(xi)
                && ((adj_indptr[xi + 1] - adj_indptr[xi]) as usize) <= non_hub_thresh
        });
    }
    n1_u.iter().any(|&x| {
        let xi = x as usize;
        ((adj_indptr[xi + 1] - adj_indptr[xi]) as usize) <= non_hub_thresh
            && n1_v.binary_search(&x).is_ok()
    })
}
