use egraph_core::{CoreError, CoreResult, CsrGraph, Scalar};

/// Compact internal super-graph representation supporting weighted self-loops across passes.
pub(super) struct WeightedLevelGraph {
    pub(super) num_nodes: usize,
    pub(super) indptr: Vec<u32>,
    pub(super) indices: Vec<u32>,
    pub(super) weights: Vec<f64>,
    pub(super) loop_weights: Vec<f64>,
    pub(super) degrees: Vec<f64>,
    pub(super) total_weight_2m: f64,
}

impl WeightedLevelGraph {
    pub(super) fn from_csr<W: Scalar>(graph: &CsrGraph<W>) -> CoreResult<Self> {
        let n = graph.num_nodes();
        let mut degrees = vec![0.0_f64; n];
        let mut weights = Vec::with_capacity(graph.nnz());
        let mut total_weight_2m = 0.0_f64;

        if let Some(ws) = graph.weights() {
            for (u, deg_u) in degrees.iter_mut().enumerate() {
                let start = graph.indptr()[u] as usize;
                let end = graph.indptr()[u + 1] as usize;
                let mut sum = 0.0_f64;
                for &w in &ws[start..end] {
                    let wf = w.to_f64();
                    if !wf.is_finite() {
                        return Err(CoreError::non_finite("louvain::graph::weights"));
                    }
                    if wf < 0.0 {
                        return Err(CoreError::invalid_input(
                            "louvain edge weights must be non-negative",
                        ));
                    }
                    weights.push(wf);
                    sum += wf;
                }
                *deg_u = sum;
                total_weight_2m += sum;
            }
        } else {
            for (u, deg_u) in degrees.iter_mut().enumerate() {
                let d = graph.degree(u) as f64;
                *deg_u = d;
                total_weight_2m += d;
            }
            weights.resize(graph.nnz(), 1.0);
        }

        Ok(Self {
            num_nodes: n,
            indptr: graph.indptr().to_vec(),
            indices: graph.indices().to_vec(),
            weights,
            loop_weights: vec![0.0_f64; n],
            degrees,
            total_weight_2m,
        })
    }

    pub(super) fn modularity(&self, comm: &[u32], sigma_tot: &[f64], resolution: f64) -> f64 {
        if self.total_weight_2m <= 0.0 {
            return 0.0;
        }
        let inv_2m = 1.0 / self.total_weight_2m;
        let mut sigma_in = vec![0.0_f64; self.num_nodes];
        for u in 0..self.num_nodes {
            let cu = comm[u] as usize;
            sigma_in[cu] += self.loop_weights[u];
            let start = self.indptr[u] as usize;
            let end = self.indptr[u + 1] as usize;
            for pos in start..end {
                let v = self.indices[pos] as usize;
                if comm[v] as usize == cu {
                    sigma_in[cu] += self.weights[pos];
                }
            }
        }

        let mut q = 0.0_f64;
        for c in 0..self.num_nodes {
            let st = sigma_tot[c];
            if st > 0.0 || sigma_in[c] > 0.0 {
                q += sigma_in[c] * inv_2m - resolution * (st * inv_2m) * (st * inv_2m);
            }
        }
        q
    }

    /// Aggregates communities `0..num_new` into the next-level super-graph without `HashMap`.
    pub(super) fn aggregate(&self, comm_compacted: &[u32], num_new: usize) -> Self {
        let mut new_loops = vec![0.0_f64; num_new];
        let mut new_degrees = vec![0.0_f64; num_new];

        // Count inter-community half-edges per super-node to allocate flat buckets.
        let mut counts = vec![0usize; num_new];
        for u in 0..self.num_nodes {
            let cu = comm_compacted[u] as usize;
            new_loops[cu] += self.loop_weights[u];
            new_degrees[cu] += self.degrees[u];
            let start = self.indptr[u] as usize;
            let end = self.indptr[u + 1] as usize;
            for pos in start..end {
                let v = self.indices[pos] as usize;
                let cv = comm_compacted[v] as usize;
                if cu == cv {
                    new_loops[cu] += self.weights[pos];
                } else {
                    counts[cu] += 1;
                }
            }
        }

        let mut row_buckets: Vec<Vec<(u32, f64)>> =
            counts.iter().map(|&c| Vec::with_capacity(c)).collect();
        for u in 0..self.num_nodes {
            let cu = comm_compacted[u] as usize;
            let start = self.indptr[u] as usize;
            let end = self.indptr[u + 1] as usize;
            for pos in start..end {
                let v = self.indices[pos] as usize;
                let cv = comm_compacted[v];
                if cu != cv as usize {
                    row_buckets[cu].push((cv, self.weights[pos]));
                }
            }
        }

        let mut new_indptr = Vec::with_capacity(num_new + 1);
        new_indptr.push(0u32);
        let mut new_indices = Vec::new();
        let mut new_weights = Vec::new();

        for bucket in &mut row_buckets {
            bucket.sort_unstable_by_key(|&(cv, _)| cv);
            let row_start = new_indices.len();
            for &(cv, w) in bucket.iter() {
                if new_indices.len() > row_start && new_indices[new_indices.len() - 1] == cv {
                    let last_idx = new_weights.len() - 1;
                    new_weights[last_idx] += w;
                } else {
                    new_indices.push(cv);
                    new_weights.push(w);
                }
            }
            new_indptr.push(new_indices.len() as u32);
        }

        Self {
            num_nodes: num_new,
            indptr: new_indptr,
            indices: new_indices,
            weights: new_weights,
            loop_weights: new_loops,
            degrees: new_degrees,
            total_weight_2m: self.total_weight_2m,
        }
    }
}

/// Renumbers labels in canonical `0..c` order of first appearance by vertex index.
pub(super) fn canonicalize_labels(labels: &mut [u32], max_id_bound: usize) -> usize {
    let mut remap = vec![u32::MAX; max_id_bound.max(1)];
    let mut next_id = 0u32;
    for label in labels.iter_mut() {
        let idx = *label as usize;
        if remap[idx] == u32::MAX {
            remap[idx] = next_id;
            next_id += 1;
        }
        *label = remap[idx];
    }
    next_id as usize
}
