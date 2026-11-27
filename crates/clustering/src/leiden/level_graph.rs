use egraph_core::{CoreError, CoreResult, CsrGraph, Scalar};

use super::config::LeidenObjective;

/// Compact internal super-graph supporting weighted self-loops and super-node sizes.
#[derive(Clone)]
pub(super) struct LeidenLevelGraph {
    pub(super) num_nodes: usize,
    pub(super) indptr: Vec<u32>,
    pub(super) indices: Vec<u32>,
    pub(super) weights: Vec<f64>,
    pub(super) loop_weights: Vec<f64>,
    pub(super) degrees: Vec<f64>,
    pub(super) node_sizes: Vec<f64>,
    pub(super) total_weight_2m: f64,
}

impl LeidenLevelGraph {
    pub(super) fn from_csr<W: Scalar>(
        graph: &CsrGraph<W>,
        objective: LeidenObjective,
    ) -> CoreResult<Self> {
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
                        return Err(CoreError::non_finite("leiden::graph::weights"));
                    }
                    if wf < 0.0 {
                        return Err(CoreError::invalid_input(
                            "leiden edge weights must be non-negative",
                        ));
                    }
                    weights.push(wf);
                    sum += wf;
                }
                if !sum.is_finite() {
                    return Err(CoreError::numerical_breakdown(
                        "edge weight sum overflows f64",
                    ));
                }
                *deg_u = sum;
                total_weight_2m += sum;
                if !total_weight_2m.is_finite() {
                    return Err(CoreError::numerical_breakdown(
                        "edge weight sum overflows f64",
                    ));
                }
            }
        } else {
            for (u, deg_u) in degrees.iter_mut().enumerate() {
                let d = graph.degree(u) as f64;
                *deg_u = d;
                total_weight_2m += d;
            }
            if !total_weight_2m.is_finite() {
                return Err(CoreError::numerical_breakdown(
                    "edge weight sum overflows f64",
                ));
            }
            weights.resize(graph.nnz(), 1.0);
        }

        if total_weight_2m > 0.0
            && objective == LeidenObjective::Modularity
            && !(1.0 / total_weight_2m).is_finite()
        {
            return Err(CoreError::numerical_breakdown(
                "total edge weight underflows f64 precision",
            ));
        }

        let node_sizes = match objective {
            LeidenObjective::Modularity => degrees.clone(),
            LeidenObjective::Cpm => vec![1.0_f64; n],
        };

        Ok(Self {
            num_nodes: n,
            indptr: graph.indptr().to_vec(),
            indices: graph.indices().to_vec(),
            weights,
            loop_weights: vec![0.0_f64; n],
            degrees,
            node_sizes,
            total_weight_2m,
        })
    }

    pub(super) fn evaluate_quality(
        &self,
        comm: &[u32],
        num_comms: usize,
        objective: LeidenObjective,
        resolution: f64,
    ) -> f64 {
        let mut sigma_in = vec![0.0_f64; num_comms];
        let mut comm_size = vec![0.0_f64; num_comms];

        for u in 0..self.num_nodes {
            let cu = comm[u] as usize;
            sigma_in[cu] += self.loop_weights[u];
            comm_size[cu] += self.node_sizes[u];
            let start = self.indptr[u] as usize;
            let end = self.indptr[u + 1] as usize;
            for pos in start..end {
                let v = self.indices[pos] as usize;
                if comm[v] as usize == cu {
                    sigma_in[cu] += self.weights[pos];
                }
            }
        }

        match objective {
            LeidenObjective::Modularity => {
                if self.total_weight_2m <= 0.0 {
                    return 0.0;
                }
                let mut q = 0.0_f64;
                for c in 0..num_comms {
                    let kc = comm_size[c];
                    if kc > 0.0 || sigma_in[c] > 0.0 {
                        let frac = kc / self.total_weight_2m;
                        q += (sigma_in[c] / self.total_weight_2m) - resolution * frac * frac;
                    }
                }
                q
            }
            LeidenObjective::Cpm => {
                let mut h = 0.0_f64;
                for c in 0..num_comms {
                    let sz = comm_size[c];
                    h += 0.5 * sigma_in[c] - resolution * (sz * (sz - 1.0) * 0.5);
                }
                h
            }
        }
    }

    /// Aggregates refined communities `0..num_new` into the next-level super-graph without `HashMap`.
    pub(super) fn aggregate(&self, refined_compacted: &[u32], num_new: usize) -> Self {
        let mut new_loops = vec![0.0_f64; num_new];
        let mut new_degrees = vec![0.0_f64; num_new];
        let mut new_sizes = vec![0.0_f64; num_new];
        let mut counts = vec![0usize; num_new];

        for u in 0..self.num_nodes {
            let cu = refined_compacted[u] as usize;
            new_loops[cu] += self.loop_weights[u];
            new_degrees[cu] += self.degrees[u];
            new_sizes[cu] += self.node_sizes[u];
            let start = self.indptr[u] as usize;
            let end = self.indptr[u + 1] as usize;
            for pos in start..end {
                let v = self.indices[pos] as usize;
                let cv = refined_compacted[v] as usize;
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
            let cu = refined_compacted[u] as usize;
            let start = self.indptr[u] as usize;
            let end = self.indptr[u + 1] as usize;
            for pos in start..end {
                let v = self.indices[pos] as usize;
                let cv = refined_compacted[v];
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
            node_sizes: new_sizes,
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
