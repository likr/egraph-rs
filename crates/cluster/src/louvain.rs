//! Pure-Rust two-phase multi-level Louvain community detection using flat CSR and
//! `VisitStamp` accumulators (zero `HashMap` in inner loops or aggregation).

use egraph_core::{CoreError, CoreResult, CsrGraph, DeterministicRng, Scalar, VisitStamp};

/// Configuration for [`louvain`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LouvainConfig {
    /// Maximum number of multi-level Louvain passes (local moving + aggregation).
    pub max_passes: usize,
    /// Modularity resolution parameter $\gamma \ge 0.0$ (`1.0` for standard modularity).
    pub resolution: f64,
    /// Minimum modularity gain required to commit a pass and aggregate communities.
    pub min_modularity_gain: f64,
    /// Deterministic PRNG seed controlling vertex visit order in Phase 1 sweeps.
    pub seed: u64,
}

impl Default for LouvainConfig {
    fn default() -> Self {
        Self {
            max_passes: 100,
            resolution: 1.0,
            min_modularity_gain: 1e-7,
            seed: 42,
        }
    }
}

/// Output of [`louvain`].
#[derive(Debug, Clone, PartialEq)]
pub struct LouvainResult {
    /// Canonical community assignment in `0..num_communities` for each vertex.
    pub labels: Vec<u32>,
    /// Number of distinct non-empty communities detected.
    pub num_communities: usize,
    /// Modularity $Q_\gamma$ of the returned partition.
    pub modularity: f64,
    /// Number of multi-level passes (local moving + aggregation) committed.
    pub passes: usize,
}

/// Compact internal super-graph representation supporting weighted self-loops across passes.
struct WeightedLevelGraph {
    num_nodes: usize,
    indptr: Vec<u32>,
    indices: Vec<u32>,
    weights: Vec<f64>,
    loop_weights: Vec<f64>,
    degrees: Vec<f64>,
    total_weight_2m: f64,
}

impl WeightedLevelGraph {
    fn from_csr<W: Scalar>(graph: &CsrGraph<W>) -> CoreResult<Self> {
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

    fn modularity(&self, comm: &[u32], sigma_tot: &[f64], resolution: f64) -> f64 {
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
    fn aggregate(&self, comm_compacted: &[u32], num_new: usize) -> Self {
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
fn canonicalize_labels(labels: &mut [u32], max_id_bound: usize) -> usize {
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

/// Runs two-phase multi-level Louvain community detection on an undirected [`CsrGraph`].
///
/// Phase 1 sweeps vertices in deterministic shuffled order (via `DeterministicRng`) and
/// accumulates neighbor community weights using a `VisitStamp` epoch array and flat `Vec`
/// scratch buffers. Phase 2 contracts communities into a super-graph without hash maps.
/// `LouvainResult::modularity` reports $Q_\gamma$ evaluated with `config.resolution`.
pub fn louvain<W: Scalar>(
    graph: &CsrGraph<W>,
    config: &LouvainConfig,
) -> CoreResult<LouvainResult> {
    if graph.is_directed() {
        return Err(CoreError::invalid_input(
            "louvain requires an undirected graph",
        ));
    }
    if config.max_passes == 0 {
        return Err(CoreError::invalid_input("max_passes must be >= 1"));
    }
    if !config.resolution.is_finite() {
        return Err(CoreError::non_finite("LouvainConfig::resolution"));
    }
    if config.resolution < 0.0 {
        return Err(CoreError::invalid_input("resolution must be >= 0.0"));
    }
    if !config.min_modularity_gain.is_finite() {
        return Err(CoreError::non_finite("LouvainConfig::min_modularity_gain"));
    }
    if config.min_modularity_gain < 0.0 {
        return Err(CoreError::invalid_input(
            "min_modularity_gain must be >= 0.0",
        ));
    }

    let n = graph.num_nodes();
    if n == 0 {
        return Ok(LouvainResult {
            labels: Vec::new(),
            num_communities: 0,
            modularity: 0.0,
            passes: 0,
        });
    }

    let mut level = WeightedLevelGraph::from_csr(graph)?;
    let mut global_labels: Vec<u32> = (0..n as u32).collect();

    if level.total_weight_2m <= 0.0 {
        return Ok(LouvainResult {
            labels: global_labels,
            num_communities: n,
            modularity: 0.0,
            passes: 0,
        });
    }

    let mut rng = DeterministicRng::from_seed_u64(config.seed);
    let mut passes = 0usize;
    let mut current_q = {
        let init_comm: Vec<u32> = (0..n as u32).collect();
        level.modularity(&init_comm, &level.degrees, config.resolution)
    };

    for _pass in 0..config.max_passes {
        let nl = level.num_nodes;
        if nl <= 1 {
            break;
        }

        let mut comm: Vec<u32> = (0..nl as u32).collect();
        let mut sigma_tot = level.degrees.clone();
        let mut order: Vec<u32> = (0..nl as u32).collect();
        let mut comm_stamp = VisitStamp::new(nl);
        let mut comm_weights = vec![0.0_f64; nl];
        let mut touched_comms: Vec<u32> = Vec::new();

        let inv_2m = 1.0 / level.total_weight_2m;
        let max_sweeps = nl.clamp(64, 500);
        let mut any_moved_in_pass = false;

        for _sweep in 0..max_sweeps {
            rng.shuffle(&mut order);
            let mut moved_in_sweep = false;

            for &u_u32 in &order {
                let u = u_u32 as usize;
                let k_u = level.degrees[u];
                let c_old = comm[u];
                let c_old_idx = c_old as usize;

                comm_stamp.clear();
                touched_comms.clear();

                let start = level.indptr[u] as usize;
                let end = level.indptr[u + 1] as usize;
                for pos in start..end {
                    let v = level.indices[pos] as usize;
                    let cv = comm[v];
                    let cv_idx = cv as usize;
                    if comm_stamp.visit(cv_idx) {
                        touched_comms.push(cv);
                    }
                    comm_weights[cv_idx] += level.weights[pos];
                }

                if touched_comms.is_empty() {
                    continue;
                }

                // Temporarily remove vertex `u` from its current community.
                sigma_tot[c_old_idx] -= k_u;
                let k_u_old = if comm_stamp.is_visited(c_old_idx) {
                    comm_weights[c_old_idx]
                } else {
                    0.0
                };
                // Proportional to modularity insertion gain relative to singleton {u}:
                // gain_term(c) = k_{u, in}(c) - gamma * k_u * sigma_tot[c] / (2m).
                let old_term = k_u_old - config.resolution * k_u * sigma_tot[c_old_idx] * inv_2m;

                let mut best_c = c_old;
                let mut best_delta_term = 0.0_f64;
                for &cand in &touched_comms {
                    if cand == c_old {
                        continue;
                    }
                    let cand_idx = cand as usize;
                    let cand_term = comm_weights[cand_idx]
                        - config.resolution * k_u * sigma_tot[cand_idx] * inv_2m;
                    let delta_term = cand_term - old_term;
                    if delta_term > best_delta_term + 1e-12
                        || ((delta_term - best_delta_term).abs() <= 1e-12
                            && delta_term > 1e-12
                            && cand < best_c)
                    {
                        best_delta_term = delta_term;
                        best_c = cand;
                    }
                }

                for &cand in &touched_comms {
                    comm_weights[cand as usize] = 0.0;
                }

                if best_c != c_old {
                    comm[u] = best_c;
                    sigma_tot[best_c as usize] += k_u;
                    moved_in_sweep = true;
                    any_moved_in_pass = true;
                } else {
                    sigma_tot[c_old_idx] += k_u;
                }
            }

            if !moved_in_sweep {
                break;
            }
        }

        if !any_moved_in_pass {
            break;
        }

        let next_q = level.modularity(&comm, &sigma_tot, config.resolution);
        if next_q - current_q <= config.min_modularity_gain {
            break;
        }
        current_q = next_q;
        passes += 1;

        let num_new = canonicalize_labels(&mut comm, nl);
        for g in &mut global_labels {
            *g = comm[*g as usize];
        }

        if num_new == nl {
            break;
        }
        level = level.aggregate(&comm, num_new);
    }

    let num_communities = canonicalize_labels(&mut global_labels, n);
    Ok(LouvainResult {
        labels: global_labels,
        num_communities,
        modularity: current_q,
        passes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_two_cliques_barbell(clique_size: usize) -> CsrGraph<f64> {
        let n = clique_size * 2;
        let mut edges = Vec::new();
        for i in 0..clique_size {
            for j in (i + 1)..clique_size {
                edges.push((i as u32, j as u32));
                edges.push(((clique_size + i) as u32, (clique_size + j) as u32));
            }
        }
        edges.push(((clique_size - 1) as u32, clique_size as u32));
        CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
    }

    #[test]
    fn test_louvain_recovers_two_cliques_and_multilevel_ring() {
        let barbell = make_two_cliques_barbell(6);
        let res = louvain(&barbell, &LouvainConfig::default()).unwrap();
        assert_eq!(res.num_communities, 2);
        assert_eq!(res.labels[..6], [0; 6]);
        assert_eq!(res.labels[6..], [1; 6]);
        assert!(res.modularity > 0.4);
        assert!(res.passes >= 1);

        // Ring of 8 cliques of size 5: multi-level aggregation merges pairs cleanly into 8 cliques.
        let mut edges = Vec::new();
        for c in 0..8u32 {
            let base = c * 5;
            for i in 0..5 {
                for j in (i + 1)..5 {
                    edges.push((base + i, base + j));
                }
            }
            let next_base = ((c + 1) % 8) * 5;
            edges.push((base + 4, next_base));
        }
        let ring = CsrGraph::<f64>::from_edges(40, &edges, None, false, false).unwrap();
        let res_ring = louvain(&ring, &LouvainConfig::default()).unwrap();
        assert_eq!(res_ring.num_communities, 8);
        assert!(res_ring.modularity > 0.7);
        for c in 0..8 {
            let first = res_ring.labels[c * 5];
            assert!(
                res_ring.labels[c * 5..(c + 1) * 5]
                    .iter()
                    .all(|&l| l == first)
            );
        }

        // Weighted f32 graph + resolution variation.
        let mut w_edges = Vec::new();
        let mut w_vals = Vec::new();
        for i in 0..4u32 {
            for j in (i + 1)..4 {
                w_edges.push((i, j));
                w_vals.push(8.0_f32);
                w_edges.push((4 + i, 4 + j));
                w_vals.push(8.0_f32);
            }
        }
        w_edges.push((3, 4));
        w_vals.push(0.1_f32);
        let wg = CsrGraph::from_edges(8, &w_edges, Some(&w_vals), false, false).unwrap();
        let w_res = louvain(&wg, &LouvainConfig::default()).unwrap();
        assert_eq!(w_res.num_communities, 2);
        assert_eq!(w_res.labels[..4], [0; 4]);
        assert_eq!(w_res.labels[4..], [1; 4]);

        let low_gamma = louvain(
            &wg,
            &LouvainConfig {
                resolution: 0.001,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(low_gamma.num_communities, 1);

        let coarse_ring = louvain(
            &ring,
            &LouvainConfig {
                resolution: 0.2,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(coarse_ring.num_communities < 8);
    }

    #[test]
    fn test_louvain_determinism_and_degenerate_cases() {
        let barbell = make_two_cliques_barbell(5);
        let cfg = LouvainConfig {
            seed: 1234,
            ..Default::default()
        };
        assert_eq!(
            louvain(&barbell, &cfg).unwrap(),
            louvain(&barbell, &cfg).unwrap()
        );

        let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        let re = louvain(&empty, &cfg).unwrap();
        assert!(re.labels.is_empty());
        assert_eq!(re.num_communities, 0);
        assert_eq!(re.modularity, 0.0);

        let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
        let rs = louvain(&single, &cfg).unwrap();
        assert_eq!(rs.labels, vec![0]);
        assert_eq!(rs.num_communities, 1);

        let edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
        let ri = louvain(&edgeless, &cfg).unwrap();
        assert_eq!(ri.labels, vec![0, 1, 2, 3]);
        assert_eq!(ri.num_communities, 4);
        assert_eq!(ri.modularity, 0.0);
    }

    #[test]
    fn test_louvain_directed_and_invalid_config_errors() {
        let cfg = LouvainConfig::default();
        let directed =
            CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], None, true, false).unwrap();
        assert!(matches!(
            louvain(&directed, &cfg),
            Err(CoreError::InvalidInput(_))
        ));

        let g = make_two_cliques_barbell(3);
        assert!(matches!(
            louvain(
                &g,
                &LouvainConfig {
                    max_passes: 0,
                    ..Default::default()
                }
            ),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            louvain(
                &g,
                &LouvainConfig {
                    resolution: -0.5,
                    ..Default::default()
                }
            ),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            louvain(
                &g,
                &LouvainConfig {
                    min_modularity_gain: f64::NAN,
                    ..Default::default()
                }
            ),
            Err(CoreError::NonFiniteValue { .. })
        ));
    }
}
