//! Pure-Rust asynchronous Label Propagation using `DeterministicRng` and `VisitStamp`.

use egraph_core::{CoreError, CoreResult, CsrGraph, DeterministicRng, Scalar, VisitStamp};

/// Configuration for [`label_propagation`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LabelPropagationConfig {
    /// Maximum number of asynchronous propagation sweeps over all vertices.
    pub max_iterations: usize,
    /// Deterministic PRNG seed controlling Fisher-Yates vertex shuffles and tie-breaking.
    pub seed: u64,
}

impl Default for LabelPropagationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            seed: 42,
        }
    }
}

/// Output of [`label_propagation`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelPropagationResult {
    /// Canonical community assignment in `0..num_communities` for each vertex.
    pub labels: Vec<u32>,
    /// Number of distinct non-empty communities detected.
    pub num_communities: usize,
    /// Number of asynchronous propagation sweeps executed.
    pub iterations: usize,
    /// Whether propagation reached a fixed point (`!changed` on the final sweep).
    pub converged: bool,
}

/// Renumbers labels into canonical `0..c` order of first appearance by vertex index and
/// returns the number of distinct communities.
fn canonicalize_labels(labels: &mut [u32], n: usize) -> usize {
    let mut remap = vec![u32::MAX; n.max(1)];
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

/// Detects communities on an undirected [`CsrGraph`] via asynchronous Label Propagation.
///
/// Each sweep shuffles vertices via Fisher-Yates (`DeterministicRng::shuffle`) and counts
/// neighbor label votes in a flat `VisitStamp`-guarded accumulator without hash maps.
/// Convergence stops early once every vertex already holds a maximum-vote neighbor label.
pub fn label_propagation<W: Scalar>(
    graph: &CsrGraph<W>,
    config: &LabelPropagationConfig,
) -> CoreResult<LabelPropagationResult> {
    if graph.is_directed() {
        return Err(CoreError::invalid_input(
            "label_propagation requires an undirected graph",
        ));
    }
    if config.max_iterations == 0 {
        return Err(CoreError::invalid_input("max_iterations must be >= 1"));
    }

    if let Some(ws) = graph.weights() {
        for &w in ws {
            let wf = w.to_f64();
            if !wf.is_finite() {
                return Err(CoreError::non_finite("label_propagation::graph::weights"));
            }
            if wf < 0.0 {
                return Err(CoreError::invalid_input(
                    "label_propagation edge weights must be non-negative",
                ));
            }
        }
    }

    let n = graph.num_nodes();
    if n == 0 {
        return Ok(LabelPropagationResult {
            labels: Vec::new(),
            num_communities: 0,
            iterations: 0,
            converged: true,
        });
    }
    if n == 1 {
        return Ok(LabelPropagationResult {
            labels: vec![0],
            num_communities: 1,
            iterations: 0,
            converged: true,
        });
    }

    let mut labels: Vec<u32> = (0..n as u32).collect();
    let mut order: Vec<u32> = (0..n as u32).collect();
    let mut vote_stamp = VisitStamp::new(n);
    let mut vote_scores = vec![0.0_f64; n];
    let mut seen_labels: Vec<u32> = Vec::new();
    let mut tied_labels: Vec<u32> = Vec::new();

    let mut rng = DeterministicRng::from_seed_u64(config.seed);
    let weights_opt = graph.weights();
    let mut iterations = 0usize;
    let mut converged = false;

    for iter in 1..=config.max_iterations {
        iterations = iter;
        rng.shuffle(&mut order);
        let mut changed = false;

        for &u_u32 in &order {
            let u = u_u32 as usize;
            let start = graph.indptr()[u] as usize;
            let end = graph.indptr()[u + 1] as usize;
            if start == end {
                continue;
            }

            vote_stamp.clear();
            seen_labels.clear();

            for pos in start..end {
                let w = match weights_opt {
                    Some(ws) => ws[pos].to_f64(),
                    None => 1.0,
                };
                if w <= 0.0 {
                    continue;
                }
                let v = graph.indices()[pos] as usize;
                let lv = labels[v];
                let lv_idx = lv as usize;
                if vote_stamp.visit(lv_idx) {
                    seen_labels.push(lv);
                }
                vote_scores[lv_idx] += w;
            }

            if seen_labels.is_empty() {
                continue;
            }

            let mut max_score = 0.0_f64;
            for &l in &seen_labels {
                let s = vote_scores[l as usize];
                if s > max_score {
                    max_score = s;
                }
            }

            let curr = labels[u];
            let tol = 1e-12 * max_score.max(1.0);
            let curr_is_max = vote_stamp.is_visited(curr as usize)
                && vote_scores[curr as usize] + tol >= max_score;

            if !curr_is_max {
                tied_labels.clear();
                for &l in &seen_labels {
                    if vote_scores[l as usize] + tol >= max_score {
                        tied_labels.push(l);
                    }
                }
                // Sort candidate ties so selection depends strictly on label IDs and RNG,
                // never on CSR neighbor iteration order.
                tied_labels.sort_unstable();
                let chosen = if tied_labels.len() == 1 {
                    tied_labels[0]
                } else {
                    tied_labels[rng.gen_range_usize(tied_labels.len())]
                };
                if chosen != curr {
                    labels[u] = chosen;
                    changed = true;
                }
            }

            for &l in &seen_labels {
                vote_scores[l as usize] = 0.0;
            }
        }

        if !changed {
            converged = true;
            break;
        }
    }

    let num_communities = canonicalize_labels(&mut labels, n);
    Ok(LabelPropagationResult {
        labels,
        num_communities,
        iterations,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_label_propagation_two_cliques_and_determinism() {
        // Two 5-cliques joined by a single bridge edge (4, 5).
        let mut edges = Vec::new();
        for i in 0..5u32 {
            for j in (i + 1)..5 {
                edges.push((i, j));
                edges.push((5 + i, 5 + j));
            }
        }
        edges.push((4, 5));
        let barbell = CsrGraph::<f64>::from_edges(10, &edges, None, false, false).unwrap();

        let cfg = LabelPropagationConfig {
            max_iterations: 100,
            seed: 42,
        };
        let r1 = label_propagation(&barbell, &cfg).unwrap();
        let r2 = label_propagation(&barbell, &cfg).unwrap();
        assert_eq!(r1, r2);
        assert_eq!(r1.num_communities, 2);
        assert!(r1.converged);
        assert!(r1.iterations >= 1 && r1.iterations <= cfg.max_iterations);
        assert_eq!(r1.labels[..5], [0, 0, 0, 0, 0]);
        assert_eq!(r1.labels[5..], [1, 1, 1, 1, 1]);

        // Weighted f32 graph preserves weighted community structure.
        let mut w_edges = Vec::new();
        let mut w_vals = Vec::new();
        for i in 0..4u32 {
            for j in (i + 1)..4 {
                w_edges.push((i, j));
                w_vals.push(5.0_f32);
                w_edges.push((4 + i, 4 + j));
                w_vals.push(5.0_f32);
            }
        }
        w_edges.push((3, 4));
        w_vals.push(0.2_f32);
        let wg = CsrGraph::from_edges(8, &w_edges, Some(&w_vals), false, false).unwrap();
        let wr = label_propagation(&wg, &cfg).unwrap();
        assert_eq!(wr.num_communities, 2);
        assert!(wr.converged);
        assert_eq!(wr.labels[..4], [0, 0, 0, 0]);
        assert_eq!(wr.labels[4..], [1, 1, 1, 1]);
    }

    #[test]
    fn test_label_propagation_disconnected_and_convergence_budget() {
        // Two disconnected 4-cliques should converge to 2 communities cleanly.
        let mut edges = Vec::new();
        for i in 0..4u32 {
            for j in (i + 1)..4 {
                edges.push((i, j));
                edges.push((4 + i, 4 + j));
            }
        }
        let disc = CsrGraph::<f64>::from_edges(8, &edges, None, false, false).unwrap();
        let res = label_propagation(&disc, &LabelPropagationConfig::default()).unwrap();
        assert_eq!(res.num_communities, 2);
        assert!(res.converged);
        assert!(res.iterations >= 2);
        assert_eq!(res.labels[..4], [0, 0, 0, 0]);
        assert_eq!(res.labels[4..], [1, 1, 1, 1]);

        // Restricting to max_iterations = 1 stops without reaching fixed-point convergence.
        let limited = label_propagation(
            &disc,
            &LabelPropagationConfig {
                max_iterations: 1,
                seed: 42,
            },
        )
        .unwrap();
        assert_eq!(limited.iterations, 1);
        assert!(!limited.converged);
    }

    #[test]
    fn test_label_propagation_degenerate_and_errors() {
        let cfg = LabelPropagationConfig::default();

        let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        let r_empty = label_propagation(&empty, &cfg).unwrap();
        assert!(r_empty.labels.is_empty());
        assert_eq!(r_empty.num_communities, 0);
        assert_eq!(r_empty.iterations, 0);
        assert!(r_empty.converged);

        let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
        let r_single = label_propagation(&single, &cfg).unwrap();
        assert_eq!(r_single.labels, vec![0]);
        assert_eq!(r_single.num_communities, 1);
        assert_eq!(r_single.iterations, 0);
        assert!(r_single.converged);

        let edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
        let r_edgeless = label_propagation(&edgeless, &cfg).unwrap();
        assert_eq!(r_edgeless.labels, vec![0, 1, 2, 3]);
        assert_eq!(r_edgeless.num_communities, 4);
        assert_eq!(r_edgeless.iterations, 1);
        assert!(r_edgeless.converged);

        let directed =
            CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], None, true, false).unwrap();
        assert!(matches!(
            label_propagation(&directed, &cfg),
            Err(CoreError::InvalidInput(_))
        ));

        let g = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
        assert!(matches!(
            label_propagation(
                &g,
                &LabelPropagationConfig {
                    max_iterations: 0,
                    seed: 42
                }
            ),
            Err(CoreError::InvalidInput(_))
        ));
    }
}
