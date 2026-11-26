use egraph_core::CsrGraph;

use super::super::*;
use crate::knn::KnnGraph;

#[test]
fn test_2hop_adaptive_perplexity_nodewise_and_median() {
    // Hub 0 connected to 1..=15; chain 15-16-17; isolated 18, 19 (N=20).
    let mut edges: Vec<(u32, u32)> = (1..=15).map(|i| (0, i)).collect();
    edges.extend([(15, 16), (16, 17)]);
    let g = CsrGraph::<f64>::from_edges(20, &edges, None, false, false).unwrap();

    let sizes = compute_2hop_neighborhood_sizes(&g);
    assert_eq!(sizes[0], 16); // 1..=15 (1-hop) + 16 (2-hop)
    assert_eq!(sizes[18], 0);
    assert_eq!(sizes[19], 0);

    let nodewise = compute_nodewise_2hop_perplexities(
        &g,
        DEFAULT_NODEWISE_PERPLEXITY_C,
        DEFAULT_NODEWISE_MIN_PERPLEXITY,
        DEFAULT_MAX_PERPLEXITY,
    )
    .unwrap();
    let upper = 19.0 / 3.0;
    assert!((nodewise[0] - upper).abs() < 1e-12); // 0.9*16=14.4 clamped to 19/3
    assert!((nodewise[18] - 5.0).abs() < 1e-12); // isolated clamped to lower=5.0

    let med = compute_median_2hop_perplexity(
        &g,
        DEFAULT_MEDIAN_PERPLEXITY_C,
        DEFAULT_MEDIAN_MIN_PERPLEXITY,
        DEFAULT_MAX_PERPLEXITY,
    )
    .unwrap();
    assert!((med - upper).abs() < 1e-12);

    // Small N=0..=3 safety
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    assert!(
        compute_nodewise_2hop_perplexities(&empty, 0.9, 5.0, 1200.0)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        compute_median_2hop_perplexity(&empty, 1.5, 15.0, 1200.0).unwrap(),
        1.0
    );
    let tiny3 = CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], None, false, false).unwrap();
    assert_eq!(
        compute_nodewise_2hop_perplexities(&tiny3, 0.9, 5.0, 1200.0).unwrap(),
        vec![1.0, 1.0, 1.0]
    );
}

#[test]
fn test_compute_knn_probabilities_entropy_symmetry_and_row_normalized() {
    // Construct non-uniform k-NN with duplicates and self-loops to verify filtering and entropy.
    let knn = KnnGraph::new_validated(
        4,
        vec![0, 5, 7, 9, 11],
        vec![
            0, 1, 1, 2, 3, // row 0: self-loop 0 + dup 1
            0, 2, // row 1
            1, 3, // row 2
            0, 2, // row 3
        ],
        vec![
            0.0, 5.0, 0.2, 1.0, 3.0, // row 0 min for 1 is 0.2
            0.2, 0.8, // row 1
            0.8, 1.2, // row 2
            3.0, 1.2, // row 3
        ],
    )
    .unwrap();
    let perps = vec![2.0, 1.5, 1.5, 1.8];
    let cfg = PerplexityConfig::default();
    let sp = compute_knn_probabilities(&knn, &perps, &cfg).unwrap();

    let mut total = 0.0;
    for i in 0..4 {
        let (cols, vals) = sp.row(i);
        assert!(cols.windows(2).all(|w| w[0] < w[1]));
        assert!(!cols.contains(&(i as u32)));
        for (&j, &p_ij) in cols.iter().zip(vals.iter()) {
            assert!(p_ij.is_finite() && p_ij > 0.0);
            let (rcols, rvals) = sp.row(j as usize);
            let pos = rcols.binary_search(&(i as u32)).unwrap();
            assert_eq!(p_ij.to_bits(), rvals[pos].to_bits());
            total += p_ij;
        }
    }
    assert!((total - 1.0).abs() < 1e-12);

    // Verify RowNormalized graph probabilities P = (D^{-1} A + A^T D^{-1}) / (2N)
    let star =
        CsrGraph::<f64>::from_edges(5, &[(0, 1), (0, 2), (0, 3), (0, 4)], None, false, false)
            .unwrap();
    let p_graph = graph_row_normalized_probabilities(&star).unwrap();
    // For leaf 1 -> hub 0: a_{1->0} = 1.0, a_{0->1} = 0.25 => P_{01} = (1 + 0.25) / 10 = 0.125.
    let (r0, v0) = p_graph.row(0);
    assert_eq!(r0, &[1, 2, 3, 4]);
    for &val in v0 {
        assert!((val - 0.125).abs() < 1e-12);
    }
    let p_total: f64 = p_graph.values.iter().sum();
    assert!((p_total - 1.0).abs() < 1e-12);

    // Verify RowNormalized mode inside compute_knn_probabilities genuinely row-normalizes
    // S = P_dir + P_dir^T before final (D_S^{-1} S + S^T D_S^{-1}) / (2N) symmetrization.
    let cfg_rn = PerplexityConfig {
        symmetrization: SymmetrizationMode::RowNormalized,
        ..Default::default()
    };
    let sp_rn = compute_knn_probabilities(&knn, &perps, &cfg_rn).unwrap();
    let sp_rn_total: f64 = sp_rn.values.iter().sum();
    assert!((sp_rn_total - 1.0).abs() < 1e-12);
    assert_eq!(sp_rn.indptr, sp.indptr);
    assert_eq!(sp_rn.indices, sp.indices);
    for i in 0..4 {
        let (cols, vals) = sp_rn.row(i);
        for (&j, &p_ij) in cols.iter().zip(vals.iter()) {
            let (rcols, rvals) = sp_rn.row(j as usize);
            let pos = rcols.binary_search(&(i as u32)).unwrap();
            assert_eq!(p_ij.to_bits(), rvals[pos].to_bits());
        }
    }
    let max_diff = sp_rn
        .values
        .iter()
        .zip(sp.values.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff > 1e-3,
        "expected RowNormalized to differ from StandardTsne on asymmetric knn, max_diff={max_diff}"
    );

    // On a regular symmetric KnnGraph, row sums of S = P_dir + P_dir^T are uniform (= 2.0),
    // so RowNormalized matches StandardTsne.
    let regular_knn = KnnGraph::new_validated(
        4,
        vec![0, 2, 4, 6, 8],
        vec![1, 3, 0, 2, 1, 3, 0, 2],
        vec![1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0],
    )
    .unwrap();
    let reg_perps = vec![1.5; 4];
    let reg_std = compute_knn_probabilities(&regular_knn, &reg_perps, &cfg).unwrap();
    let reg_rn = compute_knn_probabilities(&regular_knn, &reg_perps, &cfg_rn).unwrap();
    let reg_diff = reg_std
        .values
        .iter()
        .zip(reg_rn.values.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        reg_diff < 1e-15,
        "expected RowNormalized to match StandardTsne on regular symmetric knn, diff={reg_diff}"
    );
}

#[test]
fn test_compute_knn_probabilities_entropy_accuracy_on_distinct_row() {
    // Single node with 5 distinct neighbors: verify conditional probability perplexity matches target.
    let knn = KnnGraph::new_validated(
        6,
        vec![0, 5, 6, 7, 8, 9, 10],
        vec![1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
        vec![0.1, 0.4, 0.9, 1.6, 2.5, 0.1, 0.4, 0.9, 1.6, 2.5],
    )
    .unwrap();
    // Verify node 0 conditional probabilities via StandardTsne vs non-zero entries:
    // Note: rows 1..=5 only have node 0 so p_{0|j} = 1.0, while p_{j|0} = 2*6*P_{0j} - 1.0.
    let target_perp = 2.5;
    let perps = vec![target_perp, 1.0, 1.0, 1.0, 1.0, 1.0];
    let sp = compute_knn_probabilities(
        &knn,
        &perps,
        &PerplexityConfig {
            sigma_tolerance: 1e-9,
            sigma_iters: 80,
            ..Default::default()
        },
    )
    .unwrap();
    let (_cols, vals) = sp.row(0);
    let p_cond: Vec<f64> = vals.iter().map(|&p0j| 12.0 * p0j - 1.0).collect();
    let sum_p: f64 = p_cond.iter().sum();
    assert!((sum_p - 1.0).abs() < 1e-12);
    let entropy: f64 = p_cond
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();
    assert!((entropy.exp() - target_perp).abs() < 1e-7);
}

#[test]
fn test_partial_bfs_knn_determinism_and_tiers() {
    let mut edges = Vec::new();
    for r in 0..5 {
        for c in 0..5 {
            let u = (r * 5 + c) as u32;
            if c + 1 < 5 {
                edges.push((u, u + 1));
            }
            if r + 1 < 5 {
                edges.push((u, u + 5));
            }
        }
    }
    let g = CsrGraph::<f64>::from_edges(25, &edges, None, false, false).unwrap();

    let k1 = partial_bfs_knn(&g, 6, 42).unwrap();
    let k2 = partial_bfs_knn(&g, 6, 42).unwrap();
    assert_eq!(k1, k2);
    for i in 0..25 {
        let (nbrs, dists) = k1.row(i);
        assert_eq!(nbrs.len(), 6);
        assert!(dists.windows(2).all(|w| w[0] <= w[1]));
        assert!(!nbrs.contains(&(i as u32)));
    }

    // k == 0 returns empty rows
    let k0 = partial_bfs_knn(&g, 0, 42).unwrap();
    assert_eq!(k0.nnz(), 0);
}

#[test]
fn test_perplexity_disconnected_singleton_and_nan_rejection() {
    // Two disconnected components (0-1-2 and 3-4) + isolated node 5
    let g = CsrGraph::<f64>::from_edges(6, &[(0, 1), (1, 2), (3, 4)], None, false, false).unwrap();
    let knn = partial_bfs_knn(&g, 4, 7).unwrap();
    // Isolated node 5 has 0 reachable neighbors; component {3,4} reaches only 1 neighbor.
    assert_eq!(knn.row(5).0.len(), 0);
    assert_eq!(knn.row(3).0, &[4]);
    assert_eq!(knn.row(0).0.len(), 2);

    let probs = compute_knn_probabilities(&knn, &[1.5; 6], &PerplexityConfig::default()).unwrap();
    assert_eq!(probs.n_nodes, 6);
    assert!(probs.values.iter().all(|v| v.is_finite() && *v > 0.0));

    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    assert_eq!(partial_bfs_knn(&single, 5, 1).unwrap().nnz(), 0);
    assert!(
        compute_knn_probabilities(
            &knn,
            &[1.5, f64::NAN, 1.5, 1.5, 1.5, 1.5],
            &PerplexityConfig::default()
        )
        .is_err()
    );
    assert!(compute_nodewise_2hop_perplexities(&g, f64::NAN, 5.0, 1200.0).is_err());
    assert!(compute_median_2hop_perplexity(&g, 1.5, 15.0, -1.0).is_err());
}
