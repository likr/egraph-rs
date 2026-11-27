use egraph_core::{CoreError, CsrGraph};

use super::super::*;
use crate::knn::{extract_1hop_2hop_csr, extract_sparse_1hop_2hop_pairs};

#[test]
fn test_graph_12hop_row_normalized_probabilities_exact_and_capped() {
    // Asymmetric barbell-like graph with star hub + tail so 1-hop and 2-hop degrees vary.
    let mut edges: Vec<(u32, u32)> = (1..=6).map(|v| (0, v)).collect();
    edges.extend([(6, 7), (7, 8), (8, 9), (1, 2)]);
    let g = CsrGraph::<f64>::from_edges(10, &edges, None, false, false).unwrap();
    let n = g.num_nodes();

    // 1. Uncapped (None): algebraically equal to symmetrize_row_normalized(A1 + w2 * A2).
    let w2 = 0.5;
    let probs_uncapped = graph_12hop_row_normalized_probabilities(&g, w2, None, 42).unwrap();
    let (adj1, adj2) = extract_1hop_2hop_csr(&g).unwrap();
    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0u32);
    let mut indices = Vec::new();
    let mut weights = Vec::new();
    for u in 0..n {
        for &v in adj1.neighbors(u) {
            indices.push(v);
            weights.push(1.0);
        }
        for &v in adj2.neighbors(u) {
            indices.push(v);
            weights.push(w2);
        }
        indptr.push(indices.len() as u32);
    }
    let ref_uncapped = symmetrize_row_normalized(n, &indptr, &indices, &weights).unwrap();
    assert_eq!(probs_uncapped.indptr(), ref_uncapped.indptr());
    assert_eq!(probs_uncapped.indices(), ref_uncapped.indices());
    for (a, b) in probs_uncapped
        .values()
        .iter()
        .zip(ref_uncapped.values().iter())
    {
        assert!((a - b).abs() < 1e-14);
    }
    assert!((probs_uncapped.values().iter().sum::<f64>() - 1.0).abs() < 1e-12);

    // 2. Capped (Some(3)): algebraically equal to symmetrize_row_normalized on capped pairs.
    let cap = 3usize;
    let probs_capped = graph_12hop_row_normalized_probabilities(&g, 0.35, Some(cap), 77).unwrap();
    let pairs = extract_sparse_1hop_2hop_pairs(&g, cap, 77).unwrap();
    let mut adj_rows: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
    for (&u, &v) in pairs.u1().iter().zip(pairs.v1().iter()) {
        adj_rows[u as usize].push((v, 1.0));
        adj_rows[v as usize].push((u, 1.0));
    }
    for (&u, &v) in pairs.u2().iter().zip(pairs.v2().iter()) {
        adj_rows[u as usize].push((v, 0.35));
        adj_rows[v as usize].push((u, 0.35));
    }
    let mut c_indptr = vec![0u32];
    let mut c_indices = Vec::new();
    let mut c_weights = Vec::new();
    for row in adj_rows {
        for (v, w) in row {
            c_indices.push(v);
            c_weights.push(w);
        }
        c_indptr.push(c_indices.len() as u32);
    }
    let ref_capped = symmetrize_row_normalized(n, &c_indptr, &c_indices, &c_weights).unwrap();
    assert_eq!(probs_capped.indptr(), ref_capped.indptr());
    assert_eq!(probs_capped.indices(), ref_capped.indices());
    for (a, b) in probs_capped.values().iter().zip(ref_capped.values().iter()) {
        assert!((a - b).abs() < 1e-14);
    }

    // 3. 1-hop-only equivalence when w2 == 0.0 or max_2hop_per_node == Some(0).
    let p_w0 = graph_12hop_row_normalized_probabilities(&g, 0.0, None, 42).unwrap();
    let p_c0 = graph_12hop_row_normalized_probabilities(&g, 0.5, Some(0), 42).unwrap();
    let p_1hop = graph_row_normalized_probabilities(&g).unwrap();
    assert_eq!(p_w0.indptr(), p_1hop.indptr());
    assert_eq!(p_w0.indices(), p_1hop.indices());
    assert_eq!(p_c0.indptr(), p_1hop.indptr());
    assert_eq!(p_c0.indices(), p_1hop.indices());
    for ((&a, &b), &c) in p_w0
        .values()
        .iter()
        .zip(p_c0.values().iter())
        .zip(p_1hop.values().iter())
    {
        assert!((a - c).abs() < 1e-14);
        assert!((b - c).abs() < 1e-14);
    }

    // 4. Edge cases & validation errors.
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    assert_eq!(
        graph_12hop_row_normalized_probabilities(&empty, 0.5, None, 42)
            .unwrap()
            .nnz(),
        0
    );
    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    assert_eq!(
        graph_12hop_row_normalized_probabilities(&single, 0.5, Some(4), 42)
            .unwrap()
            .nnz(),
        0
    );
    assert!(matches!(
        graph_12hop_row_normalized_probabilities(&g, -0.1, None, 42),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        graph_12hop_row_normalized_probabilities(&g, f64::NAN, None, 42),
        Err(CoreError::NonFiniteValue { .. })
    ));
}

#[test]
fn test_graph_12hop_row_normalized_degree_decay_and_underflow_guard() {
    // Asymmetric hub + chain graph
    let mut edges: Vec<(u32, u32)> = (1..=6).map(|v| (0, v)).collect();
    edges.extend([(6, 7), (7, 8), (8, 9), (1, 2)]);
    let g = CsrGraph::<f64>::from_edges(10, &edges, None, false, false).unwrap();
    let n = g.num_nodes();
    let beta = 0.4;
    let w2 = 0.5;

    // Uncapped exact match with reference symmetrize_row_normalized((d_i * d_j)^(-beta))
    let probs_uncapped =
        graph_12hop_row_normalized_probabilities_with_degree_decay(&g, w2, None, beta, 42).unwrap();
    let (adj1, adj2) = extract_1hop_2hop_csr(&g).unwrap();
    let mut indptr = vec![0u32];
    let mut indices = Vec::new();
    let mut weights = Vec::new();
    for u in 0..n {
        let du = adj1.degree(u).max(1) as f64;
        for &v in adj1.neighbors(u) {
            let dv = adj1.degree(v as usize).max(1) as f64;
            indices.push(v);
            weights.push((du * dv).powf(-beta));
        }
        for &v in adj2.neighbors(u) {
            let dv = adj1.degree(v as usize).max(1) as f64;
            indices.push(v);
            weights.push(w2 * (du * dv).powf(-beta));
        }
        indptr.push(indices.len() as u32);
    }
    let ref_uncapped = symmetrize_row_normalized(n, &indptr, &indices, &weights).unwrap();
    assert_eq!(probs_uncapped.indptr(), ref_uncapped.indptr());
    assert_eq!(probs_uncapped.indices(), ref_uncapped.indices());
    for (&a, &b) in probs_uncapped
        .values()
        .iter()
        .zip(ref_uncapped.values().iter())
    {
        assert!((a - b).abs() < 1e-14);
    }
    assert!((probs_uncapped.values().iter().sum::<f64>() - 1.0).abs() < 1e-12);

    // Capped Some(3) exact match
    let probs_capped =
        graph_12hop_row_normalized_probabilities_with_degree_decay(&g, 0.35, Some(3), beta, 77)
            .unwrap();
    let pairs = extract_sparse_1hop_2hop_pairs(&g, 3, 77).unwrap();
    let mut deg1 = vec![0usize; n];
    for (&u, &v) in pairs.u1().iter().zip(pairs.v1().iter()) {
        deg1[u as usize] += 1;
        deg1[v as usize] += 1;
    }
    let mut adj_rows: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
    for (&u, &v) in pairs.u1().iter().zip(pairs.v1().iter()) {
        let w = ((deg1[u as usize].max(1) * deg1[v as usize].max(1)) as f64).powf(-beta);
        adj_rows[u as usize].push((v, w));
        adj_rows[v as usize].push((u, w));
    }
    for (&u, &v) in pairs.u2().iter().zip(pairs.v2().iter()) {
        let w = 0.35 * ((deg1[u as usize].max(1) * deg1[v as usize].max(1)) as f64).powf(-beta);
        adj_rows[u as usize].push((v, w));
        adj_rows[v as usize].push((u, w));
    }
    let mut c_indptr = vec![0u32];
    let mut c_indices = Vec::new();
    let mut c_weights = Vec::new();
    for row in adj_rows {
        for (v, w) in row {
            c_indices.push(v);
            c_weights.push(w);
        }
        c_indptr.push(c_indices.len() as u32);
    }
    let ref_capped = symmetrize_row_normalized(n, &c_indptr, &c_indices, &c_weights).unwrap();
    assert_eq!(probs_capped.indptr(), ref_capped.indptr());
    assert_eq!(probs_capped.indices(), ref_capped.indices());
    for (&a, &b) in probs_capped.values().iter().zip(ref_capped.values().iter()) {
        assert!((a - b).abs() < 1e-14);
    }

    // 1-hop-only branch (w2 == 0.0 and Some(0)) with degree decay
    let p_w0 = graph_12hop_row_normalized_probabilities_with_degree_decay(&g, 0.0, None, beta, 42)
        .unwrap();
    let p_c0 =
        graph_12hop_row_normalized_probabilities_with_degree_decay(&g, 0.5, Some(0), beta, 42)
            .unwrap();
    assert_eq!(p_w0, p_c0);
    assert!((p_w0.values().iter().sum::<f64>() - 1.0).abs() < 1e-12);

    // Extreme beta = 200.0 underflow test: must NEVER produce NaN/Inf in any branch
    for (w2_t, cap_t) in [(0.0, None), (0.5, None), (0.5, Some(2))] {
        let p_extreme =
            graph_12hop_row_normalized_probabilities_with_degree_decay(&g, w2_t, cap_t, 200.0, 42)
                .unwrap();
        assert!(p_extreme.values().iter().all(|v| v.is_finite() && *v > 0.0));
    }

    // Invalid degree_beta
    assert!(matches!(
        graph_12hop_row_normalized_probabilities_with_degree_decay(&g, 0.5, None, -0.1, 42),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        graph_12hop_row_normalized_probabilities_with_degree_decay(&g, 0.5, None, f64::NAN, 42),
        Err(CoreError::NonFiniteValue { .. })
    ));
}

#[test]
fn test_graph_12hop_row_normalized_probabilities_suppresses_superhub_barrel_mass() {
    // N = 130: super-hub 0 (deg = 120 > max(64, isqrt(130))) + local bridge 1-2-3-4-5 +
    // barrel leaves 6..=120 + separate connected path 121..=129 (zero isolated vertices).
    let n = 130usize;
    let mut edges: Vec<(u32, u32)> = (1..=120).map(|v| (0, v)).collect();
    edges.extend([(1, 2), (2, 3), (3, 4), (4, 5)]);
    for v in 121..129u32 {
        edges.push((v, v + 1));
    }
    let g = CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap();

    let p = graph_12hop_row_normalized_probabilities(&g, 0.5, Some(32), 42).unwrap();
    assert!((p.values().iter().sum::<f64>() - 1.0).abs() < 1e-12);

    // Barrel leaves 6..=120 must only have 1-hop probability mass with super-hub 0 (zero cross-leaf
    // or leaf-to-cluster 2-hop mass).
    for leaf in 6..=120usize {
        let (row_cols, _) = p.row(leaf);
        assert_eq!(
            row_cols,
            &[0],
            "barrel leaf {leaf} leaked 2-hop neighbors: {:?}",
            row_cols
        );
    }
}

#[test]
fn test_sparse_probabilities_from_csr_validation_and_rw_options() {
    let sp = SparseProbabilities::from_csr(
        3,
        vec![0, 1, 3, 4],
        vec![1, 0, 2, 1],
        vec![0.25, 0.25, 0.25, 0.25],
    )
    .unwrap();
    assert_eq!(sp.num_nodes(), 3);
    assert_eq!(sp.nnz(), 4);
    assert_eq!(sp.indptr(), &[0, 1, 3, 4]);
    assert_eq!(sp.row(1), (&[0, 2][..], &[0.25, 0.25][..]));

    // Validation errors
    assert!(SparseProbabilities::from_csr(2, vec![0, 1], vec![1], vec![1.0]).is_err());
    assert!(SparseProbabilities::from_csr(2, vec![1, 1, 1], vec![], vec![]).is_err());
    assert!(SparseProbabilities::from_csr(2, vec![0, 2, 1], vec![1], vec![1.0]).is_err());
    assert!(SparseProbabilities::from_csr(2, vec![0, 1, 1], vec![], vec![]).is_err());
    assert!(SparseProbabilities::from_csr(2, vec![0, 1, 1], vec![1], vec![]).is_err());
    assert!(SparseProbabilities::from_csr(2, vec![0, 1, 1], vec![2], vec![1.0]).is_err());
    assert!(SparseProbabilities::from_csr(2, vec![0, 1, 1], vec![1], vec![-0.1]).is_err());
    assert!(SparseProbabilities::from_csr(2, vec![0, 1, 1], vec![1], vec![f64::NAN]).is_err());
    assert!(SparseProbabilities::from_csr(2, vec![0, 1, 1], vec![1], vec![f64::INFINITY]).is_err());
    // Asymmetric edge (0, 1) missing (1, 0)
    assert!(matches!(
        SparseProbabilities::from_csr(2, vec![0, 1, 1], vec![1], vec![0.5]),
        Err(CoreError::InvalidCsr(_))
    ));
    // Asymmetric probability values (0, 1)=0.5 vs (1, 0)=0.25
    assert!(matches!(
        SparseProbabilities::from_csr(2, vec![0, 1, 2], vec![1, 0], vec![0.5, 0.25]),
        Err(CoreError::InvalidCsr(_))
    ));
    // Unsorted / duplicate column indices within a row
    assert!(matches!(
        SparseProbabilities::from_csr(
            3,
            vec![0, 2, 3, 4],
            vec![2, 1, 0, 0],
            vec![0.25, 0.25, 0.25, 0.25]
        ),
        Err(CoreError::InvalidCsr(_))
    ));
    assert!(matches!(
        SparseProbabilities::from_csr(
            2,
            vec![0, 2, 4],
            vec![1, 1, 0, 0],
            vec![0.25, 0.25, 0.25, 0.25]
        ),
        Err(CoreError::InvalidCsr(_))
    ));

    // graph_12hop_row_normalized_probabilities_with_options with TwoStepRandomWalk
    let mut edges: Vec<(u32, u32)> = (1..=20u32).map(|v| (0, v)).collect();
    edges.extend([(0, 21), (21, 22), (22, 23)]);
    let g = CsrGraph::<f64>::from_edges(24, &edges, None, false, false).unwrap();
    let p_rw = graph_12hop_row_normalized_probabilities_with_options(
        &g,
        0.5,
        Some(4),
        0.2,
        crate::knn::TwoHopSamplingMode::TwoStepRandomWalk,
        None,
        42,
    )
    .unwrap();
    assert!((p_rw.values().iter().sum::<f64>() - 1.0).abs() < 1e-12);
    // Node 0's leaf-filtered pot=1 <= cap=4 deterministically includes 2-hop neighbor 22.
    assert!(p_rw.row(0).0.contains(&22));
    assert!(!sp.is_empty());
    assert_eq!(sp.try_row(1).unwrap(), (&[0, 2][..], &[0.25, 0.25][..]));
    assert!(sp.try_row(3).is_err());
    assert!(SparseProbabilities::empty(3).is_empty());
    let (n, indptr, indices, values) = sp.into_parts();
    assert_eq!(n, 3);
    assert_eq!(indptr, vec![0, 1, 3, 4]);
    assert_eq!(indices, vec![1, 0, 2, 1]);
    assert_eq!(values, vec![0.25, 0.25, 0.25, 0.25]);
}

#[test]
fn test_graph_12hop_row_normalized_supports_unit_weight_csr_graph() {
    let mut edges: Vec<(u32, u32)> = (1..=6).map(|v| (0, v)).collect();
    edges.extend([(6, 7), (7, 8), (8, 9), (1, 2)]);
    let g_unit = CsrGraph::<()>::from_unweighted_edges(10, &edges, false, false).unwrap();
    let g_f64 = CsrGraph::<f64>::from_edges(10, &edges, None, false, false).unwrap();

    let p_unit = graph_12hop_row_normalized_probabilities(&g_unit, 0.5, None, 42).unwrap();
    let p_f64 = graph_12hop_row_normalized_probabilities(&g_f64, 0.5, None, 42).unwrap();
    assert_eq!(p_unit, p_f64);

    let p_unit_dd =
        graph_12hop_row_normalized_probabilities_with_degree_decay(&g_unit, 0.5, Some(3), 0.3, 42)
            .unwrap();
    let p_f64_dd =
        graph_12hop_row_normalized_probabilities_with_degree_decay(&g_f64, 0.5, Some(3), 0.3, 42)
            .unwrap();
    assert_eq!(p_unit_dd, p_f64_dd);

    let p_unit_opt = graph_12hop_row_normalized_probabilities_with_options(
        &g_unit,
        0.5,
        Some(4),
        0.2,
        crate::knn::TwoHopSamplingMode::TwoStepRandomWalk,
        Some(5),
        42,
    )
    .unwrap();
    let p_f64_opt = graph_12hop_row_normalized_probabilities_with_options(
        &g_f64,
        0.5,
        Some(4),
        0.2,
        crate::knn::TwoHopSamplingMode::TwoStepRandomWalk,
        Some(5),
        42,
    )
    .unwrap();
    assert_eq!(p_unit_opt, p_f64_opt);
}
