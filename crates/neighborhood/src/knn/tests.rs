use egraph_core::CsrGraph;

use super::*;

#[test]
fn test_extract_1hop_2hop_and_pure_2hop_knn_exact_disjoint_and_asymmetric() {
    // Star hub 0 connected to 1..=6, plus chain 6-7-8-9 and isolated node 10.
    let mut edges = vec![(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)];
    edges.extend([(6, 7), (7, 8), (8, 9)]);
    let g = CsrGraph::<f64>::from_edges(11, &edges, None, false, false).unwrap();

    let (adj1, adj2) = extract_1hop_2hop_csr(&g).unwrap();
    assert_eq!(adj1.neighbors(0), &[1, 2, 3, 4, 5, 6]);
    // Pure 2-hop of node 0 is {7} only (>=3-hop {8,9} and isolated {10} excluded).
    assert_eq!(adj2.neighbors(0), &[7]);
    // Pure 2-hop of node 1 is {2, 3, 4, 5, 6}.
    assert_eq!(adj2.neighbors(1), &[2, 3, 4, 5, 6]);
    assert!(adj1.neighbors(10).is_empty());
    assert!(adj2.neighbors(10).is_empty());

    let (knn, k2_deg, stats) = build_pure_2hop_knn(&g).unwrap();
    assert_eq!(k2_deg[0], 7);
    assert_eq!(k2_deg[10], 0);
    assert_eq!(stats.n_1hop, adj1.nnz());
    assert_eq!(stats.n_2hop, adj2.nnz());
    assert_eq!(stats.nnz, adj1.nnz() + adj2.nnz());
    assert_eq!(stats.n_truncated_1hop, 0);
    assert_eq!(stats.n_truncated_2hop, 0);
    assert_eq!(stats.n_filled_residual, 0);
    assert_eq!(stats.n_pure_2hop_rows, 11);

    let (row0_idx, row0_dist) = knn.row(0);
    assert_eq!(row0_idx, &[1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(row0_dist, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0]);

    // Directed asymmetric CSR must be symmetrized cleanly without self-loops.
    let dg = CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3)], None, true, false).unwrap();
    let (d1, d2) = extract_1hop_2hop_csr(&dg).unwrap();
    assert_eq!(d1.neighbors(1), &[0, 2]);
    assert_eq!(d2.neighbors(0), &[2]);
    assert_eq!(d2.neighbors(2), &[0]);
}

#[test]
fn test_extract_sparse_12hop_pairs_hub_safe_and_deterministic() {
    // Star graph on 500 nodes: naive 2-hop has 499*498/2 = 124,251 pairs; hub-safe caps it.
    let n = 500;
    let edges: Vec<(u32, u32)> = (1..n as u32).map(|i| (0, i)).collect();
    let g = CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap();

    let p1 = extract_sparse_1hop_2hop_pairs(&g, 8, 42).unwrap();
    let p2 = extract_sparse_1hop_2hop_pairs(&g, 8, 42).unwrap();
    assert_eq!(p1, p2);
    assert_eq!(p1.u1.len(), n - 1);
    assert!(p1.u2.len() <= n * 8);
    for (&a, &b) in p1.u2.iter().zip(p1.v2.iter()) {
        assert!(a < b);
        assert_ne!(a, 0);
    }
}

#[test]
fn test_gumap_12hop_priority_csr_and_validation() {
    let knn =
        build_gumap_12hop_priority_csr(4, &[0, 1], &[1, 2], &[2.0, 3.0], &[0], &[2], &[0.5], 1.2)
            .unwrap();
    // Incident 1-hop max for (0, 2) is max(2.0, 3.0) = 3.0 -> d2 clamped >= 1.2 * 3.0 = 3.6.
    let (r0_idx, r0_dist) = knn.row(0);
    assert_eq!(r0_idx, &[1, 2]);
    assert!((r0_dist[0] - 2.0).abs() < 1e-12);
    assert!(r0_dist[1] >= 3.6 - 1e-12);
    assert!(r0_dist[0] < r0_dist[1]);

    // Test tier2_margin < 1.0 still strictly satisfies d1 < margin * d2 and d1 < d2.
    let knn_lt =
        build_gumap_12hop_priority_csr(3, &[0, 1], &[1, 2], &[2.0, 3.0], &[0], &[2], &[1.0], 0.5)
            .unwrap();
    let d2_lt = knn_lt.row(0).1[1];
    assert!(0.5 * d2_lt > 3.0);

    // Error checks on KnnGraph and priority builder.
    assert!(KnnGraph::new_validated(2, vec![0, 1], vec![1], vec![1.0]).is_err());
    assert!(KnnGraph::new_validated(2, vec![0, 1, 1], vec![1], vec![-0.1]).is_err());
    assert!(KnnGraph::new_validated(2, vec![0, 1, 1], vec![1], vec![f64::NAN]).is_err());
    assert!(KnnGraph::new_validated(2, vec![0, 1, 1], vec![5], vec![1.0]).is_err());
    assert!(build_gumap_12hop_priority_csr(3, &[0], &[1], &[1.0], &[], &[], &[], 0.0).is_err());
}

#[test]
fn test_knn_empty_and_singleton_non_panic() {
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let (e1, e2) = extract_1hop_2hop_csr(&empty).unwrap();
    assert_eq!(e1.num_nodes(), 0);
    assert_eq!(e2.num_nodes(), 0);
    let (ke, deg_e, se) = build_pure_2hop_knn(&empty).unwrap();
    assert_eq!(ke.num_nodes(), 0);
    assert!(deg_e.is_empty());
    assert_eq!(se.nnz, 0);

    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let (ks, deg_s, ss) = build_pure_2hop_knn(&single).unwrap();
    assert_eq!(ks.num_nodes(), 1);
    assert_eq!(deg_s, vec![0]);
    assert_eq!(ss.n_pure_2hop_rows, 1);
}

#[test]
fn test_build_capped_12hop_knn_star_cap_and_ordering() {
    // Star hub 0 connected to 1..=100, plus bridge 100-101.
    let n = 102;
    let mut edges: Vec<(u32, u32)> = (1..=100).map(|i| (0, i)).collect();
    edges.push((100, 101));
    let g = CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap();

    let (knn1, k2_1, stats1) = build_capped_12hop_knn(&g, 8, 42).unwrap();
    let (knn2, k2_2, stats2) = build_capped_12hop_knn(&g, 8, 42).unwrap();
    assert_eq!(knn1, knn2);
    assert_eq!(k2_1, k2_2);
    assert_eq!(stats1, stats2);
    assert_eq!(stats1.n_1hop, 2 * edges.len());
    assert!(stats1.n_2hop <= 2 * n * 8);
    assert!(stats1.n_truncated_2hop > 0);
    for (u, &k2_u) in k2_1.iter().enumerate() {
        let (cols, dists) = knn1.row(u);
        assert_eq!(cols.len(), k2_u as usize);
        assert!(cols.windows(2).all(|w| w[0] < w[1]));
        assert!(dists.iter().all(|&d| d == 1.0 || d == 2.0));
    }

    // max_2hop_per_node = 0 keeps 1-hop only and records truncation
    let (knn_0, _, stats_0) = build_capped_12hop_knn(&g, 0, 42).unwrap();
    assert_eq!(stats_0.n_2hop, 0);
    assert_eq!(stats_0.n_1hop, knn_0.nnz());
    assert!(stats_0.n_truncated_2hop > 0);

    // Empty and singleton
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let (ke, deg_e, se) = build_capped_12hop_knn(&empty, 8, 42).unwrap();
    assert_eq!(ke.num_nodes(), 0);
    assert!(deg_e.is_empty());
    assert_eq!(se.nnz, 0);

    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let (ks, deg_s, ss) = build_capped_12hop_knn(&single, 8, 42).unwrap();
    assert_eq!(ks.num_nodes(), 1);
    assert_eq!(deg_s, vec![0]);
    assert_eq!(ss.n_pure_2hop_rows, 1);
}

#[test]
fn test_extract_sparse_12hop_pairs_degree_order_superhub_suppression_and_perm_invariance() {
    // N = 130 -> super_hub_degree_threshold(130) == 64.
    // Vertex 0 is a low-ID super-hub (deg = 120 > 64) connected to 1..=120.
    // 1..=5 form a local cluster (1-2-3-4-5), plus 5-121-122-123-124 extends outside 0.
    // 6..=120 are pure sink-barrel leaves connected only to 0.
    // 125..=129 form a disjoint path.
    let n = 130usize;
    let mut edges: Vec<(u32, u32)> = (1..=120).map(|v| (0, v)).collect();
    edges.extend([
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 121),
        (121, 122),
        (122, 123),
        (123, 124),
        (125, 126),
        (126, 127),
        (127, 128),
        (128, 129),
    ]);
    let g = CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap();
    let cap = 8usize;
    let (p_base, n_trunc) = extract_sparse_1hop_2hop_pairs_inner(&g, cap, 42).unwrap();
    assert!(n_trunc >= 115);

    let pairs_2: Vec<(u32, u32)> = p_base
        .u2
        .iter()
        .copied()
        .zip(p_base.v2.iter().copied())
        .collect();
    // Local cluster 2-hop pairs must be extracted cleanly.
    for expected in [(1, 3), (2, 4), (3, 5), (4, 121), (5, 122), (121, 123)] {
        assert!(
            pairs_2.contains(&expected),
            "missing local 2-hop pair {expected:?}"
        );
    }
    // Zero leaf-to-cluster or leaf-to-leaf barrel pairs through super-hub 0 (leaves 6..=120).
    for &(a, b) in &pairs_2 {
        assert!(
            !(6..=120).contains(&a) && !(6..=120).contains(&b),
            "barrel leaf leaked into 2-hop pair ({a}, {b})"
        );
    }

    // Apply nontrivial bijections moving super-hub 0 to N - 1 and arbitrary vertex IDs.
    let perm_reverse: Vec<u32> = (0..n as u32).map(|u| (n as u32 - 1) - u).collect();
    // gcd(73, 130) == 1, sends 0 -> 129 = N - 1.
    let perm_affine: Vec<u32> = (0..n as u32).map(|u| (u * 73 + 129) % (n as u32)).collect();
    for perm in [&perm_reverse, &perm_affine] {
        let perm_edges: Vec<(u32, u32)> = edges
            .iter()
            .map(|&(u, v)| (perm[u as usize], perm[v as usize]))
            .collect();
        let g_perm = CsrGraph::<f64>::from_edges(n, &perm_edges, None, false, false).unwrap();
        let p_perm = extract_sparse_1hop_2hop_pairs(&g_perm, cap, 42).unwrap();

        let map_pairs = |us: &[u32], vs: &[u32]| -> Vec<(u32, u32)> {
            let mut mapped: Vec<(u32, u32)> = us
                .iter()
                .zip(vs.iter())
                .map(|(&u, &v)| {
                    let pu = perm[u as usize];
                    let pv = perm[v as usize];
                    if pu < pv { (pu, pv) } else { (pv, pu) }
                })
                .collect();
            mapped.sort_unstable();
            mapped
        };

        let expected_1 = map_pairs(&p_base.u1, &p_base.v1);
        let actual_1: Vec<(u32, u32)> = p_perm
            .u1
            .iter()
            .copied()
            .zip(p_perm.v1.iter().copied())
            .collect();
        assert_eq!(actual_1, expected_1, "1-hop permutation invariance failed");

        let expected_2 = map_pairs(&p_base.u2, &p_base.v2);
        let actual_2: Vec<(u32, u32)> = p_perm
            .u2
            .iter()
            .copied()
            .zip(p_perm.v2.iter().copied())
            .collect();
        assert_eq!(actual_2, expected_2, "2-hop permutation invariance failed");
    }
}

#[test]
fn test_extract_sparse_12hop_pairs_degree_tie_and_medium_hub_boundary() {
    // N = 100 -> super_hub_degree_threshold(100) == 64, cap = 8.
    // Vertex 0 connects to:
    // - low-ID medium hub 1 (deg(1) = 40 <= 64, connected to 0 and 2..=40)
    // - high-ID low-degree bridge 90 (deg(90) = 5 <= 8, connected to 0 and 91..=94)
    let n = 100usize;
    let mut edges: Vec<(u32, u32)> = vec![(0, 1), (0, 90)];
    for v in 2..=40u32 {
        edges.push((1, v));
    }
    for v in 91..=94u32 {
        edges.push((90, v));
    }
    let g = CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap();
    let (p1, trunc1) = extract_sparse_1hop_2hop_pairs_inner(&g, 8, 42).unwrap();
    let (p2, trunc2) = extract_sparse_1hop_2hop_pairs_inner(&g, 8, 42).unwrap();
    assert_eq!(p1, p2);
    assert_eq!(trunc1, trunc2);
    assert!(trunc1 > 0);

    let pairs_2: Vec<(u32, u32)> = p1.u2.iter().copied().zip(p1.v2.iter().copied()).collect();
    // Low-degree bridge 90 (deg=5) MUST be expanded before medium-hub 1 (deg=40).
    for v in 91..=94u32 {
        assert!(
            pairs_2.contains(&(0, v)),
            "expected low-degree bridge 2-hop pair (0, {v})"
        );
    }
    // Vertex 0 collects 4 pairs from bridge 90 + 4 sampled pairs from medium hub 1 (and
    // undirected symmetrization may include additional pairs sampled by leaves 2..=40).
    let zero_deg2 = pairs_2.iter().filter(|&&(a, b)| a == 0 || b == 0).count();
    assert!((8..=(8 + 39)).contains(&zero_deg2));
    assert!(p1.u2.len() <= n * 8);

    // Even at cap = 5 (matching deg(90) = 5), low-degree bridge 90 MUST fill all 4 exact
    // neighbors before hub 1 (deg = 40) is sampled.
    let p_cap5 = extract_sparse_1hop_2hop_pairs(&g, 5, 42).unwrap();
    let pairs_cap5: Vec<(u32, u32)> = p_cap5
        .u2
        .iter()
        .copied()
        .zip(p_cap5.v2.iter().copied())
        .collect();
    for v in 91..=94u32 {
        assert!(
            pairs_cap5.contains(&(0, v)),
            "cap=5 must still select all low-degree bridge neighbors (0, {v})"
        );
    }

    // Exact boundary check for N = 100 (threshold = 64):
    // deg = 64 is retained as a medium hub (emits 2-hop pairs for node 0),
    // while deg = 65 is unconditionally suppressed (emits 0 barrel pairs for node 0).
    let edges_64: Vec<(u32, u32)> = std::iter::once((0, 1))
        .chain((2..=64u32).map(|v| (1, v)))
        .collect();
    let g_64 = CsrGraph::<f64>::from_edges(n, &edges_64, None, false, false).unwrap();
    let p_64 = extract_sparse_1hop_2hop_pairs(&g_64, 8, 42).unwrap();
    assert!(
        p_64.u2
            .iter()
            .zip(p_64.v2.iter())
            .any(|(&a, &b)| a == 0 || b == 0),
        "deg=64 should remain below/at threshold and emit 2-hop pairs"
    );

    let edges_65: Vec<(u32, u32)> = std::iter::once((0, 1))
        .chain((2..=65u32).map(|v| (1, v)))
        .collect();
    let g_65 = CsrGraph::<f64>::from_edges(n, &edges_65, None, false, false).unwrap();
    let (p_65, trunc_65) = extract_sparse_1hop_2hop_pairs_inner(&g_65, 8, 42).unwrap();
    assert!(p_65.u2.is_empty(), "deg=65 must be suppressed as super-hub");
    assert!(trunc_65 > 0);
}
