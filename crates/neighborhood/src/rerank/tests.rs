use egraph_core::{CsrGraph, DeterministicRng};

use super::*;
use crate::knn::extract_1hop_2hop_csr;

#[test]
fn test_resistance_and_diffusion_embeddings() {
    let edges: Vec<(u32, u32)> = (0..20).map(|i| (i, (i + 1) % 20)).collect();
    let cycle = CsrGraph::<f64>::from_edges(20, &edges, None, false, false).unwrap();

    let (res_coords, res_lams) =
        compute_resistance_embedding(&cycle, 8, 1e-6, 300, 42, 1e-10).unwrap();
    assert_eq!(res_lams.len(), 8);
    assert_eq!(res_coords.len(), 20 * 8);
    assert!(res_coords.iter().all(|v| v.is_finite()));

    let (diff_coords, diff_lams, t_used) =
        compute_diffusion_embedding(&cycle, 8, 0.1, None, 1e-6, 300, 42).unwrap();
    assert_eq!(diff_lams.len(), 8);
    assert_eq!(diff_coords.len(), 20 * 8);
    assert!(t_used > 0.0 && t_used.is_finite());
    assert!(diff_coords.iter().all(|v| v.is_finite()));

    // Disconnected graph: zero eigenvalue mode filtered cleanly in resistance embedding.
    let disc = CsrGraph::<f64>::from_edges(
        6,
        &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)],
        None,
        false,
        false,
    )
    .unwrap();
    let (rc_disc, rl_disc) = compute_resistance_embedding(&disc, 4, 1e-6, 200, 42, 1e-10).unwrap();
    assert_eq!(rl_disc.len(), 3);
    assert_eq!(rc_disc.len(), 6 * 3);

    // Edgeless graph -> zero nonzero eigenvalues -> empty coords.
    let iso = CsrGraph::<f64>::from_edges(5, &[], None, false, false).unwrap();
    let (rc_iso, rl_iso) = compute_resistance_embedding(&iso, 3, 1e-6, 100, 42, 1.5).unwrap();
    assert!(rl_iso.is_empty());
    assert!(rc_iso.is_empty());
}

#[test]
fn test_rerank_knn_exact_and_two_stage_kdtree() {
    let n = 150;
    let dim = 14;
    let mut rng = DeterministicRng::from_seed_u64(123);
    let mut coords = Vec::with_capacity(n * dim);
    for _ in 0..n {
        for d in 0..dim {
            // First few dimensions dominate slightly so coarse_dim=10 recovers exact k-NN well.
            let scale = if d < 10 { 2.0 } else { 0.05 };
            coords.push((rng.next_f64() - 0.5) * scale);
        }
    }

    // Exact fallback when cand_k >= n - 1
    let exact = build_rerank_knn(&coords, n, dim, 12, 10, 16).unwrap();
    // 2-stage KD-tree when cand_k < n - 1 (k=5, rerank=4 => cand_k=20 < 149)
    let two_stage = build_rerank_knn(&coords, n, dim, 5, 10, 4).unwrap();
    assert_eq!(two_stage.nnz(), n * 5);

    for i in 0..n {
        let (e_idx, e_d) = exact.row(i);
        let (t_idx, t_d) = two_stage.row(i);
        assert_eq!(t_idx, &e_idx[..5]);
        for (&a, &b) in t_d.iter().zip(e_d[..5].iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}

#[test]
fn test_hybrid_adaptive_knn_preserves_12hop_and_margin_gap() {
    // Hub 0 connected to 1..=12; chain 12-13-...-24; isolated 25 (N=26).
    let mut edges: Vec<(u32, u32)> = (1..=12).map(|i| (0, i)).collect();
    edges.extend((12..24).map(|i| (i, i + 1)));
    let g = CsrGraph::<f64>::from_edges(26, &edges, None, false, false).unwrap();

    let (coords, _) = compute_resistance_embedding(&g, 12, 1e-6, 300, 42, 1e-10).unwrap();
    let dim = coords.len() / 26;
    let mut perps = vec![4.0; 26];
    // Force small target k=6 on hub 0 (which has 13 1+2-hop neighbors) -> must NOT truncate!
    perps[0] = 2.0;

    let cfg = HybridKnnConfig {
        k_factor: 3.0,
        k_override: None,
        margin_gap: 0.15,
        coarse_dim: 8,
        rerank_factor: 4,
        max_row_k: None,
    };
    let (knn, stats) = build_hybrid_adaptive_knn(&g, &coords, dim, &perps, &cfg).unwrap();
    assert_eq!(stats.n_truncated_1hop, 0);
    assert_eq!(stats.n_truncated_2hop, 0);
    assert!(stats.n_pure_2hop_rows >= 1);
    assert_eq!(knn.row(0).0.len(), 13);

    // Verify strict max(d_{1,2-hop}) < min(d_{non-2}) on rows with residual fills.
    let (adj1, adj2) = extract_1hop_2hop_csr(&g).unwrap();
    for i in 0..26 {
        let deg12 = adj1.degree(i) + adj2.degree(i);
        let (row_idx, row_d) = knn.row(i);
        assert!(row_idx.len() >= deg12);
        if deg12 > 0 && row_idx.len() > deg12 {
            let max_12 = row_d[..deg12].iter().copied().fold(0.0_f64, f64::max);
            let min_non2 = row_d[deg12..].iter().copied().fold(f64::INFINITY, f64::min);
            assert!(max_12 < min_non2, "node {i}: {max_12} vs {min_non2}");
            assert!(min_non2 >= max_12 * 1.15 - 1e-12);
        }
    }
    // Isolated node 25 gets round(3 * 4) = 12 neighbors.
    assert_eq!(knn.row(25).0.len(), 12);

    // Twin nodes with d_2hop_max == 0.0 still enforce min_non2 > 0.
    let tri_chain = CsrGraph::<f64>::from_edges(
        6,
        &[(0, 1), (1, 2), (0, 2), (3, 4), (4, 5)],
        None,
        false,
        false,
    )
    .unwrap();
    let twin_coords = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0];
    let (knn_t, _) = build_hybrid_adaptive_knn(
        &tri_chain,
        &twin_coords,
        2,
        &[1.5; 6],
        &HybridKnnConfig::default(),
    )
    .unwrap();
    for i in 0..3 {
        let (_ri, rd) = knn_t.row(i);
        assert_eq!(rd[0], 0.0);
        assert_eq!(rd[1], 0.0);
        assert!(rd[2] > 0.0);
    }
}

#[test]
fn test_rerank_and_hybrid_empty_singleton_and_nan_rejection() {
    let empty_knn = build_rerank_knn(&[], 0, 2, 4, 2, 2).unwrap();
    assert_eq!(empty_knn.num_nodes(), 0);
    assert_eq!(empty_knn.nnz(), 0);

    let single_knn = build_rerank_knn(&[1.0, 2.0], 1, 2, 4, 2, 2).unwrap();
    assert_eq!(single_knn.num_nodes(), 1);
    assert_eq!(single_knn.nnz(), 0);

    assert!(build_rerank_knn(&[0.0, f64::NAN, 1.0, 2.0], 2, 2, 1, 2, 2).is_err());

    let g = CsrGraph::<f64>::from_edges(4, &[(0, 1), (2, 3)], None, false, false).unwrap();
    let bad_coords = vec![0.0, 0.0, 1.0, f64::NAN, 2.0, 0.0, 3.0, 0.0];
    assert!(
        build_hybrid_adaptive_knn(&g, &bad_coords, 2, &[2.0; 4], &HybridKnnConfig::default())
            .is_err()
    );
    assert!(
        build_hybrid_adaptive_knn(
            &g,
            &[0.0; 8],
            2,
            &[f64::NAN; 4],
            &HybridKnnConfig::default()
        )
        .is_err()
    );
}
