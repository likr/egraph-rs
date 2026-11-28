//! Opaque-box E2E tests for zero-copy Problem IR accessors across `TsNetSolver`,
//! `TFdpSolver`, `IbUmapSolver`, and `GuMapSolver`.

use egraph_core::{CsrGraph, IterativeSolver, Layout2d, SolverStatus};
use egraph_layout_neighbor::{
    Exact, GuMapConfig, GuMapSolver, IbUmapConfig, IbUmapSolver, TFdpConfig, TFdpSolver,
    TsNetConfig, TsNetSolver, TsNetStage,
};
use egraph_neighborhood::{
    PerplexityConfig, SparseProbabilities, build_pure_2hop_knn, compute_fuzzy_simplicial_set,
    compute_knn_probabilities,
};

fn make_grid(rows: usize, cols: usize) -> CsrGraph<f64> {
    let n = rows * cols;
    let mut edges = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let u = (r * cols + c) as u32;
            if c + 1 < cols {
                edges.push((u, u + 1));
            }
            if r + 1 < rows {
                edges.push((u, u + cols as u32));
            }
        }
    }
    CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap()
}

fn assert_csr_monotone_and_positive_f32(
    n: usize,
    indptr: &[u32],
    indices: &[u32],
    weights: &[f32],
) {
    assert_eq!(indptr.len(), n + 1);
    assert_eq!(indptr[0], 0);
    assert!(indptr.windows(2).all(|w| w[0] <= w[1]));
    assert_eq!(indptr[n] as usize, indices.len());
    assert_eq!(indices.len(), weights.len());
    assert!(indices.iter().all(|&idx| (idx as usize) < n));
    assert!(weights.iter().all(|&w| w.is_finite() && w > 0.0));
}

#[test]
fn tsnet_probabilities_csr_invariants_and_pointer_stability() {
    let g = make_grid(3, 3);
    let n = g.num_nodes();
    let cfg = TsNetConfig::pure_2hop().with_stages(vec![TsNetStage::new().with_iterations(2)]);

    let s_pure = TsNetSolver::<Layout2d<f64>>::pure_2hop(&g, cfg.clone()).unwrap();
    let s_rn =
        TsNetSolver::<Layout2d<f64>>::from_graph_row_normalized(&g, 0.5, Some(8), cfg.clone())
            .unwrap();

    let (knn, _, _) = build_pure_2hop_knn(&g).unwrap();
    let perps = vec![4.0; n];
    let probs = compute_knn_probabilities(&knn, &perps, &PerplexityConfig::default()).unwrap();
    let expected_ip = probs.indptr().to_vec();
    let expected_idx = probs.indices().to_vec();
    let expected_vals = probs.values().to_vec();
    let mut s_probs = TsNetSolver::<Layout2d<f64>>::from_probabilities(probs, cfg).unwrap();

    for solver in [&s_pure, &s_rn, &s_probs] {
        let (indptr, indices, p_vals) = solver.probabilities_csr();
        assert_eq!(indptr.len(), n + 1);
        assert_eq!(indptr[0], 0);
        assert!(indptr.windows(2).all(|w| w[0] <= w[1]));
        assert_eq!(indptr[n] as usize, indices.len());
        assert_eq!(indices.len(), p_vals.len());
        assert!(indices.iter().all(|&idx| (idx as usize) < n));
        assert!(p_vals.iter().all(|&p| p.is_finite() && p >= 0.0));

        let total_p: f64 = p_vals.iter().sum();
        assert!(
            (total_p - 1.0).abs() < 1e-10,
            "expected sum ~ 1.0, got {total_p}"
        );

        // Check sorted per-row indices, no self-loops, and symmetric P_ij == P_ji.
        for u in 0..n {
            let row_slice = &indices[indptr[u] as usize..indptr[u + 1] as usize];
            assert!(row_slice.windows(2).all(|w| w[0] < w[1]));
            assert!(!row_slice.contains(&(u as u32)));
            for pos in indptr[u] as usize..indptr[u + 1] as usize {
                let v = indices[pos] as usize;
                let p_uv = p_vals[pos];
                let rev_start = indptr[v] as usize;
                let rev_end = indptr[v + 1] as usize;
                let rev_off = indices[rev_start..rev_end]
                    .binary_search(&(u as u32))
                    .expect("probabilities_csr must be symmetric");
                assert!((p_uv - p_vals[rev_start + rev_off]).abs() < 1e-14);
            }
        }
    }

    assert_eq!(s_probs.probabilities_csr().0, expected_ip.as_slice());
    assert_eq!(s_probs.probabilities_csr().1, expected_idx.as_slice());
    assert_eq!(s_probs.probabilities_csr().2, expected_vals.as_slice());

    let ptr_before = s_probs.probabilities_csr().2.as_ptr();
    let mut layout = Layout2d::<f64>::init(n, 7, 1.0).unwrap();
    assert_eq!(s_probs.run(&mut layout).unwrap(), SolverStatus::Completed);
    let ptr_after = s_probs.probabilities_csr().2.as_ptr();
    assert_eq!(ptr_before, ptr_after);
}

#[test]
fn tfdp_edge_csr_from_graph_and_knn_and_pointer_stability() {
    let g = make_grid(3, 4);
    let n = g.num_nodes();
    let cfg = TFdpConfig::new().with_iterations(3);

    let mut s_graph =
        TFdpSolver::<Layout2d<f64>, Exact>::new_with_backend(&g, cfg.clone(), Exact::new())
            .unwrap();
    let (indptr_g, indices_g, weights_g) = s_graph.edge_csr();
    assert_csr_monotone_and_positive_f32(n, indptr_g, indices_g, weights_g);
    assert_eq!(weights_g.len(), s_graph.num_edges());

    let ptr_g = weights_g.as_ptr();
    let mut layout = Layout2d::<f64>::init(n, 13, 1.0).unwrap();
    assert_eq!(s_graph.run(&mut layout).unwrap(), SolverStatus::Completed);
    assert_eq!(s_graph.edge_csr().2.as_ptr(), ptr_g);

    let (knn, _, _) = build_pure_2hop_knn(&g).unwrap();
    let s_knn =
        TFdpSolver::<Layout2d<f64>, Exact>::from_knn_with_backend(&knn, cfg, Exact::new()).unwrap();
    let (indptr_k, indices_k, weights_k) = s_knn.edge_csr();
    assert_csr_monotone_and_positive_f32(n, indptr_k, indices_k, weights_k);
    assert_eq!(weights_k.len(), s_knn.num_edges());
}

#[test]
fn ib_umap_edge_csr_hub_eta_ab_params_and_pointer_stability() {
    let g = make_grid(3, 3);
    let (knn, _, _) = build_pure_2hop_knn(&g).unwrap();
    let probs = compute_fuzzy_simplicial_set(&knn, 64, 1e-5).unwrap();
    let degrees = vec![4_u32; 9];

    let mut damped = IbUmapSolver::<Layout2d<f64>>::from_fuzzy_set(
        probs.clone(),
        Some(&degrees),
        IbUmapConfig::new()
            .with_hub_damping(true)
            .with_iterations(2),
    )
    .unwrap();
    let (indptr, indices, weights) = damped.edge_csr();
    assert_csr_monotone_and_positive_f32(9, indptr, indices, weights);
    assert_eq!(weights.len(), damped.num_edges());
    assert_eq!(damped.hub_eta(), vec![0.25_f64; 9].as_slice());
    assert_eq!(damped.ab_params(), damped.ab());

    let ptr_w = weights.as_ptr();
    let ptr_eta = damped.hub_eta().as_ptr();
    let mut layout = Layout2d::<f64>::init(9, 17, 1.0).unwrap();
    assert_eq!(damped.run(&mut layout).unwrap(), SolverStatus::Completed);
    assert_eq!(damped.edge_csr().2.as_ptr(), ptr_w);
    assert_eq!(damped.hub_eta().as_ptr(), ptr_eta);

    let undamped = IbUmapSolver::<Layout2d<f64>>::from_fuzzy_set(
        probs,
        Some(&degrees),
        IbUmapConfig::new()
            .with_hub_damping(false)
            .with_iterations(2),
    )
    .unwrap();
    assert_eq!(undamped.hub_eta(), vec![1.0_f64; 9].as_slice());
    assert_eq!(undamped.ab_params(), undamped.ab());
}

#[test]
fn gumap_edges_ab_params_and_pointer_stability() {
    let g = make_grid(3, 3);
    let cfg = GuMapConfig::new().with_iterations(3).with_seed(13);
    let mut s_dog = GuMapSolver::<Layout2d<f64>>::new(&g, cfg.clone()).unwrap();
    let (src, tgt, w) = s_dog.edges();
    assert_eq!(src.len(), s_dog.num_edges());
    assert_eq!(tgt.len(), s_dog.num_edges());
    assert_eq!(w.len(), s_dog.num_edges());
    assert!(src.iter().all(|&u| (u as usize) < 9));
    assert!(tgt.iter().all(|&v| (v as usize) < 9));
    assert!(w.iter().all(|&x| x.is_finite() && x > 0.0));
    assert_eq!(s_dog.ab_params(), s_dog.ab());

    let ptr_w = w.as_ptr();
    let mut layout = Layout2d::<f64>::init(9, 23, 1.0).unwrap();
    assert_eq!(s_dog.run(&mut layout).unwrap(), SolverStatus::Completed);
    assert_eq!(s_dog.edges().2.as_ptr(), ptr_w);

    let (knn, _, _) = build_pure_2hop_knn(&g).unwrap();
    let probs = compute_fuzzy_simplicial_set(&knn, 64, 1e-5).unwrap();
    let s_fs = GuMapSolver::<Layout2d<f64>>::from_fuzzy_set(probs, cfg).unwrap();
    assert_eq!(s_fs.edges().0.len(), s_fs.num_edges());
    assert_eq!(s_fs.ab_params(), s_fs.ab());
}

#[test]
fn neighbor_solvers_empty_and_singleton_ir_accessors() {
    for n in [0usize, 1] {
        let g = CsrGraph::<f64>::from_edges(n, &[], None, false, false).unwrap();
        let ts = TsNetSolver::<Layout2d<f64>>::pure_2hop(
            &g,
            TsNetConfig::pure_2hop().with_stages(vec![TsNetStage::new().with_iterations(1)]),
        )
        .unwrap();
        let (ts_ip, ts_idx, ts_p) = ts.probabilities_csr();
        assert_eq!(ts_ip, vec![0_u32; n + 1].as_slice());
        assert!(ts_idx.is_empty() && ts_p.is_empty());

        let tf =
            TFdpSolver::<Layout2d<f64>>::new(&g, TFdpConfig::new().with_iterations(1)).unwrap();
        let (tf_ip, tf_idx, tf_w) = tf.edge_csr();
        assert_eq!(tf_ip, vec![0_u32; n + 1].as_slice());
        assert!(tf_idx.is_empty() && tf_w.is_empty());

        let ib =
            IbUmapSolver::<Layout2d<f64>>::new(&g, IbUmapConfig::new().with_iterations(1)).unwrap();
        let (ib_ip, ib_idx, ib_w) = ib.edge_csr();
        assert_eq!(ib_ip, vec![0_u32; n + 1].as_slice());
        assert!(ib_idx.is_empty() && ib_w.is_empty());
        assert_eq!(ib.hub_eta(), vec![1.0_f64; n].as_slice());
        let (a, b) = ib.ab_params();
        assert!(a > 0.0 && b > 0.0);

        let gu =
            GuMapSolver::<Layout2d<f64>>::new(&g, GuMapConfig::new().with_iterations(1)).unwrap();
        let (gu_s, gu_t, gu_w) = gu.edges();
        assert!(gu_s.is_empty() && gu_t.is_empty() && gu_w.is_empty());
        let (ga, gb) = gu.ab_params();
        assert!(ga > 0.0 && gb > 0.0);
    }
}

#[test]
fn neighbor_solvers_isolated_vertex_ir_invariants() {
    // Node 3 is completely isolated in a 4-node graph.
    let g = CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (0, 2)], None, false, false).unwrap();

    let ts = TsNetSolver::<Layout2d<f64>>::pure_2hop(
        &g,
        TsNetConfig::pure_2hop().with_stages(vec![TsNetStage::new().with_iterations(1)]),
    )
    .unwrap();
    let (ts_ip, _, _) = ts.probabilities_csr();
    assert_eq!(ts_ip[3], ts_ip[4]);

    let tf = TFdpSolver::<Layout2d<f64>>::new(&g, TFdpConfig::new().with_iterations(1)).unwrap();
    let (tf_ip, _, _) = tf.edge_csr();
    assert_eq!(tf_ip[3], tf_ip[4]);

    let ib =
        IbUmapSolver::<Layout2d<f64>>::new(&g, IbUmapConfig::new().with_iterations(1)).unwrap();
    let (ib_ip, _, _) = ib.edge_csr();
    assert_eq!(ib_ip[3], ib_ip[4]);
    assert_eq!(ib.hub_eta()[3], 1.0);

    let gu = GuMapSolver::<Layout2d<f64>>::new(&g, GuMapConfig::new().with_iterations(1)).unwrap();
    let (gu_src, gu_tgt, _) = gu.edges();
    assert!(!gu_src.contains(&3) && !gu_tgt.contains(&3));
}

#[test]
fn ib_umap_explicit_degrees_zero_clamped_and_damping_toggle() {
    let probs =
        SparseProbabilities::from_csr(3, vec![0, 1, 2, 2], vec![1, 0], vec![0.5, 0.5]).unwrap();
    let degs = [10_u32, 0, 5];

    let damped = IbUmapSolver::<Layout2d<f64>>::from_fuzzy_set_ref(
        &probs,
        Some(&degs),
        IbUmapConfig::new().with_hub_damping(true),
    )
    .unwrap();
    let eta = damped.hub_eta();
    assert_eq!(eta.len(), 3);
    assert!((eta[0] - 0.1).abs() < 1e-12);
    assert!((eta[1] - 1.0).abs() < 1e-12);
    assert!((eta[2] - 0.2).abs() < 1e-12);

    let undamped = IbUmapSolver::<Layout2d<f64>>::from_fuzzy_set_ref(
        &probs,
        Some(&degs),
        IbUmapConfig::new().with_hub_damping(false),
    )
    .unwrap();
    assert_eq!(undamped.hub_eta(), &[1.0, 1.0, 1.0]);
}
