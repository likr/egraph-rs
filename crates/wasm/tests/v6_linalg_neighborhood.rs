use egraph_wasm::v6::{
    JsCsrGraph, JsKnnGraph, js_build_hybrid_adaptive_knn, js_build_pure_2hop_knn,
    js_chebyshev_dog_jl_embedding, js_chebyshev_heat_spmv, js_compute_2hop_neighborhood_sizes,
    js_compute_fuzzy_simplicial_set, js_compute_horvitz_thompson_gamma,
    js_compute_knn_probabilities, js_compute_median_2hop_perplexity,
    js_compute_nodewise_2hop_perplexities, js_compute_regularized_dog_varadhan_landmarks,
    js_extract_1hop_2hop_csr, js_find_ab_params, js_koren_spectral_init, js_partial_bfs_knn,
    js_pivot_mds_init, js_select_maxmin_sp_pivots, js_smallest_laplacian_eigenpairs,
    js_spectral_moment_stats,
};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

fn make_grid(rows: u32, cols: u32) -> JsCsrGraph {
    let mut edges = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let u = r * cols + c;
            if c + 1 < cols {
                edges.push(u);
                edges.push(u + 1);
            }
            if r + 1 < rows {
                edges.push(u);
                edges.push(u + cols);
            }
        }
    }
    JsCsrGraph::from_edge_list(
        (rows * cols) as f64,
        &edges,
        None,
        &JsValue::UNDEFINED,
        None,
    )
    .expect("grid graph")
}

fn make_hub_chain_graph() -> JsCsrGraph {
    // Hub 0 connected to 1..=8; chain 8-9-10-11; isolated node 12 (N=13).
    let mut edges = Vec::new();
    for i in 1..=8u32 {
        edges.push(0);
        edges.push(i);
    }
    edges.extend([8, 9, 9, 10, 10, 11]);
    JsCsrGraph::from_edge_list(13.0, &edges, None, &JsValue::UNDEFINED, None)
        .expect("hub chain graph")
}

fn layout_global_std(layout: &egraph_wasm::v6::JsLayout2d) -> f64 {
    let n = layout.len();
    assert!(n >= 2);
    let mut sx = 0.0_f64;
    let mut sy = 0.0_f64;
    for i in 0..n {
        sx += layout.x(i as f64).unwrap() as f64;
        sy += layout.y(i as f64).unwrap() as f64;
    }
    let mx = sx / (n as f64);
    let my = sy / (n as f64);
    let mut sum_sq = 0.0_f64;
    for i in 0..n {
        let dx = (layout.x(i as f64).unwrap() as f64) - mx;
        let dy = (layout.y(i as f64).unwrap() as f64) - my;
        sum_sq += dx * dx + dy * dy;
    }
    (sum_sq / (2.0 * (n as f64))).sqrt()
}

#[wasm_bindgen_test]
fn linalg_pivots_mds_koren_eigen_and_moments() {
    let grid = make_grid(4, 4);

    // 1. selectMaxminSpPivots
    let p1 = js_select_maxmin_sp_pivots(&grid, Some(6.0), None, None).expect("pivots default");
    let p2 =
        js_select_maxmin_sp_pivots(&grid, Some(6.0), Some(42.0), Some(2.0)).expect("pivots seeded");
    assert_eq!(p1, p2);
    assert_eq!(p1.len(), 6);
    assert_eq!(
        js_select_maxmin_sp_pivots(&grid, Some(0.0), None, None)
            .unwrap()
            .len(),
        0
    );

    // 2. pivotMdsInit: default targetStd=1.0 vs explicit vs null (unnormalized)
    let mds_default =
        js_pivot_mds_init(&grid, Some(8.0), None, None, &JsValue::UNDEFINED).expect("mds default");
    assert_eq!(mds_default.len(), 16);
    assert!(mds_default.is_all_finite());
    assert!((layout_global_std(&mds_default) - 1.0).abs() < 1e-4);

    let mds_scaled = js_pivot_mds_init(
        &grid,
        Some(8.0),
        Some(42.0),
        Some(2.0),
        &JsValue::from_f64(2.5),
    )
    .expect("mds 2.5");
    assert!((layout_global_std(&mds_scaled) - 2.5).abs() < 1e-4);

    let mds_unnorm = js_pivot_mds_init(&grid, Some(8.0), Some(42.0), Some(2.0), &JsValue::NULL)
        .expect("mds unnormalized");
    assert!(mds_unnorm.is_all_finite());
    assert!((layout_global_std(&mds_unnorm) - 1.0).abs() > 0.05);

    // 3. korenSpectralInit: default targetStd=1.0 vs null
    let koren_default =
        js_koren_spectral_init(&grid, &JsValue::UNDEFINED, None, None, None).expect("koren");
    assert_eq!(koren_default.len(), 16);
    assert!(koren_default.is_all_finite());
    assert!((layout_global_std(&koren_default) - 1.0).abs() < 1e-4);

    let koren_unnorm =
        js_koren_spectral_init(&grid, &JsValue::NULL, Some(1e-5), Some(300.0), Some(42.0))
            .expect("koren unnorm");
    assert!(koren_unnorm.is_all_finite());

    // 4. smallestLaplacianEigenpairs: dense fallback (16 nodes) and Lanczos (60 nodes)
    let eig_small =
        js_smallest_laplacian_eigenpairs(&grid, Some(6.0), None, None, None).expect("eig small");
    assert_eq!(eig_small.num_nodes(), 16);
    assert_eq!(eig_small.num_modes(), 6);
    assert!(eig_small.used_dense_fallback());
    let lams = eig_small.lambdas();
    let vecs = eig_small.vectors();
    let inv_d = eig_small.inv_sqrt_deg();
    assert_eq!(lams.len(), 6);
    assert_eq!(vecs.len(), 16 * 6);
    assert_eq!(inv_d.len(), 16);
    assert!(lams.windows(2).all(|w| w[0] <= w[1] + 1e-12));
    assert!(lams[0] > 0.0);

    let grid_large = make_grid(6, 10);
    let eig_large = js_smallest_laplacian_eigenpairs(&grid_large, Some(4.0), None, None, None)
        .expect("eig lanczos");
    assert_eq!(eig_large.num_nodes(), 60);
    assert_eq!(eig_large.num_modes(), 4);
    assert!(!eig_large.used_dense_fallback());
    assert_eq!(eig_large.lambdas().len(), 4);
    assert_eq!(eig_large.vectors().len(), 60 * 4);

    // 5. spectralMomentStats
    let stats = js_spectral_moment_stats(&grid);
    assert!(stats.sigma_lambda_sq() > 0.0);
    assert!((stats.sigma_lambda() - stats.sigma_lambda_sq().sqrt()).abs() < 1e-12);
    assert!(stats.deg_cv() >= 0.0);
    assert!((stats.mean_degree() - 3.0).abs() < 1e-12);
    assert_eq!(stats.leaf_fraction(), 0.0);
}

#[wasm_bindgen_test]
fn chebyshev_heat_spmv_and_dog_jl_embedding() {
    let grid = make_grid(4, 4);
    let eig = js_smallest_laplacian_eigenpairs(&grid, Some(2.0), None, None, None).unwrap();
    let inv_d = eig.inv_sqrt_deg();

    // Stationary vector v = D^{1/2} * 1 is in nullspace of L_sym => exp(-t L_sym) v == v
    let mut vecs = Vec::with_capacity(32);
    for &id in &inv_d {
        let deg_sqrt = if id > 0.0 { 1.0 / id } else { 0.0 };
        vecs.push(deg_sqrt);
        vecs.push(1.0);
    }

    let diffused = js_chebyshev_heat_spmv(&grid, &vecs, 2.0, 2.0, None, None).expect("heat spmv");
    assert_eq!(diffused.len(), 32);
    for i in 0..16 {
        let deg_sqrt = vecs[i * 2];
        assert!((diffused[i * 2] - deg_sqrt).abs() < 1e-10);
    }

    // t = 0.0 acts as identity
    let ident =
        js_chebyshev_heat_spmv(&grid, &vecs, 2.0, 0.0, Some(10.0), Some(2.0)).expect("heat t=0");
    for (&a, &b) in ident.iter().zip(vecs.iter()) {
        assert!((a - b).abs() < 1e-12);
    }

    // 7. chebyshevDogJlEmbedding determinism and degreeNormalize difference
    let emb1 =
        js_chebyshev_dog_jl_embedding(&grid, Some(12.0), None, None, None, None, None, None, None)
            .expect("jl default");
    let emb2 = js_chebyshev_dog_jl_embedding(
        &grid,
        Some(12.0),
        Some(25.0),
        Some(42.0),
        Some(true),
        Some(1e-4),
        None,
        None,
        None,
    )
    .expect("jl explicit");
    assert_eq!(emb1.len(), 16 * 12);
    assert_eq!(emb1, emb2);
    assert!(emb1.iter().all(|v| v.is_finite()));

    let emb_unnorm = js_chebyshev_dog_jl_embedding(
        &grid,
        Some(12.0),
        Some(25.0),
        Some(42.0),
        Some(false),
        Some(1e-4),
        Some(0.5),
        Some(2.0),
        Some(0.4),
    )
    .expect("jl unnorm");
    assert_ne!(emb1, emb_unnorm);
}

#[wasm_bindgen_test]
fn neighborhood_2hop_hybrid_knn_probabilities_fuzzy_and_varadhan() {
    let g = make_hub_chain_graph();

    // 8. extract1hop2hopCsr (directed input symmetrized too)
    let two_hop = js_extract_1hop_2hop_csr(&g).expect("1+2hop csr");
    let h1 = two_hop.hop1();
    let h2 = two_hop.hop2();
    assert_eq!(h1.num_nodes(), 13);
    assert_eq!(h2.num_nodes(), 13);
    assert_eq!(h1.num_edges(), 11);
    assert!(h2.num_edges() > 0);

    // 9. buildPure2hopKnn & 10. compute2hopNeighborhoodSizes
    let pure = js_build_pure_2hop_knn(&g).expect("pure 2hop knn");
    let knn_pure = pure.knn();
    let k2_deg = pure.k2_deg();
    let sizes = js_compute_2hop_neighborhood_sizes(&g);
    assert_eq!(k2_deg, sizes);
    assert_eq!(sizes[0], 9); // 1..=8 (1-hop) + 9 (2-hop)
    assert_eq!(sizes[12], 0); // isolated node 12
    assert_eq!(knn_pure.num_nodes(), 13);
    assert_eq!(knn_pure.nnz(), pure.nnz());
    assert!(pure.avg_deg2() >= pure.avg_deg1());
    assert!(knn_pure.distances().iter().all(|&d| d == 1.0 || d == 2.0));

    // KnnGraph.fromCsr round-trip
    let knn_rt = JsKnnGraph::from_csr(
        knn_pure.num_nodes() as f64,
        &knn_pure.indptr().unwrap(),
        &knn_pure.indices(),
        &knn_pure.distances(),
    )
    .expect("KnnGraph.fromCsr");
    assert_eq!(knn_rt.num_nodes(), knn_pure.num_nodes());
    assert_eq!(knn_rt.nnz(), knn_pure.nnz());
    assert_eq!(knn_rt.indptr().unwrap(), knn_pure.indptr().unwrap());
    assert_eq!(knn_rt.indices(), knn_pure.indices());
    assert_eq!(knn_rt.distances(), knn_pure.distances());

    // 11 & 12. Perplexities
    let node_perps =
        js_compute_nodewise_2hop_perplexities(&g, None, None, None).expect("nodewise perp");
    assert_eq!(node_perps.len(), 13);
    let upper = 12.0 / 3.0; // (13 - 1) / 3 = 4.0
    for &p in &node_perps {
        assert!((p - upper).abs() < 1e-12);
    }
    let med_perp = js_compute_median_2hop_perplexity(&g, None, None, None).expect("median perp");
    assert!((med_perp - upper).abs() < 1e-12);

    // 13. buildHybridAdaptiveKnn
    let coords =
        js_chebyshev_dog_jl_embedding(&g, Some(6.0), None, None, None, None, None, None, None)
            .unwrap();
    let hybrid = js_build_hybrid_adaptive_knn(
        &g,
        &coords,
        6.0,
        &node_perps,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .expect("hybrid knn");
    let h_knn = hybrid.knn();
    assert_eq!(h_knn.num_nodes(), 13);
    assert_eq!(h_knn.nnz(), hybrid.nnz());
    assert!(hybrid.avg_row_k() >= hybrid.avg_deg2());
    assert!(hybrid.n_filled_residual() >= 1); // isolated node 12 + sparse nodes
    assert_eq!(hybrid.n_filled_residual() + hybrid.n_pure_2hop_rows(), 13);

    // 14. partialBfsKnn
    let pbfs1 = js_partial_bfs_knn(&g, Some(5.0), Some(42.0)).expect("partial bfs");
    let pbfs2 = js_partial_bfs_knn(&g, Some(5.0), None).expect("partial bfs default seed");
    assert_eq!(pbfs1, pbfs2);
    assert_eq!(pbfs1.num_nodes(), 13);

    // 15. computeKnnProbabilities
    let probs = js_compute_knn_probabilities(&h_knn, &node_perps, None, None, None, None)
        .expect("knn probs");
    assert_eq!(probs.num_nodes(), 13);
    assert_eq!(probs.nnz(), probs.indices().len());
    assert_eq!(probs.nnz(), probs.values().len());
    assert_eq!(probs.indptr().unwrap().len(), 14);
    let total_p: f64 = probs.values().iter().sum();
    assert!((total_p - 1.0).abs() < 1e-10);

    let probs_rn = js_compute_knn_probabilities(
        &h_knn,
        &node_perps,
        Some(2.0),
        Some(50.0),
        Some(1e-5),
        Some(true),
    )
    .expect("knn probs row_normalize");
    let total_rn: f64 = probs_rn.values().iter().sum();
    assert!((total_rn - 1.0).abs() < 1e-10);

    // 16. computeFuzzySimplicialSet
    let fuzzy = js_compute_fuzzy_simplicial_set(&h_knn, None, None).expect("fuzzy set");
    assert_eq!(fuzzy.num_nodes(), 13);
    assert!(fuzzy.values().iter().all(|&w| w > 1e-7 && w <= 1.0));

    // 17. findAbParams
    let ab = js_find_ab_params(None, None).expect("findAbParams");
    assert_eq!(ab.len(), 2);
    assert!((ab[0] - 1.577).abs() < 0.02);
    assert!((ab[1] - 0.895).abs() < 0.02);

    // 18. computeHorvitzThompsonGamma
    let gamma = js_compute_horvitz_thompson_gamma(1.0, 1600.0, 16.0).expect("ht gamma");
    assert!((gamma - 10.0).abs() < 1e-12);
    assert_eq!(
        js_compute_horvitz_thompson_gamma(1.0, 0.0, 16.0).unwrap(),
        0.0
    );

    // 19. computeRegularizedDogVaradhanLandmarks
    let lm1 = js_compute_regularized_dog_varadhan_landmarks(
        &g, None, None, None, None, None, None, None, None, None, None, None,
    )
    .expect("varadhan default");
    let lm2 = js_compute_regularized_dog_varadhan_landmarks(
        &g,
        None,
        Some(64.0),
        Some(42.0),
        Some(true),
        Some(0.5),
        Some(1e-4),
        Some(48.0),
        None,
        None,
        None,
        None,
    )
    .expect("varadhan explicit");
    let lm_deg25 = js_compute_regularized_dog_varadhan_landmarks(
        &g,
        Some(25.0),
        Some(64.0),
        Some(42.0),
        Some(true),
        Some(0.5),
        Some(1e-4),
        Some(48.0),
        None,
        None,
        None,
        None,
    )
    .expect("varadhan explicit degree=25");
    assert_eq!(lm_deg25.num_nodes(), 13);
    assert_eq!(lm1.num_nodes(), 13);
    assert_eq!(lm1.num_landmarks(), 13); // min(64, 13)
    assert_eq!(lm1.landmarks(), lm2.landmarks());
    assert_eq!(lm1.table(), lm2.table());
    assert_eq!(lm1.coarse_table(), lm2.coarse_table());
    assert!(lm1.used_coarse_scaffold());
    assert!(lm1.coarse_table().is_some());
    assert!(lm1.t1() > 0.0 && lm1.t2() > lm1.t1());
    assert!(lm1.alpha() >= 0.0);
    assert!(lm1.delta_psd() >= 0.0);
    assert!(lm1.beta_eff() >= 1e-4 && lm1.beta_eff() <= 0.5);
    assert!(lm1.beta_auto() >= 1e-4 && lm1.beta_auto() <= 0.5);

    let alt = lm1.alt_distances(&[0, 8], &[8, 0]).expect("alt distances");
    assert_eq!(alt.len(), 2);
    assert!((alt[0] - alt[1]).abs() < 1e-12);
    assert!(alt[0] >= 0.0 && alt[0].is_finite());

    let lm_no_coarse = js_compute_regularized_dog_varadhan_landmarks(
        &g,
        Some(20.0),
        Some(6.0),
        Some(7.0),
        Some(false),
        Some(0.5),
        Some(1e-4),
        Some(0.0),
        None,
        None,
        None,
        None,
    )
    .expect("varadhan no coarse");
    assert!(!lm_no_coarse.used_coarse_scaffold());
    assert!(lm_no_coarse.coarse_table().is_none());
}

#[wasm_bindgen_test]
fn empty_singleton_directed_and_error_handling() {
    let empty =
        JsCsrGraph::from_edge_list(0.0, &[], None, &JsValue::UNDEFINED, None).expect("empty");
    let single =
        JsCsrGraph::from_edge_list(1.0, &[], None, &JsValue::UNDEFINED, None).expect("single");

    assert!(
        js_select_maxmin_sp_pivots(&empty, Some(5.0), None, None)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        js_pivot_mds_init(&empty, None, None, None, &JsValue::UNDEFINED)
            .unwrap()
            .len(),
        0
    );
    assert_eq!(
        js_pivot_mds_init(&single, None, None, None, &JsValue::UNDEFINED)
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        js_koren_spectral_init(&empty, &JsValue::UNDEFINED, None, None, None)
            .unwrap()
            .len(),
        0
    );
    assert_eq!(
        js_koren_spectral_init(&single, &JsValue::UNDEFINED, None, None, None)
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        js_smallest_laplacian_eigenpairs(&empty, Some(4.0), None, None, None)
            .unwrap()
            .num_modes(),
        0
    );
    assert_eq!(
        js_smallest_laplacian_eigenpairs(&single, Some(4.0), None, None, None)
            .unwrap()
            .num_modes(),
        0
    );
    assert!(
        js_chebyshev_heat_spmv(&empty, &[], 1.0, 1.0, None, None)
            .unwrap()
            .is_empty()
    );
    assert!(
        js_chebyshev_dog_jl_embedding(&empty, Some(8.0), None, None, None, None, None, None, None)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        js_extract_1hop_2hop_csr(&empty).unwrap().hop1().num_nodes(),
        0
    );
    assert_eq!(js_build_pure_2hop_knn(&single).unwrap().nnz(), 0);
    assert!(js_compute_2hop_neighborhood_sizes(&empty).is_empty());
    assert!(
        js_compute_nodewise_2hop_perplexities(&empty, None, None, None)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        js_compute_median_2hop_perplexity(&single, None, None, None).unwrap(),
        1.0
    );
    assert_eq!(
        js_partial_bfs_knn(&single, Some(5.0), None).unwrap().nnz(),
        0
    );
    assert_eq!(
        js_compute_regularized_dog_varadhan_landmarks(
            &empty, None, None, None, None, None, None, None, None, None, None, None,
        )
        .unwrap()
        .num_nodes(),
        0
    );

    // Directed graph symmetrized cleanly by 1+2-hop
    let directed = JsCsrGraph::from_edge_list(
        4.0,
        &[0, 1, 1, 2, 2, 3],
        None,
        &JsValue::from_bool(true),
        Some(false),
    )
    .unwrap();
    let d_two = js_extract_1hop_2hop_csr(&directed).unwrap();
    assert!(!d_two.hop1().is_directed());
    assert_eq!(d_two.hop1().num_edges(), 3);
    assert_eq!(d_two.hop2().num_edges(), 2);

    // Adversarial error checks (NaN, bounds, mismatch, seed validation)
    let grid = make_grid(3, 3);
    assert!(js_select_maxmin_sp_pivots(&grid, Some(-1.0), None, None).is_err());
    assert!(js_select_maxmin_sp_pivots(&grid, Some(1.5), None, None).is_err());
    assert!(js_select_maxmin_sp_pivots(&grid, None, Some(-1.0), None).is_err());
    assert!(js_select_maxmin_sp_pivots(&grid, None, Some(1.5), None).is_err());
    assert!(js_select_maxmin_sp_pivots(&grid, None, Some(9007199254740992.0), None).is_err());

    assert!(js_pivot_mds_init(&grid, Some(0.0), None, None, &JsValue::UNDEFINED).is_err());
    assert!(js_pivot_mds_init(&grid, None, None, None, &JsValue::from_f64(f64::NAN)).is_err());
    assert!(js_pivot_mds_init(&grid, None, None, None, &JsValue::from_f64(0.0)).is_err());
    assert!(js_pivot_mds_init(&grid, None, None, None, &JsValue::from_str("bad")).is_err());

    assert!(js_koren_spectral_init(&grid, &JsValue::from_f64(-1.0), None, None, None).is_err());
    assert!(js_koren_spectral_init(&grid, &JsValue::UNDEFINED, Some(0.0), None, None).is_err());
    assert!(js_koren_spectral_init(&grid, &JsValue::UNDEFINED, None, Some(0.0), None).is_err());

    assert!(js_smallest_laplacian_eigenpairs(&grid, Some(0.0), None, None, None).is_err());
    assert!(js_smallest_laplacian_eigenpairs(&grid, None, Some(f64::NAN), None, None).is_err());

    assert!(js_chebyshev_heat_spmv(&grid, &[1.0; 5], 1.0, 1.0, None, None).is_err());
    assert!(js_chebyshev_heat_spmv(&grid, &[f64::NAN; 9], 1.0, 1.0, None, None).is_err());
    assert!(js_chebyshev_heat_spmv(&grid, &[1.0; 9], 1.0, -0.5, None, None).is_err());
    assert!(js_chebyshev_heat_spmv(&grid, &[1.0; 9], 1.0, 1.0, None, Some(0.0)).is_err());

    assert!(
        js_chebyshev_dog_jl_embedding(&grid, Some(0.0), None, None, None, None, None, None, None)
            .is_err()
    );
    assert!(
        js_chebyshev_dog_jl_embedding(&grid, None, None, None, None, Some(-0.1), None, None, None,)
            .is_err()
    );

    assert!(JsKnnGraph::from_csr(3.0, &[0, 1, 1], &[1], &[1.0]).is_err());
    assert!(JsKnnGraph::from_csr(3.0, &[0, 1, 1, 1], &[5], &[1.0]).is_err());
    assert!(JsKnnGraph::from_csr(3.0, &[0, 1, 1, 1], &[1], &[-0.5]).is_err());
    assert!(JsKnnGraph::from_csr(3.0, &[0, 1, 1, 1], &[1], &[f64::NAN]).is_err());

    let knn = js_partial_bfs_knn(&grid, Some(3.0), None).unwrap();
    assert!(js_compute_knn_probabilities(&knn, &[2.0; 4], None, None, None, None).is_err());
    assert!(js_compute_knn_probabilities(&knn, &[f64::NAN; 9], None, None, None, None).is_err());
    assert!(js_compute_knn_probabilities(&knn, &[2.0; 9], Some(0.0), None, None, None).is_err());
    assert!(js_compute_fuzzy_simplicial_set(&knn, Some(0.0), None).is_err());
    assert!(js_compute_fuzzy_simplicial_set(&knn, None, Some(f64::NAN)).is_err());

    assert!(js_compute_nodewise_2hop_perplexities(&grid, Some(0.0), None, None).is_err());
    assert!(js_compute_median_2hop_perplexity(&grid, None, Some(10.0), Some(5.0)).is_err());
    assert!(
        js_build_hybrid_adaptive_knn(
            &grid, &[0.0; 10], 2.0, &[2.0; 9], None, None, None, None, None, None
        )
        .is_err()
    );
    assert!(
        js_build_hybrid_adaptive_knn(
            &grid,
            &[0.0; 18],
            2.0,
            &[f64::NAN; 9],
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .is_err()
    );

    assert!(js_find_ab_params(Some(-0.1), None).is_err());
    assert!(js_find_ab_params(None, Some(0.0)).is_err());
    assert!(js_compute_horvitz_thompson_gamma(-1.0, 10.0, 2.0).is_err());
    assert!(js_compute_horvitz_thompson_gamma(f64::NAN, 10.0, 2.0).is_err());

    let lm = js_compute_regularized_dog_varadhan_landmarks(
        &grid, None, None, None, None, None, None, None, None, None, None, None,
    )
    .unwrap();
    assert!(lm.alt_distances(&[0, 1], &[2]).is_err());
    assert!(lm.alt_distances(&[0], &[99]).is_err());
    assert!(
        js_compute_regularized_dog_varadhan_landmarks(
            &grid,
            Some(0.0),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .is_err()
    );
}
