mod util;

use egraph_wasm::csr_graph::JsCsrGraph;
use egraph_wasm::dog_varadhan::js_compute_regularized_dog_varadhan_landmarks;
use egraph_wasm::linalg_chebyshev::js_chebyshev_dog_jl_embedding;
use egraph_wasm::linalg_spectral::{
    js_classical_mds_2d, js_compute_diffusion_embedding, js_compute_resistance_embedding,
    js_normalize_layout_2d, js_rd_mds, js_representative_mds, js_robust_normalize_layout_2d,
};
use egraph_wasm::neighborhood_knn::{
    js_apply_relative_distance_floor, js_build_capped_12hop_knn, js_build_gumap_12hop_priority_csr,
    js_build_hybrid_adaptive_knn, js_build_pure_2hop_knn, js_build_rerank_knn,
    js_compute_2hop_neighborhood_sizes, js_compute_fuzzy_simplicial_set,
    js_compute_horvitz_thompson_gamma, js_compute_knn_probabilities,
    js_compute_median_2hop_perplexity, js_compute_nodewise_2hop_perplexities,
    js_extract_1hop_2hop_csr, js_extract_sparse_1hop_2hop_pairs, js_find_ab_params,
    js_graph_12hop_row_normalized_probabilities, js_graph_row_normalized_probabilities,
    js_partial_bfs_knn, js_symmetrize_row_normalized,
};
use egraph_wasm::neighborhood_types::JsKnnGraph;
use util::fixtures::{make_grid, make_hub_chain_graph, make_opts};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn neighborhood_2hop_hybrid_knn_probabilities_fuzzy_and_varadhan() {
    let g = make_hub_chain_graph();

    // extract1hop2hopCsr (directed input symmetrized too)
    let two_hop = js_extract_1hop_2hop_csr(&g).expect("1+2hop csr");
    let h1 = two_hop.hop1();
    let h2 = two_hop.hop2();
    assert_eq!(h1.num_nodes(), 13);
    assert_eq!(h2.num_nodes(), 13);
    assert_eq!(h1.num_edges(), 11);
    assert!(h2.num_edges() > 0);

    // extractSparse1hop2hopPairs
    let pairs = js_extract_sparse_1hop_2hop_pairs(&g, Some(32.0), &JsValue::from_f64(42.0))
        .expect("sparse pairs");
    let pairs_def =
        js_extract_sparse_1hop_2hop_pairs(&g, None, &JsValue::UNDEFINED).expect("sparse default");
    assert_eq!(pairs, pairs_def);
    assert_eq!(pairs.num_1hop(), 11);
    assert!(pairs.num_2hop() > 0);
    assert!(pairs.u1().iter().zip(pairs.v1().iter()).all(|(a, b)| a < b));
    assert!(pairs.u2().iter().zip(pairs.v2().iter()).all(|(a, b)| a < b));
    let pairs_zero =
        js_extract_sparse_1hop_2hop_pairs(&g, Some(0.0), &JsValue::from_f64(42.0)).unwrap();
    assert_eq!(pairs_zero.num_1hop(), 11);
    assert_eq!(pairs_zero.num_2hop(), 0);
    let empty_g = JsCsrGraph::from_edge_list(0.0, &[], None, &JsValue::UNDEFINED, None).unwrap();
    let empty_p = js_extract_sparse_1hop_2hop_pairs(&empty_g, None, &JsValue::UNDEFINED).unwrap();
    assert_eq!(empty_p.num_1hop(), 0);
    assert_eq!(empty_p.num_2hop(), 0);
    assert!(js_extract_sparse_1hop_2hop_pairs(&g, Some(-1.0), &JsValue::UNDEFINED).is_err());
    assert!(js_extract_sparse_1hop_2hop_pairs(&g, None, &JsValue::from_f64(-1.0)).is_err());

    // buildPure2hopKnn & compute2hopNeighborhoodSizes
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

    // Perplexities
    let node_perps =
        js_compute_nodewise_2hop_perplexities(&g, &JsValue::UNDEFINED).expect("nodewise perp");
    assert_eq!(node_perps.len(), 13);
    let upper = 12.0 / 3.0; // (13 - 1) / 3 = 4.0
    for &p in &node_perps {
        assert!((p - upper).abs() < 1e-12);
    }
    let med_perp = js_compute_median_2hop_perplexity(&g, &JsValue::UNDEFINED).expect("median perp");
    assert!((med_perp - upper).abs() < 1e-12);

    // buildHybridAdaptiveKnn
    let coords =
        js_chebyshev_dog_jl_embedding(&g, &make_opts(&[("jlDim", JsValue::from_f64(6.0))]))
            .unwrap();
    let hybrid = js_build_hybrid_adaptive_knn(&g, &coords, 6.0, &node_perps, &JsValue::UNDEFINED)
        .expect("hybrid knn");
    let h_knn = hybrid.knn();
    assert_eq!(h_knn.num_nodes(), 13);
    assert_eq!(h_knn.nnz(), hybrid.nnz());
    assert!(hybrid.avg_row_k() >= hybrid.avg_deg2());
    assert!(hybrid.n_filled_residual() >= 1); // isolated node 12 + sparse nodes
    assert_eq!(hybrid.n_filled_residual() + hybrid.n_pure_2hop_rows(), 13);

    // partialBfsKnn
    let pbfs1 = js_partial_bfs_knn(&g, Some(5.0), &JsValue::from_f64(42.0)).expect("partial bfs");
    let pbfs2 =
        js_partial_bfs_knn(&g, Some(5.0), &JsValue::UNDEFINED).expect("partial bfs default seed");
    assert_eq!(pbfs1, pbfs2);
    assert_eq!(pbfs1.num_nodes(), 13);

    // computeKnnProbabilities
    let probs =
        js_compute_knn_probabilities(&h_knn, &node_perps, &JsValue::UNDEFINED).expect("knn probs");
    assert_eq!(probs.num_nodes(), 13);
    assert_eq!(probs.nnz(), probs.indices().len());
    assert_eq!(probs.nnz(), probs.values().len());
    assert_eq!(probs.indptr().unwrap().len(), 14);
    let total_p: f64 = probs.values().iter().sum();
    assert!((total_p - 1.0).abs() < 1e-10);

    let probs_rn = js_compute_knn_probabilities(
        &h_knn,
        &node_perps,
        &make_opts(&[
            ("power", JsValue::from_f64(2.0)),
            ("sigmaIters", JsValue::from_f64(50.0)),
            ("sigmaTolerance", JsValue::from_f64(1e-5)),
            ("rowNormalize", JsValue::from_bool(true)),
        ]),
    )
    .expect("knn probs row_normalize");
    let total_rn: f64 = probs_rn.values().iter().sum();
    assert!((total_rn - 1.0).abs() < 1e-10);

    // computeFuzzySimplicialSet
    let fuzzy = js_compute_fuzzy_simplicial_set(&h_knn, None, None).expect("fuzzy set");
    assert_eq!(fuzzy.num_nodes(), 13);
    assert!(fuzzy.values().iter().all(|&w| w > 1e-7 && w <= 1.0));

    // findAbParams
    let ab = js_find_ab_params(None, None).expect("findAbParams");
    assert_eq!(ab.len(), 2);
    assert!((ab[0] - 1.577).abs() < 0.02);
    assert!((ab[1] - 0.895).abs() < 0.02);

    // computeHorvitzThompsonGamma
    let gamma = js_compute_horvitz_thompson_gamma(1.0, 1600.0, 16.0).expect("ht gamma");
    assert!((gamma - 10.0).abs() < 1e-12);
    assert_eq!(
        js_compute_horvitz_thompson_gamma(1.0, 0.0, 16.0).unwrap(),
        0.0
    );

    // computeRegularizedDogVaradhanLandmarks
    let lm1 = js_compute_regularized_dog_varadhan_landmarks(&g, &JsValue::UNDEFINED)
        .expect("varadhan default");
    let lm2 = js_compute_regularized_dog_varadhan_landmarks(
        &g,
        &make_opts(&[
            ("numLandmarks", JsValue::from_f64(64.0)),
            ("seed", JsValue::from_f64(42.0)),
            ("degreeNormalize", JsValue::from_bool(true)),
            ("pExp", JsValue::from_f64(0.5)),
            ("epsPsd", JsValue::from_f64(1e-4)),
            ("tCoarse", JsValue::from_f64(48.0)),
        ]),
    )
    .expect("varadhan explicit");
    let lm_deg25 = js_compute_regularized_dog_varadhan_landmarks(
        &g,
        &make_opts(&[
            ("degree", JsValue::from_f64(25.0)),
            ("numLandmarks", JsValue::from_f64(64.0)),
            ("seed", JsValue::from_f64(42.0)),
            ("degreeNormalize", JsValue::from_bool(true)),
            ("pExp", JsValue::from_f64(0.5)),
            ("epsPsd", JsValue::from_f64(1e-4)),
            ("tCoarse", JsValue::from_f64(48.0)),
        ]),
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
        &make_opts(&[
            ("degree", JsValue::from_f64(20.0)),
            ("numLandmarks", JsValue::from_f64(6.0)),
            ("seed", JsValue::from_f64(7.0)),
            ("degreeNormalize", JsValue::from_bool(false)),
            ("pExp", JsValue::from_f64(0.5)),
            ("epsPsd", JsValue::from_f64(1e-4)),
            ("tCoarse", JsValue::from_f64(0.0)),
        ]),
    )
    .expect("varadhan no coarse");
    assert!(!lm_no_coarse.used_coarse_scaffold());
    assert!(lm_no_coarse.coarse_table().is_none());
}

#[wasm_bindgen_test]
fn mds_embeddings_rerank_and_priority_knn() {
    let grid = make_grid(4, 4);
    let mut cmds = js_classical_mds_2d(&grid).unwrap();
    assert_eq!(cmds.num_nodes(), 16);
    assert!(cmds.is_all_finite());
    js_normalize_layout_2d(&mut cmds, Some(1.0), &JsValue::from_f64(42.0)).unwrap();
    js_robust_normalize_layout_2d(
        &mut cmds,
        &make_opts(&[
            ("targetStd", JsValue::from_f64(1.0)),
            ("quantile", JsValue::from_f64(0.95)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();

    let rd = js_rd_mds(
        &grid,
        &make_opts(&[
            ("dims", JsValue::from_f64(3.0)),
            ("maxIterations", JsValue::from_f64(200.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    assert_eq!(rd.num_nodes(), 16);
    assert_eq!(rd.dims(), 3);
    assert!(rd.is_all_finite());

    // 4 nodes x 2 representatives
    let d_reps = [0.0, 3.0, 1.0, 2.0, 2.0, 1.0, 3.0, 0.0];
    let rep_nd = js_representative_mds(&d_reps, 4.0, 2.0, 2.0).unwrap();
    assert_eq!(rep_nd.num_nodes(), 4);
    assert_eq!(rep_nd.dims(), 2);
    assert!(rep_nd.is_all_finite());

    let res_emb = js_compute_resistance_embedding(
        &grid,
        &make_opts(&[
            ("dim", JsValue::from_f64(3.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    assert_eq!(res_emb.num_nodes(), 16);
    assert!(!res_emb.lambdas().is_empty());

    let diff_emb = js_compute_diffusion_embedding(
        &grid,
        &make_opts(&[
            ("dim", JsValue::from_f64(3.0)),
            ("p", JsValue::from_f64(0.5)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    assert_eq!(diff_emb.num_nodes(), 16);
    assert!(diff_emb.t_used().is_some());

    let rerank = js_build_rerank_knn(
        &diff_emb.coords(),
        16.0,
        diff_emb.dim() as f64,
        4.0,
        &make_opts(&[
            ("coarseDim", JsValue::from_f64(2.0)),
            ("rerankFactor", JsValue::from_f64(2.0)),
        ]),
    )
    .unwrap();
    assert_eq!(rerank.num_nodes(), 16);
    assert!(rerank.nnz() > 0);

    let row_probs = js_graph_row_normalized_probabilities(&grid).unwrap();
    assert_eq!(row_probs.num_nodes(), 16);
    let row_probs_12 = js_graph_12hop_row_normalized_probabilities(
        &grid,
        &make_opts(&[
            ("w2", JsValue::from_f64(0.5)),
            ("max2hopPerNode", JsValue::from_f64(4.0)),
            ("degreeBeta", JsValue::from_f64(0.35)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    assert_eq!(row_probs_12.num_nodes(), 16);
    let capped_knn = js_build_capped_12hop_knn(&grid, Some(4.0), &JsValue::from_f64(42.0)).unwrap();
    assert_eq!(capped_knn.knn().num_nodes(), 16);
    let sym =
        js_symmetrize_row_normalized(3.0, &[0, 1, 2, 3], &[1, 2, 0], &[1.0, 1.0, 1.0]).unwrap();
    assert_eq!(sym.num_nodes(), 3);

    let prio = js_build_gumap_12hop_priority_csr(
        3.0,
        &[0],
        &[1],
        &[1.0],
        &[0, 1],
        &[2, 2],
        &[2.0, 2.0],
        &make_opts(&[("tier2Margin", JsValue::from_f64(1.1))]),
    )
    .unwrap();
    assert_eq!(prio.num_nodes(), 3);
    assert!(prio.nnz() > 0);

    let mut d1 = [1e-6_f64, 1.0];
    let mut d2 = [2.0_f64];
    let mut d3 = [5.0_f64];
    let floor = js_apply_relative_distance_floor(
        &mut d1,
        &mut d2,
        &mut d3,
        &make_opts(&[
            ("epsDist", JsValue::from_f64(1e-3)),
            ("minRatio", JsValue::from_f64(0.1)),
        ]),
    )
    .unwrap();
    assert!(floor >= 0.04 && floor.is_finite());
    assert!(d1[0] >= floor);
}
