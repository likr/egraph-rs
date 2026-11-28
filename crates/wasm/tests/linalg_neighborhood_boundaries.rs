mod util;

use egraph_wasm::csr_graph::JsCsrGraph;
use egraph_wasm::dog_varadhan::js_compute_regularized_dog_varadhan_landmarks;
use egraph_wasm::linalg_chebyshev::{js_chebyshev_dog_jl_embedding, js_chebyshev_heat_spmv};
use egraph_wasm::linalg_spectral::{
    js_koren_spectral_init, js_pivot_mds_init, js_robust_normalize_layout_2d,
    js_select_maxmin_sp_pivots, js_smallest_laplacian_eigenpairs,
};
use egraph_wasm::neighborhood_knn::{
    js_build_capped_12hop_knn, js_build_hybrid_adaptive_knn, js_build_pure_2hop_knn,
    js_compute_2hop_neighborhood_sizes, js_compute_fuzzy_simplicial_set,
    js_compute_horvitz_thompson_gamma, js_compute_knn_probabilities,
    js_compute_median_2hop_perplexity, js_compute_nodewise_2hop_perplexities,
    js_extract_1hop_2hop_csr, js_find_ab_params, js_graph_12hop_row_normalized_probabilities,
    js_partial_bfs_knn,
};
use egraph_wasm::neighborhood_types::JsKnnGraph;
use util::fixtures::{make_grid, make_hub_chain_graph, make_opts};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn empty_singleton_directed_and_error_handling() {
    let empty =
        JsCsrGraph::from_edge_list(0.0, &[], None, &JsValue::UNDEFINED, None).expect("empty");
    let single =
        JsCsrGraph::from_edge_list(1.0, &[], None, &JsValue::UNDEFINED, None).expect("single");

    assert!(
        js_select_maxmin_sp_pivots(&empty, &make_opts(&[("numPivots", JsValue::from_f64(5.0))]))
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        js_pivot_mds_init(&empty, &JsValue::UNDEFINED)
            .unwrap()
            .len(),
        0
    );
    assert_eq!(
        js_pivot_mds_init(&single, &JsValue::UNDEFINED)
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        js_koren_spectral_init(&empty, &JsValue::UNDEFINED)
            .unwrap()
            .len(),
        0
    );
    assert_eq!(
        js_koren_spectral_init(&single, &JsValue::UNDEFINED)
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        js_smallest_laplacian_eigenpairs(
            &empty,
            &make_opts(&[("nModes", JsValue::from_f64(4.0))]),
        )
        .unwrap()
        .num_modes(),
        0
    );
    assert_eq!(
        js_smallest_laplacian_eigenpairs(
            &single,
            &make_opts(&[("nModes", JsValue::from_f64(4.0))]),
        )
        .unwrap()
        .num_modes(),
        0
    );
    assert!(
        js_chebyshev_heat_spmv(&empty, &[], 1.0, 1.0, &JsValue::UNDEFINED)
            .unwrap()
            .is_empty()
    );
    assert!(
        js_chebyshev_dog_jl_embedding(&empty, &make_opts(&[("jlDim", JsValue::from_f64(8.0))]))
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
        js_compute_nodewise_2hop_perplexities(&empty, &JsValue::UNDEFINED)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        js_compute_median_2hop_perplexity(&single, &JsValue::UNDEFINED).unwrap(),
        1.0
    );
    assert_eq!(
        js_partial_bfs_knn(&single, Some(5.0), &JsValue::UNDEFINED)
            .unwrap()
            .nnz(),
        0
    );
    assert_eq!(
        js_compute_regularized_dog_varadhan_landmarks(&empty, &JsValue::UNDEFINED)
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

    // Adversarial error checks (NaN, bounds, mismatch, number | bigint seed validation)
    let grid = make_grid(3, 3);
    let max_u64_pivots = js_select_maxmin_sp_pivots(
        &grid,
        &make_opts(&[
            ("numPivots", JsValue::from_f64(4.0)),
            ("seed", js_sys::BigInt::from(u64::MAX).into()),
        ]),
    )
    .expect("BigInt 64-bit seed supported");
    assert_eq!(max_u64_pivots.len(), 4);

    assert!(
        js_select_maxmin_sp_pivots(&grid, &make_opts(&[("numPivots", JsValue::from_f64(-1.0))]))
            .is_err()
    );
    assert!(
        js_select_maxmin_sp_pivots(&grid, &make_opts(&[("numPivots", JsValue::from_f64(1.5))]))
            .is_err()
    );
    assert!(
        js_select_maxmin_sp_pivots(&grid, &make_opts(&[("seed", JsValue::from_f64(-1.0))]))
            .is_err()
    );
    assert!(
        js_select_maxmin_sp_pivots(&grid, &make_opts(&[("seed", JsValue::from_f64(1.5))])).is_err()
    );
    assert!(
        js_select_maxmin_sp_pivots(
            &grid,
            &make_opts(&[("seed", JsValue::from_f64(9007199254740992.0))]),
        )
        .is_err()
    );

    assert!(
        js_pivot_mds_init(&grid, &make_opts(&[("numPivots", JsValue::from_f64(0.0))])).is_err()
    );
    assert!(
        js_pivot_mds_init(
            &grid,
            &make_opts(&[("targetStd", JsValue::from_f64(f64::NAN))]),
        )
        .is_err()
    );
    assert!(
        js_pivot_mds_init(&grid, &make_opts(&[("targetStd", JsValue::from_f64(0.0))])).is_err()
    );
    assert!(
        js_pivot_mds_init(
            &grid,
            &make_opts(&[("targetStd", JsValue::from_str("bad"))])
        )
        .is_err()
    );

    assert!(
        js_koren_spectral_init(&grid, &make_opts(&[("targetStd", JsValue::from_f64(-1.0))]))
            .is_err()
    );
    assert!(js_koren_spectral_init(&grid, &make_opts(&[("tol", JsValue::from_f64(0.0))])).is_err());
    assert!(
        js_koren_spectral_init(
            &grid,
            &make_opts(&[("maxIterations", JsValue::from_f64(0.0))]),
        )
        .is_err()
    );

    assert!(
        js_smallest_laplacian_eigenpairs(&grid, &make_opts(&[("nModes", JsValue::from_f64(0.0))]))
            .is_err()
    );
    assert!(
        js_smallest_laplacian_eigenpairs(
            &grid,
            &make_opts(&[("tol", JsValue::from_f64(f64::NAN))])
        )
        .is_err()
    );

    assert!(js_chebyshev_heat_spmv(&grid, &[1.0; 5], 1.0, 1.0, &JsValue::UNDEFINED).is_err());
    assert!(js_chebyshev_heat_spmv(&grid, &[f64::NAN; 9], 1.0, 1.0, &JsValue::UNDEFINED).is_err());
    assert!(js_chebyshev_heat_spmv(&grid, &[1.0; 9], 1.0, -0.5, &JsValue::UNDEFINED).is_err());
    assert!(
        js_chebyshev_heat_spmv(
            &grid,
            &[1.0; 9],
            1.0,
            1.0,
            &make_opts(&[("lambdaMax", JsValue::from_f64(0.0))]),
        )
        .is_err()
    );

    assert!(
        js_chebyshev_dog_jl_embedding(&grid, &make_opts(&[("jlDim", JsValue::from_f64(0.0))]))
            .is_err()
    );
    assert!(
        js_chebyshev_dog_jl_embedding(&grid, &make_opts(&[("epsPsd", JsValue::from_f64(-0.1))]))
            .is_err()
    );

    assert!(JsKnnGraph::from_csr(3.0, &[0, 1, 1], &[1], &[1.0]).is_err());
    assert!(JsKnnGraph::from_csr(3.0, &[0, 1, 1, 1], &[5], &[1.0]).is_err());
    assert!(JsKnnGraph::from_csr(3.0, &[0, 1, 1, 1], &[1], &[-0.5]).is_err());
    assert!(JsKnnGraph::from_csr(3.0, &[0, 1, 1, 1], &[1], &[f64::NAN]).is_err());

    let knn = js_partial_bfs_knn(&grid, Some(3.0), &JsValue::UNDEFINED).unwrap();
    assert!(js_compute_knn_probabilities(&knn, &[2.0; 4], &JsValue::UNDEFINED).is_err());
    assert!(js_compute_knn_probabilities(&knn, &[f64::NAN; 9], &JsValue::UNDEFINED).is_err());
    assert!(
        js_compute_knn_probabilities(
            &knn,
            &[2.0; 9],
            &make_opts(&[("power", JsValue::from_f64(0.0))]),
        )
        .is_err()
    );
    assert!(js_compute_fuzzy_simplicial_set(&knn, Some(0.0), None).is_err());
    assert!(js_compute_fuzzy_simplicial_set(&knn, None, Some(f64::NAN)).is_err());

    assert!(
        js_compute_nodewise_2hop_perplexities(&grid, &make_opts(&[("c", JsValue::from_f64(0.0))]))
            .is_err()
    );
    assert!(
        js_compute_median_2hop_perplexity(
            &grid,
            &make_opts(&[
                ("minPerplexity", JsValue::from_f64(10.0)),
                ("maxPerplexity", JsValue::from_f64(5.0)),
            ]),
        )
        .is_err()
    );
    assert!(
        js_build_hybrid_adaptive_knn(&grid, &[0.0; 10], 2.0, &[2.0; 9], &JsValue::UNDEFINED)
            .is_err()
    );
    assert!(
        js_build_hybrid_adaptive_knn(&grid, &[0.0; 18], 2.0, &[f64::NAN; 9], &JsValue::UNDEFINED,)
            .is_err()
    );

    assert!(js_find_ab_params(Some(-0.1), None).is_err());
    assert!(js_find_ab_params(None, Some(0.0)).is_err());
    assert!(js_compute_horvitz_thompson_gamma(-1.0, 10.0, 2.0).is_err());
    assert!(js_compute_horvitz_thompson_gamma(f64::NAN, 10.0, 2.0).is_err());

    let lm = js_compute_regularized_dog_varadhan_landmarks(&grid, &JsValue::UNDEFINED).unwrap();
    assert!(lm.alt_distances(&[0, 1], &[2]).is_err());
    assert!(lm.alt_distances(&[0], &[99]).is_err());
    assert!(
        js_compute_regularized_dog_varadhan_landmarks(
            &grid,
            &make_opts(&[("degree", JsValue::from_f64(0.0))]),
        )
        .is_err()
    );

    // Robust normalization, capped 1+2-hop kNN, and row-normalized 1+2-hop probabilities
    let mut rob = js_pivot_mds_init(
        &grid,
        &make_opts(&[
            ("numPivots", JsValue::from_f64(8.0)),
            ("seed", JsValue::from_f64(42.0)),
            ("minDegree", JsValue::from_f64(2.0)),
            ("targetStd", JsValue::NULL),
        ]),
    )
    .unwrap();
    js_robust_normalize_layout_2d(
        &mut rob,
        &make_opts(&[
            ("targetStd", JsValue::from_f64(1.5)),
            ("quantile", JsValue::from_f64(0.95)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    assert!(rob.is_all_finite());
    assert!(
        js_robust_normalize_layout_2d(
            &mut rob,
            &make_opts(&[("targetStd", JsValue::from_f64(0.0))]),
        )
        .is_err()
    );

    let hub = make_hub_chain_graph();
    let capped = js_build_capped_12hop_knn(&hub, Some(3.0), &JsValue::from_f64(42.0)).unwrap();
    assert_eq!(capped.knn().num_nodes(), 13);
    assert!(capped.n_truncated_2hop() > 0);
    assert!(capped.n_2hop() <= 2 * 13 * 3);

    let p12_opts = make_opts(&[
        ("w2", JsValue::from_f64(0.5)),
        ("max2hopPerNode", JsValue::from_f64(3.0)),
        ("degreeBeta", JsValue::from_f64(0.35)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let p12 = js_graph_12hop_row_normalized_probabilities(&hub, &p12_opts).unwrap();
    assert!((p12.values().iter().sum::<f64>() - (12.0 / 13.0)).abs() < 1e-10);
    let p12_grid = js_graph_12hop_row_normalized_probabilities(&grid, &p12_opts).unwrap();
    assert!((p12_grid.values().iter().sum::<f64>() - 1.0).abs() < 1e-10);
    let p12_ext = js_graph_12hop_row_normalized_probabilities(
        &hub,
        &make_opts(&[
            ("w2", JsValue::from_f64(0.5)),
            ("max2hopPerNode", JsValue::from_f64(3.0)),
            ("degreeBeta", JsValue::from_f64(200.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
    )
    .unwrap();
    assert!(p12_ext.values().iter().all(|v| v.is_finite()));
    assert!(
        js_graph_12hop_row_normalized_probabilities(
            &hub,
            &make_opts(&[
                ("w2", JsValue::from_f64(0.5)),
                ("degreeBeta", JsValue::from_f64(-0.1)),
            ]),
        )
        .is_err()
    );
}
