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
use util::fixtures::{make_grid, make_hub_chain_graph};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

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

    // Robust normalization, capped 1+2-hop kNN, and row-normalized 1+2-hop probabilities
    let mut rob =
        js_pivot_mds_init(&grid, Some(8.0), Some(42.0), Some(2.0), &JsValue::NULL).unwrap();
    js_robust_normalize_layout_2d(&mut rob, Some(1.5), Some(0.95), Some(42.0)).unwrap();
    assert!(rob.is_all_finite());
    assert!(js_robust_normalize_layout_2d(&mut rob, Some(0.0), None, None).is_err());

    let hub = make_hub_chain_graph();
    let capped = js_build_capped_12hop_knn(&hub, Some(3.0), Some(42.0)).unwrap();
    assert_eq!(capped.knn().num_nodes(), 13);
    assert!(capped.n_truncated_2hop() > 0);
    assert!(capped.n_2hop() <= 2 * 13 * 3);

    let p12 = js_graph_12hop_row_normalized_probabilities(
        &hub,
        Some(0.5),
        Some(3.0),
        Some(0.35),
        Some(42.0),
    )
    .unwrap();
    assert!((p12.values().iter().sum::<f64>() - (12.0 / 13.0)).abs() < 1e-10);
    let p12_grid = js_graph_12hop_row_normalized_probabilities(
        &grid,
        Some(0.5),
        Some(3.0),
        Some(0.35),
        Some(42.0),
    )
    .unwrap();
    assert!((p12_grid.values().iter().sum::<f64>() - 1.0).abs() < 1e-10);
    let p12_ext = js_graph_12hop_row_normalized_probabilities(
        &hub,
        Some(0.5),
        Some(3.0),
        Some(200.0),
        Some(42.0),
    )
    .unwrap();
    assert!(p12_ext.values().iter().all(|v| v.is_finite()));
    assert!(
        js_graph_12hop_row_normalized_probabilities(&hub, Some(0.5), None, Some(-0.1), None)
            .is_err()
    );
}
