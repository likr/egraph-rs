//! Integration tests verifying M5 WASM vs Native/Python API parity.

#![cfg(target_arch = "wasm32")]

use egraph_wasm::v6::{
    JsCsrGraph, JsGuMapSolver, JsLayout2d, JsLayoutNd, JsManifoldLayout2d, JsSgdSolver,
    JsTFdpSolver, JsTsNetSolver, js_all_pairs_bfs, js_apply_relative_distance_floor,
    js_bfs_single_source, js_build_gumap_12hop_priority_csr, js_build_pure_2hop_knn,
    js_build_rerank_knn, js_classical_mds_2d, js_compute_diffusion_embedding,
    js_compute_resistance_embedding, js_crossing_edges, js_crossing_edges_torus,
    js_crossing_summary, js_evaluate_metrics, js_gabriel_graph_edges,
    js_graph_row_normalized_probabilities, js_ideal_edge_lengths_with_unit, js_load_dataset,
    js_normalize_layout_2d, js_parse_csr_csv, js_poincare_distance, js_prism_distortion_with_edges,
    js_rd_mds, js_rectangle_overlaps, js_relative_neighborhood_edges, js_representative_mds,
    js_scale_normalized_stress_from_distances, js_spherical_distance, js_symmetrize_row_normalized,
    js_torus_distance, js_wrap_torus_point,
};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

fn make_opts(entries: &[(&str, JsValue)]) -> JsValue {
    let obj = js_sys::Object::new();
    for (k, v) in entries {
        js_sys::Reflect::set(&obj, &JsValue::from_str(k), v).unwrap();
    }
    obj.into()
}

fn make_path_graph(n: usize) -> JsCsrGraph {
    let mut edges = Vec::with_capacity((n - 1) * 2);
    for u in 0..(n as u32 - 1) {
        edges.push(u);
        edges.push(u + 1);
    }
    JsCsrGraph::from_edge_list(
        n as f64,
        &edges,
        None,
        &JsValue::from_bool(false),
        Some(true),
    )
    .unwrap()
}

fn make_grid_graph(rows: usize, cols: usize) -> JsCsrGraph {
    let n = rows * cols;
    let mut edges = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let u = (r * cols + c) as u32;
            if c + 1 < cols {
                edges.push(u);
                edges.push(u + 1);
            }
            if r + 1 < rows {
                edges.push(u);
                edges.push(u + cols as u32);
            }
        }
    }
    JsCsrGraph::from_edge_list(
        n as f64,
        &edges,
        None,
        &JsValue::from_bool(false),
        Some(true),
    )
    .unwrap()
}

#[wasm_bindgen_test]
fn csr_graph_m5_methods_and_lcc() {
    // Disconnected graph: component {0,1} and larger component {2,3,4,5}
    let edges = [0u32, 1, 2, 3, 3, 4, 4, 5, 5, 2];
    let weights = [1.5f32, 2.0, 2.0, 2.0, 2.0];
    let g = JsCsrGraph::from_edge_list(
        6.0,
        &edges,
        Some(weights.to_vec()),
        &JsValue::from_bool(false),
        Some(true),
    )
    .unwrap();

    assert_eq!(g.nnz(), 10);
    assert!(g.is_weighted());
    assert_eq!(g.degree(2.0).unwrap(), 2);
    assert_eq!(g.neighbors(2.0).unwrap(), vec![3, 5]);
    assert!(g.degree(99.0).is_err());

    let sub = g.induced_subgraph(&[2, 3, 4]).unwrap();
    assert_eq!(sub.num_nodes(), 3);
    assert_eq!(sub.num_edges(), 2);

    let lcc = g.largest_connected_component();
    assert_eq!(lcc.graph().num_nodes(), 4);
    assert_eq!(lcc.original_node_ids(), vec![2, 3, 4, 5]);
    assert_eq!(lcc.node_ids(), vec![2, 3, 4, 5]);
}

#[wasm_bindgen_test]
fn layout_2d_and_nd_m5_methods() {
    let mut l2 = JsLayout2d::init(8.0, Some(123.0)).unwrap();
    assert_eq!(l2.num_nodes(), 8);
    assert!(!l2.is_empty());
    assert!(l2.is_all_finite());
    assert_eq!(l2.copy_coords().len(), 16);

    l2.normalize_std(Some(2.0)).unwrap();
    let coords = l2.copy_coords();
    let mut mean_x = 0.0f64;
    let mut mean_y = 0.0f64;
    for i in 0..8 {
        mean_x += coords[2 * i] as f64 / 8.0;
        mean_y += coords[2 * i + 1] as f64 / 8.0;
    }
    assert!(mean_x.abs() < 1e-4 && mean_y.abs() < 1e-4);

    let nd = JsLayoutNd::zeros(4.0, 3.0).unwrap();
    assert_eq!(nd.num_nodes(), 4);
    assert_eq!(nd.dim(), 3);
    assert_eq!(nd.len(), 4);
    assert_eq!(nd.coords_len(), 12);
    assert!(nd.coords_ptr() > 0);
    assert!(!nd.is_empty());
    assert!(nd.is_all_finite());

    let nd2 = JsLayoutNd::from_coords(2.0, 3.0, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert_eq!(nd2.copy_coords(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert!(JsLayoutNd::from_coords(2.0, 3.0, &[1.0]).is_err());
}

#[wasm_bindgen_test]
fn manifold_layout_and_distances() {
    let mut hyp = JsManifoldLayout2d::hyperbolic_poincare(3.0).unwrap();
    assert_eq!(hyp.manifold(), "hyperbolic");
    assert_eq!(hyp.num_nodes(), 3);
    hyp.set_point(0.0, 0.1, 0.2).unwrap();
    hyp.set_point(1.0, -0.1, 0.05).unwrap();
    let delta = hyp.tangent_delta(0.0, 1.0).unwrap();
    assert_eq!(delta.len(), 2);
    hyp.apply_tangent_step(0.0, delta[0] * 0.1, delta[1] * 0.1)
        .unwrap();
    assert!(js_poincare_distance(0.0, 0.0, 0.3, 0.4) > 0.5);

    let sph = JsManifoldLayout2d::spherical(2.0).unwrap();
    assert_eq!(sph.manifold(), "spherical");
    assert!(
        (js_spherical_distance(0.0, 0.0, 0.0, std::f64::consts::FRAC_PI_2)
            - std::f64::consts::FRAC_PI_2)
            .abs()
            < 1e-9
    );

    let tor = JsManifoldLayout2d::torus_2d(2.0).unwrap();
    assert_eq!(tor.manifold(), "torus");
    assert!((js_torus_distance(0.05, 0.05, 0.95, 0.05) - 0.1).abs() < 1e-9);
    let wrapped = js_wrap_torus_point(1.25, -0.25);
    assert!((wrapped[0] - 0.25).abs() < 1e-9 && (wrapped[1] - 0.75).abs() < 1e-9);
}

#[wasm_bindgen_test]
fn dataset_spatial_and_metrics_m5_apis() {
    let karate = js_load_dataset("karate").unwrap();
    assert_eq!(karate.num_nodes(), 34);
    assert!(karate.num_edges() >= 78);
    assert!(js_load_dataset("nonexistent").is_err());

    let parsed = js_parse_csr_csv("3,2\n0,1\n1,2\n", Some(false), Some(true)).unwrap();
    assert_eq!(parsed.num_nodes(), 3);
    assert_eq!(parsed.num_edges(), 2);

    let rects = JsLayout2d::from_coords(&[0.0, 0.0, 0.5, 0.0, 5.0, 0.0]).unwrap();
    let overlaps = js_rectangle_overlaps(&rects, &[0.5, 0.5, 0.5], &[0.5, 0.5, 0.5]).unwrap();
    assert_eq!(overlaps, vec![0, 1]);

    // X-shaped graph for crossings
    let cross_graph = JsCsrGraph::from_edge_list(
        4.0,
        &[0, 3, 1, 2],
        None,
        &JsValue::from_bool(false),
        Some(true),
    )
    .unwrap();
    let cross_layout = JsLayout2d::from_coords(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let pairs = js_crossing_edges(&cross_graph, &cross_layout).unwrap();
    assert_eq!(pairs.len(), 1);
    let summary = js_crossing_summary(&cross_graph, &cross_layout).unwrap();
    assert_eq!(summary.count(), 1);
    assert!(summary.min_angle_rad().unwrap_or(0.0) > 0.0);
    assert!(summary.angle_cos_sq_sum() >= 0.0);

    let torus_pairs = js_crossing_edges_torus(&cross_graph, &cross_layout).unwrap();
    assert!(!torus_pairs.is_empty());

    let grid = make_grid_graph(3, 3);
    let mut grid_coords = Vec::new();
    for r in 0..3 {
        for c in 0..3 {
            grid_coords.push(c as f32);
            grid_coords.push(r as f32);
        }
    }
    let grid_layout = JsLayout2d::from_coords(&grid_coords).unwrap();
    let iel_unit = js_ideal_edge_lengths_with_unit(&grid, &grid_layout, 1.0).unwrap();
    assert!(iel_unit.is_finite());

    let targets = js_sys::Array::of2(&"sns".into(), &"idealEdgeLengths".into());
    let report =
        js_evaluate_metrics(&grid, &grid_layout, &targets, None, None, None, Some(1.0)).unwrap();
    assert!(report.scale_normalized_stress().is_some());
    assert!(report.ideal_edge_lengths().is_some());

    let gg_edges = js_gabriel_graph_edges(&grid_layout).unwrap();
    let rng_edges = js_relative_neighborhood_edges(&grid_layout).unwrap();
    assert!(!gg_edges.is_empty() && !rng_edges.is_empty());

    let apsp = js_all_pairs_bfs(&grid, Some(1.0)).unwrap();
    assert_eq!(apsp.len(), 81);
    let sssp = js_bfs_single_source(&grid, 0.0, Some(2.0)).unwrap();
    assert_eq!(sssp.len(), 9);
    assert!((sssp[8] - 8.0).abs() < 1e-9);
    let sns = js_scale_normalized_stress_from_distances(&grid_layout, &apsp, None).unwrap();
    assert!(sns.is_finite() && sns < 0.25);

    let mut adjusted = grid_layout.clone();
    adjusted.set_x(0.0, 0.05).unwrap();
    let p_dist = js_prism_distortion_with_edges(&grid_layout, &adjusted, &gg_edges).unwrap();
    assert!(p_dist.sigma_dist().is_finite() && p_dist.sigma_disp().is_finite());

    #[cfg(feature = "delaunay")]
    {
        let del_csr = egraph_wasm::v6::js_delaunay_to_csr(&grid_layout).unwrap();
        assert_eq!(del_csr.num_nodes(), 9);
        assert!(del_csr.num_edges() >= 12);
    }
}

#[wasm_bindgen_test]
fn linalg_and_neighborhood_m5_apis() {
    let grid = make_grid_graph(4, 4);
    let mut cmds = js_classical_mds_2d(&grid).unwrap();
    assert_eq!(cmds.num_nodes(), 16);
    assert!(cmds.is_all_finite());
    js_normalize_layout_2d(&mut cmds, Some(1.0), Some(42.0)).unwrap();

    let rd = js_rd_mds(&grid, Some(3.0), None, None, Some(200.0), Some(42.0)).unwrap();
    assert_eq!(rd.num_nodes(), 16);
    assert_eq!(rd.dim(), 3);
    assert!(rd.is_all_finite());

    // 4 nodes x 2 representatives
    let d_reps = [0.0, 3.0, 1.0, 2.0, 2.0, 1.0, 3.0, 0.0];
    let rep_nd = js_representative_mds(&d_reps, 4.0, 2.0, 2.0).unwrap();
    assert_eq!(rep_nd.num_nodes(), 4);
    assert_eq!(rep_nd.dim(), 2);
    assert!(rep_nd.is_all_finite());

    let res_emb =
        js_compute_resistance_embedding(&grid, Some(3.0), None, None, Some(42.0), None).unwrap();
    assert_eq!(res_emb.num_nodes(), 16);
    assert!(!res_emb.lambdas().is_empty());

    let diff_emb =
        js_compute_diffusion_embedding(&grid, Some(3.0), Some(0.5), None, None, None, Some(42.0))
            .unwrap();
    assert_eq!(diff_emb.num_nodes(), 16);
    assert!(diff_emb.t_used().is_some());

    let rerank = js_build_rerank_knn(
        &diff_emb.coords(),
        16.0,
        diff_emb.dim() as f64,
        4.0,
        Some(2.0),
        Some(2.0),
    )
    .unwrap();
    assert_eq!(rerank.num_nodes(), 16);
    assert!(rerank.nnz() > 0);

    let row_probs = js_graph_row_normalized_probabilities(&grid).unwrap();
    assert_eq!(row_probs.num_nodes(), 16);
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
        Some(1.1),
    )
    .unwrap();
    assert_eq!(prio.num_nodes(), 3);
    assert!(prio.nnz() > 0);

    let mut d1 = [1e-6_f64, 1.0];
    let mut d2 = [2.0_f64];
    let mut d3 = [5.0_f64];
    let floor =
        js_apply_relative_distance_floor(&mut d1, &mut d2, &mut d3, Some(1e-3), Some(0.1)).unwrap();
    assert!(floor >= 0.04 && floor.is_finite());
    assert!(d1[0] >= floor);
}

#[wasm_bindgen_test]
fn solvers_m5_parity_apis() {
    let grid = make_grid_graph(4, 4);
    let init = js_classical_mds_2d(&grid).unwrap();

    // TsNetSolver.fromGraphPartialBfs + lastFftStats + short FFT options
    let ts_opts = make_opts(&[
        ("backend", JsValue::from_str("fit")),
        ("iterationsStage1", JsValue::from_f64(2.0)),
        ("iterationsStage2", JsValue::from_f64(2.0)),
        ("iterationsStage3", JsValue::from_f64(2.0)),
        ("intervals", JsValue::from_f64(1.0)),
        ("interpolationOrder", JsValue::from_f64(1.0)),
        ("minGridDim", JsValue::from_f64(16.0)),
        ("maxGridDim", JsValue::from_f64(64.0)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut ts = JsTsNetSolver::from_graph_partial_bfs(
        &grid,
        Some(8.0),
        Some(6.0),
        &ts_opts,
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
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        &JsValue::UNDEFINED,
        None,
        None,
    )
    .unwrap();
    let mut l_ts = init.clone();
    assert!(ts.last_fft_stats().is_none());
    ts.run(&mut l_ts).unwrap();
    let fft_stats = ts.last_fft_stats().expect("lastFftStats after FFT run");
    assert!(fft_stats.total_fft_2d_calls() > 0.0);
    assert!(fft_stats.particle_forward_ffts() > 0.0);

    // TFdpSolver.fromKnn accepting a KnnGraph instance directly
    let knn = js_build_pure_2hop_knn(&grid).unwrap().knn();
    let tfdp_opts = make_opts(&[
        ("backend", JsValue::from_str("exact")),
        ("totalIterations", JsValue::from_f64(4.0)),
        ("seed", JsValue::from_f64(42.0)),
    ]);
    let mut tfdp = JsTFdpSolver::from_knn(
        &JsValue::from(knn.clone()),
        &tfdp_opts,
        &JsValue::UNDEFINED,
        &JsValue::UNDEFINED,
    )
    .unwrap();
    let mut l_tfdp = init.clone();
    tfdp.run(&mut l_tfdp).unwrap();
    assert!(l_tfdp.is_all_finite());

    // SgdSolver.dogStats + optional fromPairs args
    let mut sgd = JsSgdSolver::sparse_barrier_dog(
        &grid,
        &make_opts(&[
            ("tMax", JsValue::from_f64(5.0)),
            ("seed", JsValue::from_f64(42.0)),
        ]),
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
    .unwrap();
    let d_stats = sgd.dog_stats().expect("dogStats from sparseBarrierDog");
    assert!(d_stats.degree() > 0);
    assert!(d_stats.t2() > d_stats.t1() && d_stats.t1() > 0.0);
    let mut l_sgd = init.clone();
    sgd.run(&mut l_sgd).unwrap();

    let mut pairs_opt = JsSgdSolver::from_pairs(
        2.0,
        &[0],
        &[1],
        &[1.5],
        None,
        None,
        None,
        &JsValue::from_f64(4.0),
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let mut l2 = JsLayout2d::from_coords(&[0.0, 0.0, 0.5, 0.0]).unwrap();
    pairs_opt.run(&mut l2).unwrap();
    assert!(l2.is_all_finite());

    // GuMapSolver.fromGraphDog positional minDegree/jlDim + fromFuzzySet sigmaIters/sigmaTolerance
    let path = make_path_graph(6);
    let gm_pos = JsGuMapSolver::from_graph_dog(
        &path,
        &JsValue::UNDEFINED,
        &JsValue::from_f64(5.0),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(42.0),
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
        Some(2.0),
        Some(8.0),
    )
    .unwrap();
    assert!(gm_pos.num_edges() > 0);
}
