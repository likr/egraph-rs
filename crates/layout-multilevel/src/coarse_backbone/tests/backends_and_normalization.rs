use super::super::*;
use super::make_three_cluster_chain;
use egraph_clustering::{CoarseLouvainConfig, align_clustered_centroids};
use egraph_core::{CsrGraph, Layout2d, LayoutGenerator};
use egraph_linalg_mds::robust_normalize_layout_2d;

#[test]
fn test_pitfall5_three_cluster_chain_no_tunneling_and_rcf_mode() {
    let g = make_three_cluster_chain(24);
    let cfg = CoarseBackboneConfig::new()
        .with_louvain(CoarseLouvainConfig::new().with_max_coarse_nodes(3))
        .with_c0(0.8)
        .with_kappa(1.0)
        .with_mu_ov(2.0)
        .with_m_gap(0.1)
        .with_target_std(None)
        .with_distance_backend(CoarseDistanceBackend::ShortestPath);

    let res_sp = CoarseBackbone::<f64, f64>::from_config(&g, cfg.clone())
        .generate()
        .unwrap();
    assert_eq!(res_sp.stats.num_coarse_nodes, 3);
    assert_eq!(res_sp.stats.coarse_overlap_pairs_final, 0);

    // Identify clusters A, B, C (B is the large central cluster).
    let sizes = &res_sp.hierarchy.coarsest.coarse_sizes;
    let b_id = sizes.iter().enumerate().max_by_key(|&(_, s)| *s).unwrap().0;
    assert!(sizes[b_id] >= 20, "expected large cluster B, got {sizes:?}");
    let ac_ids: Vec<usize> = (0..3).filter(|&i| i != b_id).collect();
    let a_id = ac_ids[0];
    let c_id = ac_ids[1];
    let dx = res_sp.coarse_layout.x()[a_id] - res_sp.coarse_layout.x()[c_id];
    let dy = res_sp.coarse_layout.y()[a_id] - res_sp.coarse_layout.y()[c_id];
    let dist_ac = dx.hypot(dy);
    assert!(
        dist_ac >= 2.0 * res_sp.radii[b_id],
        "Pitfall #5 tunneling detected: dist(A,C)={dist_ac} < 2*r_B={}",
        2.0 * res_sp.radii[b_id]
    );

    // RHGH + ResistanceCurvatureFlow (weights_are_lengths=false) determinism check.
    let cfg_rcf = cfg
        .with_edge_length_mode(CoarseEdgeLengthMode::ResistanceCurvatureFlow)
        .with_distance_backend(CoarseDistanceBackend::Rhgh)
        .with_target_std(Some(1.0));
    let r1 = CoarseBackbone::<f64, f64>::from_config(&g, cfg_rcf.clone())
        .generate()
        .unwrap();
    let r2 = CoarseBackbone::<f64, f64>::from_config(&g, cfg_rcf)
        .generate()
        .unwrap();
    assert_eq!(r1.fine_layout, r2.fine_layout);
    assert_eq!(r1.coarse_layout, r2.coarse_layout);
    assert_eq!(r1.stats.coarse_overlap_pairs_final, 0);

    // Centroid tether alignment keeps fine centroids aligned with coarse_layout after normalization.
    let mut tethered = r1.fine_layout.clone();
    align_clustered_centroids(
        &mut tethered,
        &r1.coarse_layout,
        &r1.hierarchy.coarsest.node_to_coarse,
        1.0,
    )
    .unwrap();
    assert!(tethered.is_all_finite());
}

#[test]
fn test_ablation_zero_values_and_chebyshev_dog_and_edge_cases() {
    let g = make_three_cluster_chain(12);
    let ablation_cfg = CoarseBackboneConfig::new()
        .with_distance_backend(CoarseDistanceBackend::ChebyshevDog)
        .with_c0(0.0)
        .with_kappa(0.0)
        .with_gamma_w(0.0)
        .with_gamma_m(0.0)
        .with_mu_ov(0.0)
        .with_m_gap(0.0)
        .with_jitter_scale(0.0)
        .with_gamma_tether(0.0)
        .with_target_std(Some(1.0));
    let res = CoarseBackbone::<f32, f64>::from_config(&g, ablation_cfg)
        .generate()
        .unwrap();
    assert!(res.fine_layout.is_all_finite());
    assert_eq!(res.stats.max_radius, 0.0);

    // Empty, singleton, edgeless, and invalid config handling.
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    assert_eq!(
        CoarseBackbone::<f64, f64>::new(&empty)
            .generate()
            .unwrap()
            .fine_layout
            .len(),
        0
    );
    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    assert_eq!(
        CoarseBackbone::<f64, f64>::new(&single)
            .generate()
            .unwrap()
            .fine_layout
            .len(),
        1
    );
    let edgeless = CsrGraph::<f64>::from_edges(5, &[], None, false, false).unwrap();
    let re = CoarseBackbone::<f64, f64>::new(&edgeless)
        .generate()
        .unwrap();
    assert!(re.fine_layout.is_all_finite());

    let bad_cfg = CoarseBackboneConfig::new().with_alpha(-0.1);
    assert!(
        CoarseBackbone::<f64, f64>::from_config(&g, bad_cfg)
            .generate()
            .is_err()
    );

    // Directed graph rejected cleanly.
    let dir_g = CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], None, true, false).unwrap();
    assert!(CoarseBackbone::<f64, f64>::new(&dir_g).generate().is_err());
}

#[test]
fn test_collinear_two_cluster_consistent_normalization_f64_and_f32() {
    // Regression reproducer: 20-node two-10-clique graph connected by a single bridge (9, 10).
    let mut edges = Vec::new();
    for i in 0..10u32 {
        for j in (i + 1)..10u32 {
            edges.push((i, j));
            edges.push((10 + i, 10 + j));
        }
    }
    edges.push((9, 10));
    let g64 = CsrGraph::<f64>::from_edges(20, &edges, None, false, false).unwrap();
    let g32 = CsrGraph::<f32>::from_edges(20, &edges, None, false, false).unwrap();

    let cfg = CoarseBackboneConfig::new()
        .with_louvain(CoarseLouvainConfig::new().with_max_coarse_nodes(2))
        .with_jitter_scale(0.0)
        .with_target_std(Some(1.0))
        .with_seed(42);

    // f64 check: scale_factor_applied != 1.0 and cluster mean == coarse_layout[0] within 1e-12.
    let r64 = CoarseBackbone::<f64, f64>::from_config(&g64, cfg.clone())
        .generate()
        .unwrap();
    assert_eq!(r64.stats.num_coarse_nodes, 2);
    assert!(
        (r64.stats.scale_factor_applied - 1.0).abs() > 1e-3,
        "expected non-unit scale_factor_applied, got {}",
        r64.stats.scale_factor_applied
    );
    for cid in 0..2u32 {
        let mut sx = 0.0_f64;
        let mut sy = 0.0_f64;
        let mut cnt = 0usize;
        for (i, &c) in r64.hierarchy.coarsest.node_to_coarse.iter().enumerate() {
            if c == cid {
                sx += r64.fine_layout.x()[i];
                sy += r64.fine_layout.y()[i];
                cnt += 1;
            }
        }
        let mx = sx / (cnt as f64);
        let my = sy / (cnt as f64);
        let diff = (mx - r64.coarse_layout.x()[cid as usize])
            .hypot(my - r64.coarse_layout.y()[cid as usize]);
        assert!(
            diff < 1e-12,
            "f64 cluster {cid} centroid mismatch: diff={diff}"
        );
    }

    // f32 check: scale_factor_applied != 1.0 and cluster mean == coarse_layout[0] within 1e-5.
    let r32 = CoarseBackbone::<f32, f32>::from_config(&g32, cfg)
        .generate()
        .unwrap();
    assert_eq!(r32.stats.num_coarse_nodes, 2);
    assert!(
        (r32.stats.scale_factor_applied - 1.0).abs() > 1e-3,
        "expected non-unit scale_factor_applied in f32, got {}",
        r32.stats.scale_factor_applied
    );
    for cid in 0..2u32 {
        let mut sx = 0.0_f64;
        let mut sy = 0.0_f64;
        let mut cnt = 0usize;
        for (i, &c) in r32.hierarchy.coarsest.node_to_coarse.iter().enumerate() {
            if c == cid {
                sx += r32.fine_layout.x()[i] as f64;
                sy += r32.fine_layout.y()[i] as f64;
                cnt += 1;
            }
        }
        let mx = sx / (cnt as f64);
        let my = sy / (cnt as f64);
        let diff = (mx - r32.coarse_layout.x()[cid as usize] as f64)
            .hypot(my - r32.coarse_layout.y()[cid as usize] as f64);
        assert!(
            diff < 1e-5,
            "f32 cluster {cid} centroid mismatch: diff={diff}"
        );
    }

    // Direct vertical 1D collinear extract_robust_transform oracle.
    let mut vert_before = Layout2d::from_xy(vec![0.0_f64, 0.0, 0.0], vec![-3.0, 0.0, 3.0]).unwrap();
    let before_snap = vert_before.clone();
    robust_normalize_layout_2d(&mut vert_before, 1.0, 0.95, 42).unwrap();
    let (cx, cy, sc) = extract_robust_transform(&before_snap, &vert_before)
        .expect("vertical 1D layout should yield valid transform");
    assert!(cx.abs() < 1e-12 && cy.abs() < 1e-12);
    assert!((sc - 1.0).abs() > 1e-3);
}

#[test]
fn test_zero_weight_coarse_edges_guarded_in_rcf_mode() {
    // Three 4-cliques: C0-C1 has positive edge weight 1.0; C1-C2 has explicit 0.0 edge weight.
    let mut edges = Vec::new();
    let mut weights = Vec::new();
    for c in 0..3u32 {
        let off = c * 4;
        for i in 0..4u32 {
            for j in (i + 1)..4u32 {
                edges.push((off + i, off + j));
                weights.push(1.0_f64);
            }
        }
    }
    edges.push((3, 4));
    weights.push(1.0_f64);
    edges.push((7, 8));
    weights.push(0.0_f64);

    let g_mixed = CsrGraph::<f64>::from_edges(12, &edges, Some(&weights), false, false).unwrap();
    let cfg = CoarseBackboneConfig::new()
        .with_louvain(CoarseLouvainConfig::new().with_max_coarse_nodes(3))
        .with_edge_length_mode(CoarseEdgeLengthMode::ResistanceCurvatureFlow)
        .with_distance_backend(CoarseDistanceBackend::Rhgh);
    let res_mixed = CoarseBackbone::<f64, f64>::from_config(&g_mixed, cfg.clone())
        .generate()
        .unwrap();
    assert!(res_mixed.fine_layout.is_all_finite());
    assert!(res_mixed.coarse_layout.is_all_finite());

    // Also test f32 path with all-zero inter-cluster weights.
    let all_zero_weights = vec![0.0_f32; edges.len()];
    let g_zero32 =
        CsrGraph::<f32>::from_edges(12, &edges, Some(&all_zero_weights), false, false).unwrap();
    let res_zero = CoarseBackbone::<f32, f32>::from_config(&g_zero32, cfg)
        .generate()
        .unwrap();
    assert!(res_zero.fine_layout.is_all_finite());
    assert!(res_zero.coarse_layout.is_all_finite());
}
