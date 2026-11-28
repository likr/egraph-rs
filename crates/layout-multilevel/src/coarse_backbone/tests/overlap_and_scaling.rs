use super::super::overlap::{count_disk_overlaps, resolve_residual_disk_overlaps};
use super::super::*;
use super::{make_three_cluster_chain, make_unequal_clusters_graph};
use egraph_clustering::CoarseLouvainConfig;
use egraph_core::{CsrGraph, Layout2d, LayoutGenerator, Scalar};

#[test]
fn test_mass_scaled_hinges_and_non_edge_clearance() {
    // 1. Unequal large clusters: mass-scaled hinges eliminate coarse overlap in both f64 and f32.
    let big = make_three_cluster_chain(64);
    let cfg = CoarseBackboneConfig::new()
        .with_louvain(CoarseLouvainConfig::new().with_max_coarse_nodes(3))
        .with_c0(0.8)
        .with_gamma_m(0.5)
        .with_mu_ov(1.0)
        .with_m_gap(0.1)
        .with_target_std(None)
        .with_distance_backend(CoarseDistanceBackend::ShortestPath);

    let res64 = CoarseBackbone::<f64, f64>::from_config(&big, cfg.clone())
        .generate()
        .unwrap();
    assert_eq!(res64.stats.num_coarse_nodes, 3);
    assert_eq!(res64.stats.coarse_overlap_pairs_final, 0);
    assert!(res64.fine_layout.is_all_finite());

    let big32 = make_three_cluster_chain(64);
    let res32 = CoarseBackbone::<f32, f64>::from_config(&big32, cfg)
        .generate()
        .unwrap();
    assert_eq!(res32.stats.num_coarse_nodes, 3);
    assert_eq!(res32.stats.coarse_overlap_pairs_final, 0);
    assert!(res32.fine_layout.is_all_finite());

    // 2. Disconnected large cliques (m_l == 0, non-edge springs + hinges) preserve non-edge
    //    clearance >= radii[u] + radii[v] + m_gap without collapsing.
    let mut disc_edges = Vec::new();
    for c in 0..4u32 {
        let off = c * 12;
        for i in 0..12u32 {
            for j in (i + 1)..12u32 {
                disc_edges.push((off + i, off + j));
            }
        }
    }
    let g_disc = CsrGraph::<f64>::from_edges(48, &disc_edges, None, false, false).unwrap();
    let cfg_disc = CoarseBackboneConfig::new()
        .with_louvain(CoarseLouvainConfig::new().with_max_coarse_nodes(4))
        .with_c0(0.8)
        .with_gamma_m(0.5)
        .with_mu_ov(1.0)
        .with_m_gap(0.1)
        .with_coarse_neg_per_node(4)
        .with_target_std(None);
    let res_disc = CoarseBackbone::<f64, f64>::from_config(&g_disc, cfg_disc.clone())
        .generate()
        .unwrap();
    assert_eq!(res_disc.stats.num_coarse_nodes, 4);
    assert_eq!(res_disc.stats.coarse_overlap_pairs_final, 0);
    for u in 0..4usize {
        for v in (u + 1)..4usize {
            let d = (res_disc.coarse_layout.x()[u] - res_disc.coarse_layout.x()[v])
                .hypot(res_disc.coarse_layout.y()[u] - res_disc.coarse_layout.y()[v]);
            let min_req = res_disc.radii[u] + res_disc.radii[v] + cfg_disc.m_gap;
            assert!(
                d >= min_req - 1e-6,
                "disconnected cluster pair ({u}, {v}) collapsed: d={d} < min_req={min_req}"
            );
        }
    }

    // 3. Edge cases: tiny m_gap with c0=0.0 in f32, c0=0.0 & m_gap=0.0, and mu_ov=0.0.
    for (c0, m_gap, mu_ov) in [(0.0, 1e-8, 1.0), (0.0, 0.0, 1.0), (0.8, 0.1, 0.0)] {
        let edge_cfg = CoarseBackboneConfig::new()
            .with_louvain(CoarseLouvainConfig::new().with_max_coarse_nodes(3))
            .with_c0(c0)
            .with_m_gap(m_gap)
            .with_mu_ov(mu_ov)
            .with_target_std(None);
        let r_f32 = CoarseBackbone::<f32, f64>::from_config(&big, edge_cfg)
            .generate()
            .unwrap();
        assert!(
            r_f32.fine_layout.is_all_finite() && r_f32.coarse_layout.is_all_finite(),
            "non-finite output for c0={c0}, m_gap={m_gap}, mu_ov={mu_ov}"
        );
    }
}

#[test]
fn test_unequal_clusters_all_topologies_f64_and_f32() {
    // Unequal-cluster stress suite: 10x100 + 20x10 across chain, star_chain, disconnected,
    // mixed_2comp x gamma_m in {0.0, 0.5, 1.0} x {Rhgh, ChebyshevDog, ShortestPath} x {f64, f32}.
    for topology in ["chain", "star_chain", "disconnected", "mixed_2comp"] {
        let g = make_unequal_clusters_graph(10, 100, 20, 10, topology);
        for gamma_m in [0.0, 0.5, 1.0] {
            for backend in [
                CoarseDistanceBackend::Rhgh,
                CoarseDistanceBackend::ChebyshevDog,
                CoarseDistanceBackend::ShortestPath,
            ] {
                let cfg = CoarseBackboneConfig::new()
                    .with_louvain(CoarseLouvainConfig::new().with_max_coarse_nodes(30))
                    .with_c0(0.8)
                    .with_m_gap(0.1)
                    .with_gamma_m(gamma_m)
                    .with_distance_backend(backend)
                    .with_target_std(None);

                let r64 = CoarseBackbone::<f64, f64>::from_config(&g, cfg.clone())
                    .generate()
                    .unwrap();
                assert_eq!(
                    r64.stats.coarse_overlap_pairs_final, 0,
                    "f64 overlap for topology={topology}, gamma_m={gamma_m}, backend={backend:?}"
                );
                assert!(r64.fine_layout.is_all_finite() && r64.coarse_layout.is_all_finite());

                let r32 = CoarseBackbone::<f32, f64>::from_config(&g, cfg.clone())
                    .generate()
                    .unwrap();
                assert_eq!(
                    r32.stats.coarse_overlap_pairs_final, 0,
                    "f32 overlap for topology={topology}, gamma_m={gamma_m}, backend={backend:?}"
                );
                assert!(r32.fine_layout.is_all_finite() && r32.coarse_layout.is_all_finite());

                // Verify normalized fine_layout remains finite and well-scaled around target_std.
                if gamma_m == 0.5 && backend == CoarseDistanceBackend::Rhgh {
                    let norm_cfg = cfg.with_target_std(Some(1.0));
                    let r_norm = CoarseBackbone::<f64, f64>::from_config(&g, norm_cfg)
                        .generate()
                        .unwrap();
                    assert_eq!(r_norm.stats.coarse_overlap_pairs_final, 0);
                    assert!(r_norm.fine_layout.is_all_finite());
                    assert!(r_norm.coarse_layout.is_all_finite());
                    assert!(
                        r_norm.stats.scale_factor_applied.is_finite()
                            && r_norm.stats.scale_factor_applied > 0.0
                    );
                    let n_f = r_norm.fine_layout.len() as f64;
                    let fx = r_norm.fine_layout.x();
                    let fy = r_norm.fine_layout.y();
                    let mx = fx.iter().sum::<f64>() / n_f;
                    let my = fy.iter().sum::<f64>() / n_f;
                    let var = fx
                        .iter()
                        .zip(fy.iter())
                        .map(|(&x, &y)| (x - mx).powi(2) + (y - my).powi(2))
                        .sum::<f64>()
                        / n_f;
                    let std_2d = var.sqrt();
                    assert!(
                        std_2d > 0.2 && std_2d < 5.0,
                        "unexpected fine_layout std={std_2d} for topology={topology}"
                    );
                }
            }
        }
    }

    // Smaller disconnected unequal clusters: 2x100 + 4x10.
    let g_small_disc = make_unequal_clusters_graph(2, 100, 4, 10, "disconnected");
    for gamma_m in [0.0, 0.5, 1.0] {
        let cfg = CoarseBackboneConfig::new()
            .with_louvain(CoarseLouvainConfig::new().with_max_coarse_nodes(6))
            .with_c0(0.8)
            .with_m_gap(0.1)
            .with_gamma_m(gamma_m)
            .with_target_std(None);
        let r64 = CoarseBackbone::<f64, f64>::from_config(&g_small_disc, cfg.clone())
            .generate()
            .unwrap();
        assert_eq!(
            r64.stats.coarse_overlap_pairs_final, 0,
            "2x100+4x10 f64 overlap for gamma_m={gamma_m}"
        );
        let r32 = CoarseBackbone::<f32, f64>::from_config(&g_small_disc, cfg)
            .generate()
            .unwrap();
        assert_eq!(
            r32.stats.coarse_overlap_pairs_final, 0,
            "2x100+4x10 f32 overlap for gamma_m={gamma_m}"
        );
    }

    // mu_ov == 0.0 must leave both SGD hinges and post-SGD overlap projection disabled.
    let g_star = make_unequal_clusters_graph(10, 100, 20, 10, "star_chain");
    let cfg_no_ov = CoarseBackboneConfig::new()
        .with_louvain(CoarseLouvainConfig::new().with_max_coarse_nodes(30))
        .with_c0(0.8)
        .with_m_gap(0.1)
        .with_mu_ov(0.0)
        .with_target_std(None);
    let r_no_ov = CoarseBackbone::<f64, f64>::from_config(&g_star, cfg_no_ov)
        .generate()
        .unwrap();
    assert!(r_no_ov.fine_layout.is_all_finite() && r_no_ov.coarse_layout.is_all_finite());
    assert!(
        r_no_ov.stats.coarse_overlap_pairs_final > 0,
        "expected residual overlaps when mu_ov=0.0 disables both barrier hinges and projection"
    );
}

#[test]
fn test_resolve_residual_disk_overlaps_coincident_and_zero_noop() {
    // Coincident disks resolve cleanly in both f64 and f32 without NaNs/Inf.
    let radii = vec![2.0, 2.0, 1.0, 1.0];
    let sizes = vec![100, 100, 10, 10];
    let m_gap = 0.1;
    let half_boxes64: Vec<f64> = radii.iter().map(|&r| r + 0.5 * m_gap).collect();
    let mut l64 = Layout2d::<f64>::zeros(4);
    resolve_residual_disk_overlaps(&mut l64, &radii, m_gap, &half_boxes64, &sizes).unwrap();
    assert!(l64.is_all_finite());
    assert_eq!(
        count_disk_overlaps(&l64, &radii, m_gap, &half_boxes64).unwrap(),
        0
    );

    let half_boxes32: Vec<f32> = radii.iter().map(|&r| (r + 0.5 * m_gap) as f32).collect();
    let mut l32 = Layout2d::<f32>::zeros(4);
    resolve_residual_disk_overlaps(&mut l32, &radii, m_gap, &half_boxes32, &sizes).unwrap();
    assert!(l32.is_all_finite());
    assert_eq!(
        count_disk_overlaps(&l32, &radii, m_gap, &half_boxes32).unwrap(),
        0
    );

    // Already non-overlapping layout is an exact bit-level no-op.
    let orig = Layout2d::from_xy(vec![0.0_f64, 10.0], vec![0.0, 0.0]).unwrap();
    let mut copy = orig.clone();
    let hb2 = vec![2.05_f64, 2.05];
    resolve_residual_disk_overlaps(&mut copy, &[2.0, 2.0], 0.1, &hb2, &[10, 10]).unwrap();
    assert_eq!(copy, orig);
}

#[test]
fn test_large_n_l_default_config_zero_overlaps_f64_and_f32() {
    fn brute_force_overlap_count<S: Scalar>(
        layout: &Layout2d<S>,
        radii: &[f64],
        m_gap: f64,
        scale: f64,
        tol: f64,
    ) -> usize {
        let n = layout.len();
        let xs = layout.x();
        let ys = layout.y();
        let mut ov = 0usize;
        for u in 0..n {
            let xu = xs[u].to_f64();
            let yu = ys[u].to_f64();
            for v in (u + 1)..n {
                let req = (radii[u] + radii[v] + m_gap) * scale * (1.0 - tol);
                let d = (xu - xs[v].to_f64()).hypot(yu - ys[v].to_f64());
                if d < req {
                    ov += 1;
                }
            }
        }
        ov
    }

    let cases = [
        ("300-disc-default", 10, 50, 290, 4, "disconnected", None),
        ("500-disc-default", 10, 50, 490, 4, "disconnected", None),
        ("1000-disc-default", 10, 50, 990, 4, "disconnected", None),
        ("500-star_all-res10", 1, 50, 499, 10, "star_all", Some(10.0)),
    ];

    for (label, n_big, big_sz, n_small, small_sz, topo, res_opt) in cases {
        let g = make_unequal_clusters_graph(n_big, big_sz, n_small, small_sz, topo);
        let cfg = if let Some(res) = res_opt {
            CoarseBackboneConfig::new()
                .with_louvain(CoarseLouvainConfig::new().with_resolution(res))
        } else {
            CoarseBackboneConfig::default()
        };
        let expected_nl = n_big + n_small;

        let r64 = CoarseBackbone::<f64, f64>::from_config(&g, cfg.clone())
            .generate()
            .unwrap();
        assert_eq!(
            r64.stats.num_coarse_nodes, expected_nl,
            "{label} f64 unexpected num_coarse_nodes"
        );
        assert_eq!(
            r64.stats.coarse_overlap_pairs_final, 0,
            "{label} f64 residual overlaps: init={} -> final={}",
            r64.stats.coarse_overlap_pairs_initial, r64.stats.coarse_overlap_pairs_final
        );
        assert_eq!(
            brute_force_overlap_count(
                &r64.coarse_layout,
                &r64.radii,
                cfg.m_gap,
                r64.stats.scale_factor_applied,
                1e-9,
            ),
            0,
            "{label} f64 brute-force oracle detected overlap"
        );
        assert!(
            r64.fine_layout.is_all_finite() && r64.coarse_layout.is_all_finite(),
            "{label} f64 produced non-finite coordinates"
        );

        let r32 = CoarseBackbone::<f32, f64>::from_config(&g, cfg.clone())
            .generate()
            .unwrap();
        assert_eq!(
            r32.stats.num_coarse_nodes, expected_nl,
            "{label} f32 unexpected num_coarse_nodes"
        );
        assert_eq!(
            r32.stats.coarse_overlap_pairs_final, 0,
            "{label} f32 residual overlaps: init={} -> final={}",
            r32.stats.coarse_overlap_pairs_initial, r32.stats.coarse_overlap_pairs_final
        );
        assert_eq!(
            brute_force_overlap_count(
                &r32.coarse_layout,
                &r32.radii,
                cfg.m_gap,
                r32.stats.scale_factor_applied,
                1e-5,
            ),
            0,
            "{label} f32 brute-force oracle detected overlap"
        );
        assert!(
            r32.fine_layout.is_all_finite() && r32.coarse_layout.is_all_finite(),
            "{label} f32 produced non-finite coordinates"
        );
    }

    // Also test 500-disc high-radius t_max=1/80 and 1000-star_all-res10 stress cases.
    let g500 = make_unequal_clusters_graph(10, 50, 490, 4, "disconnected");
    for t_max in [1usize, 80usize] {
        let cfg_hi = CoarseBackboneConfig::new()
            .with_c0(1.0)
            .with_m_gap(0.2)
            .with_gamma_m(1.0)
            .with_coarse_t_max(t_max)
            .with_target_std(None);
        let r_hi = CoarseBackbone::<f64, f64>::from_config(&g500, cfg_hi.clone())
            .generate()
            .unwrap();
        assert_eq!(
            r_hi.stats.coarse_overlap_pairs_final, 0,
            "500-disc high-radius t_max={t_max} residual overlaps: init={} -> final={}",
            r_hi.stats.coarse_overlap_pairs_initial, r_hi.stats.coarse_overlap_pairs_final
        );
        assert_eq!(
            brute_force_overlap_count(&r_hi.coarse_layout, &r_hi.radii, cfg_hi.m_gap, 1.0, 0.0),
            0
        );
        let r_hi32 = CoarseBackbone::<f32, f64>::from_config(&g500, cfg_hi.clone())
            .generate()
            .unwrap();
        assert_eq!(
            r_hi32.stats.coarse_overlap_pairs_final, 0,
            "500-disc high-radius f32 t_max={t_max} residual overlaps: init={} -> final={}",
            r_hi32.stats.coarse_overlap_pairs_initial, r_hi32.stats.coarse_overlap_pairs_final
        );
        assert_eq!(
            brute_force_overlap_count(&r_hi32.coarse_layout, &r_hi32.radii, cfg_hi.m_gap, 1.0, 0.0),
            0
        );
    }

    let g1000_star = make_unequal_clusters_graph(1, 50, 999, 4, "star_all");
    let cfg_star1000 = CoarseBackboneConfig::new()
        .with_louvain(CoarseLouvainConfig::new().with_resolution(10.0))
        .with_target_std(None);
    let r_star1000 = CoarseBackbone::<f64, f64>::from_config(&g1000_star, cfg_star1000.clone())
        .generate()
        .unwrap();
    assert_eq!(
        r_star1000.stats.coarse_overlap_pairs_final,
        0,
        "1000-star_all-res10 f64 residual overlaps: nl={}, init={} -> final={}",
        r_star1000.stats.num_coarse_nodes,
        r_star1000.stats.coarse_overlap_pairs_initial,
        r_star1000.stats.coarse_overlap_pairs_final
    );
    assert_eq!(
        brute_force_overlap_count(
            &r_star1000.coarse_layout,
            &r_star1000.radii,
            cfg_star1000.m_gap,
            1.0,
            0.0
        ),
        0
    );

    // Verify 2-cluster disconnected collinearity stays preserved with area-scaled init.
    let g2 = make_unequal_clusters_graph(1, 50, 1, 10, "disconnected");
    let cfg2 = CoarseBackboneConfig::new()
        .with_c0(0.8)
        .with_m_gap(0.1)
        .with_target_std(None);
    let r2_64 = CoarseBackbone::<f64, f64>::from_config(&g2, cfg2.clone())
        .generate()
        .unwrap();
    assert_eq!(r2_64.stats.num_coarse_nodes, 2);
    assert_eq!(r2_64.stats.coarse_overlap_pairs_final, 0);
    assert!(
        (r2_64.coarse_layout.y()[0] - r2_64.coarse_layout.y()[1]).abs() < 1e-12,
        "2-cluster disconnected collinearity broken in f64: {:?}",
        r2_64.coarse_layout.y()
    );
    let r2_32 = CoarseBackbone::<f32, f64>::from_config(&g2, cfg2)
        .generate()
        .unwrap();
    assert_eq!(r2_32.stats.num_coarse_nodes, 2);
    assert_eq!(r2_32.stats.coarse_overlap_pairs_final, 0);
    assert!(
        (r2_32.coarse_layout.y()[0] - r2_32.coarse_layout.y()[1]).abs() < 1e-5,
        "2-cluster disconnected collinearity broken in f32: {:?}",
        r2_32.coarse_layout.y()
    );
}
