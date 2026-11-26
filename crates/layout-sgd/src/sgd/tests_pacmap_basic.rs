use egraph_core::{CsrGraph, IterativeSolver, Layout2d, SolverStatus};
use egraph_linalg_mds::{PivotMdsConfig, pivot_mds};
use egraph_neighborhood::extract_1hop_2hop_csr;

use super::super::sgd::{PacmapGraphConfig, SgdSolver};
use super::test_support::{barbell_graph, circle_layout, compute_np2, grid_graph};

#[test]
fn test_pacmap_graph_three_tier_three_phase_and_bit_exact() {
    let grid = grid_graph(4, 4);
    let cfg = PacmapGraphConfig {
        n_neighbors: 6,
        mn_ratio: 0.5,
        fp_ratio: 2.0,
        phase1_iters: 30,
        phase2_iters: 30,
        phase3_iters: 60,
        learning_rate: 0.5,
        seed: 42,
        keep_all_1hop: false,
        ..Default::default()
    };

    // Check 3-tier pair invariants on grid(4, 4).
    let s_check = SgdSolver::<f64>::pacmap_graph(&grid, cfg).unwrap();
    let st = s_check.pacmap.as_ref().unwrap();
    assert!(!st.nb_pairs.is_empty());
    assert!(!st.mn_pairs.is_empty());
    assert!(!st.fp_pairs.is_empty());
    assert_eq!(s_check.total_iterations(), 120);

    let (adj1, adj2) = extract_1hop_2hop_csr(&grid).unwrap();
    for &(u, v) in &st.nb_pairs {
        assert_ne!(u, v);
        let ui = u as usize;
        assert!(
            adj1.neighbors(ui).contains(&v) || adj2.neighbors(ui).contains(&v),
            "NB pair ({u}, {v}) must be 1-hop or 2-hop"
        );
    }
    for &(u, v) in &st.mn_pairs {
        assert_ne!(u, v);
        let ui = u as usize;
        assert!(!adj1.neighbors(ui).contains(&v));
        assert!(!adj2.neighbors(ui).contains(&v));
    }
    let mut fp_per_u: Vec<Vec<u32>> = vec![Vec::new(); grid.num_nodes()];
    for &(u, v) in &st.fp_pairs {
        assert_ne!(u, v);
        let ui = u as usize;
        assert!(!adj1.neighbors(ui).contains(&v));
        fp_per_u[ui].push(v);
    }
    // On grid(4, 4) (n = 16, n_neighbors = 6, n_fp = 12), FP fallback guarantees
    // min(12, 16 - 1 - |N1(u) U nb_u|) unique FP pairs per vertex (9 for interior/edge, 10 for corners).
    for (u, row) in fp_per_u.iter_mut().enumerate() {
        let expected = (16 - 1 - (adj1.degree(u) + adj2.degree(u)).min(6)).min(12);
        assert_eq!(row.len(), expected);
        row.sort_unstable();
        row.dedup();
        assert_eq!(row.len(), expected);
    }
    // On grid(8, 8) (n = 64, max_deg = 4 <= 16), sufficient non-2-hop vertices exist so
    // FP strictly excludes both 1-hop and 2-hop neighbors.
    let grid8 = grid_graph(8, 8);
    let s8 = SgdSolver::<f64>::pacmap_graph(&grid8, cfg).unwrap();
    let st8 = s8.pacmap.as_ref().unwrap();
    let (adj1_8, adj2_8) = extract_1hop_2hop_csr(&grid8).unwrap();
    for &(u, v) in &st8.fp_pairs {
        assert_ne!(u, v);
        let ui = u as usize;
        assert!(!adj1_8.neighbors(ui).contains(&v));
        assert!(
            !adj2_8.neighbors(ui).contains(&v),
            "grid(8, 8) FP pair ({u}, {v}) must not be 2-hop"
        );
    }

    // Bit-exact step() == run() for f64 and f32 across all 3 phases.
    let mut s64_run = SgdSolver::<f64>::pacmap_graph(&grid, cfg).unwrap();
    let mut s64_step = SgdSolver::<f64>::pacmap_graph(&grid, cfg).unwrap();
    let mut l64_run = pivot_mds(&grid, &PivotMdsConfig::default()).unwrap();
    let mut l64_step = l64_run.clone();
    s64_run.run(&mut l64_run).unwrap();
    s64_step.run_steps(&mut l64_step, 35).unwrap(); // across phase 1 -> 2
    s64_step.run_steps(&mut l64_step, 35).unwrap(); // across phase 2 -> 3
    while !s64_step.status().is_finished() {
        s64_step.step(&mut l64_step).unwrap();
    }
    assert_eq!(s64_step.iteration(), 120);
    for i in 0..grid.num_nodes() {
        assert_eq!(l64_run.x()[i].to_bits(), l64_step.x()[i].to_bits());
        assert_eq!(l64_run.y()[i].to_bits(), l64_step.y()[i].to_bits());
    }

    let mut s32_run = SgdSolver::<f32>::pacmap_graph(&grid, cfg).unwrap();
    let mut s32_step = SgdSolver::<f32>::pacmap_graph(&grid, cfg).unwrap();
    let mut l32_run = circle_layout::<f32>(grid.num_nodes(), 1.0);
    let mut l32_step = l32_run.clone();
    s32_run.run(&mut l32_run).unwrap();
    s32_step.run_steps(&mut l32_step, 45).unwrap();
    s32_step.run(&mut l32_step).unwrap();
    for i in 0..grid.num_nodes() {
        assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
        assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
    }

    // Barbell separation & NP2 quality with default PaCMAP-Graph parameters on a 20-node barbell.
    let barbell20 = barbell_graph(10);
    let mut bb_solver =
        SgdSolver::<f64>::pacmap_graph(&barbell20, PacmapGraphConfig::default()).unwrap();
    let mut bb_layout = circle_layout::<f64>(barbell20.num_nodes(), 0.25);
    bb_solver.run(&mut bb_layout).unwrap();
    assert!(bb_layout.is_all_finite());
    let c0x = bb_layout.x()[..10].iter().sum::<f64>() / 10.0;
    let c0y = bb_layout.y()[..10].iter().sum::<f64>() / 10.0;
    let c1x = bb_layout.x()[10..].iter().sum::<f64>() / 10.0;
    let c1y = bb_layout.y()[10..].iter().sum::<f64>() / 10.0;
    let sep = (c0x - c1x).hypot(c0y - c1y);
    assert!(sep > 1.0, "expected separated barbell cliques, got {sep}");
    assert!(compute_np2(&barbell20, &bb_layout) >= 0.85);

    // Weighted and directed graph paths work cleanly.
    let dwg = CsrGraph::<f64>::from_edges(
        6,
        &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],
        Some(&[1.0, 2.0, 1.5, 1.0, 2.5, 0.8]),
        true,
        false,
    )
    .unwrap();
    let mut dw_pm = SgdSolver::<f64>::pacmap_graph(&dwg, cfg).unwrap();
    let mut dw_l = circle_layout::<f64>(6, 1.0);
    dw_pm.run(&mut dw_l).unwrap();
    assert!(dw_l.is_all_finite());

    // Edge cases & invalid configs.
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    assert_eq!(
        SgdSolver::<f64>::pacmap_graph(&empty, cfg)
            .unwrap()
            .run(&mut Layout2d::zeros(0))
            .unwrap(),
        SolverStatus::Completed
    );
    assert_eq!(
        SgdSolver::<f64>::pacmap_graph(&single, cfg)
            .unwrap()
            .run(&mut Layout2d::zeros(1))
            .unwrap(),
        SolverStatus::Completed
    );
    assert_eq!(
        SgdSolver::<f64>::pacmap_graph(
            &grid,
            PacmapGraphConfig {
                phase1_iters: 0,
                phase2_iters: 0,
                phase3_iters: 0,
                ..cfg
            }
        )
        .unwrap()
        .status(),
        SolverStatus::Completed
    );
    for bad in [
        PacmapGraphConfig {
            n_neighbors: 0,
            ..cfg
        },
        PacmapGraphConfig {
            mn_ratio: -0.1,
            ..cfg
        },
        PacmapGraphConfig {
            fp_ratio: f64::NAN,
            ..cfg
        },
        PacmapGraphConfig {
            learning_rate: 0.0,
            ..cfg
        },
        PacmapGraphConfig {
            max_1hop_per_node: Some(0),
            ..cfg
        },
        PacmapGraphConfig {
            degree_weight_beta: -0.1,
            ..cfg
        },
        PacmapGraphConfig {
            degree_weight_beta: f64::NAN,
            ..cfg
        },
    ] {
        assert!(SgdSolver::<f64>::pacmap_graph(&grid, bad).is_err());
    }
}

#[test]
fn test_pacmap_graph_keep_all_1hop_and_fp_2hop_exclusion() {
    // Hub 0 connected to 1..=20 plus a path 20-21-22-23-24-25 (so leaves have deg < n_neighbors
    // and vertices near the hub have 2-hop neighbors outside truncated nb_u).
    let mut edges: Vec<(u32, u32)> = (1..=20).map(|v| (0, v)).collect();
    edges.extend([(20, 21), (21, 22), (22, 23), (23, 24), (24, 25)]);
    let n = 26usize;
    let unweighted = CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap();
    let weights: Vec<f64> = (0..edges.len()).map(|i| 1.0 + 0.1 * (i as f64)).collect();
    let weighted = CsrGraph::<f64>::from_edges(n, &edges, Some(&weights), false, false).unwrap();

    let (adj1, adj2) = extract_1hop_2hop_csr(&unweighted).unwrap();
    let cfg_trunc = PacmapGraphConfig {
        n_neighbors: 6,
        mn_ratio: 0.5,
        fp_ratio: 2.0,
        phase1_iters: 5,
        phase2_iters: 5,
        phase3_iters: 5,
        seed: 17,
        keep_all_1hop: false,
        ..Default::default()
    };
    let cfg_keep = PacmapGraphConfig {
        keep_all_1hop: true,
        ..cfg_trunc
    };

    for g in [&unweighted, &weighted] {
        let s_trunc = SgdSolver::<f64>::pacmap_graph(g, cfg_trunc).unwrap();
        let st_trunc = s_trunc.pacmap.as_ref().unwrap();
        // Zero duplicate undirected pairs in nb_pairs (all canonical u < v).
        for w in st_trunc.nb_pairs.windows(2) {
            assert!(w[0] < w[1], "duplicate or unsorted nb_pair: {w:?}");
        }
        assert!(st_trunc.nb_pairs.iter().all(|&(u, v)| u < v));
        // Hub 0 has 20 1-hop neighbors (1..=20), all selected reciprocally by leaves 1..=19
        // (and node 21 may backfill 0 as a 2-hop neighbor), deduplicated with zero duplicates.
        let hub_inc_trunc = st_trunc
            .nb_pairs
            .iter()
            .filter(|&&(u, v)| u == 0 || v == 0)
            .count();
        assert!(hub_inc_trunc >= 20);

        let s_keep = SgdSolver::<f64>::pacmap_graph(g, cfg_keep).unwrap();
        let st_keep = s_keep.pacmap.as_ref().unwrap();
        for w in st_keep.nb_pairs.windows(2) {
            assert!(w[0] < w[1], "duplicate or unsorted nb_pair: {w:?}");
        }
        let hub_nb_keep: Vec<u32> = st_keep
            .nb_pairs
            .iter()
            .filter_map(|&(u, v)| {
                if u == 0 {
                    Some(v)
                } else if v == 0 {
                    Some(u)
                } else {
                    None
                }
            })
            .collect();
        assert!(hub_nb_keep.len() >= 20);
        for v in 1..=20u32 {
            assert!(hub_nb_keep.contains(&v));
        }
        // Leaf node 1 has 1-hop degree 1 < 6, so 2-hop backfill adds at least 5 pairs -> >= 6 incident.
        let leaf_nb_keep = st_keep
            .nb_pairs
            .iter()
            .filter(|&&(u, v)| u == 1 || v == 1)
            .count();
        assert!(leaf_nb_keep >= 6);

        // FP must strictly exclude 1-hop edges even when deg > n_neighbors, and tail vertices
        // with non-hub 2-hop neighbors (u >= 22) still exclude adj2.
        for &(u, v) in &st_trunc.fp_pairs {
            assert_ne!(u, v);
            let ui = u as usize;
            assert!(!adj1.neighbors(ui).contains(&v));
            if ui >= 22 {
                assert!(!adj2.neighbors(ui).contains(&v));
            }
        }
        for &(u, v) in &st_keep.fp_pairs {
            assert_ne!(u, v);
            let ui = u as usize;
            assert!(!adj1.neighbors(ui).contains(&v));
            if ui >= 22 {
                assert!(!adj2.neighbors(ui).contains(&v));
            }
        }
    }

    // Larger non-hub graph (n = 70, all degrees <= 10 <= non_hub_thresh = 16):
    // deg(0) = 10 > n_neighbors = 4, and FP strictly excludes both adj1 and adj2.
    let mut nh_edges: Vec<(u32, u32)> = (1..=10u32).map(|v| (0, v)).collect();
    for v in 10..69u32 {
        nh_edges.push((v, v + 1));
    }
    let g_nh = CsrGraph::<f64>::from_edges(70, &nh_edges, None, false, false).unwrap();
    let (nh_adj1, nh_adj2) = extract_1hop_2hop_csr(&g_nh).unwrap();
    let nh_cfg_trunc = PacmapGraphConfig {
        n_neighbors: 4,
        mn_ratio: 0.5,
        fp_ratio: 2.0,
        phase1_iters: 2,
        phase2_iters: 2,
        phase3_iters: 2,
        seed: 17,
        keep_all_1hop: false,
        ..Default::default()
    };
    let nh_cfg_keep = PacmapGraphConfig {
        keep_all_1hop: true,
        ..nh_cfg_trunc
    };
    for nh_cfg in [nh_cfg_trunc, nh_cfg_keep] {
        let s_nh = SgdSolver::<f64>::pacmap_graph(&g_nh, nh_cfg).unwrap();
        let st_nh = s_nh.pacmap.as_ref().unwrap();
        assert!(!st_nh.fp_pairs.is_empty());
        for &(u, v) in &st_nh.fp_pairs {
            assert_ne!(u, v);
            let ui = u as usize;
            assert!(!nh_adj1.neighbors(ui).contains(&v));
            assert!(
                !nh_adj2.neighbors(ui).contains(&v),
                "non-hub FP pair ({u}, {v}) must not be 2-hop"
            );
        }
    }
}
