use egraph_core::{
    CoreError, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, Scalar, SolverStatus,
};

use super::*;
use crate::continuous_fdp::{ForceAtlas2Config, ForceAtlas2Solver};

fn seeded_layout<S: Scalar>(n: usize, seed: u64) -> Layout2d<S> {
    let mut rng = DeterministicRng::from_seed_u64(seed);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        x.push(S::from_f64(2.0 * (rng.next_f64() - 0.5)));
        y.push(S::from_f64(2.0 * (rng.next_f64() - 0.5)));
    }
    Layout2d::from_xy(x, y).unwrap()
}

fn planted_asymmetric_clusters(small_k: usize, large_k: usize) -> CsrGraph<f64> {
    let n = small_k + large_k;
    let mut edges = Vec::new();
    for i in 0..small_k {
        for j in (i + 1)..small_k {
            edges.push((i as u32, j as u32));
        }
    }
    for i in 0..large_k {
        for j in (i + 1)..large_k {
            edges.push(((small_k + i) as u32, (small_k + j) as u32));
        }
    }
    // Single bridge edge connecting the small dense cluster to the large dense cluster.
    edges.push(((small_k - 1) as u32, small_k as u32));
    CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
}

fn cluster_compactness_and_separation(
    layout: &Layout2d<f64>,
    small_k: usize,
    large_k: usize,
) -> (f64, f64, f64) {
    let mut sx = 0.0;
    let mut sy = 0.0;
    for i in 0..small_k {
        sx += layout.x()[i];
        sy += layout.y()[i];
    }
    sx /= small_k as f64;
    sy /= small_k as f64;

    let mut lx = 0.0;
    let mut ly = 0.0;
    for i in small_k..(small_k + large_k) {
        lx += layout.x()[i];
        ly += layout.y()[i];
    }
    lx /= large_k as f64;
    ly /= large_k as f64;

    let inter = (sx - lx).hypot(sy - ly);

    let mut small_diam = 0.0_f64;
    for i in 0..small_k {
        for j in (i + 1)..small_k {
            let d = (layout.x()[i] - layout.x()[j]).hypot(layout.y()[i] - layout.y()[j]);
            small_diam = small_diam.max(d);
        }
    }
    let mut large_diam = 0.0_f64;
    for i in small_k..(small_k + large_k) {
        for j in (i + 1)..(small_k + large_k) {
            let d = (layout.x()[i] - layout.x()[j]).hypot(layout.y()[i] - layout.y()[j]);
            large_diam = large_diam.max(d);
        }
    }
    (small_diam, large_diam, inter)
}

#[test]
fn test_linlog_config_defaults_and_validation() {
    let cfg = LinLogConfig::default();
    assert_eq!(cfg.iterations, 200);
    assert_eq!(cfg.repulsion_model, LinLogRepulsion::EdgeRepulsion);
    assert_eq!(cfg.attraction_exponent, 0.0);
    assert_eq!(cfg.repulsion_exponent, -1.0);
    assert_eq!(cfg.theta, 1.2);
    assert!(cfg.use_barnes_hut);
    assert_eq!(cfg.gravity, 0.01);
    assert_eq!(cfg.edge_weight_influence, 1.0);
    assert_eq!(cfg.max_displacement, None);
    assert_eq!(cfg.seed, 42);
    cfg.validate().unwrap();

    for bad in [
        LinLogConfig {
            attraction_exponent: -1.0,
            repulsion_exponent: -1.0,
            ..cfg
        },
        LinLogConfig {
            attraction_exponent: -2.0,
            repulsion_exponent: -1.0,
            ..cfg
        },
        LinLogConfig {
            attraction_exponent: f64::NAN,
            ..cfg
        },
        LinLogConfig {
            repulsion_exponent: f64::NEG_INFINITY,
            ..cfg
        },
        LinLogConfig { theta: -0.1, ..cfg },
        LinLogConfig {
            gravity: -0.01,
            ..cfg
        },
        LinLogConfig {
            edge_weight_influence: -1.0,
            ..cfg
        },
        LinLogConfig {
            max_displacement: Some(0.0),
            ..cfg
        },
        LinLogConfig {
            max_displacement: Some(f64::NAN),
            ..cfg
        },
    ] {
        assert!(bad.validate().is_err());
    }
}

#[test]
fn test_linlog_pinhole_collapse_edge_vs_node_and_distinct_from_fa2_linlog() {
    let small_k = 5;
    let large_k = 25;
    let graph = planted_asymmetric_clusters(small_k, large_k);
    let init = seeded_layout::<f64>(graph.num_nodes(), 11);

    // 1. True Noack Edge-Repulsion LinLog exhibits extreme pinhole collapse of the small cluster
    //    relative to inter-cluster separation and the large cluster.
    let mut l_edge = init.clone();
    let mut s_edge = LinLogSolver::new(
        &graph,
        LinLogConfig {
            iterations: 300,
            repulsion_model: LinLogRepulsion::EdgeRepulsion,
            use_barnes_hut: false,
            ..Default::default()
        },
    )
    .unwrap();
    let e0 = s_edge.energy(&l_edge).unwrap();
    assert_eq!(s_edge.run(&mut l_edge).unwrap(), SolverStatus::Completed);
    let e1 = s_edge.energy(&l_edge).unwrap();
    assert!(e1 < e0, "expected LinLog energy decrease: {e0} -> {e1}");

    let (small_d_edge, large_d_edge, inter_edge) =
        cluster_compactness_and_separation(&l_edge, small_k, large_k);
    let pinhole_ratio_edge = inter_edge / small_d_edge.max(1e-12);
    assert!(
        pinhole_ratio_edge > 20.0,
        "expected strong pinhole collapse separation/small_diam > 20, got {pinhole_ratio_edge:.3} (small={small_d_edge:.3}, large={large_d_edge:.3}, inter={inter_edge:.3})"
    );

    // 2. Node-Repulsion LinLog produces distinctly less asymmetric cluster expansion than Edge-Repulsion.
    let mut l_node = init.clone();
    let mut s_node = LinLogSolver::new(
        &graph,
        LinLogConfig {
            iterations: 300,
            repulsion_model: LinLogRepulsion::NodeRepulsion,
            use_barnes_hut: false,
            ..Default::default()
        },
    )
    .unwrap();
    s_node.run(&mut l_node).unwrap();
    let (small_d_node, _large_d_node, inter_node) =
        cluster_compactness_and_separation(&l_node, small_k, large_k);
    assert!(
        inter_edge > inter_node * 2.0,
        "expected EdgeRepulsion inter-cluster distance ({inter_edge:.3}) >> NodeRepulsion ({inter_node:.3})"
    );
    assert!(
        (inter_edge / small_d_edge.max(1e-12)) > (inter_node / small_d_node.max(1e-12)) * 1.5,
        "expected EdgeRepulsion pinhole ratio > 1.5x NodeRepulsion"
    );

    // 3. ForceAtlas2(linlog_mode: true) uses ln(1 + d) attraction + (deg + 1) repulsion,
    //    producing distinctly different cluster compactness/separation than true Noack LinLog.
    //    3a. Compare EdgeRepulsion LinLog against FA2(linlog_mode: true) with matched gravity.
    let mut l_fa2_matched = init.clone();
    ForceAtlas2Solver::new(
        &graph,
        ForceAtlas2Config {
            iterations: 300,
            linlog_mode: true,
            use_barnes_hut: false,
            gravity: 0.01,
            scaling_ratio: 1.0,
            ..Default::default()
        },
    )
    .unwrap()
    .run(&mut l_fa2_matched)
    .unwrap();
    let (small_d_fa2, _large_d_fa2, inter_fa2) =
        cluster_compactness_and_separation(&l_fa2_matched, small_k, large_k);
    let pinhole_ratio_fa2 = inter_fa2 / small_d_fa2.max(1e-12);

    assert!(
        (inter_edge - inter_fa2).abs() / inter_edge.max(1.0) > 0.25,
        "expected distinct separation between Noack LinLog ({inter_edge:.3}) and FA2 linlog_mode ({inter_fa2:.3})"
    );
    assert!(
        pinhole_ratio_edge > pinhole_ratio_fa2 * 1.25,
        "expected Noack LinLog pinhole ratio ({pinhole_ratio_edge:.3}) > 1.25x FA2 linlog_mode ({pinhole_ratio_fa2:.3})"
    );

    //    3b. Compare default FA2(linlog_mode: true) against Noack NodeRepulsion and EdgeRepulsion:
    //        Noack NodeRepulsion collapses dense clusters tightly while FA2(linlog_mode: true)
    //        keeps both intra-cluster diameters larger and inter-cluster ratio much smaller.
    let mut l_fa2_default = init.clone();
    ForceAtlas2Solver::new(
        &graph,
        ForceAtlas2Config {
            iterations: 300,
            linlog_mode: true,
            use_barnes_hut: false,
            ..Default::default()
        },
    )
    .unwrap()
    .run(&mut l_fa2_default)
    .unwrap();
    let (small_d_fa2_def, _large_d_fa2_def, inter_fa2_def) =
        cluster_compactness_and_separation(&l_fa2_default, small_k, large_k);
    let pinhole_ratio_fa2_def = inter_fa2_def / small_d_fa2_def.max(1e-12);
    assert!(
        pinhole_ratio_edge > pinhole_ratio_fa2_def * 3.0,
        "expected Noack LinLog pinhole ratio ({pinhole_ratio_edge:.3}) > 3x default FA2 linlog_mode ({pinhole_ratio_fa2_def:.3})"
    );
    assert!(
        small_d_node < small_d_fa2_def * 0.5,
        "expected NodeRepulsion small cluster diameter ({small_d_node:.3}) < 0.5x default FA2 linlog_mode ({small_d_fa2_def:.3})"
    );
}

#[test]
fn test_linlog_barnes_hut_vs_exact_and_general_polylog_exponents() {
    let graph = planted_asymmetric_clusters(4, 6);
    let init = seeded_layout::<f64>(graph.num_nodes(), 77);

    // 1. Coulomb r = -1.0: ArenaQuadtree2d with theta = 0.0 matches exact O(N^2).
    let mut l_exact = init.clone();
    let mut l_bh0 = init.clone();
    LinLogSolver::new(
        &graph,
        LinLogConfig {
            iterations: 20,
            use_barnes_hut: false,
            ..Default::default()
        },
    )
    .unwrap()
    .run(&mut l_exact)
    .unwrap();
    LinLogSolver::new(
        &graph,
        LinLogConfig {
            iterations: 20,
            use_barnes_hut: true,
            theta: 0.0,
            ..Default::default()
        },
    )
    .unwrap()
    .run(&mut l_bh0)
    .unwrap();
    for i in 0..graph.num_nodes() {
        assert!((l_exact.x()[i] - l_bh0.x()[i]).abs() < 1e-8);
        assert!((l_exact.y()[i] - l_bh0.y()[i]).abs() < 1e-8);
    }

    // 2. General (a=2.0, r=-1.0) 3-PolyLog (FR-like energy) and (a=0.5, r=-1.5):
    //    theta = 0.0 matches exact, and default theta = 1.2 converges cleanly without NaN/explosion.
    for (a_exp, r_exp) in [(2.0, -1.0), (0.5, -1.5)] {
        let mut l_gen_ex = init.clone();
        let mut l_gen_bh0 = init.clone();
        let mut l_gen_bh = init.clone();
        let base_cfg = LinLogConfig {
            iterations: 25,
            attraction_exponent: a_exp,
            repulsion_exponent: r_exp,
            ..Default::default()
        };
        LinLogSolver::new(
            &graph,
            LinLogConfig {
                use_barnes_hut: false,
                ..base_cfg
            },
        )
        .unwrap()
        .run(&mut l_gen_ex)
        .unwrap();
        LinLogSolver::new(
            &graph,
            LinLogConfig {
                use_barnes_hut: true,
                theta: 0.0,
                ..base_cfg
            },
        )
        .unwrap()
        .run(&mut l_gen_bh0)
        .unwrap();
        let mut s_bh = LinLogSolver::new(
            &graph,
            LinLogConfig {
                use_barnes_hut: true,
                theta: 1.2,
                ..base_cfg
            },
        )
        .unwrap();
        let e_before = s_bh.energy(&l_gen_bh).unwrap();
        s_bh.run(&mut l_gen_bh).unwrap();
        let e_after = s_bh.energy(&l_gen_bh).unwrap();
        assert!(l_gen_bh.is_all_finite());
        assert!(
            e_after < e_before,
            "expected energy decrease for (a={a_exp}, r={r_exp}): {e_before} -> {e_after}"
        );
        for i in 0..graph.num_nodes() {
            assert!((l_gen_ex.x()[i] - l_gen_bh0.x()[i]).abs() < 1e-8);
            assert!((l_gen_ex.y()[i] - l_gen_bh0.y()[i]).abs() < 1e-8);
        }
    }
}

#[test]
fn test_linlog_step_vs_run_bit_exact_and_edge_cases() {
    let graph = planted_asymmetric_clusters(4, 8);
    let cfg = LinLogConfig {
        iterations: 30,
        ..Default::default()
    };

    // f64 bit-exact step vs run
    let mut l64_r = seeded_layout::<f64>(12, 5);
    let mut l64_s = l64_r.clone();
    let mut s64_r = LinLogSolver::<f64>::new(&graph, cfg).unwrap();
    let mut s64_s = LinLogSolver::<f64>::new(&graph, cfg).unwrap();
    s64_r.run(&mut l64_r).unwrap();
    assert_eq!(
        s64_s.run_steps(&mut l64_s, 11).unwrap(),
        SolverStatus::Running
    );
    while !s64_s.status().is_finished() {
        s64_s.step(&mut l64_s).unwrap();
    }
    for i in 0..12 {
        assert_eq!(l64_r.x()[i].to_bits(), l64_s.x()[i].to_bits());
        assert_eq!(l64_r.y()[i].to_bits(), l64_s.y()[i].to_bits());
    }

    // f32 bit-exact step vs run
    let mut l32_r = seeded_layout::<f32>(12, 5);
    let mut l32_s = l32_r.clone();
    let mut s32_r = LinLogSolver::<f32>::new(&graph, cfg).unwrap();
    let mut s32_s = LinLogSolver::<f32>::new(&graph, cfg).unwrap();
    s32_r.run(&mut l32_r).unwrap();
    while !s32_s.status().is_finished() {
        s32_s.step(&mut l32_s).unwrap();
    }
    for i in 0..12 {
        assert_eq!(l32_r.x()[i].to_bits(), l32_s.x()[i].to_bits());
        assert_eq!(l32_r.y()[i].to_bits(), l32_s.y()[i].to_bits());
    }

    // Degenerate graphs: N=0, N=1, edgeless, collapsed start, mixed connected+isolated, directed reciprocal.
    for n in [0usize, 1, 5] {
        let g = CsrGraph::<f64>::from_edges(n, &[], None, false, false).unwrap();
        let mut l = Layout2d::<f64>::zeros(n);
        let mut s = LinLogSolver::new(&g, cfg).unwrap();
        assert!(s.run(&mut l).unwrap().is_finished());
        assert!(l.is_all_finite());
    }

    let mixed =
        CsrGraph::<f64>::from_edges(5, &[(0, 1), (1, 2), (2, 0)], None, false, false).unwrap();
    let mut l_mixed = Layout2d::<f64>::zeros(5);
    let mut s_mixed = LinLogSolver::new(&mixed, cfg).unwrap();
    assert_eq!(s_mixed.masses()[3], ISOLATED_NODE_MASS);
    assert_eq!(s_mixed.masses()[4], ISOLATED_NODE_MASS);
    assert!(s_mixed.run(&mut l_mixed).unwrap().is_finished());
    assert!(l_mixed.is_all_finite());

    let dir_g = CsrGraph::from_edges(
        4,
        &[(0, 1), (1, 0), (1, 2), (2, 3)],
        Some(&[2.0, 4.0, 1.5, 3.0]),
        true,
        false,
    )
    .unwrap();
    let mut l_dir = seeded_layout::<f64>(4, 9);
    assert!(
        LinLogSolver::new(&dir_g, cfg)
            .unwrap()
            .run(&mut l_dir)
            .unwrap()
            .is_finished()
    );
    assert!(l_dir.is_all_finite());

    // Zero-iteration immediate completion & error paths.
    let s_zero = LinLogSolver::<f64>::new(
        &mixed,
        LinLogConfig {
            iterations: 0,
            ..cfg
        },
    )
    .unwrap();
    assert_eq!(s_zero.status(), SolverStatus::Completed);

    let neg_g = CsrGraph::from_edges(2, &[(0, 1)], Some(&[-1.0]), false, false).unwrap();
    assert!(matches!(
        LinLogSolver::<f64>::new(&neg_g, cfg),
        Err(CoreError::InvalidInput(_))
    ));

    let mut s_err = LinLogSolver::<f64>::new(&mixed, cfg).unwrap();
    let mut wrong = Layout2d::<f64>::zeros(2);
    assert!(matches!(
        s_err.step(&mut wrong),
        Err(CoreError::DimensionMismatch { .. })
    ));
    let mut bad = Layout2d::from_xy(vec![0.0, f64::NAN, 0.0, 0.0, 0.0], vec![0.0; 5]).unwrap();
    assert!(matches!(
        s_err.step(&mut bad),
        Err(CoreError::NonFiniteValue { .. })
    ));
}
