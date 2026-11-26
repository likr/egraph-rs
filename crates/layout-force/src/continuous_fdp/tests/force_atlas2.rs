use egraph_core::{CoreError, CsrGraph, IterativeSolver, Layout2d, Scalar, SolverStatus};

use super::super::force_atlas2::MAX_FA2_DISPLACEMENT;
use super::super::*;
use super::{barbell_graph, grid_graph, seeded_layout};

fn cluster_separation_ratio<S: Scalar>(layout: &Layout2d<S>, k: usize) -> f64 {
    let mut c0x = 0.0;
    let mut c0y = 0.0;
    let mut c1x = 0.0;
    let mut c1y = 0.0;
    for i in 0..k {
        c0x += layout.x()[i].to_f64();
        c0y += layout.y()[i].to_f64();
        c1x += layout.x()[k + i].to_f64();
        c1y += layout.y()[k + i].to_f64();
    }
    c0x /= k as f64;
    c0y /= k as f64;
    c1x /= k as f64;
    c1y /= k as f64;
    let inter = (c0x - c1x).hypot(c0y - c1y);

    let mut intra = 0.0;
    for i in 0..k {
        intra += (layout.x()[i].to_f64() - c0x).hypot(layout.y()[i].to_f64() - c0y);
        intra += (layout.x()[k + i].to_f64() - c1x).hypot(layout.y()[k + i].to_f64() - c1y);
    }
    intra /= (2 * k) as f64;
    inter / intra.max(1e-12)
}

#[test]
fn test_force_atlas2_barnes_hut_vs_exact_and_modes() {
    let graph = barbell_graph(6);
    let init = seeded_layout::<f64>(graph.num_nodes(), 7);

    // 1. Barnes-Hut with theta=0.0 matches Exact to machine-level precision.
    let mut l_exact = init.clone();
    let mut l_bh_zero = init.clone();
    let cfg_exact = ForceAtlas2Config {
        iterations: 20,
        use_barnes_hut: false,
        ..Default::default()
    };
    let cfg_bh_zero = ForceAtlas2Config {
        iterations: 20,
        theta: 0.0,
        use_barnes_hut: true,
        ..Default::default()
    };
    ForceAtlas2Solver::new(&graph, cfg_exact)
        .unwrap()
        .run(&mut l_exact)
        .unwrap();
    ForceAtlas2Solver::new(&graph, cfg_bh_zero)
        .unwrap()
        .run(&mut l_bh_zero)
        .unwrap();
    for i in 0..graph.num_nodes() {
        assert!((l_exact.x()[i] - l_bh_zero.x()[i]).abs() < 1e-8);
        assert!((l_exact.y()[i] - l_bh_zero.y()[i]).abs() < 1e-8);
    }

    // 2. Standard Barnes-Hut (theta=0.5), LinLog, strong gravity, outbound attraction distribution.
    for cfg in [
        ForceAtlas2Config {
            iterations: 150,
            use_barnes_hut: true,
            ..Default::default()
        },
        ForceAtlas2Config {
            iterations: 150,
            linlog_mode: true,
            strong_gravity: true,
            gravity: 0.5,
            outbound_attraction_distribution: true,
            ..Default::default()
        },
    ] {
        let mut layout = init.clone();
        let mut solver = ContinuousFdp::force_atlas2(&graph, cfg).unwrap();
        assert_eq!(solver.run(&mut layout).unwrap(), SolverStatus::Completed);
        assert!(layout.is_all_finite());
        let sep = cluster_separation_ratio(&layout, 6);
        assert!(
            sep > 1.5,
            "expected barbell cluster separation > 1.5, got {sep}"
        );
    }
}

#[test]
fn test_force_atlas2_step_vs_run_bit_exact_f64_and_f32() {
    let graph = grid_graph(4, 4);
    let cfg = ForceAtlas2Config {
        iterations: 25,
        linlog_mode: true,
        outbound_attraction_distribution: true,
        ..Default::default()
    };

    // f64 bit-exact check
    let mut l_run = seeded_layout::<f64>(16, 99);
    let mut l_step = l_run.clone();
    let mut s_run = ForceAtlas2Solver::<f64>::new(&graph, cfg).unwrap();
    let mut s_step = ForceAtlas2Solver::<f64>::new(&graph, cfg).unwrap();
    assert_eq!(s_run.run(&mut l_run).unwrap(), SolverStatus::Completed);
    assert_eq!(
        s_step.run_steps(&mut l_step, 10).unwrap(),
        SolverStatus::Running
    );
    assert_eq!(s_step.iteration(), 10);
    while !s_step.status().is_finished() {
        s_step.step(&mut l_step).unwrap();
    }
    assert_eq!(s_step.iteration(), 25);
    for i in 0..16 {
        assert_eq!(l_run.x()[i].to_bits(), l_step.x()[i].to_bits());
        assert_eq!(l_run.y()[i].to_bits(), l_step.y()[i].to_bits());
    }

    // f32 bit-exact check via ContinuousFdp facade
    let mut l32_run = seeded_layout::<f32>(16, 99);
    let mut l32_step = l32_run.clone();
    let mut s32_run = ContinuousFdp::<f32>::force_atlas2(&graph, cfg).unwrap();
    let mut s32_step = ContinuousFdp::<f32>::force_atlas2(&graph, cfg).unwrap();
    s32_run.run(&mut l32_run).unwrap();
    while !s32_step.status().is_finished() {
        s32_step.step(&mut l32_step).unwrap();
    }
    for i in 0..16 {
        assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
        assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
    }
}

#[test]
fn test_force_atlas2_scale_adaptive_max_displacement_and_step_parity() {
    let grid_16 = grid_graph(4, 4);
    let solver_small =
        ForceAtlas2Solver::<f64>::new(&grid_16, ForceAtlas2Config::default()).unwrap();
    assert_eq!(solver_small.max_displacement(), MAX_FA2_DISPLACEMENT);

    // N = 90_000 sparse path graph scales max_displacement to 50 * sqrt(90) ~= 474.3416.
    let n_large = 90_000usize;
    let large_edges: Vec<(u32, u32)> = (0..n_large - 1)
        .map(|i| (i as u32, (i + 1) as u32))
        .collect();
    let g_large = CsrGraph::<f64>::from_edges(n_large, &large_edges, None, false, false).unwrap();
    let solver_large =
        ForceAtlas2Solver::<f64>::new(&g_large, ForceAtlas2Config::default()).unwrap();
    let expected_large = MAX_FA2_DISPLACEMENT * ((n_large as f64) / 1000.0).sqrt();
    assert!((solver_large.max_displacement() - expected_large).abs() < 1e-12);
    assert!((solver_large.max_displacement() - 474.3416490252569).abs() < 1e-9);

    // Explicit override clamps single-step displacement to <= max_displacement.
    let pair_g = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
    let cap_val = 12.5_f64;
    let mut s_cap = ForceAtlas2Solver::<f64>::new(
        &pair_g,
        ForceAtlas2Config {
            iterations: 1,
            max_displacement: Some(cap_val),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(s_cap.max_displacement(), cap_val);
    let init_pair = Layout2d::from_xy(vec![-500.0, 500.0], vec![0.0, 0.0]).unwrap();
    let mut l_cap = init_pair.clone();
    s_cap.step(&mut l_cap).unwrap();
    for i in 0..2 {
        let disp = (l_cap.x()[i] - init_pair.x()[i]).hypot(l_cap.y()[i] - init_pair.y()[i]);
        assert!(
            disp <= cap_val + 1e-12,
            "vertex {i} moved {disp} > {cap_val}"
        );
    }

    // step == run bit-exactness (f64 and f32) for both None and Some(2.5).
    for max_disp in [None, Some(2.5)] {
        let cfg = ForceAtlas2Config {
            iterations: 12,
            max_displacement: max_disp,
            ..Default::default()
        };
        let mut l64_r = seeded_layout::<f64>(16, 21);
        let mut l64_s = l64_r.clone();
        let mut s64_r = ForceAtlas2Solver::<f64>::new(&grid_16, cfg).unwrap();
        let mut s64_s = ForceAtlas2Solver::<f64>::new(&grid_16, cfg).unwrap();
        s64_r.run(&mut l64_r).unwrap();
        s64_s.run_steps(&mut l64_s, 4).unwrap();
        while !s64_s.status().is_finished() {
            s64_s.step(&mut l64_s).unwrap();
        }
        for i in 0..16 {
            assert_eq!(l64_r.x()[i].to_bits(), l64_s.x()[i].to_bits());
            assert_eq!(l64_r.y()[i].to_bits(), l64_s.y()[i].to_bits());
        }

        let mut l32_r = seeded_layout::<f32>(16, 21);
        let mut l32_s = l32_r.clone();
        let mut s32_r = ForceAtlas2Solver::<f32>::new(&grid_16, cfg).unwrap();
        let mut s32_s = ForceAtlas2Solver::<f32>::new(&grid_16, cfg).unwrap();
        s32_r.run(&mut l32_r).unwrap();
        s32_s.run_steps(&mut l32_s, 4).unwrap();
        while !s32_s.status().is_finished() {
            s32_s.step(&mut l32_s).unwrap();
        }
        for i in 0..16 {
            assert_eq!(l32_r.x()[i].to_bits(), l32_s.x()[i].to_bits());
            assert_eq!(l32_r.y()[i].to_bits(), l32_s.y()[i].to_bits());
        }
    }
}

fn assert_copy<T: Copy>(_: T) {}

fn count_disk_overlaps<S: Scalar>(layout: &Layout2d<S>, radii: &[f64]) -> usize {
    let n = layout.len();
    let mut overlaps = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = (layout.x()[i].to_f64() - layout.x()[j].to_f64())
                .hypot(layout.y()[i].to_f64() - layout.y()[j].to_f64());
            if d < radii[i] + radii[j] - 1e-6 {
                overlaps += 1;
            }
        }
    }
    overlaps
}

#[test]
fn test_force_atlas2_prevent_overlap_resolves_overlaps_and_validation() {
    let cfg_default = ForceAtlas2Config::default();
    assert!(!cfg_default.prevent_overlap);
    assert_eq!(cfg_default.default_node_radius, 1.0);
    assert_copy(cfg_default);

    // Invalid default_node_radius rejected.
    for bad_r in [-0.1, f64::NAN, f64::INFINITY] {
        assert!(
            ForceAtlas2Config {
                default_node_radius: bad_r,
                ..cfg_default
            }
            .validate()
            .is_err()
        );
    }

    // with_node_radii validation.
    let g4 = grid_graph(2, 2);
    assert!(matches!(
        ForceAtlas2Solver::<f64>::with_node_radii(&g4, cfg_default, vec![1.0; 3]),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        ForceAtlas2Solver::<f64>::with_node_radii(&g4, cfg_default, vec![1.0, -0.5, 1.0, 1.0]),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        ForceAtlas2Solver::<f64>::with_node_radii(&g4, cfg_default, vec![1.0, f64::NAN, 1.0, 1.0]),
        Err(CoreError::NonFiniteValue { .. })
    ));

    // 1. Overlapping initial layout on a 4x4 grid with heterogeneous radii -> 0 overlaps
    //    under both Exact and Barnes-Hut.
    let grid = grid_graph(4, 4);
    let radii: Vec<f64> = (0..16).map(|i| 0.8 + 0.1 * ((i % 4) as f64)).collect();
    let mut init_overlap = seeded_layout::<f64>(16, 42);
    // Compress initial positions so many pairs overlap initially.
    for i in 0..16 {
        init_overlap.x_mut()[i] *= 0.25;
        init_overlap.y_mut()[i] *= 0.25;
    }
    assert!(count_disk_overlaps(&init_overlap, &radii) > 10);

    for use_bh in [false, true] {
        let mut layout = init_overlap.clone();
        let cfg = ForceAtlas2Config {
            iterations: 300,
            use_barnes_hut: use_bh,
            prevent_overlap: true,
            linlog_mode: use_bh,
            ..Default::default()
        };
        let mut solver =
            ForceAtlas2Solver::<f64>::with_node_radii(&grid, cfg, radii.clone()).unwrap();
        assert_eq!(solver.node_radii(), radii.as_slice());
        assert_eq!(solver.run(&mut layout).unwrap(), SolverStatus::Completed);
        assert!(layout.is_all_finite());
        assert_eq!(
            count_disk_overlaps(&layout, &radii),
            0,
            "expected 0 disk overlaps with use_barnes_hut={use_bh}"
        );
    }

    // 2. Barnes-Hut with theta=0.0 matches Exact even with prevent_overlap=true and coincident starts.
    let mut l_exact = Layout2d::<f64>::zeros(6);
    let mut l_bh0 = Layout2d::<f64>::zeros(6);
    let g6 = barbell_graph(3);
    let r6 = vec![1.5; 6];
    ForceAtlas2Solver::with_node_radii(
        &g6,
        ForceAtlas2Config {
            iterations: 25,
            use_barnes_hut: false,
            prevent_overlap: true,
            ..Default::default()
        },
        r6.clone(),
    )
    .unwrap()
    .run(&mut l_exact)
    .unwrap();
    ForceAtlas2Solver::with_node_radii(
        &g6,
        ForceAtlas2Config {
            iterations: 25,
            use_barnes_hut: true,
            theta: 0.0,
            prevent_overlap: true,
            ..Default::default()
        },
        r6.clone(),
    )
    .unwrap()
    .run(&mut l_bh0)
    .unwrap();
    for i in 0..6 {
        assert!((l_exact.x()[i] - l_bh0.x()[i]).abs() < 1e-8);
        assert!((l_exact.y()[i] - l_bh0.y()[i]).abs() < 1e-8);
    }

    // 3. Bit-exact step() == run() for f64 and f32 with prevent_overlap=true.
    for use_bh in [false, true] {
        let cfg = ForceAtlas2Config {
            iterations: 25,
            use_barnes_hut: use_bh,
            prevent_overlap: true,
            default_node_radius: 1.25,
            ..Default::default()
        };
        let mut l64_r = seeded_layout::<f64>(16, 88);
        let mut l64_s = l64_r.clone();
        let mut s64_r = ForceAtlas2Solver::<f64>::new(&grid, cfg).unwrap();
        let mut s64_s = ForceAtlas2Solver::<f64>::new(&grid, cfg).unwrap();
        s64_r.run(&mut l64_r).unwrap();
        s64_s.run_steps(&mut l64_s, 9).unwrap();
        while !s64_s.status().is_finished() {
            s64_s.step(&mut l64_s).unwrap();
        }
        for i in 0..16 {
            assert_eq!(l64_r.x()[i].to_bits(), l64_s.x()[i].to_bits());
            assert_eq!(l64_r.y()[i].to_bits(), l64_s.y()[i].to_bits());
        }

        let mut l32_r = seeded_layout::<f32>(16, 88);
        let mut l32_s = l32_r.clone();
        let mut s32_r = ForceAtlas2Solver::<f32>::new(&grid, cfg).unwrap();
        let mut s32_s = ForceAtlas2Solver::<f32>::new(&grid, cfg).unwrap();
        s32_r.run(&mut l32_r).unwrap();
        while !s32_s.status().is_finished() {
            s32_s.step(&mut l32_s).unwrap();
        }
        for i in 0..16 {
            assert_eq!(l32_r.x()[i].to_bits(), l32_s.x()[i].to_bits());
            assert_eq!(l32_r.y()[i].to_bits(), l32_s.y()[i].to_bits());
        }
    }
}
