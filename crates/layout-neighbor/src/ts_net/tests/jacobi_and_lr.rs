use egraph_core::{CsrGraph, IterativeSolver, Layout2d, SolverStatus};
use egraph_neighborhood::{SparseProbabilities, SymmetrizationMode};

use super::super::config::compute_stage_lr;
use super::super::{TsNetBackend, TsNetConfig, TsNetSolver, TsNetStage};
use super::{make_barbell_graph, make_grid_graph, seeded_layout};

#[test]
fn test_stage_lr_stability_cap_and_velocity_rescaling() {
    // 1. For small graphs (N < 100), all stages are capped at 2.0 * N even when lambda_c > 0.
    let g = make_barbell_graph(3); // n = 6
    let cfg_small = TsNetConfig {
        backend: TsNetBackend::Exact,
        iterations_stage1: 2,
        iterations_stage2: 2,
        iterations_stage3: 2,
        lambda_c_stage1: 0.1,
        lambda_c_stage2: 0.01,
        lambda_c_stage3: 0.0,
        lambda_r_stage3: 0.6,
        learning_rate: Some(200.0),
        momentum: 0.8,
        ..TsNetConfig::pure_2hop()
    };
    let mut solver_small = TsNetSolver::<f64>::from_graph_pure_2hop(&g, cfg_small).unwrap();
    let mut layout_small = seeded_layout::<f64>(6, 0.5, 31);
    solver_small.step(&mut layout_small).unwrap();
    assert!((solver_small.current_stage_lr() - 12.0).abs() < 1e-10);
    solver_small.step(&mut layout_small).unwrap();
    solver_small.step(&mut layout_small).unwrap();
    assert!((solver_small.current_stage_lr() - 12.0).abs() < 1e-10);
    solver_small.step(&mut layout_small).unwrap();
    solver_small.step(&mut layout_small).unwrap();
    assert!((solver_small.current_stage_lr() - 12.0).abs() < 1e-10);
    assert!(layout_small.is_all_finite());

    // 2. For N = 30 (N < 100), Stage 1/2 cap is 2*30 = 60.0 -> Stage 3 (lambda_r > 0) caps at 50.0
    // and rescales velocity by 50/60.
    let g30 = make_grid_graph(5, 6); // n = 30
    let cfg_30 = TsNetConfig {
        backend: TsNetBackend::Exact,
        iterations_stage1: 2,
        iterations_stage2: 2,
        iterations_stage3: 2,
        lambda_c_stage1: 0.1,
        lambda_c_stage2: 0.01,
        lambda_c_stage3: 0.0,
        lambda_r_stage3: 0.6,
        learning_rate: Some(200.0),
        momentum: 0.8,
        ..TsNetConfig::pure_2hop()
    };
    let mut solver30 = TsNetSolver::<f64>::from_graph_pure_2hop(&g30, cfg_30).unwrap();
    let mut l30 = seeded_layout::<f64>(30, 0.5, 31);
    solver30.step(&mut l30).unwrap();
    assert!((solver30.current_stage_lr() - 60.0).abs() < 1e-10);
    solver30.step(&mut l30).unwrap();
    solver30.step(&mut l30).unwrap();
    assert!((solver30.current_stage_lr() - 60.0).abs() < 1e-10);
    solver30.step(&mut l30).unwrap();
    let vel_x_end_s2 = solver30.vel_x[0];
    solver30.step(&mut l30).unwrap();
    assert!((solver30.current_stage_lr() - 50.0).abs() < 1e-10);
    assert!(vel_x_end_s2.is_finite());
    assert!(l30.is_all_finite());

    // 3. For N >= 100 (N = 120), Stage 1 stability cap max_stable_lr = 16.83 * 120 = 2019.6
    // applies without the N < 100 2*N cap.
    let g120 = make_grid_graph(10, 12); // n = 120
    let cfg_120 = TsNetConfig {
        backend: TsNetBackend::BarnesHut,
        iterations_stage1: 2,
        iterations_stage2: 2,
        iterations_stage3: 2,
        lambda_c_stage1: 0.1,
        lambda_c_stage2: 0.01,
        lambda_c_stage3: 0.0,
        lambda_r_stage3: 0.6,
        learning_rate: Some(2500.0),
        momentum: 0.8,
        ..TsNetConfig::pure_2hop()
    };
    let mut solver120 = TsNetSolver::<f64>::from_graph_pure_2hop(&g120, cfg_120).unwrap();
    let mut l120 = seeded_layout::<f64>(120, 0.5, 31);
    solver120.step(&mut l120).unwrap();
    assert!((solver120.current_stage_lr() - 2019.6).abs() < 1e-8);
    solver120.step(&mut l120).unwrap();
    solver120.step(&mut l120).unwrap();
    assert!((solver120.current_stage_lr() - 2500.0).abs() < 1e-8);
    solver120.step(&mut l120).unwrap();
    solver120.step(&mut l120).unwrap();
    assert!((solver120.current_stage_lr() - 50.0).abs() < 1e-10);
    assert!(l120.is_all_finite());
}

#[test]
fn test_auto_lr_scaling_stage3_small_n_cap_and_row_normalized_pure_2hop() {
    // 1. Small-N Stage 3 with lambda_c == 0.0 and lambda_r == 0.0 is capped at 2 * N.
    let g6 = make_barbell_graph(3); // N = 6
    let cfg_s3 = TsNetConfig {
        backend: TsNetBackend::Exact,
        iterations_stage1: 0,
        iterations_stage2: 0,
        iterations_stage3: 2,
        lambda_c_stage3: 0.0,
        lambda_r_stage3: 0.0,
        learning_rate: None,
        ..TsNetConfig::pure_2hop()
    };
    let mut solver6 = TsNetSolver::<f64>::from_graph_pure_2hop(&g6, cfg_s3).unwrap();
    let mut l6 = seeded_layout::<f64>(6, 0.5, 13);
    solver6.step(&mut l6).unwrap();
    assert!((solver6.current_stage_lr() - 12.0).abs() < 1e-12);
    assert!(l6.is_all_finite());

    // 2. Large-N Stage 3 (N = 3000) resolves to N/3 = 1000.0 and is never clipped by 2 * N.
    let stage3 = TsNetStage {
        iterations: 500,
        exaggeration: 1.0,
        lambda_kl: 1.0,
        lambda_c: 0.0,
        lambda_r: 0.0,
    };
    let cfg_auto = TsNetConfig::pure_2hop();
    let base_3000 = cfg_auto.resolve_learning_rate(3000);
    assert!((base_3000 - 1000.0).abs() < 1e-12);
    let stage3_lr_3000 = compute_stage_lr(&stage3, 3000, base_3000, cfg_auto.momentum);
    assert!((stage3_lr_3000 - 1000.0).abs() < 1e-12);

    // 2b. Boundary checks at n = 99 (capped at 198.0 across stages) and n = 100 (200.0).
    let stage1_c = TsNetStage {
        iterations: 10,
        exaggeration: 4.0,
        lambda_kl: 1.0,
        lambda_c: 0.1,
        lambda_r: 0.0,
    };
    let stage2_c = TsNetStage {
        iterations: 10,
        exaggeration: 1.0,
        lambda_kl: 1.0,
        lambda_c: 0.01,
        lambda_r: 0.0,
    };
    let stage3_no_r = TsNetStage {
        iterations: 10,
        exaggeration: 1.0,
        lambda_kl: 1.0,
        lambda_c: 0.0,
        lambda_r: 0.0,
    };
    let lr99 = cfg_auto.resolve_learning_rate(99);
    assert!((lr99 - 200.0).abs() < 1e-12);
    assert!((compute_stage_lr(&stage1_c, 99, lr99, cfg_auto.momentum) - 198.0).abs() < 1e-12);
    assert!((compute_stage_lr(&stage2_c, 99, lr99, cfg_auto.momentum) - 198.0).abs() < 1e-12);
    assert!((compute_stage_lr(&stage3_no_r, 99, lr99, cfg_auto.momentum) - 198.0).abs() < 1e-12);

    let lr100 = cfg_auto.resolve_learning_rate(100);
    assert!((lr100 - 200.0).abs() < 1e-12);
    assert!((compute_stage_lr(&stage1_c, 100, lr100, cfg_auto.momentum) - 200.0).abs() < 1e-12);
    assert!((compute_stage_lr(&stage2_c, 100, lr100, cfg_auto.momentum) - 200.0).abs() < 1e-12);
    assert!((compute_stage_lr(&stage3_no_r, 100, lr100, cfg_auto.momentum) - 200.0).abs() < 1e-12);
    // With explicit larger learning_rate=500.0 at n=100, Stage 1/2 use 500.0 (not capped at 200.0).
    assert!((compute_stage_lr(&stage1_c, 100, 500.0, cfg_auto.momentum) - 500.0).abs() < 1e-12);
    assert!((compute_stage_lr(&stage2_c, 100, 500.0, cfg_auto.momentum) - 500.0).abs() < 1e-12);

    // 3. RowNormalized vs StandardTsne in from_graph_pure_2hop on asymmetric star+tail graph.
    let mut edges: Vec<(u32, u32)> = (1..=6).map(|v| (0, v)).collect();
    edges.extend([(6, 7), (7, 8), (8, 9)]);
    let star_tail = CsrGraph::<f64>::from_edges(10, &edges, None, false, false).unwrap();
    let s_std = TsNetSolver::<f64>::from_graph_pure_2hop(&star_tail, cfg_auto).unwrap();
    let s_rn = TsNetSolver::<f64>::from_graph_pure_2hop(
        &star_tail,
        TsNetConfig {
            symmetrization: SymmetrizationMode::RowNormalized,
            ..cfg_auto
        },
    )
    .unwrap();
    let max_p_diff = s_std
        .p_vals
        .iter()
        .zip(s_rn.p_vals.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_p_diff > 1e-4,
        "expected RowNormalized pure_2hop p_vals to differ from StandardTsne, diff={max_p_diff}"
    );
}

#[test]
fn test_jacobi_precondition_weights_regular_isolated_and_hub() {
    // 1. Regular cycle graph (N=16, deg=2): P_{i.} == 1/N, so inv_jacobi[i] == 1.0 everywhere.
    let n_reg = 16;
    let reg_edges: Vec<(u32, u32)> = (0..n_reg as u32)
        .map(|i| (i, (i + 1) % (n_reg as u32)))
        .collect();
    let cycle = CsrGraph::<f64>::from_edges(n_reg, &reg_edges, None, false, false).unwrap();
    let cfg_on = TsNetConfig::pure_2hop();
    let s_reg_p2h = TsNetSolver::<f64>::from_graph_pure_2hop(&cycle, cfg_on).unwrap();
    let s_reg_rn =
        TsNetSolver::<f64>::from_graph_row_normalized(&cycle, 0.5, None, cfg_on).unwrap();
    assert_eq!(s_reg_p2h.inv_jacobi().len(), n_reg);
    assert_eq!(s_reg_rn.inv_jacobi().len(), n_reg);
    for &w in s_reg_p2h.inv_jacobi().iter().chain(s_reg_rn.inv_jacobi()) {
        assert!(
            (w - 1.0).abs() < 1e-12,
            "expected 1.0 on regular graph, got {w}"
        );
    }

    // 2. Disconnected graph with isolated vertex (triangle 0-1-2 + isolated vertex 3).
    let disc =
        CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 0)], None, false, false).unwrap();
    let s_disc = TsNetSolver::<f64>::from_graph_row_normalized(&disc, 0.0, None, cfg_on).unwrap();
    assert_eq!(s_disc.inv_jacobi().len(), 4);
    assert_eq!(s_disc.inv_jacobi()[3], 1.0);
    for &w in s_disc.inv_jacobi() {
        assert!(w.is_finite() && w > 0.0 && w <= 1.0);
    }

    // 3. Star graph (N=20, hub 0, leaves 1..19): P_{0.} = 0.5, N * P_{0.} = 10 -> inv_jacobi[0] = 0.1.
    let n_star = 20usize;
    let star_edges: Vec<(u32, u32)> = (1..n_star as u32).map(|v| (0, v)).collect();
    let star = CsrGraph::<f64>::from_edges(n_star, &star_edges, None, false, false).unwrap();
    let s_star = TsNetSolver::<f64>::from_graph_row_normalized(&star, 0.0, None, cfg_on).unwrap();
    assert!((s_star.inv_jacobi()[0] - 0.1).abs() < 1e-12);
    for leaf in 1..n_star {
        assert!((s_star.inv_jacobi()[leaf] - 1.0).abs() < 1e-12);
    }

    // 4. Self-loops in raw SparseProbabilities are skipped when computing inv_jacobi.
    let self_loop_probs = SparseProbabilities {
        n_nodes: 4,
        indptr: vec![0, 2, 3, 4, 4],
        indices: vec![0, 1, 0, 2],
        values: vec![100.0, 0.5, 0.5, 0.1],
    };
    let s_sl = TsNetSolver::<f64>::from_probabilities(self_loop_probs, cfg_on).unwrap();
    assert!((s_sl.inv_jacobi()[0] - 0.5).abs() < 1e-12);
    assert!((s_sl.inv_jacobi()[1] - 0.5).abs() < 1e-12);
    assert!((s_sl.inv_jacobi()[2] - 1.0).abs() < 1e-12);
    assert!((s_sl.inv_jacobi()[3] - 1.0).abs() < 1e-12);

    // 5. jacobi_precondition = false leaves inv_jacobi empty and preserves f64/f32 step == run_steps.
    let cfg_off = TsNetConfig {
        backend: TsNetBackend::Exact,
        jacobi_precondition: false,
        iterations_stage1: 6,
        iterations_stage2: 4,
        iterations_stage3: 4,
        ..TsNetConfig::pure_2hop()
    };
    let mut s64_run =
        TsNetSolver::<f64>::from_graph_row_normalized(&star, 0.0, None, cfg_off).unwrap();
    let mut s64_step =
        TsNetSolver::<f64>::from_graph_row_normalized(&star, 0.0, None, cfg_off).unwrap();
    assert!(s64_run.inv_jacobi().is_empty());
    let mut l64_run = seeded_layout::<f64>(n_star, 0.5, 17);
    let mut l64_step = l64_run.clone();
    s64_run.run(&mut l64_run).unwrap();
    s64_step.run_steps(&mut l64_step, 5).unwrap();
    while !s64_step.status().is_finished() {
        s64_step.step(&mut l64_step).unwrap();
    }
    for i in 0..n_star {
        assert_eq!(l64_run.x()[i].to_bits(), l64_step.x()[i].to_bits());
        assert_eq!(l64_run.y()[i].to_bits(), l64_step.y()[i].to_bits());
    }

    let mut s32_run =
        TsNetSolver::<f32>::from_graph_row_normalized(&star, 0.0, None, cfg_off).unwrap();
    let mut s32_step =
        TsNetSolver::<f32>::from_graph_row_normalized(&star, 0.0, None, cfg_off).unwrap();
    assert!(s32_run.inv_jacobi().is_empty());
    let mut l32_run = seeded_layout::<f32>(n_star, 0.5, 17);
    let mut l32_step = l32_run.clone();
    s32_run.run(&mut l32_run).unwrap();
    for _ in 0..14 {
        s32_step.step(&mut l32_step).unwrap();
    }
    for i in 0..n_star {
        assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
        assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
    }
}

#[test]
fn test_jacobi_precondition_prevents_hub_overshoot_and_512_grid_blowup() {
    // Sink-barrel graph (N=1200): hub 0 connected to 600 leaves (1..=600) plus a tail/ring (601..=1199).
    let n = 1200usize;
    let mut edges: Vec<(u32, u32)> = (1..=600u32).map(|v| (0, v)).collect();
    for u in 600..1199u32 {
        edges.push((u, u + 1));
    }
    edges.push((1199, 601));
    let barrel = CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap();

    let make_init = || {
        let mut l = seeded_layout::<f64>(n, 0.5, 42);
        let (xs, ys) = l.xy_mut();
        xs[0] = 1.0;
        ys[0] = 0.0;
        let mx = xs.iter().sum::<f64>() / (n as f64);
        let my = ys.iter().sum::<f64>() / (n as f64);
        for i in 0..n {
            xs[i] -= mx;
            ys[i] -= my;
        }
        l
    };
    let init_f64 = make_init();
    let init_f32 = Layout2d::<f32>::from_xy(
        init_f64.x().iter().map(|&v| v as f32).collect(),
        init_f64.y().iter().map(|&v| v as f32).collect(),
    )
    .unwrap();

    // Baseline without Jacobi: Stage 1 immediately ejects hub 0 and blows up ibfft grid to 512.
    let cfg_off = TsNetConfig {
        backend: TsNetBackend::FftInterpolation,
        jacobi_precondition: false,
        iterations_stage1: 10,
        iterations_stage2: 0,
        iterations_stage3: 0,
        ..TsNetConfig::pure_2hop()
    };
    let mut s_off =
        TsNetSolver::<f64>::from_graph_row_normalized(&barrel, 0.0, None, cfg_off).unwrap();
    let mut l_off = init_f64.clone();
    s_off.run_steps(&mut l_off, 2).unwrap();
    let r_hub_off = l_off.x()[0].hypot(l_off.y()[0]);
    let spec_off = s_off.fit_field.as_ref().unwrap().grid_spec().unwrap();
    assert!(
        r_hub_off > 100.0,
        "expected baseline hub ejection without Jacobi, got r_hub_off={r_hub_off}"
    );
    assert!(
        spec_off.nx == 512 || spec_off.ny == 512,
        "expected baseline 512 grid blowup without Jacobi, got {spec_off:?}"
    );

    // With Jacobi preconditioning (f64 and f32): bounded hub displacement, nx/ny < 512, step == run_steps.
    let cfg_on = TsNetConfig {
        jacobi_precondition: true,
        ..cfg_off
    };

    let mut s64_run =
        TsNetSolver::<f64>::from_graph_row_normalized(&barrel, 0.0, None, cfg_on).unwrap();
    let mut s64_step =
        TsNetSolver::<f64>::from_graph_row_normalized(&barrel, 0.0, None, cfg_on).unwrap();
    let mut l64_run = init_f64.clone();
    let mut l64_step = init_f64;
    assert_eq!(
        s64_run.run_steps(&mut l64_run, 10).unwrap(),
        SolverStatus::Completed
    );
    for _ in 0..10 {
        s64_step.step(&mut l64_step).unwrap();
    }
    assert!(l64_run.is_all_finite());
    let r_hub_64 = l64_run.x()[0].hypot(l64_run.y()[0]);
    let spec_64 = s64_run.fit_field.as_ref().unwrap().grid_spec().unwrap();
    assert!(
        r_hub_64 < 20.0,
        "expected bounded hub displacement (f64), got {r_hub_64}"
    );
    assert!(
        spec_64.nx < 512 && spec_64.ny < 512,
        "expected grid dims < 512 with Jacobi (f64), got {spec_64:?}"
    );
    for i in 0..n {
        assert_eq!(l64_run.x()[i].to_bits(), l64_step.x()[i].to_bits());
        assert_eq!(l64_run.y()[i].to_bits(), l64_step.y()[i].to_bits());
    }

    let mut s32_run =
        TsNetSolver::<f32>::from_graph_row_normalized(&barrel, 0.0, None, cfg_on).unwrap();
    let mut s32_step =
        TsNetSolver::<f32>::from_graph_row_normalized(&barrel, 0.0, None, cfg_on).unwrap();
    let mut l32_run = init_f32.clone();
    let mut l32_step = init_f32;
    assert_eq!(
        s32_run.run_steps(&mut l32_run, 10).unwrap(),
        SolverStatus::Completed
    );
    for _ in 0..10 {
        s32_step.step(&mut l32_step).unwrap();
    }
    assert!(l32_run.is_all_finite());
    let r_hub_32 = (l32_run.x()[0] as f64).hypot(l32_run.y()[0] as f64);
    let spec_32 = s32_run.fit_field.as_ref().unwrap().grid_spec().unwrap();
    assert!(
        r_hub_32 < 20.0,
        "expected bounded hub displacement (f32), got {r_hub_32}"
    );
    assert!(
        spec_32.nx < 512 && spec_32.ny < 512,
        "expected grid dims < 512 with Jacobi (f32), got {spec_32:?}"
    );
    for i in 0..n {
        assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
        assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
    }
}
