use egraph_core::{CsrGraph, IterativeSolver, Layout2d, SolverStatus};
use egraph_neighborhood::{SparseProbabilities, SymmetrizationMode};

use crate::backends::{BarnesHut, Exact, FitFft};

use super::super::config::compute_stage_lr;
use super::super::{TsNetConfig, TsNetSolver, TsNetStage};
use super::{make_barbell_graph, make_grid_graph, make_stages, seeded_layout};

#[test]
fn test_stage_lr_stability_cap_and_velocity_rescaling() {
    let g = make_barbell_graph(3); // n = 6
    let cfg_small = TsNetConfig::pure_2hop()
        .with_stages(make_stages(2, 2, 2, 0.6))
        .with_learning_rate(Some(200.0))
        .with_momentum(0.8);
    let mut solver_small =
        TsNetSolver::<Layout2d<f64>, Exact>::from_graph_pure_2hop(&g, cfg_small).unwrap();
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

    let g30 = make_grid_graph(5, 6); // n = 30
    let cfg_30 = TsNetConfig::pure_2hop()
        .with_stages(make_stages(2, 2, 2, 0.6))
        .with_learning_rate(Some(200.0))
        .with_momentum(0.8);
    let mut solver30 =
        TsNetSolver::<Layout2d<f64>, Exact>::from_graph_pure_2hop(&g30, cfg_30).unwrap();
    let mut l30 = seeded_layout::<f64>(30, 0.5, 31);
    solver30.step(&mut l30).unwrap();
    assert!((solver30.current_stage_lr() - 60.0).abs() < 1e-10);
    solver30.step(&mut l30).unwrap();
    solver30.step(&mut l30).unwrap();
    assert!((solver30.current_stage_lr() - 60.0).abs() < 1e-10);
    solver30.step(&mut l30).unwrap();
    let vel_x_end_s2 = solver30.space.vel_x().unwrap()[0];
    solver30.step(&mut l30).unwrap();
    assert!((solver30.current_stage_lr() - 50.0).abs() < 1e-10);
    assert!(vel_x_end_s2.is_finite());
    assert!(l30.is_all_finite());

    let g120 = make_grid_graph(10, 12); // n = 120
    let cfg_120 = TsNetConfig::pure_2hop()
        .with_stages(make_stages(2, 2, 2, 0.6))
        .with_learning_rate(Some(2500.0))
        .with_momentum(0.8);
    let mut solver120 =
        TsNetSolver::<Layout2d<f64>, BarnesHut>::from_graph_pure_2hop(&g120, cfg_120).unwrap();
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
    let g6 = make_barbell_graph(3); // N = 6
    let cfg_s3 = TsNetConfig::pure_2hop()
        .with_stages(make_stages(0, 0, 2, 0.0))
        .with_learning_rate(None);
    let mut solver6 =
        TsNetSolver::<Layout2d<f64>, Exact>::from_graph_pure_2hop(&g6, cfg_s3).unwrap();
    let mut l6 = seeded_layout::<f64>(6, 0.5, 13);
    solver6.step(&mut l6).unwrap();
    assert!((solver6.current_stage_lr() - 10.0).abs() < 1e-12);
    assert!(l6.is_all_finite());

    let stage3 = TsNetStage::new().with_iterations(500);
    let cfg_auto = TsNetConfig::pure_2hop();
    let base_3000 = cfg_auto.resolve_learning_rate(3000);
    assert!((base_3000 - 1000.0).abs() < 1e-12);
    let stage3_lr_3000 = compute_stage_lr(&stage3, 3000, base_3000, cfg_auto.momentum);
    assert!((stage3_lr_3000 - 1000.0).abs() < 1e-12);

    let stage1_c = TsNetStage::new()
        .with_iterations(10)
        .with_exaggeration(4.0)
        .with_lambda_c(0.1);
    let stage2_c = TsNetStage::new().with_iterations(10).with_lambda_c(0.01);
    let stage3_no_r = TsNetStage::new().with_iterations(10);
    let lr99 = cfg_auto.resolve_learning_rate(99);
    assert!((lr99 - 33.0).abs() < 1e-12);
    assert!((compute_stage_lr(&stage1_c, 99, lr99, cfg_auto.momentum) - 33.0).abs() < 1e-12);
    assert!((compute_stage_lr(&stage2_c, 99, lr99, cfg_auto.momentum) - 33.0).abs() < 1e-12);
    assert!((compute_stage_lr(&stage3_no_r, 99, lr99, cfg_auto.momentum) - 33.0).abs() < 1e-12);

    let lr100 = cfg_auto.resolve_learning_rate(100);
    assert!((lr100 - 100.0 / 3.0).abs() < 1e-12);
    assert!(
        (compute_stage_lr(&stage1_c, 100, lr100, cfg_auto.momentum) - 100.0 / 3.0).abs() < 1e-12
    );
    assert!(
        (compute_stage_lr(&stage2_c, 100, lr100, cfg_auto.momentum) - 100.0 / 3.0).abs() < 1e-12
    );
    assert!(
        (compute_stage_lr(&stage3_no_r, 100, lr100, cfg_auto.momentum) - 100.0 / 3.0).abs() < 1e-12
    );
    assert!((compute_stage_lr(&stage1_c, 100, 500.0, cfg_auto.momentum) - 500.0).abs() < 1e-12);
    assert!((compute_stage_lr(&stage2_c, 100, 500.0, cfg_auto.momentum) - 500.0).abs() < 1e-12);

    let mut edges: Vec<(u32, u32)> = (1..=6).map(|v| (0, v)).collect();
    edges.extend([(6, 7), (7, 8), (8, 9)]);
    let star_tail = CsrGraph::<f64>::from_edges(10, &edges, None, false, false).unwrap();
    let s_std =
        TsNetSolver::<Layout2d<f64>>::from_graph_pure_2hop(&star_tail, cfg_auto.clone()).unwrap();
    let s_rn = TsNetSolver::<Layout2d<f64>>::from_graph_pure_2hop(
        &star_tail,
        cfg_auto.with_symmetrization(SymmetrizationMode::RowNormalized),
    )
    .unwrap();
    let max_p_diff = s_std
        .core
        .p_vals
        .iter()
        .zip(s_rn.core.p_vals.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_p_diff > 1e-4,
        "expected RowNormalized pure_2hop p_vals to differ from StandardTsne, diff={max_p_diff}"
    );
}

#[test]
fn test_jacobi_precondition_weights_regular_isolated_and_hub() {
    let n_reg = 16;
    let reg_edges: Vec<(u32, u32)> = (0..n_reg as u32)
        .map(|i| (i, (i + 1) % (n_reg as u32)))
        .collect();
    let cycle = CsrGraph::<f64>::from_edges(n_reg, &reg_edges, None, false, false).unwrap();
    let cfg_on = TsNetConfig::pure_2hop();
    let s_reg_p2h =
        TsNetSolver::<Layout2d<f64>>::from_graph_pure_2hop(&cycle, cfg_on.clone()).unwrap();
    let s_reg_rn =
        TsNetSolver::<Layout2d<f64>>::from_graph_row_normalized(&cycle, 0.5, None, cfg_on.clone())
            .unwrap();
    assert_eq!(s_reg_p2h.inv_jacobi().len(), n_reg);
    assert_eq!(s_reg_rn.inv_jacobi().len(), n_reg);
    for &w in s_reg_p2h.inv_jacobi().iter().chain(s_reg_rn.inv_jacobi()) {
        assert!(
            (w - 1.0).abs() < 1e-12,
            "expected 1.0 on regular graph, got {w}"
        );
    }

    let disc =
        CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 0)], None, false, false).unwrap();
    let s_disc =
        TsNetSolver::<Layout2d<f64>>::from_graph_row_normalized(&disc, 0.0, None, cfg_on.clone())
            .unwrap();
    assert_eq!(s_disc.inv_jacobi().len(), 4);
    assert_eq!(s_disc.inv_jacobi()[3], 1.0);
    for &w in s_disc.inv_jacobi() {
        assert!(w.is_finite() && w > 0.0 && w <= 1.0);
    }

    let n_star = 20usize;
    let star_edges: Vec<(u32, u32)> = (1..n_star as u32).map(|v| (0, v)).collect();
    let star = CsrGraph::<f64>::from_edges(n_star, &star_edges, None, false, false).unwrap();
    let s_star =
        TsNetSolver::<Layout2d<f64>>::from_graph_row_normalized(&star, 0.0, None, cfg_on.clone())
            .unwrap();
    assert!((s_star.inv_jacobi()[0] - 0.1).abs() < 1e-12);
    for leaf in 1..n_star {
        assert!((s_star.inv_jacobi()[leaf] - 1.0).abs() < 1e-12);
    }

    let self_loop_probs = SparseProbabilities::from_csr(
        4,
        vec![0, 2, 3, 4, 4],
        vec![0, 1, 0, 2],
        vec![100.0, 0.5, 0.5, 0.1],
    )
    .unwrap();
    let s_sl = TsNetSolver::<Layout2d<f64>>::from_probabilities(self_loop_probs, cfg_on).unwrap();
    assert!((s_sl.inv_jacobi()[0] - 0.5).abs() < 1e-12);
    assert!((s_sl.inv_jacobi()[1] - 0.5).abs() < 1e-12);
    assert!((s_sl.inv_jacobi()[2] - 1.0).abs() < 1e-12);
    assert!((s_sl.inv_jacobi()[3] - 1.0).abs() < 1e-12);

    let cfg_off = TsNetConfig::pure_2hop()
        .with_jacobi_precondition(false)
        .with_stages(make_stages(6, 4, 4, 0.0));
    let mut s64_run = TsNetSolver::<Layout2d<f64>, Exact>::from_graph_row_normalized(
        &star,
        0.0,
        None,
        cfg_off.clone(),
    )
    .unwrap();
    let mut s64_step = TsNetSolver::<Layout2d<f64>, Exact>::from_graph_row_normalized(
        &star,
        0.0,
        None,
        cfg_off.clone(),
    )
    .unwrap();
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

    let mut s32_run = TsNetSolver::<Layout2d<f32>, Exact>::from_graph_row_normalized(
        &star,
        0.0,
        None,
        cfg_off.clone(),
    )
    .unwrap();
    let mut s32_step =
        TsNetSolver::<Layout2d<f32>, Exact>::from_graph_row_normalized(&star, 0.0, None, cfg_off)
            .unwrap();
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

    let cfg_off = TsNetConfig::pure_2hop()
        .with_jacobi_precondition(false)
        .with_stages(make_stages(10, 0, 0, 0.0));
    let mut s_off = TsNetSolver::<Layout2d<f64>, FitFft>::from_graph_row_normalized(
        &barrel,
        0.0,
        None,
        cfg_off.clone(),
    )
    .unwrap();
    let mut l_off = init_f64.clone();
    s_off.run_steps(&mut l_off, 2).unwrap();
    let r_hub_off = l_off.x()[0].hypot(l_off.y()[0]);
    let spec_off = s_off.backend_state.fit_field.grid_spec().unwrap();
    assert!(
        r_hub_off > 100.0,
        "expected baseline hub ejection without Jacobi, got r_hub_off={r_hub_off}"
    );
    assert!(
        spec_off.nx == 512 || spec_off.ny == 512,
        "expected baseline 512 grid blowup without Jacobi, got {spec_off:?}"
    );

    let cfg_on = cfg_off.with_jacobi_precondition(true);

    let mut s64_run = TsNetSolver::<Layout2d<f64>, FitFft>::from_graph_row_normalized(
        &barrel,
        0.0,
        None,
        cfg_on.clone(),
    )
    .unwrap();
    let mut s64_step = TsNetSolver::<Layout2d<f64>, FitFft>::from_graph_row_normalized(
        &barrel,
        0.0,
        None,
        cfg_on.clone(),
    )
    .unwrap();
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
    let spec_64 = s64_run.backend_state.fit_field.grid_spec().unwrap();
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

    let mut s32_run = TsNetSolver::<Layout2d<f32>, FitFft>::from_graph_row_normalized(
        &barrel,
        0.0,
        None,
        cfg_on.clone(),
    )
    .unwrap();
    let mut s32_step =
        TsNetSolver::<Layout2d<f32>, FitFft>::from_graph_row_normalized(&barrel, 0.0, None, cfg_on)
            .unwrap();
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
    let spec_32 = s32_run.backend_state.fit_field.grid_spec().unwrap();
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
