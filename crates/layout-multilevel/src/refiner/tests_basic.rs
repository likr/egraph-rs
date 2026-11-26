use super::*;
use egraph_core::IterativeSolver;

pub(super) fn grid_graph(rows: usize, cols: usize) -> CsrGraph<f64> {
    let n = rows * cols;
    let mut edges = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let u = (r * cols + c) as u32;
            if c + 1 < cols {
                edges.push((u, u + 1));
            }
            if r + 1 < rows {
                edges.push((u, u + cols as u32));
            }
        }
    }
    CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
}

#[test]
fn test_refiner_config_defaults_and_validation() {
    let cfg = SpatialRefinerConfig::default();
    assert_eq!(cfg.rounds, 4);
    assert_eq!(cfg.k_cap, 32);
    assert_eq!(cfg.margin_scale, 1.2);
    assert_eq!(cfg.w_anchor, 1.0);
    assert_eq!(cfg.w_anchor2, 0.0);
    assert_eq!(cfg.w_fn, 0.0);
    assert_eq!(cfg.w_fp, 0.2);
    assert_eq!(cfg.fn_shrink, 0.88);
    assert_eq!(cfg.round_decay, 0.75);
    assert_eq!(cfg.k_cap_decay, 1.0);
    assert_eq!(cfg.k_cap_min, 1);
    assert_eq!(cfg.t_max, 20);
    assert_eq!(cfg.eps, 0.05);
    assert!(cfg.hinge_fp);
    assert_eq!(cfg.min_sep_ratio, 0.0);
    assert_eq!(cfg.w_sep, 0.0);
    assert_eq!(cfg.seed, 42);
    cfg.validate().unwrap();

    let bad_cases = [
        SpatialRefinerConfig { k_cap: 0, ..cfg },
        SpatialRefinerConfig {
            margin_scale: 0.0,
            ..cfg
        },
        SpatialRefinerConfig { w_fp: -0.1, ..cfg },
        SpatialRefinerConfig { w_fn: -0.1, ..cfg },
        SpatialRefinerConfig {
            w_anchor: f64::NAN,
            ..cfg
        },
        SpatialRefinerConfig {
            w_anchor2: -0.1,
            ..cfg
        },
        SpatialRefinerConfig {
            min_sep_ratio: -0.1,
            ..cfg
        },
        SpatialRefinerConfig {
            min_sep_ratio: f64::NAN,
            ..cfg
        },
        SpatialRefinerConfig { w_sep: -0.1, ..cfg },
        SpatialRefinerConfig {
            w_sep: f64::INFINITY,
            ..cfg
        },
        SpatialRefinerConfig {
            k_cap_decay: 0.0,
            ..cfg
        },
        SpatialRefinerConfig {
            k_cap_decay: 1.5,
            ..cfg
        },
        SpatialRefinerConfig {
            k_cap_min: 0,
            ..cfg
        },
        SpatialRefinerConfig {
            fn_shrink: 0.0,
            ..cfg
        },
        SpatialRefinerConfig {
            fn_shrink: 1.5,
            ..cfg
        },
        SpatialRefinerConfig {
            round_decay: 0.0,
            ..cfg
        },
        SpatialRefinerConfig { eps: 0.0, ..cfg },
    ];
    for bad in bad_cases {
        assert!(bad.validate().is_err());
    }
}

#[test]
fn test_refiner_leaves_clean_layout_undisturbed_and_pushes_intruders_apart() {
    // 1. Clean 3x3 grid embedded at exact lattice positions: zero >2-hop intruders -> early Converged.
    let grid = grid_graph(3, 3);
    let mut clean_x = Vec::with_capacity(9);
    let mut clean_y = Vec::with_capacity(9);
    for r in 0..3 {
        for c in 0..3 {
            clean_x.push(c as f64);
            clean_y.push(r as f64);
        }
    }
    let clean_orig = Layout2d::from_xy(clean_x, clean_y).unwrap();
    let mut clean_layout = clean_orig.clone();

    let cfg = SpatialRefinerConfig::default();
    let mut clean_refiner = SpatialIntruderRefiner::<f64>::new(&grid, cfg).unwrap();
    assert_eq!(
        clean_refiner.run(&mut clean_layout).unwrap(),
        SolverStatus::Converged
    );
    assert_eq!(clean_refiner.stats().rounds_executed, 0);
    assert_eq!(clean_refiner.stats().n_fp_total, 0);
    assert_eq!(clean_refiner.stats().round_stats.len(), 1);
    assert!(clean_refiner.stats().round_stats[0].skipped);
    for i in 0..9 {
        assert_eq!(clean_layout.x()[i].to_bits(), clean_orig.x()[i].to_bits());
        assert_eq!(clean_layout.y()[i].to_bits(), clean_orig.y()[i].to_bits());
    }

    // 2. Introduce a spatial intruder: place far corner node 8 (4 hops from 0) right next to node 0.
    let mut intruder_layout = clean_orig;
    intruder_layout.x_mut()[8] = 0.08;
    intruder_layout.y_mut()[8] = 0.08;
    let dist_before = (intruder_layout.x()[0] - intruder_layout.x()[8])
        .hypot(intruder_layout.y()[0] - intruder_layout.y()[8]);

    let mut intr_refiner = SpatialIntruderRefiner::<f64>::new(&grid, cfg).unwrap();
    let status = intr_refiner.run(&mut intruder_layout).unwrap();
    assert!(status.is_finished());
    assert!(intruder_layout.is_all_finite());
    assert!(intr_refiner.stats().n_fp_total > 0);
    assert!(intr_refiner.stats().rounds_executed >= 1);

    let dist_after = (intruder_layout.x()[0] - intruder_layout.x()[8])
        .hypot(intruder_layout.y()[0] - intruder_layout.y()[8]);
    assert!(
        dist_after > dist_before * 2.0,
        "expected spatial intruder (0, 8) pushed apart: before={dist_before}, after={dist_after}"
    );
}

#[test]
fn test_refiner_step_vs_run_bit_exact_f64_and_f32_and_two_sided_fn() {
    let grid = grid_graph(3, 3);
    let mut base_x = Vec::with_capacity(9);
    let mut base_y = Vec::with_capacity(9);
    for r in 0..3 {
        for c in 0..3 {
            base_x.push(c as f64);
            base_y.push(r as f64);
        }
    }
    // Place node 8 near node 0 and stretch node 2 slightly so both FP and FN triggers work.
    base_x[8] = 0.1;
    base_y[8] = 0.1;
    base_x[2] = 4.0;

    for hinge_fp in [true, false] {
        let cfg = SpatialRefinerConfig {
            rounds: 3,
            w_fn: 0.2,
            hinge_fp,
            seed: 77,
            ..Default::default()
        };

        // f64 bit-exact parity
        let mut l_run = Layout2d::from_xy(base_x.clone(), base_y.clone()).unwrap();
        let mut l_step = l_run.clone();
        let mut r_run = SpatialIntruderRefiner::<f64>::new(&grid, cfg).unwrap();
        let mut r_step = SpatialIntruderRefiner::<f64>::new(&grid, cfg).unwrap();

        assert_eq!(r_run.total_iterations(), 3);
        let s_run = r_run.run(&mut l_run).unwrap();
        let s_step1 = r_step.step(&mut l_step).unwrap();
        assert!(!s_step1.is_finished() || r_step.iteration() == r_run.iteration());
        let s_step = r_step.run(&mut l_step).unwrap();
        assert_eq!(s_run, s_step);
        assert_eq!(r_run.stats(), r_step.stats());
        for i in 0..9 {
            assert_eq!(l_run.x()[i].to_bits(), l_step.x()[i].to_bits());
            assert_eq!(l_run.y()[i].to_bits(), l_step.y()[i].to_bits());
        }

        // f32 bit-exact parity
        let bx32: Vec<f32> = base_x.iter().map(|&v| v as f32).collect();
        let by32: Vec<f32> = base_y.iter().map(|&v| v as f32).collect();
        let mut l32_run = Layout2d::from_xy(bx32.clone(), by32.clone()).unwrap();
        let mut l32_step = l32_run.clone();
        let mut r32_run = SpatialIntruderRefiner::<f32>::new(&grid, cfg).unwrap();
        let mut r32_step = SpatialIntruderRefiner::<f32>::new(&grid, cfg).unwrap();
        r32_run.run(&mut l32_run).unwrap();
        while !r32_step.status().is_finished() {
            r32_step.step(&mut l32_step).unwrap();
        }
        for i in 0..9 {
            assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
            assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
        }
    }
}

#[test]
fn test_refiner_edge_cases_and_errors() {
    let cfg = SpatialRefinerConfig::default();

    // Empty, singleton, 2-node, and edgeless graphs converge cleanly without panic.
    for n in [0usize, 1, 2, 4] {
        let g = if n == 2 {
            CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap()
        } else {
            CsrGraph::<f64>::from_edges(n, &[], None, false, false).unwrap()
        };
        let mut r = SpatialIntruderRefiner::<f64>::new(&g, cfg).unwrap();
        let mut l = Layout2d::<f64>::zeros(n);
        assert_eq!(r.run(&mut l).unwrap(), SolverStatus::Converged);
        assert_eq!(r.stats().rounds_executed, 0);
    }

    // rounds == 0 completes immediately, t_max == 0 converges immediately.
    let grid = grid_graph(3, 3);
    let r_zero_rounds =
        SpatialIntruderRefiner::<f64>::new(&grid, SpatialRefinerConfig { rounds: 0, ..cfg })
            .unwrap();
    assert_eq!(r_zero_rounds.status(), SolverStatus::Completed);
    assert_eq!(r_zero_rounds.total_iterations(), 0);

    let mut r_zero_tmax =
        SpatialIntruderRefiner::<f64>::new(&grid, SpatialRefinerConfig { t_max: 0, ..cfg })
            .unwrap();
    let mut l9 = Layout2d::<f64>::zeros(9);
    assert_eq!(r_zero_tmax.step(&mut l9).unwrap(), SolverStatus::Converged);

    // Dimension mismatch & non-finite layout rejected.
    let mut r_err = SpatialIntruderRefiner::<f64>::new(&grid, cfg).unwrap();
    let mut l_wrong = Layout2d::<f64>::zeros(4);
    assert!(matches!(
        r_err.step(&mut l_wrong),
        Err(CoreError::DimensionMismatch { .. })
    ));
    let mut l_nan = Layout2d::from_xy(vec![f64::NAN; 9], vec![0.0; 9]).unwrap();
    assert!(matches!(
        r_err.step(&mut l_nan),
        Err(CoreError::NonFiniteValue { .. })
    ));
}
