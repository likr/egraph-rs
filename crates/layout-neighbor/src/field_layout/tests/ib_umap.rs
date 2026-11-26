use egraph_core::{IterativeSolver, Layout2d, SolverStatus};
use egraph_neighborhood::{DogVaradhanConfig, SparseProbabilities};

use super::super::{FieldLayoutSolver, FieldRepulsionBackend, IbUmapConfig};
use super::{barbell_graph, grid_graph, seeded_layout, star_graph};

#[test]
fn test_ib_umap_step_vs_run_bit_exact_f64_and_f32_all_backends() {
    let grid = grid_graph(4, 4);
    let dog_cfg = DogVaradhanConfig::default();
    for backend in [
        FieldRepulsionBackend::IbFft,
        FieldRepulsionBackend::BarnesHut,
        FieldRepulsionBackend::Exact,
    ] {
        let cfg = IbUmapConfig {
            backend,
            n_epochs: 20,
            seed: 55,
            ..Default::default()
        };

        let mut s_run = FieldLayoutSolver::<f64>::ib_umap(&grid, 32, &dog_cfg, cfg).unwrap();
        let mut s_step = FieldLayoutSolver::<f64>::ib_umap(&grid, 32, &dog_cfg, cfg).unwrap();
        let mut l_run = seeded_layout::<f64>(grid.num_nodes(), 2.0, 23);
        let mut l_step = l_run.clone();

        assert_eq!(s_run.run(&mut l_run).unwrap(), SolverStatus::Completed);
        assert_eq!(
            s_step.run_steps(&mut l_step, 5).unwrap(),
            SolverStatus::Running
        );
        while !s_step.status().is_finished() {
            s_step.step(&mut l_step).unwrap();
        }
        for i in 0..grid.num_nodes() {
            assert_eq!(l_run.x()[i].to_bits(), l_step.x()[i].to_bits());
            assert_eq!(l_run.y()[i].to_bits(), l_step.y()[i].to_bits());
        }

        let mut s32_run = FieldLayoutSolver::<f32>::ib_umap(&grid, 32, &dog_cfg, cfg).unwrap();
        let mut s32_step = FieldLayoutSolver::<f32>::ib_umap(&grid, 32, &dog_cfg, cfg).unwrap();
        let mut l32_run = seeded_layout::<f32>(grid.num_nodes(), 2.0, 23);
        let mut l32_step = l32_run.clone();
        s32_run.run(&mut l32_run).unwrap();
        for _ in 0..20 {
            s32_step.step(&mut l32_step).unwrap();
        }
        for i in 0..grid.num_nodes() {
            assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
            assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
        }
    }
}

#[test]
fn test_ib_umap_ibfft_vs_exact_and_barbell_separation() {
    let barbell = barbell_graph(6);
    let dog_cfg = DogVaradhanConfig::default();
    let init = seeded_layout::<f64>(barbell.num_nodes(), 2.5, 202);

    let mut s_exact = FieldLayoutSolver::<f64>::ib_umap(
        &barbell,
        32,
        &dog_cfg,
        IbUmapConfig {
            backend: FieldRepulsionBackend::Exact,
            n_epochs: 25,
            ..Default::default()
        },
    )
    .unwrap();
    let mut s_ibfft = FieldLayoutSolver::<f64>::ib_umap(
        &barbell,
        32,
        &dog_cfg,
        IbUmapConfig {
            backend: FieldRepulsionBackend::IbFft,
            n_epochs: 25,
            ..Default::default()
        },
    )
    .unwrap();
    let mut s_bh_exact = FieldLayoutSolver::<f64>::ib_umap(
        &barbell,
        32,
        &dog_cfg,
        IbUmapConfig {
            backend: FieldRepulsionBackend::BarnesHut,
            n_epochs: 25,
            theta: 0.0,
            ..Default::default()
        },
    )
    .unwrap();

    let mut l_exact = init.clone();
    let mut l_ibfft = init.clone();
    let mut l_bh = init.clone();
    s_exact.run(&mut l_exact).unwrap();
    s_ibfft.run(&mut l_ibfft).unwrap();
    s_bh_exact.run(&mut l_bh).unwrap();

    for i in 0..barbell.num_nodes() {
        let bh_err = (l_bh.x()[i] - l_exact.x()[i]).hypot(l_bh.y()[i] - l_exact.y()[i]);
        assert!(bh_err < 1e-10, "ibUMAP BH(theta=0) err={bh_err}");
        let fft_err = (l_ibfft.x()[i] - l_exact.x()[i]).hypot(l_ibfft.y()[i] - l_exact.y()[i]);
        assert!(fft_err < 0.08, "ibUMAP IbFft vs Exact err={fft_err}");
    }

    // Full 150-epoch run from all-zeros separates the two barbell cliques cleanly.
    let mut full_solver = FieldLayoutSolver::<f64>::ib_umap(
        &barbell,
        32,
        &dog_cfg,
        IbUmapConfig {
            n_epochs: 150,
            ..Default::default()
        },
    )
    .unwrap();
    let mut layout = Layout2d::<f64>::zeros(barbell.num_nodes());
    assert_eq!(
        full_solver.run(&mut layout).unwrap(),
        SolverStatus::Completed
    );
    assert!(layout.is_all_finite());

    let c0x = layout.x()[..6].iter().sum::<f64>() / 6.0;
    let c0y = layout.y()[..6].iter().sum::<f64>() / 6.0;
    let c1x = layout.x()[6..].iter().sum::<f64>() / 6.0;
    let c1y = layout.y()[6..].iter().sum::<f64>() / 6.0;
    let centroid_dist = (c0x - c1x).hypot(c0y - c1y);
    let mut intra = 0.0_f64;
    for i in 0..6 {
        intra += (layout.x()[i] - c0x).hypot(layout.y()[i] - c0y);
        intra += (layout.x()[6 + i] - c1x).hypot(layout.y()[6 + i] - c1y);
    }
    intra /= 12.0;
    assert!(
        centroid_dist > intra * 1.5,
        "expected barbell separation: centroid_dist={centroid_dist}, intra={intra}"
    );
}

#[test]
fn test_ib_umap_hub_damping_and_tau_clipping() {
    let star = star_graph(24);
    let probs = SparseProbabilities {
        n_nodes: star.num_nodes(),
        indptr: {
            let mut p = Vec::with_capacity(star.num_nodes() + 1);
            p.push(0);
            p.push(24);
            for i in 1..star.num_nodes() {
                p.push(24 + i);
            }
            p
        },
        indices: {
            let mut idx: Vec<u32> = (1..star.num_nodes() as u32).collect();
            idx.resize(48, 0);
            idx
        },
        values: vec![1.0; 48],
    };

    // Place hub far away at (100, 0) and leaves around (0, 0) to trigger large hub attraction
    // and verify both hub_damping eta_i and tau_t = tau_scale * alpha_t step clipping.
    let mut xs = vec![0.0_f64; star.num_nodes()];
    let mut ys = vec![0.0_f64; star.num_nodes()];
    xs[0] = 100.0;
    for i in 1..star.num_nodes() {
        let angle = std::f64::consts::TAU * ((i - 1) as f64) / 24.0;
        xs[i] = angle.cos();
        ys[i] = angle.sin();
    }
    let init = Layout2d::from_xy(xs, ys).unwrap();

    let cfg_damped = IbUmapConfig {
        backend: FieldRepulsionBackend::Exact,
        n_epochs: 10,
        learning_rate: 1.0,
        tau_scale: 4.0,
        hub_damping: true,
        ..Default::default()
    };
    let cfg_undamped = IbUmapConfig {
        hub_damping: false,
        ..cfg_damped
    };
    let cfg_tight_tau = IbUmapConfig {
        tau_scale: 0.25,
        hub_damping: false,
        ..cfg_damped
    };

    let s_damped = FieldLayoutSolver::<f64>::ib_umap_from_fuzzy_set(
        probs.clone(),
        Some(&[24; 25]), // explicit degrees: even stronger damping test below uses natural degrees
        cfg_damped,
    )
    .unwrap();
    assert_eq!(s_damped.num_nodes(), 25);
    assert_eq!(s_damped.num_edges(), 48);
    assert!(s_damped.ab().is_some());
    assert_eq!(s_damped.backend(), FieldRepulsionBackend::Exact);
    assert!((s_damped.current_rho() - 1.0).abs() < 1e-12);

    // Natural degrees from fuzzy set: hub deg = 24, leaf deg = 1.
    let mut s_nat_damped =
        FieldLayoutSolver::<f64>::ib_umap_from_fuzzy_set(probs.clone(), None, cfg_damped).unwrap();
    let mut s_undamped =
        FieldLayoutSolver::<f64>::ib_umap_from_fuzzy_set(probs.clone(), None, cfg_undamped)
            .unwrap();
    let mut s_tight =
        FieldLayoutSolver::<f64>::ib_umap_from_fuzzy_set(probs, None, cfg_tight_tau).unwrap();

    let mut l_damped = init.clone();
    let mut l_undamped = init.clone();
    let mut l_tight = init.clone();

    s_nat_damped.step(&mut l_damped).unwrap();
    s_undamped.step(&mut l_undamped).unwrap();
    s_tight.step(&mut l_tight).unwrap();

    // Relative displacement of hub vs centroid shift must be strictly smaller with hub damping.
    let mean_x_before = init.x().iter().sum::<f64>() / (star.num_nodes() as f64);
    let hub_rel_before = init.x()[0] - mean_x_before;
    let hub_rel_damped = l_damped.x()[0]; // post-step layouts are zero-mean centered
    let hub_rel_undamped = l_undamped.x()[0];
    let hub_rel_tight = l_tight.x()[0];

    let damped_hub_delta = (hub_rel_before - hub_rel_damped).abs();
    let undamped_hub_delta = (hub_rel_before - hub_rel_undamped).abs();
    let tight_hub_delta = (hub_rel_before - hub_rel_tight).abs();

    assert!(
        damped_hub_delta < undamped_hub_delta,
        "expected hub damping to reduce step: damped={damped_hub_delta}, undamped={undamped_hub_delta}"
    );
    assert!(
        tight_hub_delta < undamped_hub_delta,
        "expected tighter tau_scale=0.25 to clip step: tight={tight_hub_delta}, undamped={undamped_hub_delta}"
    );
}
