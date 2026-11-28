use egraph_core::{CoreError, CoreResult, CsrGraph, IterativeSolver, Layout2d, SolverStatus};
use egraph_neighborhood::{DogVaradhanConfig, SparseProbabilities};

use crate::backends::{BarnesHut, Exact, IbFft, IbUmapBackend2d};

use super::super::{IbUmapConfig, IbUmapRepulsion, IbUmapSolver};
use super::{barbell_graph, grid_graph, seeded_layout, star_graph};

#[test]
fn test_ib_umap_step_vs_run_bit_exact_f64_and_f32_all_backends() {
    let grid = grid_graph(4, 4);
    let dog_cfg = DogVaradhanConfig::default();
    for backend in [
        IbUmapBackend2d::IbFft(IbFft::new()),
        IbUmapBackend2d::BarnesHut(BarnesHut::default()),
        IbUmapBackend2d::Exact(Exact::new()),
    ] {
        let cfg = IbUmapConfig::new().with_iterations(20).with_seed(55);

        let mut s_run =
            IbUmapSolver::<Layout2d<f64>, IbUmapBackend2d>::from_graph_dog_with_backend(
                &grid,
                32,
                &dog_cfg,
                cfg.clone(),
                backend.clone(),
            )
            .unwrap();
        let mut s_step =
            IbUmapSolver::<Layout2d<f64>, IbUmapBackend2d>::from_graph_dog_with_backend(
                &grid,
                32,
                &dog_cfg,
                cfg.clone(),
                backend.clone(),
            )
            .unwrap();
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

        let mut s32_run =
            IbUmapSolver::<Layout2d<f32>, IbUmapBackend2d>::from_graph_dog_with_backend(
                &grid,
                32,
                &dog_cfg,
                cfg.clone(),
                backend.clone(),
            )
            .unwrap();
        let mut s32_step =
            IbUmapSolver::<Layout2d<f32>, IbUmapBackend2d>::from_graph_dog_with_backend(
                &grid, 32, &dog_cfg, cfg, backend,
            )
            .unwrap();
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

    let mut s_exact = IbUmapSolver::<Layout2d<f64>, Exact>::from_graph_dog(
        &barbell,
        32,
        &dog_cfg,
        IbUmapConfig::new().with_iterations(25),
    )
    .unwrap();
    let mut s_ibfft = IbUmapSolver::<Layout2d<f64>, IbFft>::from_graph_dog(
        &barbell,
        32,
        &dog_cfg,
        IbUmapConfig::new().with_iterations(25),
    )
    .unwrap();
    let mut s_bh_exact = IbUmapSolver::<Layout2d<f64>, BarnesHut>::from_graph_dog_with_backend(
        &barbell,
        32,
        &dog_cfg,
        IbUmapConfig::new().with_iterations(25),
        BarnesHut::new(0.0),
    )
    .unwrap();

    let mut l_exact = init.clone();
    let mut l_ibfft = init.clone();
    let mut l_bh = init.clone();
    s_exact.run(&mut l_exact).unwrap();
    s_ibfft.run(&mut l_ibfft).unwrap();
    s_bh_exact.run(&mut l_bh).unwrap();
    assert!(s_ibfft.total_fft_2d_calls() > 0);
    assert!(s_ibfft.last_fft_stats().is_some());

    for i in 0..barbell.num_nodes() {
        let bh_err = (l_bh.x()[i] - l_exact.x()[i]).hypot(l_bh.y()[i] - l_exact.y()[i]);
        assert!(bh_err < 1e-10, "ibUMAP BH(theta=0) err={bh_err}");
        let fft_err = (l_ibfft.x()[i] - l_exact.x()[i]).hypot(l_ibfft.y()[i] - l_exact.y()[i]);
        assert!(fft_err < 0.08, "ibUMAP IbFft vs Exact err={fft_err}");
    }

    // Full 150-epoch run from all-zeros separates the two barbell cliques cleanly.
    let mut full_solver = IbUmapSolver::<Layout2d<f64>>::from_graph_dog(
        &barbell,
        32,
        &dog_cfg,
        IbUmapConfig::new().with_iterations(150),
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
fn test_ib_umap_hub_damping_tau_clipping_and_ir_accessors() {
    let star = star_graph(24);
    let probs = SparseProbabilities::from_csr(
        star.num_nodes(),
        {
            let mut p = Vec::with_capacity(star.num_nodes() + 1);
            p.push(0);
            p.push(24);
            for i in 1..star.num_nodes() {
                p.push((24 + i) as u32);
            }
            p
        },
        {
            let mut idx: Vec<u32> = (1..star.num_nodes() as u32).collect();
            idx.resize(48, 0);
            idx
        },
        vec![1.0; 48],
    )
    .unwrap();

    let mut xs = vec![0.0_f64; star.num_nodes()];
    let mut ys = vec![0.0_f64; star.num_nodes()];
    xs[0] = 100.0;
    for i in 1..star.num_nodes() {
        let angle = std::f64::consts::TAU * ((i - 1) as f64) / 24.0;
        xs[i] = angle.cos();
        ys[i] = angle.sin();
    }
    let init = Layout2d::from_xy(xs, ys).unwrap();

    let cfg_damped = IbUmapConfig::new()
        .with_iterations(10)
        .with_learning_rate(1.0)
        .with_tau_scale(4.0)
        .with_hub_damping(true);
    let cfg_undamped = cfg_damped.clone().with_hub_damping(false);
    let cfg_tight_tau = cfg_damped
        .clone()
        .with_tau_scale(0.25)
        .with_hub_damping(false);

    let s_damped = IbUmapSolver::<Layout2d<f64>, Exact>::from_fuzzy_set(
        probs.clone(),
        Some(&[24; 25]),
        cfg_damped.clone(),
    )
    .unwrap();
    assert_eq!(s_damped.num_nodes(), 25);
    assert_eq!(s_damped.num_edges(), 48);
    assert!(s_damped.a() > 0.0 && s_damped.b() > 0.0);
    assert_eq!(s_damped.ab_params(), s_damped.ab());
    assert_eq!(s_damped.backend(), &Exact::new());
    let (indptr, indices, weights) = s_damped.edge_csr();
    assert_eq!(indptr.len(), 26);
    assert_eq!(indices.len(), 48);
    assert_eq!(weights.len(), 48);
    assert_eq!(s_damped.hub_eta().len(), 25);
    assert!((s_damped.hub_eta()[0] - 1.0 / 24.0).abs() < 1e-12);

    let mut s_nat_damped =
        IbUmapSolver::<Layout2d<f64>, Exact>::from_fuzzy_set(probs.clone(), None, cfg_damped)
            .unwrap();
    let mut s_undamped =
        IbUmapSolver::<Layout2d<f64>, Exact>::from_fuzzy_set(probs.clone(), None, cfg_undamped)
            .unwrap();
    let mut s_tight =
        IbUmapSolver::<Layout2d<f64>, Exact>::from_fuzzy_set(probs, None, cfg_tight_tau).unwrap();
    assert!(
        s_undamped
            .hub_eta()
            .iter()
            .all(|&v| (v - 1.0).abs() < 1e-12)
    );

    let mut l_damped = init.clone();
    let mut l_undamped = init.clone();
    let mut l_tight = init.clone();

    let csr_ptr_before = s_nat_damped.edge_csr().0.as_ptr();
    s_nat_damped.step(&mut l_damped).unwrap();
    assert_eq!(s_nat_damped.edge_csr().0.as_ptr(), csr_ptr_before);
    s_undamped.step(&mut l_undamped).unwrap();
    s_tight.step(&mut l_tight).unwrap();

    let mean_x_before = init.x().iter().sum::<f64>() / (star.num_nodes() as f64);
    let hub_rel_before = init.x()[0] - mean_x_before;
    let hub_rel_damped = l_damped.x()[0];
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

    // Empty graph CSR invariant on IbUmapSolver::edge_csr / hub_eta.
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let s_empty = IbUmapSolver::<Layout2d<f64>>::new(&empty, IbUmapConfig::new()).unwrap();
    let (e_indptr, e_indices, e_weights) = s_empty.edge_csr();
    assert_eq!(e_indptr, &[0]);
    assert!(e_indices.is_empty());
    assert!(e_weights.is_empty());
    assert!(s_empty.hub_eta().is_empty());
    assert!(s_empty.ab_params().0 > 0.0 && s_empty.ab_params().1 > 0.0);
}

#[derive(Clone, Copy, Debug)]
enum RepulsionErrorKind {
    BackendUnavailable,
    DeviceLimitExceeded,
    NumericalBreakdown,
}

#[derive(Clone, Debug)]
struct FailingRepulsionBackend {
    kind: RepulsionErrorKind,
    fail_on_call: usize,
}

#[derive(Clone, Debug)]
struct FailingRepulsionState {
    calls: usize,
}

impl IbUmapRepulsion<Layout2d<f64>> for FailingRepulsionBackend {
    type State = FailingRepulsionState;

    fn validate_config(&self) -> CoreResult<()> {
        Ok(())
    }

    fn init_state(&self, _num_nodes: usize) -> CoreResult<Self::State> {
        Ok(FailingRepulsionState { calls: 0 })
    }

    fn compute_repulsion(
        &self,
        state: &mut Self::State,
        _layout: &Layout2d<f64>,
        _a: f64,
        _b: f64,
        _c_k: f64,
        _repulsive_eps: f64,
        rep_forces: &mut [f64],
    ) -> CoreResult<()> {
        state.calls += 1;
        if state.calls == self.fail_on_call {
            return Err(match self.kind {
                RepulsionErrorKind::BackendUnavailable => {
                    CoreError::backend_unavailable("simulated unavailable backend")
                }
                RepulsionErrorKind::DeviceLimitExceeded => {
                    CoreError::device_limit_exceeded("simulated device limit exceeded")
                }
                RepulsionErrorKind::NumericalBreakdown => {
                    CoreError::numerical_breakdown("simulated numerical breakdown")
                }
            });
        }
        rep_forces.fill(0.0);
        Ok(())
    }
}

#[test]
fn test_ib_umap_terminal_failure_sets_failed_status_and_makes_step_noop() {
    let grid = grid_graph(3, 3);
    let dog_cfg = DogVaradhanConfig::default();

    for kind in [
        RepulsionErrorKind::BackendUnavailable,
        RepulsionErrorKind::DeviceLimitExceeded,
        RepulsionErrorKind::NumericalBreakdown,
    ] {
        for fail_on_call in [1_usize, 2_usize] {
            let mut solver =
                IbUmapSolver::<Layout2d<f64>, FailingRepulsionBackend>::from_graph_dog_with_backend(
                    &grid,
                    16,
                    &dog_cfg,
                    IbUmapConfig::new().with_iterations(10).with_seed(17),
                    FailingRepulsionBackend { kind, fail_on_call },
                )
                .unwrap();
            let mut layout = seeded_layout::<f64>(grid.num_nodes(), 2.0, 11);

            for expected_iter in 1..fail_on_call {
                assert_eq!(solver.step(&mut layout).unwrap(), SolverStatus::Running);
                assert_eq!(solver.iteration(), expected_iter);
            }

            let err = solver.step(&mut layout).unwrap_err();
            match kind {
                RepulsionErrorKind::BackendUnavailable => {
                    assert!(matches!(err, CoreError::BackendUnavailable(_)));
                }
                RepulsionErrorKind::DeviceLimitExceeded => {
                    assert!(matches!(err, CoreError::DeviceLimitExceeded(_)));
                }
                RepulsionErrorKind::NumericalBreakdown => {
                    assert!(matches!(err, CoreError::NumericalBreakdown(_)));
                }
            }

            assert_eq!(solver.status(), SolverStatus::Failed);
            assert!(solver.status().is_finished());
            assert!(solver.status().is_failed());
            assert_eq!(solver.iteration(), fail_on_call - 1);
            assert_eq!(solver.backend_state().calls, fail_on_call);

            let snapshot = layout.clone();
            assert_eq!(solver.step(&mut layout).unwrap(), SolverStatus::Failed);
            assert_eq!(
                solver.run_steps(&mut layout, 3).unwrap(),
                SolverStatus::Failed
            );
            assert_eq!(solver.run(&mut layout).unwrap(), SolverStatus::Failed);

            assert_eq!(solver.backend_state().calls, fail_on_call);
            assert_eq!(solver.iteration(), fail_on_call - 1);
            for i in 0..grid.num_nodes() {
                assert_eq!(layout.x()[i].to_bits(), snapshot.x()[i].to_bits());
                assert_eq!(layout.y()[i].to_bits(), snapshot.y()[i].to_bits());
            }
        }
    }
}

#[test]
fn test_ib_umap_repulsive_eps_propagates_and_matches_across_backends() {
    let grid = grid_graph(3, 3);
    let n = grid.num_nodes();
    let init = seeded_layout::<f64>(n, 0.5, 41);

    let cfg_small_eps = IbUmapConfig::new()
        .with_iterations(4)
        .with_repulsive_eps(0.01)
        .with_seed(42);
    let cfg_large_eps = IbUmapConfig::new()
        .with_iterations(4)
        .with_repulsive_eps(0.5)
        .with_seed(42);

    let mut s_ex_small =
        IbUmapSolver::<Layout2d<f64>, Exact>::new(&grid, cfg_small_eps.clone()).unwrap();
    let mut s_ex_large =
        IbUmapSolver::<Layout2d<f64>, Exact>::new(&grid, cfg_large_eps.clone()).unwrap();
    let mut s_bh_large = IbUmapSolver::<Layout2d<f64>, BarnesHut>::new_with_backend(
        &grid,
        cfg_large_eps.clone(),
        BarnesHut::new(0.0),
    )
    .unwrap();
    let mut s_fft_large = IbUmapSolver::<Layout2d<f64>, IbFft>::new(&grid, cfg_large_eps).unwrap();

    let mut l_ex_small = init.clone();
    let mut l_ex_large = init.clone();
    let mut l_bh_large = init.clone();
    let mut l_fft_large = init.clone();

    s_ex_small.run(&mut l_ex_small).unwrap();
    s_ex_large.run(&mut l_ex_large).unwrap();
    s_bh_large.run(&mut l_bh_large).unwrap();
    s_fft_large.run(&mut l_fft_large).unwrap();

    assert_ne!(l_ex_small, l_ex_large);
    for i in 0..n {
        let bh_err =
            (l_bh_large.x()[i] - l_ex_large.x()[i]).hypot(l_bh_large.y()[i] - l_ex_large.y()[i]);
        assert!(bh_err < 1e-10, "BH(theta=0) vs Exact err={bh_err}");
        let fft_err =
            (l_fft_large.x()[i] - l_ex_large.x()[i]).hypot(l_fft_large.y()[i] - l_ex_large.y()[i]);
        assert!(fft_err < 0.08, "IbFft vs Exact err={fft_err}");
    }

    // Direct IbUmapRepulsion::compute_repulsion rejects non-finite or non-positive repulsive_eps.
    let mut forces = vec![0.0_f64; n * 2];
    let mut ex_state =
        <Exact as IbUmapRepulsion<Layout2d<f64>>>::init_state(&Exact::new(), n).unwrap();
    assert!(matches!(
        Exact::new().compute_repulsion(&mut ex_state, &init, 1.5, 0.9, 4.0, 0.0, &mut forces),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        Exact::new().compute_repulsion(&mut ex_state, &init, 1.5, 0.9, 4.0, f64::NAN, &mut forces),
        Err(CoreError::NonFiniteValue { .. })
    ));
}
