//! Opaque-box E2E tests for `IbUmapSolver<L, B>`, `DefaultIbUmapBackend`,
//! `IbUmapRepulsion<L>`, `IbUmapBackend2d`, and custom plugin error propagation.

use std::cell::Cell;
use std::rc::Rc;

use egraph_core::{CoreError, CoreResult, CsrGraph, IterativeSolver, Layout2d, SolverStatus};
use egraph_layout_neighbor::{
    BarnesHut, DefaultIbUmapBackend, Exact, IbFft, IbUmapBackend2d, IbUmapConfig, IbUmapRepulsion,
    IbUmapSolver,
};
use egraph_neighborhood::{
    DogVaradhanConfig, build_pure_2hop_knn, compute_fuzzy_simplicial_set,
    compute_regularized_dog_varadhan_landmarks,
};

fn make_grid(rows: usize, cols: usize) -> CsrGraph<f64> {
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
    CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap()
}

fn assert_default_ib_umap_backend<L: DefaultIbUmapBackend>() {}

#[derive(Debug, Clone)]
struct CountingZeroRepulsion {
    validate_calls: Rc<Cell<usize>>,
    init_calls: Rc<Cell<usize>>,
    compute_calls: Rc<Cell<usize>>,
}

impl CountingZeroRepulsion {
    fn new() -> Self {
        Self {
            validate_calls: Rc::new(Cell::new(0)),
            init_calls: Rc::new(Cell::new(0)),
            compute_calls: Rc::new(Cell::new(0)),
        }
    }
}

impl IbUmapRepulsion<Layout2d<f64>> for CountingZeroRepulsion {
    type State = usize;

    fn validate_config(&self) -> CoreResult<()> {
        self.validate_calls.set(self.validate_calls.get() + 1);
        Ok(())
    }

    fn init_state(&self, num_nodes: usize) -> CoreResult<Self::State> {
        self.init_calls.set(self.init_calls.get() + 1);
        Ok(num_nodes)
    }

    fn compute_repulsion(
        &self,
        _state: &mut Self::State,
        layout: &Layout2d<f64>,
        _a: f64,
        _b: f64,
        _c_k: f64,
        rep_forces: &mut [f64],
    ) -> CoreResult<()> {
        self.compute_calls.set(self.compute_calls.get() + 1);
        assert_eq!(rep_forces.len(), layout.len() * 2);
        rep_forces.fill(0.0);
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
enum FaultPhase {
    ValidateUnavailable,
    InitDeviceLimit,
    StepBackendUnavailableAfter(usize),
    StepDeviceLimitAfter(usize),
    StepNumericalBreakdownAfter(usize),
}

#[derive(Debug, Clone)]
struct FaultingIbUmapRepulsion {
    phase: FaultPhase,
    compute_calls: Rc<Cell<usize>>,
}

impl FaultingIbUmapRepulsion {
    fn new(phase: FaultPhase) -> Self {
        Self {
            phase,
            compute_calls: Rc::new(Cell::new(0)),
        }
    }
}

impl IbUmapRepulsion<Layout2d<f64>> for FaultingIbUmapRepulsion {
    type State = usize;

    fn validate_config(&self) -> CoreResult<()> {
        if matches!(self.phase, FaultPhase::ValidateUnavailable) {
            return Err(CoreError::backend_unavailable("no device adapter"));
        }
        Ok(())
    }

    fn init_state(&self, num_nodes: usize) -> CoreResult<Self::State> {
        if matches!(self.phase, FaultPhase::InitDeviceLimit) {
            return Err(CoreError::device_limit_exceeded("node buffer too large"));
        }
        Ok(num_nodes)
    }

    fn compute_repulsion(
        &self,
        _state: &mut Self::State,
        _layout: &Layout2d<f64>,
        _a: f64,
        _b: f64,
        _c_k: f64,
        rep_forces: &mut [f64],
    ) -> CoreResult<()> {
        let count = self.compute_calls.get() + 1;
        self.compute_calls.set(count);
        match self.phase {
            FaultPhase::StepBackendUnavailableAfter(k) if count >= k => {
                Err(CoreError::backend_unavailable("device lost during step"))
            }
            FaultPhase::StepDeviceLimitAfter(k) if count >= k => {
                Err(CoreError::device_limit_exceeded("grid allocation overflow"))
            }
            FaultPhase::StepNumericalBreakdownAfter(k) if count >= k => Err(
                CoreError::numerical_breakdown("synthetic repulsion breakdown"),
            ),
            _ => {
                rep_forces.fill(0.0);
                Ok(())
            }
        }
    }
}

#[test]
fn ib_umap_default_type_params_and_all_backends_2d_f64_f32() {
    assert_default_ib_umap_backend::<Layout2d<f64>>();
    assert_default_ib_umap_backend::<Layout2d<f32>>();
    let _: <Layout2d<f64> as DefaultIbUmapBackend>::Backend = IbFft::new();

    let g = make_grid(4, 4);
    let cfg = IbUmapConfig::new().with_iterations(6).with_seed(17);

    let mut default_solver: IbUmapSolver =
        IbUmapSolver::<Layout2d<f64>>::new(&g, cfg.clone()).unwrap();
    let mut default_layout = Layout2d::<f64>::init(16, 11, 2.0).unwrap();
    assert_eq!(
        default_solver.run(&mut default_layout).unwrap(),
        SolverStatus::Completed
    );
    assert!(default_layout.is_all_finite());

    for wrapper in [
        IbUmapBackend2d::IbFft(IbFft::new()),
        IbUmapBackend2d::BarnesHut(BarnesHut::new(0.5)),
        IbUmapBackend2d::Exact(Exact::new()),
    ] {
        let mut s_run_64 = IbUmapSolver::<Layout2d<f64>, IbUmapBackend2d>::new_with_backend(
            &g,
            cfg.clone(),
            wrapper.clone(),
        )
        .unwrap();
        let mut s_step_64 = IbUmapSolver::<Layout2d<f64>, IbUmapBackend2d>::new_with_backend(
            &g,
            cfg.clone(),
            wrapper.clone(),
        )
        .unwrap();
        let mut l_run_64 = Layout2d::<f64>::init(16, 11, 2.0).unwrap();
        let mut l_step_64 = l_run_64.clone();
        assert_eq!(
            s_run_64.run(&mut l_run_64).unwrap(),
            SolverStatus::Completed
        );
        while !s_step_64.status().is_finished() {
            s_step_64.step(&mut l_step_64).unwrap();
        }
        assert_eq!(l_run_64, l_step_64);

        let mut s_32 = IbUmapSolver::<Layout2d<f32>, IbUmapBackend2d>::new_with_backend(
            &g,
            cfg.clone(),
            wrapper,
        )
        .unwrap();
        let mut l_32 = Layout2d::<f32>::init(16, 11, 2.0).unwrap();
        assert_eq!(s_32.run(&mut l_32).unwrap(), SolverStatus::Completed);
        assert!(l_32.is_all_finite());
    }
}

#[test]
fn ib_umap_all_with_backend_constructors_parity() {
    let g = make_grid(3, 4);
    let n = g.num_nodes();
    let cfg = IbUmapConfig::new().with_iterations(5).with_seed(42);
    let dog_cfg = DogVaradhanConfig::default()
        .with_n_landmarks(6)
        .with_seed(42);
    let (landmarks, _) = compute_regularized_dog_varadhan_landmarks(&g, &dog_cfg).unwrap();
    let (knn, _, _) = build_pure_2hop_knn(&g).unwrap();
    let probs = compute_fuzzy_simplicial_set(&knn, 64, 1e-5).unwrap();

    let mut s_lm_bh = IbUmapSolver::<Layout2d<f64>, BarnesHut>::from_landmarks_with_backend(
        &g,
        &landmarks,
        16,
        cfg.clone(),
        BarnesHut::new(0.0),
    )
    .unwrap();
    let mut s_lm_ex = IbUmapSolver::<Layout2d<f64>, Exact>::from_landmarks_with_backend(
        &g,
        &landmarks,
        16,
        cfg.clone(),
        Exact::new(),
    )
    .unwrap();
    assert_eq!(s_lm_bh.ab(), s_lm_ex.ab());
    assert_eq!(s_lm_bh.num_edges(), s_lm_ex.num_edges());

    let mut l_lm_bh = Layout2d::<f64>::init(n, 7, 1.0).unwrap();
    let mut l_lm_ex = l_lm_bh.clone();
    assert_eq!(s_lm_bh.run(&mut l_lm_bh).unwrap(), SolverStatus::Completed);
    assert_eq!(s_lm_ex.run(&mut l_lm_ex).unwrap(), SolverStatus::Completed);
    for i in 0..n {
        assert!((l_lm_bh.x()[i] - l_lm_ex.x()[i]).abs() < 1e-12);
        assert!((l_lm_bh.y()[i] - l_lm_ex.y()[i]).abs() < 1e-12);
    }

    let mut s_dog = IbUmapSolver::<Layout2d<f64>, Exact>::from_graph_dog_with_backend(
        &g,
        16,
        &dog_cfg,
        cfg.clone(),
        Exact::new(),
    )
    .unwrap();
    let mut s_knn = IbUmapSolver::<Layout2d<f64>, Exact>::from_knn_with_backend(
        &knn,
        64,
        1e-5,
        cfg.clone(),
        Exact::new(),
    )
    .unwrap();
    let mut s_fset = IbUmapSolver::<Layout2d<f64>, Exact>::from_fuzzy_set_with_backend(
        probs.clone(),
        None,
        cfg.clone(),
        Exact::new(),
    )
    .unwrap();
    let mut s_fref = IbUmapSolver::<Layout2d<f64>, Exact>::from_fuzzy_set_ref_with_backend(
        &probs,
        None,
        cfg,
        Exact::new(),
    )
    .unwrap();

    assert_eq!(s_knn.num_edges(), s_fset.num_edges());
    assert_eq!(s_fset.num_edges(), s_fref.num_edges());

    let base = Layout2d::<f64>::init(n, 7, 1.0).unwrap();
    let mut l_dog = base.clone();
    let mut l_knn = base.clone();
    let mut l_fset = base.clone();
    let mut l_fref = base;
    assert_eq!(s_dog.run(&mut l_dog).unwrap(), SolverStatus::Completed);
    assert_eq!(s_knn.run(&mut l_knn).unwrap(), SolverStatus::Completed);
    assert_eq!(s_fset.run(&mut l_fset).unwrap(), SolverStatus::Completed);
    assert_eq!(s_fref.run(&mut l_fref).unwrap(), SolverStatus::Completed);
    assert_eq!(l_fset, l_fref);
    for i in 0..n {
        assert!((l_knn.x()[i] - l_fset.x()[i]).abs() < 1e-12);
        assert!((l_knn.y()[i] - l_fset.y()[i]).abs() < 1e-12);
    }
}

#[test]
fn ib_umap_custom_external_repulsion_trait_impl() {
    let g = make_grid(3, 3);
    let backend = CountingZeroRepulsion::new();
    let cfg = IbUmapConfig::new().with_iterations(4).with_seed(9);
    let mut solver =
        IbUmapSolver::<Layout2d<f64>, CountingZeroRepulsion>::new_with_backend(&g, cfg, backend)
            .unwrap();
    assert!(solver.backend().validate_calls.get() >= 1);
    assert_eq!(solver.backend().init_calls.get(), 1);
    assert_eq!(solver.backend().compute_calls.get(), 0);

    let mut layout = Layout2d::<f64>::init(9, 2, 1.0).unwrap();
    assert_eq!(solver.run(&mut layout).unwrap(), SolverStatus::Completed);
    assert_eq!(solver.backend().compute_calls.get(), 4);
    assert!(layout.is_all_finite());
}

#[test]
fn ib_umap_custom_repulsion_error_propagation_and_failure_status() {
    let g = make_grid(3, 3);
    let cfg = IbUmapConfig::new().with_iterations(4).with_seed(9);

    assert!(matches!(
        IbUmapSolver::<Layout2d<f64>, FaultingIbUmapRepulsion>::new_with_backend(
            &g,
            cfg.clone(),
            FaultingIbUmapRepulsion::new(FaultPhase::ValidateUnavailable),
        ),
        Err(CoreError::BackendUnavailable(_))
    ));
    assert!(matches!(
        IbUmapSolver::<Layout2d<f64>, FaultingIbUmapRepulsion>::new_with_backend(
            &g,
            cfg.clone(),
            FaultingIbUmapRepulsion::new(FaultPhase::InitDeviceLimit),
        ),
        Err(CoreError::DeviceLimitExceeded(_))
    ));

    // Mid-step BackendUnavailable transitions solver to terminal SolverStatus::Failed.
    let mut s_unavail = IbUmapSolver::<Layout2d<f64>, FaultingIbUmapRepulsion>::new_with_backend(
        &g,
        cfg.clone(),
        FaultingIbUmapRepulsion::new(FaultPhase::StepBackendUnavailableAfter(2)),
    )
    .unwrap();
    let mut l_unavail = Layout2d::<f64>::init(9, 1, 1.0).unwrap();
    assert_eq!(
        s_unavail.step(&mut l_unavail).unwrap(),
        SolverStatus::Running
    );
    let l_unavail_before_fail = l_unavail.clone();
    assert!(matches!(
        s_unavail.step(&mut l_unavail),
        Err(CoreError::BackendUnavailable(_))
    ));
    assert_eq!(s_unavail.status(), SolverStatus::Failed);
    assert!(s_unavail.status().is_failed());
    assert_eq!(s_unavail.iteration(), 1);
    assert_eq!(s_unavail.backend().compute_calls.get(), 2);
    assert_eq!(l_unavail, l_unavail_before_fail);
    assert_eq!(
        s_unavail.step(&mut l_unavail).unwrap(),
        SolverStatus::Failed
    );
    assert_eq!(s_unavail.backend().compute_calls.get(), 2);
    assert_eq!(l_unavail, l_unavail_before_fail);

    // Mid-step DeviceLimitExceeded transitions solver to terminal SolverStatus::Failed.
    let mut s_limit = IbUmapSolver::<Layout2d<f64>, FaultingIbUmapRepulsion>::new_with_backend(
        &g,
        cfg.clone(),
        FaultingIbUmapRepulsion::new(FaultPhase::StepDeviceLimitAfter(1)),
    )
    .unwrap();
    let mut l_limit = Layout2d::<f64>::init(9, 2, 1.0).unwrap();
    let l_limit_initial = l_limit.clone();
    assert!(matches!(
        s_limit.run(&mut l_limit),
        Err(CoreError::DeviceLimitExceeded(_))
    ));
    assert_eq!(s_limit.status(), SolverStatus::Failed);
    assert!(s_limit.status().is_failed());
    assert_eq!(s_limit.iteration(), 0);
    assert_eq!(s_limit.backend().compute_calls.get(), 1);
    assert_eq!(l_limit, l_limit_initial);
    assert_eq!(s_limit.step(&mut l_limit).unwrap(), SolverStatus::Failed);
    assert_eq!(s_limit.backend().compute_calls.get(), 1);
    assert_eq!(l_limit, l_limit_initial);

    // Mid-step NumericalBreakdown transitions solver to terminal SolverStatus::Failed.
    let mut s_breakdown = IbUmapSolver::<Layout2d<f64>, FaultingIbUmapRepulsion>::new_with_backend(
        &g,
        cfg,
        FaultingIbUmapRepulsion::new(FaultPhase::StepNumericalBreakdownAfter(2)),
    )
    .unwrap();
    let mut l_breakdown = Layout2d::<f64>::init(9, 3, 1.0).unwrap();
    assert_eq!(
        s_breakdown.step(&mut l_breakdown).unwrap(),
        SolverStatus::Running
    );
    assert!(matches!(
        s_breakdown.step(&mut l_breakdown),
        Err(CoreError::NumericalBreakdown(_))
    ));
    assert_eq!(s_breakdown.status(), SolverStatus::Failed);
    assert!(s_breakdown.status().is_failed());
    assert_eq!(s_breakdown.backend().compute_calls.get(), 2);
    assert_eq!(
        s_breakdown.step(&mut l_breakdown).unwrap(),
        SolverStatus::Failed
    );
    assert_eq!(s_breakdown.backend().compute_calls.get(), 2);
}

#[test]
fn ib_umap_config_builders_resolve_ab_and_backend_stats() {
    let cfg = IbUmapConfig::new()
        .with_iterations(3)
        .with_learning_rate(0.8)
        .with_min_dist(0.2)
        .with_spread(1.5)
        .with_a(Some(1.75))
        .with_b(Some(0.9))
        .with_c_k(3.5)
        .with_tau_scale(2.5)
        .with_hub_damping(false)
        .with_seed(99);
    assert_eq!(cfg.resolve_ab().unwrap(), (1.75, 0.9));
    cfg.validate().unwrap();

    let g = make_grid(3, 3);
    let mut s_fft =
        IbUmapSolver::<Layout2d<f64>, IbFft>::new_with_backend(&g, cfg.clone(), IbFft::new())
            .unwrap();
    let mut s_bh = IbUmapSolver::<Layout2d<f64>, BarnesHut>::new_with_backend(
        &g,
        cfg.clone(),
        BarnesHut::new(0.5),
    )
    .unwrap();
    let mut s_ex =
        IbUmapSolver::<Layout2d<f64>, Exact>::new_with_backend(&g, cfg, Exact::new()).unwrap();

    assert_eq!(s_bh.backend().theta(), 0.5);
    let mut l_fft = Layout2d::<f64>::init(9, 1, 1.0).unwrap();
    let mut l_bh = l_fft.clone();
    let mut l_ex = l_fft.clone();
    s_fft.step(&mut l_fft).unwrap();
    s_bh.step(&mut l_bh).unwrap();
    s_ex.step(&mut l_ex).unwrap();
    assert!(s_fft.last_fft_stats().is_some());
    assert!(s_fft.total_fft_2d_calls() > 0);
    assert!(s_bh.last_fft_stats().is_none());
    assert!(s_ex.last_fft_stats().is_none());
}
