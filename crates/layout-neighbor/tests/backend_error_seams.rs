use egraph_core::{
    CoreError, CoreResult, CsrGraph, IterativeSolver, Layout2d, Layout3d, LayoutNd, LayoutStorage,
    SolverStatus,
};
use egraph_layout_neighbor::{
    IbUmapConfig, IbUmapRepulsion, IbUmapSolver, TFdpConfig, TFdpRepulsion, TFdpSolver,
    TsNetConfig, TsNetRepulsion, TsNetSolver,
};
use egraph_neighborhood::SparseProbabilities;

#[derive(Debug, Clone, Copy)]
enum FailureKind {
    BackendUnavailable,
    DeviceLimitExceeded,
}

impl FailureKind {
    fn to_error(self, ctx: &str) -> CoreError {
        match self {
            Self::BackendUnavailable => CoreError::backend_unavailable(ctx),
            Self::DeviceLimitExceeded => CoreError::device_limit_exceeded(ctx),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct FailingNeighborBackend {
    kind: FailureKind,
    fail_on_call: usize,
}

impl<L: LayoutStorage + ?Sized> IbUmapRepulsion<L> for FailingNeighborBackend {
    type State = usize;

    fn validate_config(&self) -> CoreResult<()> {
        Ok(())
    }

    fn init_state(&self, _num_nodes: usize) -> CoreResult<Self::State> {
        Ok(0)
    }

    fn compute_repulsion(
        &self,
        state: &mut Self::State,
        _layout: &L,
        _a: f64,
        _b: f64,
        _c_k: f64,
        _repulsive_eps: f64,
        rep_forces: &mut [f64],
    ) -> CoreResult<()> {
        *state += 1;
        if *state >= self.fail_on_call {
            return Err(self.kind.to_error("simulated ib_umap failure"));
        }
        rep_forces.fill(0.0);
        Ok(())
    }
}

impl<L: LayoutStorage + ?Sized> TFdpRepulsion<L> for FailingNeighborBackend {
    type State = usize;

    fn validate_config(&self) -> CoreResult<()> {
        Ok(())
    }

    fn init_state(&self, _num_nodes: usize) -> CoreResult<Self::State> {
        Ok(0)
    }

    fn compute_repulsion(
        &self,
        state: &mut Self::State,
        _layout: &L,
        _gamma: f64,
        rep_forces: &mut [f64],
    ) -> CoreResult<()> {
        *state += 1;
        if *state >= self.fail_on_call {
            return Err(self.kind.to_error("simulated t_fdp failure"));
        }
        rep_forces.fill(0.0);
        Ok(())
    }
}

impl<L: LayoutStorage + ?Sized> TsNetRepulsion<L> for FailingNeighborBackend {
    type State = usize;

    fn validate_config(&self) -> CoreResult<()> {
        Ok(())
    }

    fn init_state(
        &self,
        _num_nodes: usize,
        _epsilon_r: f64,
        _any_entropy: bool,
        _init_entropy: bool,
    ) -> CoreResult<Self::State> {
        Ok(0)
    }

    fn compute_repulsion(
        &self,
        state: &mut Self::State,
        _layout: &L,
        _epsilon_r: f64,
        _compute_entropy: bool,
        rep_forces: &mut [f64],
        ent_forces: &mut [f64],
    ) -> CoreResult<()> {
        *state += 1;
        if *state >= self.fail_on_call {
            return Err(self.kind.to_error("simulated ts_net failure"));
        }
        rep_forces.fill(0.0);
        ent_forces.fill(0.0);
        Ok(())
    }
}

fn path4() -> CsrGraph<f64> {
    CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3)], None, false, false).unwrap()
}

#[test]
fn test_step0_zero_layout_non_mutation_on_backend_error() {
    let g = path4();
    for kind in [
        FailureKind::BackendUnavailable,
        FailureKind::DeviceLimitExceeded,
    ] {
        let backend = FailingNeighborBackend {
            kind,
            fail_on_call: 1,
        };

        let mut ib64 =
            IbUmapSolver::<Layout2d<f64>, _>::new_with_backend(&g, IbUmapConfig::new(), backend)
                .unwrap();
        let mut l_ib64 = Layout2d::<f64>::zeros(4);
        assert!(ib64.step(&mut l_ib64).is_err());
        assert_eq!(ib64.status(), SolverStatus::Failed);
        assert!(l_ib64.x().iter().all(|&v| v.to_bits() == 0));
        assert!(l_ib64.y().iter().all(|&v| v.to_bits() == 0));

        let mut ib32 =
            IbUmapSolver::<Layout2d<f32>, _>::new_with_backend(&g, IbUmapConfig::new(), backend)
                .unwrap();
        let mut l_ib32 = Layout2d::<f32>::zeros(4);
        assert!(ib32.step(&mut l_ib32).is_err());
        assert!(l_ib32.x().iter().all(|&v| v.to_bits() == 0));
        assert!(l_ib32.y().iter().all(|&v| v.to_bits() == 0));

        let mut tf2 =
            TFdpSolver::<Layout2d<f64>, _>::new_with_backend(&g, TFdpConfig::new(), backend)
                .unwrap();
        let mut l_tf2 = Layout2d::<f64>::zeros(4);
        assert!(tf2.step(&mut l_tf2).is_err());
        assert_eq!(tf2.status(), SolverStatus::Failed);
        assert!(l_tf2.x().iter().all(|&v| v.to_bits() == 0));
        assert!(l_tf2.y().iter().all(|&v| v.to_bits() == 0));

        let mut tf3 =
            TFdpSolver::<Layout3d<f64>, _>::new_with_backend(&g, TFdpConfig::new(), backend)
                .unwrap();
        let mut l_tf3 = Layout3d::<f64>::zeros(4);
        assert!(tf3.step(&mut l_tf3).is_err());
        assert_eq!(tf3.status(), SolverStatus::Failed);
        assert!(l_tf3.x().iter().all(|&v| v.to_bits() == 0));
        assert!(l_tf3.y().iter().all(|&v| v.to_bits() == 0));
        assert!(l_tf3.z().iter().all(|&v| v.to_bits() == 0));

        let mut tfnd =
            TFdpSolver::<LayoutNd<f64>, _>::new_with_backend(&g, TFdpConfig::new(), backend)
                .unwrap();
        let mut l_tfnd = LayoutNd::<f64>::zeros(4, 5).unwrap();
        assert!(tfnd.step(&mut l_tfnd).is_err());
        assert_eq!(tfnd.status(), SolverStatus::Failed);
        assert!(l_tfnd.coords().iter().all(|&v| v.to_bits() == 0));
    }
}

#[test]
fn test_tfdp_and_tsnet_status_failed_on_step0_and_mid_run_backend_errors() {
    let g = path4();
    for kind in [
        FailureKind::BackendUnavailable,
        FailureKind::DeviceLimitExceeded,
    ] {
        for fail_on_call in [1usize, 2usize] {
            let backend = FailingNeighborBackend { kind, fail_on_call };

            let mut tf =
                TFdpSolver::<Layout2d<f64>, _>::new_with_backend(&g, TFdpConfig::new(), backend)
                    .unwrap();
            let mut l_tf = Layout2d::<f64>::init(4, 7, 1.0).unwrap();
            if fail_on_call == 2 {
                assert_eq!(tf.step(&mut l_tf).unwrap(), SolverStatus::Running);
            }
            let err_tf = tf.step(&mut l_tf).unwrap_err();
            assert!(matches!(
                err_tf,
                CoreError::BackendUnavailable(_) | CoreError::DeviceLimitExceeded(_)
            ));
            assert_eq!(tf.status(), SolverStatus::Failed);
            assert_eq!(tf.step(&mut l_tf).unwrap(), SolverStatus::Failed);

            let mut ts = TsNetSolver::<Layout2d<f64>, _>::new_with_backend(
                &g,
                TsNetConfig::pure_2hop(),
                backend,
            )
            .unwrap();
            let mut l_ts = Layout2d::<f64>::init(4, 7, 1.0).unwrap();
            if fail_on_call == 2 {
                assert_eq!(ts.step(&mut l_ts).unwrap(), SolverStatus::Running);
            }
            let err_ts = ts.step(&mut l_ts).unwrap_err();
            assert!(matches!(
                err_ts,
                CoreError::BackendUnavailable(_) | CoreError::DeviceLimitExceeded(_)
            ));
            assert_eq!(ts.status(), SolverStatus::Failed);
            assert_eq!(ts.step(&mut l_ts).unwrap(), SolverStatus::Failed);
        }
    }
}

#[test]
fn test_f64_to_f32_narrowing_overflow_rejected_at_construction() {
    let probs =
        SparseProbabilities::from_csr(2, vec![0, 1, 2], vec![1, 0], vec![1e50_f64, 1e50_f64])
            .unwrap();
    assert!(matches!(
        IbUmapSolver::<Layout2d<f64>>::from_fuzzy_set_ref(&probs, None, IbUmapConfig::new()),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        IbUmapSolver::<Layout2d<f64>>::from_fuzzy_set(probs, None, IbUmapConfig::new()),
        Err(CoreError::NonFiniteValue { .. })
    ));

    let g_big = CsrGraph::<f64>::from_edges(2, &[(0, 1)], Some(&[1e50_f64]), false, false).unwrap();
    assert!(matches!(
        TFdpSolver::<Layout2d<f64>>::new(&g_big, TFdpConfig::new()),
        Err(CoreError::NonFiniteValue { .. })
    ));
}
