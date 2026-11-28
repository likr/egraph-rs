use egraph_core::{
    CoreError, CoreResult, CsrGraph, IterativeSolver, Layout2d, RiemannianLayout, SolverStatus,
};
#[cfg(feature = "baselines")]
use egraph_core::{Layout3d, LayoutNd, LayoutStorage, RepulsionBackend};
#[cfg(feature = "baselines")]
use egraph_layout_stress::{MaxentRepulsion, MaxentStressConfig, MaxentStressSolver};
use egraph_layout_stress::{
    StressMajorizationConfig, StressMajorizationSolver, StressMajorizationStrategy,
};

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

#[cfg(feature = "baselines")]
#[derive(Debug, Clone, Copy)]
struct FailingMaxentBackend {
    kind: FailureKind,
    fail_on_call: usize,
}

#[cfg(feature = "baselines")]
#[derive(Debug, Clone)]
struct FailingMaxentEvaluator {
    kind: FailureKind,
    fail_on_call: usize,
    calls: usize,
}

#[cfg(feature = "baselines")]
impl<L: LayoutStorage + ?Sized> RepulsionBackend<L> for FailingMaxentEvaluator {
    fn accumulate_repulsion(&mut self, _layout: &L, forces: &mut [f64]) -> CoreResult<Option<f64>> {
        self.calls += 1;
        if self.calls >= self.fail_on_call {
            return Err(self.kind.to_error("simulated maxent failure"));
        }
        forces.fill(0.0);
        Ok(None)
    }
}

#[cfg(feature = "baselines")]
impl<L: LayoutStorage + ?Sized> MaxentRepulsion<L> for FailingMaxentBackend {
    type Evaluator = FailingMaxentEvaluator;

    fn validate(&self) -> CoreResult<()> {
        Ok(())
    }

    fn build_evaluator(
        &self,
        _num_nodes: usize,
        _q: f64,
        _eps_sq: f64,
    ) -> CoreResult<Self::Evaluator> {
        Ok(FailingMaxentEvaluator {
            kind: self.kind,
            fail_on_call: self.fail_on_call,
            calls: 0,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct FailingStressStrategy {
    kind: FailureKind,
    fail_on_call: usize,
}

impl<L: RiemannianLayout + ?Sized> StressMajorizationStrategy<L> for FailingStressStrategy {
    type State = usize;

    fn init_state(&self, _num_nodes: usize, _weights: &[f64]) -> CoreResult<Self::State> {
        Ok(0)
    }

    fn validate_target(
        &self,
        _state: &Self::State,
        num_nodes: usize,
        target: &L,
    ) -> CoreResult<()> {
        if target.len() != num_nodes {
            return Err(CoreError::dimension_mismatch(
                num_nodes,
                target.len(),
                "FailingStressStrategy::validate_target",
            ));
        }
        target.validate_finite()
    }

    fn compute_stress(
        &self,
        _state: &Self::State,
        _num_nodes: usize,
        _dists: &[f64],
        _weights: &[f64],
        _layout: &L,
    ) -> CoreResult<f64> {
        Ok(1.0)
    }

    fn step(
        &self,
        state: &mut Self::State,
        _num_nodes: usize,
        _dists: &[f64],
        _weights: &[f64],
        _config: &StressMajorizationConfig,
        _prev_stress: f64,
        _layout: &mut L,
    ) -> CoreResult<Option<f64>> {
        *state += 1;
        if *state >= self.fail_on_call {
            return Err(self.kind.to_error("simulated stress_majorization failure"));
        }
        Ok(Some(0.5))
    }
}

fn path4() -> CsrGraph<f64> {
    CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3)], None, false, false).unwrap()
}

#[cfg(feature = "baselines")]
#[test]
fn test_maxent_step0_collapsed_layout_non_mutation_on_backend_error() {
    let g = path4();
    for kind in [
        FailureKind::BackendUnavailable,
        FailureKind::DeviceLimitExceeded,
    ] {
        let backend = FailingMaxentBackend {
            kind,
            fail_on_call: 1,
        };

        let mut mx2 = MaxentStressSolver::<Layout2d<f64>, _>::from_graph_with_backend(
            &g,
            MaxentStressConfig::default(),
            backend,
        )
        .unwrap();
        let mut l2 = Layout2d::<f64>::zeros(4);
        assert!(mx2.step(&mut l2).is_err());
        assert_eq!(mx2.status(), SolverStatus::Failed);
        assert!(l2.x().iter().all(|&v| v.to_bits() == 0));
        assert!(l2.y().iter().all(|&v| v.to_bits() == 0));

        let mut mx3 = MaxentStressSolver::<Layout3d<f64>, _>::from_graph_with_backend(
            &g,
            MaxentStressConfig::default(),
            backend,
        )
        .unwrap();
        let mut l3 = Layout3d::<f64>::zeros(4);
        assert!(mx3.step(&mut l3).is_err());
        assert_eq!(mx3.status(), SolverStatus::Failed);
        assert!(l3.x().iter().all(|&v| v.to_bits() == 0));
        assert!(l3.y().iter().all(|&v| v.to_bits() == 0));
        assert!(l3.z().iter().all(|&v| v.to_bits() == 0));

        let mut mxnd = MaxentStressSolver::<LayoutNd<f64>, _>::from_graph_with_backend(
            &g,
            MaxentStressConfig::default(),
            backend,
        )
        .unwrap();
        let mut lnd = LayoutNd::<f64>::zeros(4, 4).unwrap();
        assert!(mxnd.step(&mut lnd).is_err());
        assert_eq!(mxnd.status(), SolverStatus::Failed);
        assert!(lnd.coords().iter().all(|&v| v.to_bits() == 0));
    }
}

#[test]
fn test_stress_majorization_status_failed_on_backend_errors() {
    let g = path4();
    for kind in [
        FailureKind::BackendUnavailable,
        FailureKind::DeviceLimitExceeded,
    ] {
        for fail_on_call in [1usize, 2usize] {
            let sm_strategy = FailingStressStrategy { kind, fail_on_call };
            let mut sm = StressMajorizationSolver::<Layout2d<f64>, _>::from_graph_with_strategy(
                &g,
                StressMajorizationConfig::default(),
                sm_strategy,
            )
            .unwrap();
            let mut l_sm = Layout2d::<f64>::init(4, 2, 1.0).unwrap();
            if fail_on_call == 2 {
                assert_eq!(sm.step(&mut l_sm).unwrap(), SolverStatus::Running);
            }
            assert!(sm.step(&mut l_sm).is_err());
            assert_eq!(sm.status(), SolverStatus::Failed);
            assert_eq!(sm.step(&mut l_sm).unwrap(), SolverStatus::Failed);
        }
    }
}

#[cfg(feature = "baselines")]
#[test]
fn test_maxent_status_failed_on_backend_errors() {
    let g = path4();
    for kind in [
        FailureKind::BackendUnavailable,
        FailureKind::DeviceLimitExceeded,
    ] {
        for fail_on_call in [1usize, 2usize] {
            let mx_backend = FailingMaxentBackend { kind, fail_on_call };
            let mut mx = MaxentStressSolver::<Layout2d<f64>, _>::from_graph_with_backend(
                &g,
                MaxentStressConfig::default(),
                mx_backend,
            )
            .unwrap();
            let mut l_mx = Layout2d::<f64>::init(4, 1, 1.0).unwrap();
            if fail_on_call == 2 {
                assert_eq!(mx.step(&mut l_mx).unwrap(), SolverStatus::Running);
            }
            assert!(mx.step(&mut l_mx).is_err());
            assert_eq!(mx.status(), SolverStatus::Failed);
            assert_eq!(mx.step(&mut l_mx).unwrap(), SolverStatus::Failed);
        }
    }
}
