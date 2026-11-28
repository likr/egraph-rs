use egraph_core::{
    CoreError, CoreResult, CsrGraph, IterativeSolver, Layout2d, Layout3d, LayoutNd, LayoutStorage,
    SolverStatus,
};
#[cfg(feature = "baselines")]
use egraph_core::{DeterministicRng, RiemannianLayout};
use egraph_layout_force::{
    ForceAtlas2Config, ForceAtlas2Solver, ForceRepulsion, ForceRepulsionParams, YifanHuConfig,
    YifanHuLevelCoords, YifanHuRepulsion, YifanHuSolver,
};
#[cfg(feature = "baselines")]
use egraph_layout_force::{
    FrRepulsion, FruchtermanReingoldConfig, FruchtermanReingoldSolver, LinLogConfig,
    LinLogRepulsion, LinLogRepulsionBackend, LinLogSolver,
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

#[derive(Debug, Clone, Copy)]
struct FailingForceBackend {
    kind: FailureKind,
    fail_on_call: usize,
}

impl<L: LayoutStorage + ?Sized> ForceRepulsion<L> for FailingForceBackend {
    type State = usize;

    fn validate(&self) -> CoreResult<()> {
        Ok(())
    }

    fn init_state(&self, _num_nodes: usize) -> CoreResult<Self::State> {
        Ok(0)
    }

    fn accumulate_repulsion(
        &self,
        state: &mut Self::State,
        _layout: &L,
        _params: &ForceRepulsionParams<'_>,
        _forces: &mut [f64],
    ) -> CoreResult<()> {
        *state += 1;
        if *state >= self.fail_on_call {
            return Err(self.kind.to_error("simulated force_atlas2 failure"));
        }
        Ok(())
    }
}

#[cfg(feature = "baselines")]
impl<L: LayoutStorage + ?Sized> LinLogRepulsionBackend<L> for FailingForceBackend {
    type State = usize;

    fn validate(&self) -> CoreResult<()> {
        Ok(())
    }

    fn init_state(&self, _num_nodes: usize) -> CoreResult<Self::State> {
        Ok(0)
    }

    fn accumulate_repulsion(
        &self,
        state: &mut Self::State,
        _layout: &L,
        _masses: &[f64],
        _repulsion_exponent: f64,
        _eps_sq: f64,
        _forces: &mut [f64],
    ) -> CoreResult<()> {
        *state += 1;
        if *state >= self.fail_on_call {
            return Err(self.kind.to_error("simulated linlog failure"));
        }
        Ok(())
    }
}

impl<L: LayoutStorage + ?Sized> YifanHuRepulsion<L> for FailingForceBackend {
    type State = usize;

    fn validate(&self) -> CoreResult<()> {
        Ok(())
    }

    fn init_state(&self, _num_nodes: usize) -> CoreResult<Self::State> {
        Ok(0)
    }

    fn accumulate_level_repulsion(
        &self,
        state: &mut Self::State,
        _coords: &YifanHuLevelCoords<'_>,
        _eps_sq: f64,
        _rep_scale: f64,
        _forces: &mut [f64],
    ) -> CoreResult<()> {
        *state += 1;
        if *state >= self.fail_on_call {
            return Err(self.kind.to_error("simulated yifan_hu failure"));
        }
        Ok(())
    }
}

#[cfg(feature = "baselines")]
impl<L: RiemannianLayout + ?Sized> FrRepulsion<L> for FailingForceBackend {
    type State = usize;

    fn validate(&self) -> CoreResult<()> {
        Ok(())
    }

    fn init_state(&self, _num_nodes: usize) -> CoreResult<Self::State> {
        Ok(0)
    }

    fn accumulate_repulsion(
        &self,
        state: &mut Self::State,
        _layout: &L,
        _k: f64,
        _iter: usize,
        _rng: &mut DeterministicRng,
        _disp: &mut [f64],
    ) -> CoreResult<()> {
        *state += 1;
        if *state >= self.fail_on_call {
            return Err(self.kind.to_error("simulated fr failure"));
        }
        Ok(())
    }
}

fn path4() -> CsrGraph<f64> {
    CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3)], None, false, false).unwrap()
}

#[test]
fn test_fa2_step0_collapsed_layout_non_mutation_on_backend_error() {
    let g = path4();
    for kind in [
        FailureKind::BackendUnavailable,
        FailureKind::DeviceLimitExceeded,
    ] {
        let backend = FailingForceBackend {
            kind,
            fail_on_call: 1,
        };

        let mut fa2 = ForceAtlas2Solver::<Layout2d<f64>, _>::with_backend(
            &g,
            ForceAtlas2Config::new(),
            backend,
        )
        .unwrap();
        let mut l_fa2 = Layout2d::<f64>::zeros(4);
        assert!(fa2.step(&mut l_fa2).is_err());
        assert_eq!(fa2.status(), SolverStatus::Failed);
        assert!(l_fa2.x().iter().all(|&v| v.to_bits() == 0));
        assert!(l_fa2.y().iter().all(|&v| v.to_bits() == 0));

        let mut fa3 = ForceAtlas2Solver::<Layout3d<f64>, _>::with_backend(
            &g,
            ForceAtlas2Config::new(),
            backend,
        )
        .unwrap();
        let mut l_fa3 = Layout3d::<f64>::zeros(4);
        assert!(fa3.step(&mut l_fa3).is_err());
        assert_eq!(fa3.status(), SolverStatus::Failed);
        assert!(l_fa3.x().iter().all(|&v| v.to_bits() == 0));
        assert!(l_fa3.y().iter().all(|&v| v.to_bits() == 0));
        assert!(l_fa3.z().iter().all(|&v| v.to_bits() == 0));

        let mut fand = ForceAtlas2Solver::<LayoutNd<f64>, _>::with_backend(
            &g,
            ForceAtlas2Config::new(),
            backend,
        )
        .unwrap();
        let mut l_fand = LayoutNd::<f64>::zeros(4, 4).unwrap();
        assert!(fand.step(&mut l_fand).is_err());
        assert_eq!(fand.status(), SolverStatus::Failed);
        assert!(l_fand.coords().iter().all(|&v| v.to_bits() == 0));
    }
}

#[cfg(feature = "baselines")]
#[test]
fn test_linlog_step0_collapsed_layout_non_mutation_on_backend_error() {
    let g = path4();
    for kind in [
        FailureKind::BackendUnavailable,
        FailureKind::DeviceLimitExceeded,
    ] {
        let backend = FailingForceBackend {
            kind,
            fail_on_call: 1,
        };

        let mut ll2 =
            LinLogSolver::<Layout2d<f64>, _>::with_backend(&g, LinLogConfig::new(), backend)
                .unwrap();
        let mut l_ll2 = Layout2d::<f64>::zeros(4);
        assert!(ll2.step(&mut l_ll2).is_err());
        assert_eq!(ll2.status(), SolverStatus::Failed);
        assert!(l_ll2.x().iter().all(|&v| v.to_bits() == 0));
        assert!(l_ll2.y().iter().all(|&v| v.to_bits() == 0));

        let mut ll3 =
            LinLogSolver::<Layout3d<f64>, _>::with_backend(&g, LinLogConfig::new(), backend)
                .unwrap();
        let mut l_ll3 = Layout3d::<f64>::zeros(4);
        assert!(ll3.step(&mut l_ll3).is_err());
        assert_eq!(ll3.status(), SolverStatus::Failed);
        assert!(l_ll3.x().iter().all(|&v| v.to_bits() == 0));
        assert!(l_ll3.y().iter().all(|&v| v.to_bits() == 0));
        assert!(l_ll3.z().iter().all(|&v| v.to_bits() == 0));

        let mut llnd =
            LinLogSolver::<LayoutNd<f64>, _>::with_backend(&g, LinLogConfig::new(), backend)
                .unwrap();
        let mut l_llnd = LayoutNd::<f64>::zeros(4, 4).unwrap();
        assert!(llnd.step(&mut l_llnd).is_err());
        assert_eq!(llnd.status(), SolverStatus::Failed);
        assert!(l_llnd.coords().iter().all(|&v| v.to_bits() == 0));
    }
}

#[test]
fn test_fa2_and_yifan_hu_status_failed_on_backend_errors() {
    let g = path4();
    for kind in [
        FailureKind::BackendUnavailable,
        FailureKind::DeviceLimitExceeded,
    ] {
        for fail_on_call in [1usize, 2usize] {
            let backend = FailingForceBackend { kind, fail_on_call };

            let mut fa = ForceAtlas2Solver::<Layout2d<f64>, _>::with_backend(
                &g,
                ForceAtlas2Config::new(),
                backend,
            )
            .unwrap();
            let mut l_fa = Layout2d::<f64>::init(4, 1, 1.0).unwrap();
            if fail_on_call == 2 {
                assert_eq!(fa.step(&mut l_fa).unwrap(), SolverStatus::Running);
            }
            assert!(fa.step(&mut l_fa).is_err());
            assert_eq!(fa.status(), SolverStatus::Failed);
            assert_eq!(fa.step(&mut l_fa).unwrap(), SolverStatus::Failed);

            let mut yh =
                YifanHuSolver::<Layout2d<f64>, _>::with_backend(&g, YifanHuConfig::new(), backend)
                    .unwrap();
            let mut l_yh = Layout2d::<f64>::init(4, 3, 1.0).unwrap();
            if fail_on_call == 2 {
                assert_eq!(yh.step(&mut l_yh).unwrap(), SolverStatus::Running);
            }
            assert!(yh.step(&mut l_yh).is_err());
            assert_eq!(yh.status(), SolverStatus::Failed);
            assert_eq!(yh.step(&mut l_yh).unwrap(), SolverStatus::Failed);
        }
    }
}

#[cfg(feature = "baselines")]
#[test]
fn test_linlog_and_fr_status_failed_on_backend_errors() {
    let g = path4();
    for kind in [
        FailureKind::BackendUnavailable,
        FailureKind::DeviceLimitExceeded,
    ] {
        for fail_on_call in [1usize, 2usize] {
            let backend = FailingForceBackend { kind, fail_on_call };

            let mut ll =
                LinLogSolver::<Layout2d<f64>, _>::with_backend(&g, LinLogConfig::new(), backend)
                    .unwrap();
            let mut l_ll = Layout2d::<f64>::init(4, 2, 1.0).unwrap();
            if fail_on_call == 2 {
                assert_eq!(ll.step(&mut l_ll).unwrap(), SolverStatus::Running);
            }
            assert!(ll.step(&mut l_ll).is_err());
            assert_eq!(ll.status(), SolverStatus::Failed);
            assert_eq!(ll.step(&mut l_ll).unwrap(), SolverStatus::Failed);

            let mut fr = FruchtermanReingoldSolver::<Layout2d<f64>, _>::with_backend(
                &g,
                FruchtermanReingoldConfig::new(),
                backend,
            )
            .unwrap();
            let mut l_fr = Layout2d::<f64>::init(4, 4, 1.0).unwrap();
            if fail_on_call == 2 {
                assert_eq!(fr.step(&mut l_fr).unwrap(), SolverStatus::Running);
            }
            assert!(fr.step(&mut l_fr).is_err());
            assert_eq!(fr.status(), SolverStatus::Failed);
            assert_eq!(fr.step(&mut l_fr).unwrap(), SolverStatus::Failed);
        }
    }
}

#[test]
fn test_force_atlas2_rejects_powf_overflow_at_construction() {
    let g_powf = CsrGraph::<f64>::from_edges(
        3,
        &[(0, 1), (1, 2)],
        Some(&[1e200_f64, 1e200_f64]),
        false,
        false,
    )
    .unwrap();
    let fa_cfg = ForceAtlas2Config::new().with_edge_weight_influence(2.0);
    assert!(matches!(
        ForceAtlas2Solver::<Layout2d<f64>>::new(&g_powf, fa_cfg),
        Err(CoreError::NonFiniteValue { .. })
    ));
}

#[cfg(feature = "baselines")]
#[test]
fn test_linlog_rejects_powf_and_mass_overflow_at_construction() {
    let g_powf = CsrGraph::<f64>::from_edges(
        3,
        &[(0, 1), (1, 2)],
        Some(&[1e200_f64, 1e200_f64]),
        false,
        false,
    )
    .unwrap();
    let ll_cfg = LinLogConfig::new()
        .with_edge_weight_influence(2.0)
        .with_repulsion_model(LinLogRepulsion::EdgeRepulsion);
    assert!(matches!(
        LinLogSolver::<Layout2d<f64>>::new(&g_powf, ll_cfg),
        Err(CoreError::NonFiniteValue { .. })
    ));

    let g_sum = CsrGraph::<f64>::from_edges(
        3,
        &[(0, 1), (1, 2)],
        Some(&[1e308_f64, 1e308_f64]),
        false,
        false,
    )
    .unwrap();
    let ll_sum_cfg = LinLogConfig::new()
        .with_edge_weight_influence(1.0)
        .with_repulsion_model(LinLogRepulsion::EdgeRepulsion);
    assert!(matches!(
        LinLogSolver::<Layout2d<f64>>::new(&g_sum, ll_sum_cfg),
        Err(CoreError::NonFiniteValue { .. })
    ));
}
