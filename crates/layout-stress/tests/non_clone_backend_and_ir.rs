use egraph_core::{
    CoreResult, CsrGraph, IterativeSolver, Layout2d, RiemannianLayout, SolverStatus,
};
#[cfg(feature = "baselines")]
use egraph_core::{LayoutStorage, RepulsionBackend};
#[cfg(feature = "baselines")]
use egraph_layout_stress::{MaxentRepulsion, MaxentStressConfig, MaxentStressSolver};
use egraph_layout_stress::{
    StressMajorizationConfig, StressMajorizationSolver, StressMajorizationStrategy,
};

#[derive(Debug)]
struct NonCloneStressState {
    steps: usize,
}

#[derive(Debug, Clone, Copy)]
struct NonCloneStressStrategy;

impl<L: RiemannianLayout + ?Sized> StressMajorizationStrategy<L> for NonCloneStressStrategy {
    type State = NonCloneStressState;

    fn init_state(&self, _num_nodes: usize, _weights: &[f64]) -> CoreResult<Self::State> {
        Ok(NonCloneStressState { steps: 0 })
    }

    fn validate_target(
        &self,
        _state: &Self::State,
        num_nodes: usize,
        target: &L,
    ) -> CoreResult<()> {
        if target.len() != num_nodes {
            return Err(egraph_core::CoreError::dimension_mismatch(
                num_nodes,
                target.len(),
                "NonCloneStressStrategy::validate_target",
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
        num_nodes: usize,
        _dists: &[f64],
        _weights: &[f64],
        _config: &StressMajorizationConfig,
        prev_stress: f64,
        _layout: &mut L,
    ) -> CoreResult<Option<f64>> {
        if num_nodes <= 1 {
            return Ok(None);
        }
        state.steps += 1;
        Ok(Some(prev_stress * 0.5))
    }
}

#[cfg(feature = "baselines")]
#[derive(Debug)]
struct NonCloneMaxentEvaluator {
    calls: usize,
}

#[cfg(feature = "baselines")]
impl<L: LayoutStorage + ?Sized> RepulsionBackend<L> for NonCloneMaxentEvaluator {
    fn accumulate_repulsion(&mut self, _layout: &L, forces: &mut [f64]) -> CoreResult<Option<f64>> {
        self.calls += 1;
        forces.fill(0.0);
        Ok(None)
    }
}

#[cfg(feature = "baselines")]
#[derive(Debug, Clone, Copy)]
struct NonCloneMaxentBackend;

#[cfg(feature = "baselines")]
impl<L: LayoutStorage + ?Sized> MaxentRepulsion<L> for NonCloneMaxentBackend {
    type Evaluator = NonCloneMaxentEvaluator;

    fn build_evaluator(
        &self,
        _num_nodes: usize,
        _q: f64,
        _eps_sq: f64,
    ) -> CoreResult<Self::Evaluator> {
        Ok(NonCloneMaxentEvaluator { calls: 0 })
    }
}

#[test]
fn non_clone_stress_strategy_state_compiles_and_runs() {
    let g = CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3)], None, false, false).unwrap();
    let mut solver = StressMajorizationSolver::<Layout2d<f64>, _>::from_graph_with_strategy(
        &g,
        StressMajorizationConfig::new().with_max_iterations(3),
        NonCloneStressStrategy,
    )
    .unwrap();
    let mut layout = Layout2d::<f64>::init(4, 7, 1.0).unwrap();
    assert_eq!(solver.run(&mut layout).unwrap(), SolverStatus::Completed);
    assert_eq!(solver.iteration(), 3);

    // Conditional Clone still works on standard Clone strategies.
    let std_solver =
        StressMajorizationSolver::<Layout2d<f64>>::from_graph(&g, StressMajorizationConfig::new())
            .unwrap();
    let cloned = std_solver.clone();
    assert_eq!(cloned.distances(), std_solver.distances());
    assert_eq!(cloned.weights(), std_solver.weights());
}

#[test]
fn stress_majorization_distances_and_weights_getters() {
    let g =
        CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2)], Some(&[2.0, 3.0]), false, false).unwrap();
    let solver =
        StressMajorizationSolver::<Layout2d<f64>>::from_graph(&g, StressMajorizationConfig::new())
            .unwrap();
    let d = solver.distances();
    let w = solver.weights();
    assert_eq!(d.len(), 16);
    assert_eq!(w.len(), 16);

    // Diagonal and disconnected vertex 3 must be 0.0.
    for i in 0..4 {
        assert_eq!(d[i * 4 + i], 0.0);
        assert_eq!(w[i * 4 + i], 0.0);
        assert_eq!(d[3 * 4 + i], 0.0);
        assert_eq!(w[3 * 4 + i], 0.0);
    }
    // Pair (0, 2) has distance 5.0 and weight 1/25.
    assert!((d[2] - 5.0).abs() < 1e-12);
    assert!((d[2 * 4] - 5.0).abs() < 1e-12);
    assert!((w[2] - 1.0 / 25.0).abs() < 1e-12);
    assert!((w[2 * 4] - 1.0 / 25.0).abs() < 1e-12);
}

#[cfg(feature = "baselines")]
#[test]
fn non_clone_maxent_evaluator_and_ir_getters() {
    // Disconnected graph: triangle (0, 1, 2) + edge (3, 4) + isolated node 5.
    let g = CsrGraph::<f64>::from_edges(
        6,
        &[(0, 1), (1, 2), (0, 2), (3, 4)],
        Some(&[1.0, 2.0, 1.5, 4.0]),
        false,
        false,
    )
    .unwrap();

    let mut non_clone_solver = MaxentStressSolver::<Layout2d<f64>, _>::from_graph_with_backend(
        &g,
        MaxentStressConfig::new()
            .with_stages(1)
            .with_steps_per_stage(2),
        NonCloneMaxentBackend,
    )
    .unwrap();
    let mut layout = Layout2d::<f64>::init(6, 5, 1.0).unwrap();
    assert_eq!(
        non_clone_solver.run(&mut layout).unwrap(),
        SolverStatus::Completed
    );
    assert!(layout.is_all_finite());

    let std_solver =
        MaxentStressSolver::<Layout2d<f64>>::new(&g, MaxentStressConfig::new()).unwrap();
    let cloned = std_solver.clone();
    assert_eq!(cloned.num_edges(), 4);
    assert_eq!(std_solver.num_edges(), 4);

    let edges: Vec<_> = std_solver.iter_edges().collect();
    assert_eq!(edges.len(), 4);
    assert_eq!((edges[3].0, edges[3].1), (3, 4));
    assert!((edges[3].2 - 4.0).abs() < 1e-12);
    assert!((edges[3].3 - 1.0 / 16.0).abs() < 1e-12);

    let comp_ids = std_solver.component_ids();
    assert_eq!(comp_ids.len(), 6);
    assert_eq!(comp_ids[0], comp_ids[1]);
    assert_eq!(comp_ids[1], comp_ids[2]);
    assert_eq!(comp_ids[3], comp_ids[4]);
    assert_ne!(comp_ids[0], comp_ids[3]);
    assert_ne!(comp_ids[0], comp_ids[5]);
    assert_ne!(comp_ids[3], comp_ids[5]);
    assert_eq!(std_solver.component_sizes(), &[3, 2, 1]);
    assert_eq!(std_solver.amg_solver().num_nodes(), 6);
}
