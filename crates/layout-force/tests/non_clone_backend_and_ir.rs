use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, IterativeSolver, Layout2d, LayoutStorage, SolverStatus,
};
#[cfg(feature = "baselines")]
use egraph_core::{DeterministicRng, MetricSpace, RiemannianLayout};
use egraph_layout_force::{
    ForceAtlas2Config, ForceAtlas2Solver, ForceRepulsion, ForceRepulsionParams, YifanHuConfig,
    YifanHuLevelCoords, YifanHuRepulsion, YifanHuSolver,
};
#[cfg(feature = "baselines")]
use egraph_layout_force::{
    FrRepulsion, FruchtermanReingoldConfig, FruchtermanReingoldSolver, LinLogConfig,
    LinLogRepulsion, LinLogRepulsionBackend, LinLogSolver,
};

#[derive(Debug)]
struct NonCloneState {
    calls: usize,
}

#[derive(Debug, Clone, Copy)]
struct NonCloneStateBackend;

impl<L: LayoutStorage + ?Sized> ForceRepulsion<L> for NonCloneStateBackend {
    type State = NonCloneState;

    fn validate(&self) -> CoreResult<()> {
        Ok(())
    }

    fn init_state(&self, _num_nodes: usize) -> CoreResult<Self::State> {
        Ok(NonCloneState { calls: 0 })
    }

    fn accumulate_repulsion(
        &self,
        state: &mut Self::State,
        _layout: &L,
        _params: &ForceRepulsionParams<'_>,
        _forces: &mut [f64],
    ) -> CoreResult<()> {
        state.calls += 1;
        Ok(())
    }
}

impl<L: LayoutStorage + ?Sized> YifanHuRepulsion<L> for NonCloneStateBackend {
    type State = NonCloneState;

    fn validate(&self) -> CoreResult<()> {
        Ok(())
    }

    fn init_state(&self, _num_nodes: usize) -> CoreResult<Self::State> {
        Ok(NonCloneState { calls: 0 })
    }

    fn accumulate_level_repulsion(
        &self,
        state: &mut Self::State,
        _coords: &YifanHuLevelCoords<'_>,
        _eps_sq: f64,
        _rep_scale: f64,
        _forces: &mut [f64],
    ) -> CoreResult<()> {
        state.calls += 1;
        Ok(())
    }
}

#[cfg(feature = "baselines")]
impl<L: LayoutStorage + ?Sized> LinLogRepulsionBackend<L> for NonCloneStateBackend {
    type State = NonCloneState;

    fn validate(&self) -> CoreResult<()> {
        Ok(())
    }

    fn init_state(&self, _num_nodes: usize) -> CoreResult<Self::State> {
        Ok(NonCloneState { calls: 0 })
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
        state.calls += 1;
        Ok(())
    }
}

#[cfg(feature = "baselines")]
impl<L: RiemannianLayout + ?Sized> FrRepulsion<L> for NonCloneStateBackend {
    type State = NonCloneState;

    fn validate(&self) -> CoreResult<()> {
        Ok(())
    }

    fn init_state(&self, _num_nodes: usize) -> CoreResult<Self::State> {
        Ok(NonCloneState { calls: 0 })
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
        state.calls += 1;
        Ok(())
    }
}

#[cfg(feature = "baselines")]
#[derive(Debug)]
struct NonCloneRiemannian2d(Layout2d<f64>);

#[cfg(feature = "baselines")]
impl LayoutStorage for NonCloneRiemannian2d {
    type Scalar = f64;

    fn len(&self) -> usize {
        self.0.len()
    }

    fn validate_finite(&self) -> CoreResult<()> {
        self.0.validate_finite()
    }
}

#[cfg(feature = "baselines")]
impl MetricSpace for NonCloneRiemannian2d {
    fn distance(&self, i: usize, j: usize) -> f64 {
        self.0.distance(i, j)
    }
}

#[cfg(feature = "baselines")]
impl RiemannianLayout for NonCloneRiemannian2d {
    fn tangent_dim(&self) -> usize {
        self.0.tangent_dim()
    }

    fn log_map_into(&self, i: usize, j: usize, out: &mut [f64]) -> CoreResult<()> {
        self.0.log_map_into(i, j, out)
    }

    fn exp_map(&mut self, i: usize, delta: &[f64]) -> CoreResult<()> {
        self.0.exp_map(i, delta)
    }

    fn parallel_transport_into(&self, i: usize, delta: &[f64], vec: &mut [f64]) -> CoreResult<()> {
        self.0.parallel_transport_into(i, delta, vec)
    }
}

#[derive(Debug, Clone)]
struct NonCloneLayoutMarker {
    _phantom: PhantomData<NonCloneState>,
}

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

#[test]
fn non_clone_repulsion_state_works_across_force_solvers() {
    let g = make_grid(3, 3);
    let _marker = NonCloneLayoutMarker {
        _phantom: PhantomData,
    };

    let mut fa2 = ForceAtlas2Solver::<Layout2d<f64>, _>::with_backend(
        &g,
        ForceAtlas2Config::new().with_iterations(4),
        NonCloneStateBackend,
    )
    .unwrap();
    let mut l_fa2 = Layout2d::<f64>::init(9, 1, 1.0).unwrap();
    assert_eq!(fa2.run(&mut l_fa2).unwrap(), SolverStatus::Completed);
    assert!(l_fa2.is_all_finite());

    let mut yh = YifanHuSolver::<Layout2d<f64>, _>::with_backend(
        &g,
        YifanHuConfig::new().with_max_iterations_per_level(3),
        NonCloneStateBackend,
    )
    .unwrap();
    let mut l_yh = Layout2d::<f64>::init(9, 2, 1.0).unwrap();
    assert_eq!(yh.run(&mut l_yh).unwrap(), SolverStatus::Completed);
    assert!(l_yh.is_all_finite());

    // Conditional Clone still works on standard Clone backends.
    let fa2_std = ForceAtlas2Solver::<Layout2d<f64>>::new(&g, ForceAtlas2Config::new()).unwrap();
    let fa2_cloned = fa2_std.clone();
    assert_eq!(fa2_cloned.num_edges(), fa2_std.num_edges());
}

#[cfg(feature = "baselines")]
#[test]
fn non_clone_state_and_non_clone_layout_work_for_fr_and_linlog() {
    let g = make_grid(3, 3);

    let mut ll = LinLogSolver::<Layout2d<f64>, _>::with_backend(
        &g,
        LinLogConfig::new().with_iterations(4),
        NonCloneStateBackend,
    )
    .unwrap();
    let mut l_ll = Layout2d::<f64>::init(9, 3, 1.0).unwrap();
    assert_eq!(ll.run(&mut l_ll).unwrap(), SolverStatus::Completed);
    assert!(l_ll.is_all_finite());

    // FR supports both non-Clone backend State AND non-Clone custom RiemannianLayout.
    let mut fr = FruchtermanReingoldSolver::<NonCloneRiemannian2d, _>::with_backend(
        &g,
        FruchtermanReingoldConfig::new().with_iterations(4),
        NonCloneStateBackend,
    )
    .unwrap();
    let mut l_fr = NonCloneRiemannian2d(Layout2d::<f64>::init(9, 4, 1.0).unwrap());
    assert_eq!(fr.run(&mut l_fr).unwrap(), SolverStatus::Completed);
    assert!(l_fr.0.is_all_finite());

    // Even with a Clone backend (Exact), FR solver with non-Clone layout L can be cloned!
    let fr_exact = FruchtermanReingoldSolver::<NonCloneRiemannian2d>::new(
        &g,
        FruchtermanReingoldConfig::new().with_iterations(3),
    )
    .unwrap();
    let mut fr_cloned = fr_exact.clone();
    let mut l_fr2 = NonCloneRiemannian2d(Layout2d::<f64>::init(9, 5, 1.0).unwrap());
    assert_eq!(fr_cloned.run(&mut l_fr2).unwrap(), SolverStatus::Completed);
}

#[test]
fn fa2_and_yifan_hu_ir_getters_expose_full_state() {
    let g = make_grid(5, 5);
    let mut fa2 =
        ForceAtlas2Solver::<Layout2d<f64>>::new(&g, ForceAtlas2Config::new().with_iterations(3))
            .unwrap();
    assert_eq!(fa2.num_edges(), g.num_edges());
    assert_eq!(fa2.num_edges(), fa2.edges().0.len());
    assert!((fa2.global_speed() - 1.0).abs() < 1e-12);
    let mut l_fa2 = Layout2d::<f64>::init(25, 1, 1.0).unwrap();
    fa2.step(&mut l_fa2).unwrap();
    assert!(fa2.global_speed().is_finite() && fa2.global_speed() > 0.0);

    let mut yh = YifanHuSolver::<Layout2d<f64>>::new(
        &g,
        YifanHuConfig::new()
            .with_min_coarsen_size(3)
            .with_max_iterations_per_level(3),
    )
    .unwrap();
    let num_levels = yh.num_levels();
    assert!(num_levels >= 2);
    assert_eq!(yh.level_graph(0).unwrap().num_nodes(), 25);
    assert!((yh.level_optimal_distance(0).unwrap() - yh.optimal_distance()).abs() < 1e-12);
    assert_eq!(yh.fine_to_level(0), Some(&[][..]));
    assert!(yh.level_graph(num_levels).is_none());
    assert!(yh.level_optimal_distance(num_levels).is_none());
    assert!(yh.level_coarse_of_finer(num_levels).is_none());
    assert!(yh.level_cluster_counts_of_finer(num_levels).is_none());
    assert!(yh.fine_to_level(num_levels).is_none());

    for lvl in 0..(num_levels - 1) {
        let coarse_map = yh.level_coarse_of_finer(lvl).unwrap();
        let counts = yh.level_cluster_counts_of_finer(lvl).unwrap();
        let fine_n = yh.level_graph(lvl).unwrap().num_nodes();
        let coarse_n = yh.level_graph(lvl + 1).unwrap().num_nodes();
        assert_eq!(coarse_map.len(), fine_n);
        assert_eq!(counts.len(), coarse_n);
        assert_eq!(counts.iter().map(|&c| c as usize).sum::<usize>(), fine_n);
        let f2l = yh.fine_to_level(lvl + 1).unwrap();
        assert_eq!(f2l.len(), 25);
        assert!(f2l.iter().all(|&c| (c as usize) < coarse_n));
        assert!(yh.level_optimal_distance(lvl + 1).unwrap() > 0.0);
    }
    assert!(yh.level_coarse_of_finer(num_levels - 1).is_none());
    assert!(yh.level_cluster_counts_of_finer(num_levels - 1).is_none());
    assert!(yh.current_level_step_length() > 0.0);

    let mut l_yh = Layout2d::<f64>::init(25, 2, 1.0).unwrap();
    yh.step(&mut l_yh).unwrap();
    assert!(yh.current_level_step_length().is_finite() && yh.current_level_step_length() > 0.0);
    // Ensure CoreError stays imported cleanly if needed
    let _ = std::mem::size_of::<CoreError>();
}

#[cfg(feature = "baselines")]
#[test]
fn fr_and_linlog_ir_getters_expose_edges_temperature_and_masses() {
    let g = CsrGraph::<f64>::from_edges(
        4,
        &[(0, 1), (1, 2), (2, 3)],
        Some(&[1.0, 2.0, 3.0]),
        false,
        false,
    )
    .unwrap();

    let fr = FruchtermanReingoldSolver::<Layout2d<f64>>::new(
        &g,
        FruchtermanReingoldConfig::new()
            .with_area(Some(16.0))
            .with_initial_temperature(Some(2.5)),
    )
    .unwrap();
    assert_eq!(fr.num_edges(), 3);
    let (fu, fv, fw) = fr.edges();
    assert_eq!(fu, &[0, 1, 2]);
    assert_eq!(fv, &[1, 2, 3]);
    assert_eq!(fw, &[1.0, 2.0, 3.0]);
    assert!((fr.initial_temperature() - 2.5).abs() < 1e-12);
    assert!((fr.current_temperature() - 2.5).abs() < 1e-12);

    let mut ll_edge = LinLogSolver::<Layout2d<f64>>::new(
        &g,
        LinLogConfig::new()
            .with_repulsion_model(LinLogRepulsion::EdgeRepulsion)
            .with_iterations(4),
    )
    .unwrap();
    assert_eq!(ll_edge.num_edges(), 3);
    let (lu, lv, lw) = ll_edge.edges();
    assert_eq!(lu, &[0, 1, 2]);
    assert_eq!(lv, &[1, 2, 3]);
    assert_eq!(lw, &[1.0, 2.0, 3.0]);
    // Weighted degrees: 1.0, 3.0, 5.0, 3.0 -> sum = 12.0.
    assert!((ll_edge.total_mass() - 12.0).abs() < 1e-12);
    assert!((ll_edge.global_speed() - 1.0).abs() < 1e-12);
    let mut l_ll = Layout2d::<f64>::init(4, 9, 1.0).unwrap();
    ll_edge.step(&mut l_ll).unwrap();
    assert!(ll_edge.global_speed().is_finite() && ll_edge.global_speed() > 0.0);

    let ll_node = LinLogSolver::<Layout2d<f64>>::new(
        &g,
        LinLogConfig::new().with_repulsion_model(LinLogRepulsion::NodeRepulsion),
    )
    .unwrap();
    assert!((ll_node.total_mass() - 4.0).abs() < 1e-12);
}
