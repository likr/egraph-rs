use egraph::clustering::{LeidenConfig, leiden};
use egraph::core::{
    CsrGraph, DeterministicRng, EuclideanLayout, IterativeSolver, Layout2d, LayoutGenerator,
    RiemannianLayout,
};
use egraph::layout::{
    BackboneConfig, BackboneSolver, DEFAULT_MAX_2HOP_PER_NODE, DefaultForceBackend,
    DefaultFrBackend, DefaultMaxentBackend, DefaultStressStrategy, DefaultTFdpBackend,
    DefaultTsNetBackend, DrGraphConfig, DrGraphSolver, ForceAtlas2Config, ForceAtlas2Solver,
    ForceRepulsion, FrRepulsion, FruchtermanReingoldConfig, FruchtermanReingoldSolver, GemConfig,
    GemSolver, GraphCneConfig, GraphCneSolver, GripConfig, GripSolver, GroupInABoxConfig,
    GroupInABoxSolver, GuMapConfig, GuMapSolver, GuMapTarget, IbUmapConfig, IbUmapSolver,
    KamadaKawaiConfig, KamadaKawaiSolver, LargeVisConfig, LargeVisSolver, LinLogConfig,
    LinLogRepulsionBackend, LinLogSolver, MaxentRepulsion, MaxentStressConfig, MaxentStressSolver,
    OrtmannSparseStressConfig, PacmapGraphConfig, PacmapGraphSolver, SelConfig, SelLayoutSolver,
    SfcConfig, SfcLayout, SgdScheduleConfig, SgdSolver, SnapTFdpConfig, SnapTFdpSolver,
    SparseBarrierConfig, StressMajorizationConfig, StressMajorizationSolver,
    StressMajorizationStrategy, TFdpConfig, TFdpRepulsion, TFdpSolver, TsNetConfig, TsNetRepulsion,
    TsNetSolver, YifanHuConfig, YifanHuRepulsion, YifanHuSolver,
};
use egraph::linalg::{Hde, KorenSpectral, PivotMds, normalize_layout_2d};
use egraph::neighborhood::{DogVaradhanConfig, TwoHopSamplingMode};

use crate::args::{InitMethod, LayoutMethod};
use crate::io::{CliError, CliResult};

/// Builds an initial 2D layout using the requested [`InitMethod`].
pub fn build_initial_layout(
    graph: &CsrGraph<f64>,
    init: InitMethod,
    seed: u64,
) -> CliResult<Layout2d<f64>> {
    let n = graph.num_nodes();
    match init {
        InitMethod::PivotMds => Ok(PivotMds::<Layout2d<f64>>::new(graph)
            .with_seed(seed)
            .generate()?),
        InitMethod::Koren => Ok(KorenSpectral::<Layout2d<f64>>::new(graph)
            .with_target_std(Some(1.0))
            .with_tol(1e-6)
            .with_max_iters(500)
            .with_seed(seed)
            .generate()?),
        InitMethod::Random => {
            let mut rng = DeterministicRng::from_seed_u64(seed);
            let mut xs = Vec::with_capacity(n);
            let mut ys = Vec::with_capacity(n);
            for _ in 0..n {
                xs.push(rng.next_f64() * 2.0 - 1.0);
                ys.push(rng.next_f64() * 2.0 - 1.0);
            }
            let mut layout = Layout2d::from_xy(xs, ys)?;
            normalize_layout_2d(&mut layout, 1.0, seed)?;
            Ok(layout)
        }
    }
}

/// Returns `true` when `method` is backed by [`SgdSolver`].
pub fn is_sgd_solver_method(method: LayoutMethod) -> bool {
    matches!(
        method,
        LayoutMethod::SgdBarrierDog | LayoutMethod::SgdFull | LayoutMethod::SgdOrtmann
    )
}

/// Constructs an [`SgdSolver`] for any [`SgdSolver`]-backed [`LayoutMethod`].
pub fn build_sgd_solver<L: RiemannianLayout + 'static>(
    graph: &CsrGraph<f64>,
    method: LayoutMethod,
    seed: u64,
) -> CliResult<SgdSolver<L>> {
    match method {
        LayoutMethod::SgdBarrierDog => {
            let dog_cfg = DogVaradhanConfig::new().with_seed(seed);
            let barrier_cfg = SparseBarrierConfig::new().with_seed(seed);
            let (solver, _barrier_stats, _dog_stats) =
                SgdSolver::sparse_barrier_dog(graph, &dog_cfg, &barrier_cfg)?;
            Ok(solver)
        }
        LayoutMethod::SgdFull => Ok(SgdSolver::full(graph, SgdScheduleConfig::default(), seed)?),
        LayoutMethod::SgdOrtmann => {
            let cfg = OrtmannSparseStressConfig::new().with_seed(seed);
            Ok(SgdSolver::ortmann_sparse_stress(graph, cfg)?)
        }
        other => Err(CliError::InvalidInput(format!(
            "method {:?} is not an SgdSolver preset",
            other
        ))),
    }
}

/// Runs a Riemannian-compatible layout method on `layout`. Returns `Ok(false)` when `method`
/// requires a Euclidean or 2D-only solver.
pub fn run_riemannian_method<L>(
    graph: &CsrGraph<f64>,
    layout: &mut L,
    method: LayoutMethod,
    seed: u64,
) -> CliResult<bool>
where
    L: RiemannianLayout
        + GuMapTarget
        + DefaultTsNetBackend
        + DefaultStressStrategy
        + DefaultFrBackend
        + 'static,
    <L as DefaultTsNetBackend>::Backend: TsNetRepulsion<L> + Default,
    <L as DefaultStressStrategy>::Strategy: StressMajorizationStrategy<L> + Default,
    <L as DefaultFrBackend>::Backend: FrRepulsion<L> + Default,
{
    if is_sgd_solver_method(method) {
        let mut solver = build_sgd_solver(graph, method, seed)?;
        solver.run(layout)?;
        return Ok(true);
    }
    match method {
        LayoutMethod::SgdSnap => {
            let cfg = SnapTFdpConfig::new().with_seed(seed);
            let mut solver = SnapTFdpSolver::new(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::SgdPacmap => {
            let cfg = PacmapGraphConfig::new().with_seed(seed);
            let mut solver = PacmapGraphSolver::new(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::Largevis => {
            let cfg = LargeVisConfig::new().with_seed(seed);
            let mut solver = LargeVisSolver::new(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::Drgraph => {
            let cfg = DrGraphConfig::new().with_seed(seed);
            let mut solver = DrGraphSolver::new(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::GraphCne => {
            let cfg = GraphCneConfig::new().with_seed(seed);
            let mut solver = GraphCneSolver::new(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::TsnetPure2hop => {
            let cfg = TsNetConfig::pure_2hop().with_seed(seed);
            let mut solver = TsNetSolver::<L>::from_graph_pure_2hop(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::TsnetRn => {
            let cfg = TsNetConfig::new().with_seed(seed);
            let mut solver = TsNetSolver::<L>::from_graph_row_normalized_with_options(
                graph,
                0.5,
                Some(DEFAULT_MAX_2HOP_PER_NODE),
                0.0,
                TwoHopSamplingMode::TwoStepRandomWalk,
                None,
                cfg,
            )?;
            solver.run(layout)?;
        }
        LayoutMethod::Gumap => {
            let dog_cfg = DogVaradhanConfig::new().with_seed(seed);
            let g_cfg = GuMapConfig::new().with_seed(seed);
            let mut solver = GuMapSolver::from_graph_dog(graph, 32, &dog_cfg, g_cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::Stress => {
            let mut solver = StressMajorizationSolver::<L>::from_graph(
                graph,
                StressMajorizationConfig::default(),
            )?;
            solver.run(layout)?;
        }
        LayoutMethod::Kk => {
            let mut solver = KamadaKawaiSolver::from_graph(graph, KamadaKawaiConfig::default())?;
            solver.run(layout)?;
        }
        LayoutMethod::Fr => {
            let cfg = FruchtermanReingoldConfig::new().with_seed(seed);
            let mut solver = FruchtermanReingoldSolver::<L>::new(graph, cfg)?;
            solver.run(layout)?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}

/// Runs a Euclidean-compatible layout method on `layout` (excluding `Hde` and 2D-only presets).
/// Returns `Ok(false)` when `method` is `Hde` or a 2D-only solver.
pub fn run_euclidean_method<L>(
    graph: &CsrGraph<f64>,
    layout: &mut L,
    method: LayoutMethod,
    seed: u64,
) -> CliResult<bool>
where
    L: EuclideanLayout
        + GuMapTarget
        + DefaultTsNetBackend
        + DefaultTFdpBackend
        + DefaultForceBackend
        + DefaultFrBackend
        + DefaultMaxentBackend
        + DefaultStressStrategy
        + 'static,
    <L as DefaultTsNetBackend>::Backend: TsNetRepulsion<L> + Default,
    <L as DefaultTFdpBackend>::Backend: TFdpRepulsion<L> + Default,
    <L as DefaultForceBackend>::Backend:
        ForceRepulsion<L> + LinLogRepulsionBackend<L> + YifanHuRepulsion<L> + Default,
    <L as DefaultFrBackend>::Backend: FrRepulsion<L> + Default,
    <L as DefaultMaxentBackend>::Backend: MaxentRepulsion<L> + Default,
    <L as DefaultStressStrategy>::Strategy: StressMajorizationStrategy<L> + Default,
{
    if run_riemannian_method(graph, layout, method, seed)? {
        return Ok(true);
    }
    match method {
        LayoutMethod::Tfdp => {
            let cfg = TFdpConfig::new().with_seed(seed);
            let mut solver = TFdpSolver::<L>::new(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::Fa2 => {
            let cfg = ForceAtlas2Config::new().with_seed(seed);
            let mut solver = ForceAtlas2Solver::<L>::new(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::Fa2Overlap => {
            let cfg = ForceAtlas2Config::new()
                .with_prevent_overlap(true)
                .with_seed(seed);
            let mut solver = ForceAtlas2Solver::<L>::new(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::YifanHu => {
            let cfg = YifanHuConfig::new().with_seed(seed);
            let mut solver = YifanHuSolver::<L>::new(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::Linlog => {
            let cfg = LinLogConfig::new().with_seed(seed);
            let mut solver = LinLogSolver::<L>::new(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::Maxent => {
            let cfg = MaxentStressConfig::new().with_seed(seed);
            let mut solver = MaxentStressSolver::<L>::new(graph, cfg)?;
            solver.run(layout)?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}

/// Runs the selected layout solver in-place on `layout`.
pub fn run_layout_method(
    graph: &CsrGraph<f64>,
    layout: &mut Layout2d<f64>,
    method: LayoutMethod,
    seed: u64,
) -> CliResult<()> {
    run_layout_method_with_labels(graph, layout, method, None, seed)
}

/// Runs the selected layout solver in-place on `layout`, forwarding optional `labels`
/// (used by [`LayoutMethod::Gib`]; auto-clustered via Leiden if `None`).
pub fn run_layout_method_with_labels(
    graph: &CsrGraph<f64>,
    layout: &mut Layout2d<f64>,
    method: LayoutMethod,
    labels: Option<&[u32]>,
    seed: u64,
) -> CliResult<()> {
    if run_euclidean_method(graph, layout, method, seed)? {
        return Ok(());
    }
    match method {
        LayoutMethod::Ibumap => {
            let dog_cfg = DogVaradhanConfig::new().with_seed(seed);
            let cfg = IbUmapConfig::new().with_seed(seed);
            let mut solver = IbUmapSolver::from_graph_dog(graph, 32, &dog_cfg, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::Backbone => {
            let cfg = BackboneConfig::new().with_seed(seed);
            let mut solver = BackboneSolver::new(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::Gib => {
            let resolved_labels = match labels {
                Some(lbls) => lbls.to_vec(),
                None => {
                    let l_cfg = LeidenConfig::new().with_seed(seed);
                    leiden(graph, &l_cfg)?.labels
                }
            };
            let cfg = GroupInABoxConfig::new().with_seed(seed);
            let mut solver = GroupInABoxSolver::new(graph, &resolved_labels, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::Hde => {
            *layout = Hde::<Layout2d<f64>>::new(graph)
                .with_seed(seed)
                .generate()?;
        }
        LayoutMethod::Grip => {
            let cfg = GripConfig::new().with_seed(seed);
            let mut solver = GripSolver::new(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::Gem => {
            let cfg = GemConfig::new().with_seed(seed);
            let mut solver = GemSolver::new(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::Sel => {
            let cfg = SelConfig::new().with_seed(seed);
            let mut solver = SelLayoutSolver::new(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::Sfc => {
            let cfg = SfcConfig::new().with_seed(seed);
            let mut generator = SfcLayout::<f64, f64>::from_config(graph, cfg);
            *layout = generator.generate()?;
        }
        other => unreachable!("unhandled 2D layout method {:?}", other),
    }
    Ok(())
}
