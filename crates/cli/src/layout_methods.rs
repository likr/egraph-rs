use egraph::prelude::{
    BackboneConfig, BackboneSolver, CsrGraph, DEFAULT_MAX_2HOP_PER_NODE, DeterministicRng,
    DogVaradhanConfig, DrGraphConfig, FieldLayoutSolver, ForceAtlas2Config, ForceAtlas2Solver,
    FruchtermanReingoldConfig, FruchtermanReingoldSolver, GemConfig, GemSolver, GraphCneConfig,
    GripConfig, GripSolver, GroupInABoxConfig, GroupInABoxSolver, GuMapConfig, GuMapSolver,
    HdeConfig, IbUmapConfig, IterativeSolver, KamadaKawaiConfig, KamadaKawaiSolver, LargeVisConfig,
    Layout2d, LeidenConfig, LinLogConfig, LinLogSolver, MaxentStressConfig, MaxentStressSolver,
    OrtmannSparseStressConfig, PacmapGraphConfig, PivotMdsConfig, SelConfig, SelLayoutSolver,
    SfcConfig, SfcLayoutSolver, SgdScheduleConfig, SgdSolver, SnapTFdpConfig, SparseBarrierConfig,
    StressMajorizationConfig, StressMajorizationSolver, TFdpConfig, TsNetBackend, TsNetConfig,
    TsNetSolver, TwoHopSamplingMode, YifanHuConfig, YifanHuSolver, hde, koren_spectral_init,
    leiden, normalize_layout_2d, pivot_mds,
};

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
        InitMethod::PivotMds => {
            let cfg = PivotMdsConfig {
                seed,
                ..PivotMdsConfig::default()
            };
            Ok(pivot_mds(graph, &cfg)?)
        }
        InitMethod::Koren => Ok(koren_spectral_init(graph, Some(1.0), 1e-6, 500, seed)?),
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
pub fn is_sgd_method(method: LayoutMethod) -> bool {
    matches!(
        method,
        LayoutMethod::SgdBarrierDog
            | LayoutMethod::SgdFull
            | LayoutMethod::SgdSnap
            | LayoutMethod::SgdPacmap
            | LayoutMethod::SgdOrtmann
            | LayoutMethod::Largevis
            | LayoutMethod::Drgraph
            | LayoutMethod::GraphCne
    )
}

/// Constructs an [`SgdSolver`] for any SGD-backed [`LayoutMethod`].
pub fn build_sgd_solver(
    graph: &CsrGraph<f64>,
    method: LayoutMethod,
    seed: u64,
) -> CliResult<SgdSolver<f64>> {
    match method {
        LayoutMethod::SgdBarrierDog => {
            let dog_cfg = DogVaradhanConfig {
                seed,
                ..DogVaradhanConfig::default()
            };
            let barrier_cfg = SparseBarrierConfig {
                seed,
                ..SparseBarrierConfig::default()
            };
            let (solver, _barrier_stats, _dog_stats) =
                SgdSolver::sparse_barrier_dog(graph, &dog_cfg, &barrier_cfg)?;
            Ok(solver)
        }
        LayoutMethod::SgdFull => Ok(SgdSolver::full(graph, SgdScheduleConfig::default(), seed)?),
        LayoutMethod::SgdSnap => {
            let cfg = SnapTFdpConfig {
                seed,
                ..SnapTFdpConfig::default()
            };
            Ok(SgdSolver::snap_t_fdp(graph, cfg)?)
        }
        LayoutMethod::SgdPacmap => {
            let cfg = PacmapGraphConfig {
                seed,
                ..PacmapGraphConfig::default()
            };
            Ok(SgdSolver::pacmap_graph(graph, cfg)?)
        }
        LayoutMethod::SgdOrtmann => {
            let cfg = OrtmannSparseStressConfig {
                seed,
                ..OrtmannSparseStressConfig::default()
            };
            Ok(SgdSolver::ortmann_sparse_stress(graph, cfg)?)
        }
        LayoutMethod::Largevis => {
            let cfg = LargeVisConfig {
                seed,
                ..LargeVisConfig::default()
            };
            Ok(SgdSolver::largevis(graph, cfg)?)
        }
        LayoutMethod::Drgraph => {
            let cfg = DrGraphConfig {
                seed,
                ..DrGraphConfig::default()
            };
            Ok(SgdSolver::drgraph(graph, cfg)?)
        }
        LayoutMethod::GraphCne => {
            let cfg = GraphCneConfig {
                seed,
                ..GraphCneConfig::default()
            };
            Ok(SgdSolver::graph_cne(graph, cfg)?)
        }
        other => Err(CliError::InvalidInput(format!(
            "method {:?} is not an SgdSolver preset",
            other
        ))),
    }
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
    if is_sgd_method(method) {
        let mut solver = build_sgd_solver(graph, method, seed)?;
        solver.run(layout)?;
        return Ok(());
    }
    match method {
        LayoutMethod::TsnetPure2hop => {
            let cfg = TsNetConfig {
                seed,
                ..TsNetConfig::pure_2hop()
            };
            let mut solver = TsNetSolver::from_graph_pure_2hop(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::TsnetRn => {
            let cfg = TsNetConfig {
                backend: TsNetBackend::Auto,
                seed,
                ..TsNetConfig::default()
            };
            let mut solver = TsNetSolver::from_graph_row_normalized_with_options(
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
            let dog_cfg = DogVaradhanConfig {
                seed,
                ..DogVaradhanConfig::default()
            };
            let g_cfg = GuMapConfig {
                seed,
                ..GuMapConfig::default()
            };
            let mut solver = GuMapSolver::from_graph_dog(graph, 32, &dog_cfg, g_cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::Tfdp => {
            let cfg = TFdpConfig {
                seed,
                ..TFdpConfig::default()
            };
            let mut solver = FieldLayoutSolver::t_fdp(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::Ibumap => {
            let dog_cfg = DogVaradhanConfig {
                seed,
                ..DogVaradhanConfig::default()
            };
            let cfg = IbUmapConfig {
                seed,
                ..IbUmapConfig::default()
            };
            let mut solver = FieldLayoutSolver::ib_umap(graph, 32, &dog_cfg, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::Fa2 => {
            let cfg = ForceAtlas2Config {
                seed,
                ..ForceAtlas2Config::default()
            };
            let mut solver = ForceAtlas2Solver::new(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::YifanHu => {
            let cfg = YifanHuConfig {
                seed,
                ..YifanHuConfig::default()
            };
            let mut solver = YifanHuSolver::new(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::Stress => {
            let mut solver =
                StressMajorizationSolver::from_graph(graph, StressMajorizationConfig::default())?;
            solver.run(layout)?;
        }
        LayoutMethod::Kk => {
            let mut solver = KamadaKawaiSolver::from_graph(graph, KamadaKawaiConfig::default())?;
            solver.run(layout)?;
        }
        LayoutMethod::Linlog => {
            let cfg = LinLogConfig {
                seed,
                ..LinLogConfig::default()
            };
            let mut solver = LinLogSolver::new(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::Fr => {
            let cfg = FruchtermanReingoldConfig {
                seed,
                ..FruchtermanReingoldConfig::default()
            };
            let mut solver = FruchtermanReingoldSolver::new(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::Fa2Overlap => {
            let cfg = ForceAtlas2Config {
                prevent_overlap: true,
                seed,
                ..ForceAtlas2Config::default()
            };
            let mut solver = ForceAtlas2Solver::new(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::Backbone => {
            let cfg = BackboneConfig {
                seed,
                ..BackboneConfig::default()
            };
            let mut solver = BackboneSolver::new(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::Gib => {
            let resolved_labels = match labels {
                Some(lbls) => lbls.to_vec(),
                None => {
                    let l_cfg = LeidenConfig {
                        seed,
                        ..LeidenConfig::default()
                    };
                    leiden(graph, &l_cfg)?.labels
                }
            };
            let cfg = GroupInABoxConfig {
                seed,
                ..GroupInABoxConfig::default()
            };
            let mut solver = GroupInABoxSolver::new(graph, resolved_labels, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::Hde => {
            let cfg = HdeConfig {
                seed,
                ..HdeConfig::default()
            };
            *layout = hde(graph, &cfg)?;
        }
        LayoutMethod::Maxent => {
            let cfg = MaxentStressConfig {
                seed,
                ..MaxentStressConfig::default()
            };
            let mut solver = MaxentStressSolver::new(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::Grip => {
            let cfg = GripConfig {
                seed,
                ..GripConfig::default()
            };
            let mut solver = GripSolver::new(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::Gem => {
            let cfg = GemConfig {
                seed,
                ..GemConfig::default()
            };
            let mut solver = GemSolver::new(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::Sel => {
            let cfg = SelConfig {
                seed,
                ..SelConfig::default()
            };
            let mut solver = SelLayoutSolver::new(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::Sfc => {
            let cfg = SfcConfig {
                seed,
                ..SfcConfig::default()
            };
            let mut solver = SfcLayoutSolver::new(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::SgdBarrierDog
        | LayoutMethod::SgdFull
        | LayoutMethod::SgdSnap
        | LayoutMethod::SgdPacmap
        | LayoutMethod::SgdOrtmann
        | LayoutMethod::Largevis
        | LayoutMethod::Drgraph
        | LayoutMethod::GraphCne => unreachable!("handled by is_sgd_method branch"),
    }
    Ok(())
}
