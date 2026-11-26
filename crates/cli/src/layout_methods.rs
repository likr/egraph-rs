use egraph::prelude::{
    BackboneConfig, BackboneSolver, CsrGraph, DeterministicRng, DogVaradhanConfig, DrGraphConfig,
    FieldLayoutSolver, ForceAtlas2Config, ForceAtlas2Solver, FruchtermanReingoldConfig,
    FruchtermanReingoldSolver, GemConfig, GemSolver, GraphCneConfig, GripConfig, GripSolver,
    GroupInABoxConfig, GroupInABoxSolver, GuMapConfig, GuMapSolver, HdeConfig, IbUmapConfig,
    IterativeSolver, KamadaKawaiConfig, KamadaKawaiSolver, LargeVisConfig, Layout2d, LeidenConfig,
    LinLogConfig, LinLogSolver, MaxentStressConfig, MaxentStressSolver, OrtmannSparseStressConfig,
    PacmapGraphConfig, PivotMdsConfig, SelConfig, SelLayoutSolver, SfcConfig, SfcLayoutSolver,
    SgdScheduleConfig, SgdSolver, SnapTFdpConfig, SparseBarrierConfig, StressMajorizationConfig,
    StressMajorizationSolver, TFdpConfig, TsNetConfig, TsNetSolver, YifanHuConfig, YifanHuSolver,
    hde, koren_spectral_init, leiden, normalize_layout_2d, pivot_mds,
};

use crate::args::{InitMethod, LayoutMethod};
use crate::io::CliResult;

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

/// Runs the selected layout solver in-place on `layout`.
pub fn run_layout_method(
    graph: &CsrGraph<f64>,
    layout: &mut Layout2d<f64>,
    method: LayoutMethod,
    seed: u64,
) -> CliResult<()> {
    run_layout_method_with_communities(graph, layout, method, None, seed)
}

/// Runs the selected layout solver in-place on `layout`, forwarding optional `community_labels`
/// (used by [`LayoutMethod::Gib`]; auto-clustered via Leiden if `None`).
pub fn run_layout_method_with_communities(
    graph: &CsrGraph<f64>,
    layout: &mut Layout2d<f64>,
    method: LayoutMethod,
    community_labels: Option<&[u32]>,
    seed: u64,
) -> CliResult<()> {
    match method {
        LayoutMethod::TsnetPure2hop => {
            let cfg = TsNetConfig {
                seed,
                ..TsNetConfig::pure_2hop()
            };
            let mut solver = TsNetSolver::from_graph_pure_2hop(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::SgdBarrierDog => {
            let dog_cfg = DogVaradhanConfig {
                seed,
                ..DogVaradhanConfig::default()
            };
            let barrier_cfg = SparseBarrierConfig {
                seed,
                ..SparseBarrierConfig::default()
            };
            let (mut solver, _barrier_stats, _dog_stats) =
                SgdSolver::sparse_barrier_dog(graph, &dog_cfg, &barrier_cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::SgdFull => {
            let mut solver = SgdSolver::full(graph, SgdScheduleConfig::default(), seed)?;
            solver.run(layout)?;
        }
        LayoutMethod::SgdSnap => {
            let cfg = SnapTFdpConfig {
                seed,
                ..SnapTFdpConfig::default()
            };
            let mut solver = SgdSolver::snap_t_fdp(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::SgdPacmap => {
            let cfg = PacmapGraphConfig {
                seed,
                ..PacmapGraphConfig::default()
            };
            let mut solver = SgdSolver::pacmap_graph(graph, cfg)?;
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
            let labels = match community_labels {
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
            let mut solver = GroupInABoxSolver::new(graph, labels, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::Hde => {
            let cfg = HdeConfig {
                seed,
                ..HdeConfig::default()
            };
            *layout = hde(graph, &cfg)?;
        }
        LayoutMethod::SgdOrtmann => {
            let cfg = OrtmannSparseStressConfig {
                seed,
                ..OrtmannSparseStressConfig::default()
            };
            let mut solver = SgdSolver::ortmann_sparse_stress(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::Maxent => {
            let cfg = MaxentStressConfig {
                seed,
                ..MaxentStressConfig::default()
            };
            let mut solver = MaxentStressSolver::new(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::Largevis => {
            let cfg = LargeVisConfig {
                seed,
                ..LargeVisConfig::default()
            };
            let mut solver = SgdSolver::largevis(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::Drgraph => {
            let cfg = DrGraphConfig {
                seed,
                ..DrGraphConfig::default()
            };
            let mut solver = SgdSolver::drgraph(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::GraphCne => {
            let cfg = GraphCneConfig {
                seed,
                ..GraphCneConfig::default()
            };
            let mut solver = SgdSolver::graph_cne(graph, cfg)?;
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
    }
    Ok(())
}
