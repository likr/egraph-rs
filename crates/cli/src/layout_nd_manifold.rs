use egraph::prelude::{
    BfsWorkspace, CoreError, CsrGraph, DEFAULT_MAX_2HOP_PER_NODE, DijkstraWorkspace,
    DogVaradhanConfig, ForceAtlas2Config, ForceAtlas2Solver, FrBackend, FruchtermanReingoldConfig,
    FruchtermanReingoldSolver, GuMapConfig, GuMapSolver, HdeConfig, IterativeSolver,
    KamadaKawaiConfig, KamadaKawaiSolver, Layout2d, LayoutNd, LinLogConfig, LinLogSolver,
    ManifoldKind, ManifoldLayout2d, MaxentStressConfig, MaxentStressSolver, SgdScheduleConfig,
    StressMajorizationConfig, StressMajorizationSolver, TsNetBackend, TsNetConfig, TsNetSolver,
    TwoHopSamplingMode, hde_nd, rd_mds, representative_mds, select_maxmin_sp_pivots,
};

use crate::args::{InitMethod, LayoutMethod};
use crate::io::{CliError, CliResult};
use crate::layout_methods::{build_initial_layout, build_sgd_solver, is_sgd_method};

/// Builds an initial `dims`-dimensional [`LayoutNd<f64>`] using the requested [`InitMethod`].
pub fn build_initial_nd_layout(
    graph: &CsrGraph<f64>,
    init: InitMethod,
    dims: usize,
    seed: u64,
) -> CliResult<LayoutNd<f64>> {
    if dims == 0 {
        return Err(CliError::Core(CoreError::invalid_input(
            "dims must be >= 1",
        )));
    }
    let n = graph.num_nodes();
    if n <= 1 {
        return Ok(LayoutNd::zeros(n, dims)?);
    }
    match init {
        InitMethod::PivotMds => {
            let pivots = select_maxmin_sp_pivots(graph, 50, seed, 2)?;
            let k = pivots.len();
            if k < 2 {
                let mut layout = LayoutNd::init(n, dims, seed, 2.0)?;
                layout.center_and_normalize_std()?;
                return Ok(layout);
            }
            let total = n
                .checked_mul(k)
                .ok_or_else(|| CoreError::invalid_input("pivot_mds distance matrix overflow"))?;
            let mut distances_nk = vec![f64::INFINITY; total];
            if graph.is_weighted() {
                let mut dist_col = vec![f64::INFINITY; n];
                let mut ws = DijkstraWorkspace::new(n);
                for (col, &p) in pivots.iter().enumerate() {
                    ws.dijkstra_from(graph, p as usize, &mut dist_col)?;
                    for i in 0..n {
                        distances_nk[i * k + col] = dist_col[i];
                    }
                }
            } else {
                let mut dist_col = vec![u32::MAX; n];
                let mut ws = BfsWorkspace::new(n);
                for (col, &p) in pivots.iter().enumerate() {
                    ws.bfs_from_u32(graph, p as usize, &mut dist_col)?;
                    for i in 0..n {
                        let d = dist_col[i];
                        distances_nk[i * k + col] = if d == u32::MAX {
                            f64::INFINITY
                        } else {
                            d as f64
                        };
                    }
                }
            }
            let mut layout = representative_mds(&distances_nk, n, k, dims)?;
            layout.center_and_normalize_std()?;
            if layout.coords().iter().all(|&v| v == 0.0) {
                layout = LayoutNd::init(n, dims, seed, 2.0)?;
                layout.center_and_normalize_std()?;
            }
            Ok(layout)
        }
        InitMethod::Koren => {
            let mut layout = rd_mds(graph, dims, 1e-10, 1e-6, 500, seed)?;
            layout.center_and_normalize_std()?;
            if layout.coords().iter().all(|&v| v == 0.0) {
                layout = LayoutNd::init(n, dims, seed, 2.0)?;
                layout.center_and_normalize_std()?;
            }
            Ok(layout)
        }
        InitMethod::Random => {
            let mut layout = LayoutNd::init(n, dims, seed, 2.0)?;
            layout.center_and_normalize_std()?;
            Ok(layout)
        }
    }
}

/// Runs an N-D-compatible layout solver in-place on `layout`.
pub fn run_layout_method_nd(
    graph: &CsrGraph<f64>,
    layout: &mut LayoutNd<f64>,
    method: LayoutMethod,
    seed: u64,
) -> CliResult<()> {
    let dims = layout.dims();
    if is_sgd_method(method) {
        let mut solver = build_sgd_solver(graph, method, seed)?;
        solver.run(layout)?;
        return Ok(());
    }
    match method {
        LayoutMethod::TsnetPure2hop => {
            let cfg = TsNetConfig {
                backend: TsNetBackend::Auto,
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
        LayoutMethod::Stress => {
            let mut solver =
                StressMajorizationSolver::from_graph(graph, StressMajorizationConfig::default())?;
            solver.run(layout)?;
        }
        LayoutMethod::Kk => {
            let mut solver = KamadaKawaiSolver::from_graph(graph, KamadaKawaiConfig::default())?;
            solver.run(layout)?;
        }
        LayoutMethod::Maxent => {
            let cfg = MaxentStressConfig {
                use_barnes_hut: dims == 2 || dims == 3,
                seed,
                ..MaxentStressConfig::default()
            };
            let mut solver = MaxentStressSolver::new(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::Fr => {
            let mut cfg = FruchtermanReingoldConfig {
                seed,
                ..FruchtermanReingoldConfig::default()
            };
            if dims != 2 && dims != 3 {
                cfg.backend = FrBackend::Exact;
            }
            let mut solver = FruchtermanReingoldSolver::new(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::Linlog => {
            let cfg = LinLogConfig {
                use_barnes_hut: dims == 2 || dims == 3,
                seed,
                ..LinLogConfig::default()
            };
            let mut solver = LinLogSolver::new(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::Fa2 | LayoutMethod::Fa2Overlap => {
            let cfg = ForceAtlas2Config {
                use_barnes_hut: dims == 2 || dims == 3,
                prevent_overlap: method == LayoutMethod::Fa2Overlap,
                seed,
                ..ForceAtlas2Config::default()
            };
            let mut solver = ForceAtlas2Solver::new(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::YifanHu => {
            let cfg = egraph::prelude::YifanHuConfig {
                seed,
                ..egraph::prelude::YifanHuConfig::default()
            };
            let mut solver = egraph::prelude::YifanHuSolver::new(graph, cfg)?;
            solver.run(layout)?;
        }
        LayoutMethod::Hde => {
            let cfg = HdeConfig {
                output_dims: dims,
                seed,
                ..HdeConfig::default()
            };
            *layout = hde_nd(graph, &cfg)?;
        }
        other => {
            return Err(CliError::Core(CoreError::unsupported(format!(
                "layout method {:?} does not support N-D LayoutNd (dims={dims})",
                other
            ))));
        }
    }
    Ok(())
}

/// Builds an initial [`ManifoldLayout2d<f64>`] with the requested manifold geometry and metric.
pub fn build_initial_manifold_layout(
    graph: &CsrGraph<f64>,
    kind: ManifoldKind,
    init: InitMethod,
    manifold_scale: f64,
    torus_aspect: f64,
    seed: u64,
) -> CliResult<ManifoldLayout2d<f64>> {
    let n = graph.num_nodes();
    let mut layout = match init {
        InitMethod::Random => ManifoldLayout2d::init(kind, n, seed, 0.25)?,
        InitMethod::PivotMds => {
            let base = build_initial_layout(graph, InitMethod::PivotMds, seed)?;
            if kind == ManifoldKind::HyperbolicPoincare {
                ManifoldLayout2d::from_layout_scaled(kind, &base, 0.85)?
            } else {
                ManifoldLayout2d::from_layout(kind, base)?
            }
        }
        InitMethod::Koren => {
            if kind == ManifoldKind::Torus2d {
                ManifoldLayout2d::torus_spectral_phase_init(graph, seed)?
            } else {
                let base = build_initial_layout(graph, InitMethod::Koren, seed)?;
                if kind == ManifoldKind::HyperbolicPoincare {
                    ManifoldLayout2d::from_layout_scaled(kind, &base, 0.85)?
                } else {
                    ManifoldLayout2d::from_layout(kind, base)?
                }
            }
        }
    };
    if kind == ManifoldKind::Torus2d {
        layout.set_torus_extents(manifold_scale * torus_aspect, manifold_scale)?;
    } else {
        layout.set_scale(manifold_scale)?;
    }
    Ok(layout)
}

/// Runs a 2D SGD solver across optional schedule windows (`reset_schedule` + `run_schedule_window`).
pub fn run_sgd_with_windows_2d(
    graph: &CsrGraph<f64>,
    layout: &mut Layout2d<f64>,
    method: LayoutMethod,
    seed: u64,
    schedule_windows: Option<usize>,
    schedule_window_iters: Option<usize>,
) -> CliResult<()> {
    let mut solver = build_sgd_solver(graph, method, seed)?;
    if schedule_windows.is_none() && schedule_window_iters.is_none() {
        solver.run(layout)?;
        return Ok(());
    }
    let windows = schedule_windows.unwrap_or(1);
    let default_iters = solver.total_iterations().max(1);
    let window_iters = schedule_window_iters.unwrap_or(default_iters);
    for w in 0..windows {
        if w > 0 || schedule_window_iters.is_some() {
            let cfg = SgdScheduleConfig::new(window_iters);
            solver.reset_schedule(cfg, seed.wrapping_add(w as u64))?;
        }
        solver.run_schedule_window(layout, 0, window_iters)?;
    }
    Ok(())
}

/// Runs a manifold-compatible layout solver in-place on `layout`, optionally applying schedule
/// windows (`reset_schedule` + `run_schedule_window_manifold`) and final geodesic rescaling.
#[allow(clippy::too_many_arguments)]
pub fn run_layout_method_manifold(
    graph: &CsrGraph<f64>,
    layout: &mut ManifoldLayout2d<f64>,
    method: LayoutMethod,
    seed: u64,
    schedule_windows: Option<usize>,
    schedule_window_iters: Option<usize>,
    manifold_scale: f64,
    rescale_final: Option<f64>,
) -> CliResult<()> {
    if is_sgd_method(method) {
        let mut solver = build_sgd_solver(graph, method, seed)?;
        if schedule_windows.is_none() && schedule_window_iters.is_none() {
            solver.run(layout)?;
        } else {
            let windows = schedule_windows.unwrap_or(1);
            let default_iters = solver.total_iterations().max(1);
            let window_iters = schedule_window_iters.unwrap_or(default_iters);
            for w in 0..windows {
                if w > 0 || schedule_window_iters.is_some() {
                    let cfg = SgdScheduleConfig::new(window_iters);
                    solver.reset_schedule(cfg, seed.wrapping_add(w as u64))?;
                }
                solver.run_schedule_window_manifold(layout, 0, window_iters)?;
                if let Some(final_scale) = rescale_final
                    && w + 1 < windows
                {
                    let frac = ((w + 1) as f64) / (windows as f64);
                    let next_scale = manifold_scale + (final_scale - manifold_scale) * frac;
                    layout.rescale_geodesic(next_scale)?;
                }
            }
        }
        if let Some(final_scale) = rescale_final {
            layout.rescale_geodesic(final_scale)?;
        }
        return Ok(());
    }

    if schedule_windows.is_some() || schedule_window_iters.is_some() {
        return Err(CliError::InvalidInput(format!(
            "--schedule-windows / --schedule-window-iters are only supported for SGD methods, got {:?}",
            method
        )));
    }

    match method {
        LayoutMethod::TsnetPure2hop => {
            let cfg = TsNetConfig {
                backend: TsNetBackend::Auto,
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
        LayoutMethod::Stress => {
            let mut solver =
                StressMajorizationSolver::from_graph(graph, StressMajorizationConfig::default())?;
            solver.run(layout)?;
        }
        LayoutMethod::Kk => {
            let mut solver = KamadaKawaiSolver::from_graph(graph, KamadaKawaiConfig::default())?;
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
        other => {
            return Err(CliError::Core(CoreError::unsupported(format!(
                "layout method {:?} does not support ManifoldLayout2d",
                other
            ))));
        }
    }

    if let Some(final_scale) = rescale_final {
        layout.rescale_geodesic(final_scale)?;
    }
    Ok(())
}
