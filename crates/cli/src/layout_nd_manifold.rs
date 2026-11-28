use egraph::core::{
    CoreError, CsrGraph, IterativeSolver, Layout2d, Layout3d, LayoutGenerator, LayoutNd,
    ManifoldKind, ManifoldLayout2d,
};
use egraph::layout::SgdScheduleConfig;
use egraph::linalg::{Hde, PivotMds, RdMds, normalize_euclidean_layout};

use crate::args::{InitMethod, LayoutMethod};
use crate::io::{CliError, CliResult};
use crate::layout_methods::{
    build_initial_layout, build_sgd_solver, is_sgd_solver_method, run_euclidean_method,
    run_riemannian_method,
};

/// Builds an initial 3D [`Layout3d<f64>`] using the requested [`InitMethod`].
pub fn build_initial_3d_layout(
    graph: &CsrGraph<f64>,
    init: InitMethod,
    seed: u64,
) -> CliResult<Layout3d<f64>> {
    let n = graph.num_nodes();
    match init {
        InitMethod::PivotMds => Ok(PivotMds::<Layout3d<f64>>::new(graph)
            .with_seed(seed)
            .generate()?),
        InitMethod::Koren => Ok(RdMds::<Layout3d<f64>>::new(graph)
            .with_eig_eps(1e-10)
            .with_tol(1e-6)
            .with_max_iters(500)
            .with_seed(seed)
            .with_target_std(Some(1.0))
            .generate()?),
        InitMethod::Random => {
            let mut layout = Layout3d::init(n, seed, 2.0)?;
            if n > 1 {
                normalize_euclidean_layout(&mut layout, 1.0, seed)?;
            }
            Ok(layout)
        }
    }
}

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
    match init {
        InitMethod::PivotMds => Ok(PivotMds::<LayoutNd<f64>>::new(graph)
            .with_dims(dims)
            .with_seed(seed)
            .generate()?),
        InitMethod::Koren => Ok(RdMds::<LayoutNd<f64>>::new(graph)
            .with_dims(dims)
            .with_eig_eps(1e-10)
            .with_tol(1e-6)
            .with_max_iters(500)
            .with_seed(seed)
            .with_target_std(Some(1.0))
            .generate()?),
        InitMethod::Random => {
            let mut layout = LayoutNd::init(n, dims, seed, 2.0)?;
            if n > 1 {
                normalize_euclidean_layout(&mut layout, 1.0, seed)?;
            }
            Ok(layout)
        }
    }
}

/// Runs a 3D-compatible layout solver in-place on [`Layout3d<f64>`].
pub fn run_layout_method_3d(
    graph: &CsrGraph<f64>,
    layout: &mut Layout3d<f64>,
    method: LayoutMethod,
    seed: u64,
) -> CliResult<()> {
    if run_euclidean_method(graph, layout, method, seed)? {
        return Ok(());
    }
    match method {
        LayoutMethod::Hde => {
            *layout = Hde::<Layout3d<f64>>::new(graph)
                .with_seed(seed)
                .generate()?;
            Ok(())
        }
        other => Err(CliError::Core(CoreError::unsupported(format!(
            "layout method {:?} does not support 3D Layout3d",
            other
        )))),
    }
}

/// Runs an N-D-compatible layout solver in-place on [`LayoutNd<f64>`].
pub fn run_layout_method_nd(
    graph: &CsrGraph<f64>,
    layout: &mut LayoutNd<f64>,
    method: LayoutMethod,
    seed: u64,
) -> CliResult<()> {
    let dims = layout.dims();
    if run_euclidean_method(graph, layout, method, seed)? {
        return Ok(());
    }
    match method {
        LayoutMethod::Hde => {
            *layout = Hde::<LayoutNd<f64>>::new(graph)
                .with_dims(dims)
                .with_seed(seed)
                .generate()?;
            Ok(())
        }
        other => Err(CliError::Core(CoreError::unsupported(format!(
            "layout method {:?} does not support N-D LayoutNd (dims={dims})",
            other
        )))),
    }
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
    let default_iters = solver.total_iterations().unwrap_or(1).max(1);
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
/// windows (`reset_schedule` + `run_schedule_window`) and final geodesic rescaling.
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
    if is_sgd_solver_method(method) {
        let mut solver = build_sgd_solver(graph, method, seed)?;
        if schedule_windows.is_none() && schedule_window_iters.is_none() {
            solver.run(layout)?;
        } else {
            let windows = schedule_windows.unwrap_or(1);
            let default_iters = solver.total_iterations().unwrap_or(1).max(1);
            let window_iters = schedule_window_iters.unwrap_or(default_iters);
            for w in 0..windows {
                if w > 0 || schedule_window_iters.is_some() {
                    let cfg = SgdScheduleConfig::new(window_iters);
                    solver.reset_schedule(cfg, seed.wrapping_add(w as u64))?;
                }
                solver.run_schedule_window(layout, 0, window_iters)?;
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

    if !run_riemannian_method(graph, layout, method, seed)? {
        return Err(CliError::Core(CoreError::unsupported(format!(
            "layout method {:?} does not support ManifoldLayout2d",
            method
        ))));
    }

    if let Some(final_scale) = rescale_final {
        layout.rescale_geodesic(final_scale)?;
    }
    Ok(())
}
