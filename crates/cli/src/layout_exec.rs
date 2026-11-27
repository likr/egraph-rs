use std::time::Instant;

use egraph::prelude::{
    ClusterRefineConfig, ClusterRefineMethod, ClusterRefineSolver, CoreError, ForbidConfig,
    ForbidOverlapRemoval, ForbidVariant, GTreeConfig, GTreeOverlapRemoval, IterativeSolver,
    LeidenConfig, ManifoldKind, PfsPrimeConfig, PfsPrimeOverlapRemoval, PrismConfig,
    PrismOverlapRemoval, SpatialIntruderRefiner, SpatialRefinerConfig, TwinDeclutterConfig,
    TwinDeclutterer, leiden,
};

use crate::args::{ClusterRefineArg, LayoutArgs, LayoutMethod, ManifoldArg, OverlapRemovalArg};
use crate::io::{CliError, CliResult, resolve_graph};
use crate::layout_methods::{build_initial_layout, is_sgd_method, run_layout_method_with_labels};
use crate::layout_nd_manifold::{
    build_initial_manifold_layout, build_initial_nd_layout, run_layout_method_manifold,
    run_layout_method_nd, run_sgd_with_windows_2d,
};
use crate::metrics_targets::load_labels_from_path;
use crate::output::LayoutOutput;

fn validate_layout_args(args: &LayoutArgs, effective_overlap: OverlapRemovalArg) -> CliResult<()> {
    if args.dims == 0 {
        return Err(CliError::Core(CoreError::invalid_input(
            "dims must be >= 1",
        )));
    }
    if !args.manifold_scale.is_finite() || args.manifold_scale <= 0.0 {
        return Err(CliError::Core(CoreError::invalid_input(
            "manifold-scale must be finite and > 0.0",
        )));
    }
    if !args.torus_aspect.is_finite() || args.torus_aspect <= 0.0 {
        return Err(CliError::Core(CoreError::invalid_input(
            "torus-aspect must be finite and > 0.0",
        )));
    }
    if args.dims != 2 && args.manifold != ManifoldArg::None {
        return Err(CliError::InvalidInput(
            "--dims != 2 cannot be combined with --manifold".to_string(),
        ));
    }
    if args.torus_auto_pan && args.manifold != ManifoldArg::Torus {
        return Err(CliError::InvalidInput(
            "--torus-auto-pan requires --manifold torus".to_string(),
        ));
    }
    if let Some(final_scale) = args.manifold_rescale_final {
        if args.manifold == ManifoldArg::None {
            return Err(CliError::InvalidInput(
                "--manifold-rescale-final requires --manifold != none".to_string(),
            ));
        }
        if !final_scale.is_finite() || final_scale <= 0.0 {
            return Err(CliError::Core(CoreError::invalid_input(
                "manifold-rescale-final must be finite and > 0.0",
            )));
        }
    }
    if matches!(args.schedule_windows, Some(0)) {
        return Err(CliError::InvalidInput(
            "--schedule-windows must be >= 1".to_string(),
        ));
    }
    if matches!(args.schedule_window_iters, Some(0)) {
        return Err(CliError::InvalidInput(
            "--schedule-window-iters must be >= 1".to_string(),
        ));
    }
    if (args.schedule_windows.is_some() || args.schedule_window_iters.is_some())
        && (args.dims != 2 || !is_sgd_method(args.method))
    {
        return Err(CliError::InvalidInput(
            "--schedule-windows / --schedule-window-iters require --dims 2 and an SGD method"
                .to_string(),
        ));
    }
    if (args.dims != 2 || args.manifold != ManifoldArg::None)
        && (args.refine
            || args.cluster_refine != ClusterRefineArg::None
            || effective_overlap != OverlapRemovalArg::None)
    {
        return Err(CliError::InvalidInput(
            "2D refinement and overlap removal cannot be combined with --dims != 2 or --manifold"
                .to_string(),
        ));
    }
    if effective_overlap != OverlapRemovalArg::None
        && (!args.prism_radius.is_finite() || args.prism_radius <= 0.0)
    {
        return Err(CliError::Core(CoreError::invalid_input(
            "prism-radius must be finite and > 0.0",
        )));
    }
    Ok(())
}

/// Executes the `layout` subcommand and returns the structured JSON output.
pub fn execute_layout(args: &LayoutArgs) -> CliResult<LayoutOutput> {
    if args.prism
        && matches!(
            args.overlap_removal,
            Some(mode) if mode != OverlapRemovalArg::Prism
        )
    {
        return Err(CliError::InvalidInput(
            "--prism cannot be combined with a conflicting --overlap-removal mode".to_string(),
        ));
    }
    let effective_overlap = match (args.prism, args.overlap_removal) {
        (true, _) => OverlapRemovalArg::Prism,
        (false, Some(mode)) => mode,
        (false, None) => OverlapRemovalArg::None,
    };
    validate_layout_args(args, effective_overlap)?;

    let t0 = Instant::now();
    let (graph, embedded_layout, source) =
        resolve_graph(args.input.as_deref(), args.dataset.as_deref(), None)?;

    if args.dims != 2 {
        let mut nd_layout = build_initial_nd_layout(&graph, args.init, args.dims, args.seed)?;
        run_layout_method_nd(&graph, &mut nd_layout, args.method, args.seed)?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let n = graph.num_nodes();
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut coords = Vec::with_capacity(n);
        let mut coords_nd = Vec::with_capacity(n);
        for i in 0..n {
            let row = nd_layout.node(i)?;
            let xi = row[0];
            let yi = if args.dims >= 2 { row[1] } else { 0.0 };
            x.push(xi);
            y.push(yi);
            coords.push([xi, yi]);
            coords_nd.push(row.to_vec());
        }
        return Ok(LayoutOutput {
            source,
            method: args.method,
            init: args.init,
            dims: args.dims,
            manifold: None,
            manifold_scale: None,
            torus_aspect: None,
            seed: args.seed,
            num_nodes: n,
            num_edges: graph.num_edges(),
            refined: false,
            cluster_refine: ClusterRefineArg::None,
            prism_applied: false,
            prism_overlaps_remaining: None,
            overlap_removal: OverlapRemovalArg::None,
            overlaps_remaining: None,
            elapsed_ms,
            x,
            y,
            coords,
            coords_nd: Some(coords_nd),
        });
    }

    if let Some(kind) = args.manifold.to_manifold_kind() {
        let mut manifold_layout = build_initial_manifold_layout(
            &graph,
            kind,
            args.init,
            args.manifold_scale,
            args.torus_aspect,
            args.seed,
        )?;
        run_layout_method_manifold(
            &graph,
            &mut manifold_layout,
            args.method,
            args.seed,
            args.schedule_windows,
            args.schedule_window_iters,
            args.manifold_scale,
            args.manifold_rescale_final,
        )?;
        if args.torus_auto_pan && kind == ManifoldKind::Torus2d {
            manifold_layout = manifold_layout.torus_auto_pan(&graph)?;
        }
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let x = manifold_layout.x().to_vec();
        let y = manifold_layout.y().to_vec();
        let coords = x.iter().zip(y.iter()).map(|(&xi, &yi)| [xi, yi]).collect();
        return Ok(LayoutOutput {
            source,
            method: args.method,
            init: args.init,
            dims: 2,
            manifold: Some(args.manifold),
            manifold_scale: Some(manifold_layout.scale()),
            torus_aspect: if kind == ManifoldKind::Torus2d {
                Some(manifold_layout.aspect())
            } else {
                None
            },
            seed: args.seed,
            num_nodes: graph.num_nodes(),
            num_edges: graph.num_edges(),
            refined: false,
            cluster_refine: ClusterRefineArg::None,
            prism_applied: false,
            prism_overlaps_remaining: None,
            overlap_removal: OverlapRemovalArg::None,
            overlaps_remaining: None,
            elapsed_ms,
            x,
            y,
            coords,
            coords_nd: None,
        });
    }

    let mut labels = args
        .labels
        .as_deref()
        .map(|path| load_labels_from_path(path, graph.num_nodes()))
        .transpose()?;

    if labels.is_none()
        && (args.method == LayoutMethod::Gib || args.cluster_refine != ClusterRefineArg::None)
    {
        let l_cfg = LeidenConfig {
            seed: args.seed,
            ..LeidenConfig::default()
        };
        labels = Some(leiden(&graph, &l_cfg)?.labels);
    }

    let mut layout = match embedded_layout {
        Some(l) => l,
        None => build_initial_layout(&graph, args.init, args.seed)?,
    };

    if args.schedule_windows.is_some() || args.schedule_window_iters.is_some() {
        run_sgd_with_windows_2d(
            &graph,
            &mut layout,
            args.method,
            args.seed,
            args.schedule_windows,
            args.schedule_window_iters,
        )?;
    } else {
        run_layout_method_with_labels(
            &graph,
            &mut layout,
            args.method,
            labels.as_deref(),
            args.seed,
        )?;
    }

    if args.refine {
        let ref_cfg = SpatialRefinerConfig {
            seed: args.seed,
            ..SpatialRefinerConfig::default()
        };
        let mut refiner = SpatialIntruderRefiner::new(&graph, ref_cfg)?;
        refiner.run(&mut layout)?;
    }

    if args.cluster_refine != ClusterRefineArg::None {
        let method = match args.cluster_refine {
            ClusterRefineArg::None => unreachable!(),
            ClusterRefineArg::Kmeans => ClusterRefineMethod::Kmeans,
            ClusterRefineArg::Hac => ClusterRefineMethod::Hac,
        };
        let cr_labels = labels.clone().unwrap_or_default();
        let cr_cfg = ClusterRefineConfig {
            method,
            seed: args.seed,
            ..ClusterRefineConfig::default()
        };
        let mut refiner = ClusterRefineSolver::from_graph(&graph, cr_labels, cr_cfg)?;
        refiner.run(&mut layout)?;
    }

    let (prism_overlaps_remaining, overlaps_remaining) = match effective_overlap {
        OverlapRemovalArg::None => (None, None),
        OverlapRemovalArg::Prism => {
            let prism_cfg = PrismConfig {
                seed: args.seed,
                ..PrismConfig::default()
            };
            let radii = vec![args.prism_radius; graph.num_nodes()];
            let mut prism = PrismOverlapRemoval::from_radii(radii, prism_cfg)?;
            prism.run(&mut layout)?;
            let rem = Some(prism.last_overlap_count());
            (rem, rem)
        }
        OverlapRemovalArg::Gtree => {
            let gtree_cfg = GTreeConfig {
                seed: args.seed,
                ..GTreeConfig::default()
            };
            let radii = vec![args.prism_radius; graph.num_nodes()];
            let mut gtree = GTreeOverlapRemoval::from_radii(radii, gtree_cfg)?;
            gtree.run(&mut layout)?;
            (None, Some(gtree.last_overlap_count()))
        }
        OverlapRemovalArg::Forbid
        | OverlapRemovalArg::ForbidPrime
        | OverlapRemovalArg::SparseForbid => {
            let variant = match effective_overlap {
                OverlapRemovalArg::Forbid => ForbidVariant::Forbid,
                OverlapRemovalArg::ForbidPrime => ForbidVariant::ForbidPrime,
                OverlapRemovalArg::SparseForbid => ForbidVariant::SparseForbid,
                _ => unreachable!(),
            };
            let forbid_cfg = ForbidConfig {
                variant,
                seed: args.seed,
                ..ForbidConfig::default()
            };
            let radii = vec![args.prism_radius; graph.num_nodes()];
            let mut forbid = ForbidOverlapRemoval::from_radii(radii, forbid_cfg)?;
            forbid.run(&mut layout)?;
            (None, Some(forbid.last_overlap_count()))
        }
        OverlapRemovalArg::PfsPrime => {
            let pfs_cfg = PfsPrimeConfig {
                seed: args.seed,
                ..PfsPrimeConfig::default()
            };
            let radii = vec![args.prism_radius; graph.num_nodes()];
            let mut pfs = PfsPrimeOverlapRemoval::from_radii(radii, pfs_cfg)?;
            pfs.run(&mut layout)?;
            (None, Some(pfs.last_overlap_count()))
        }
        OverlapRemovalArg::TwinDeclutter => {
            let mut declutterer = TwinDeclutterer::new(TwinDeclutterConfig::default())?;
            declutterer.run(&mut layout)?;
            (None, None)
        }
    };

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let x = layout.x().to_vec();
    let y = layout.y().to_vec();
    let coords = x.iter().zip(y.iter()).map(|(&xi, &yi)| [xi, yi]).collect();

    Ok(LayoutOutput {
        source,
        method: args.method,
        init: args.init,
        dims: 2,
        manifold: None,
        manifold_scale: None,
        torus_aspect: None,
        seed: args.seed,
        num_nodes: graph.num_nodes(),
        num_edges: graph.num_edges(),
        refined: args.refine,
        cluster_refine: args.cluster_refine,
        prism_applied: effective_overlap == OverlapRemovalArg::Prism,
        prism_overlaps_remaining,
        overlap_removal: effective_overlap,
        overlaps_remaining,
        elapsed_ms,
        x,
        y,
        coords,
        coords_nd: None,
    })
}
