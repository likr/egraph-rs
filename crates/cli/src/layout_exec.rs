use std::time::Instant;

use egraph::prelude::{
    ClusterRefineConfig, ClusterRefineMethod, ClusterRefineSolver, CoreError, ForbidConfig,
    ForbidOverlapRemoval, ForbidVariant, GTreeConfig, GTreeOverlapRemoval, IterativeSolver,
    LeidenConfig, PfsPrimeConfig, PfsPrimeOverlapRemoval, PrismConfig, PrismOverlapRemoval,
    SpatialIntruderRefiner, SpatialRefinerConfig, leiden,
};

use crate::args::{ClusterRefineArg, LayoutArgs, LayoutMethod, OverlapRemovalArg};
use crate::io::{CliError, CliResult, resolve_graph};
use crate::layout_methods::{build_initial_layout, run_layout_method_with_communities};
use crate::metrics_exec::load_community_labels_from_path;
use crate::output::LayoutOutput;

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
    if effective_overlap != OverlapRemovalArg::None
        && (!args.prism_radius.is_finite() || args.prism_radius <= 0.0)
    {
        return Err(CliError::Core(CoreError::invalid_input(
            "prism-radius must be finite and > 0.0",
        )));
    }
    let t0 = Instant::now();
    let (graph, embedded_layout, source) =
        resolve_graph(args.input.as_deref(), args.dataset.as_deref(), None)?;

    let mut community_labels = args
        .community_labels
        .as_deref()
        .map(|path| load_community_labels_from_path(path, graph.num_nodes()))
        .transpose()?;

    if community_labels.is_none()
        && (args.method == LayoutMethod::Gib || args.cluster_refine != ClusterRefineArg::None)
    {
        let l_cfg = LeidenConfig {
            seed: args.seed,
            ..LeidenConfig::default()
        };
        community_labels = Some(leiden(&graph, &l_cfg)?.labels);
    }

    let mut layout = match embedded_layout {
        Some(l) => l,
        None => build_initial_layout(&graph, args.init, args.seed)?,
    };

    run_layout_method_with_communities(
        &graph,
        &mut layout,
        args.method,
        community_labels.as_deref(),
        args.seed,
    )?;

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
        let labels = community_labels.clone().unwrap_or_default();
        let cr_cfg = ClusterRefineConfig {
            method,
            seed: args.seed,
            ..ClusterRefineConfig::default()
        };
        let mut refiner = ClusterRefineSolver::from_graph(&graph, labels, cr_cfg)?;
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
    };

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let x = layout.x().to_vec();
    let y = layout.y().to_vec();
    let coords = x.iter().zip(y.iter()).map(|(&xi, &yi)| [xi, yi]).collect();

    Ok(LayoutOutput {
        source,
        method: args.method,
        init: args.init,
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
    })
}
