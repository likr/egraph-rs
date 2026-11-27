use std::time::Instant;

use egraph::prelude::{
    ClusterAreaConfig, ClusterBaselineConfig, CoreError, MetricConfig, MetricInputs, MetricTarget,
    cluster_area_faithfulness, cluster_area_faithfulness_detail, evaluate_metrics_nd_with_inputs,
    evaluate_metrics_with_inputs, layout_degeneracy, layout_degeneracy_manifold,
    layout_degeneracy_nd,
};

use crate::args::MetricsArgs;
use crate::io::{
    CliError, CliResult, LoadedLayout, load_any_layout_from_path, load_layout_from_path,
    resolve_graph,
};
use crate::metrics_targets::{
    load_labels_from_path, load_positive_f64_series_from_path, parse_metric_targets,
    push_unique_target,
};
use crate::output::MetricsOutput;

/// Executes the `metrics` subcommand and returns the structured JSON output.
pub fn execute_metrics(args: &MetricsArgs) -> CliResult<MetricsOutput> {
    if !args.overlap_radius.is_finite() || args.overlap_radius <= 0.0 {
        return Err(CliError::Core(CoreError::invalid_input(
            "overlap-radius must be finite and > 0.0",
        )));
    }
    if args.sprawlter_half_widths.is_some() != args.sprawlter_half_heights.is_some() {
        return Err(CliError::Core(CoreError::invalid_input(
            "sprawlter-half-widths and sprawlter-half-heights must be provided together",
        )));
    }

    let (graph, _, source) = resolve_graph(args.input.as_deref(), args.dataset.as_deref(), None)?;
    let n = graph.num_nodes();
    let loaded_layout = load_any_layout_from_path(&args.layout, n)?;
    let original_layout = args
        .original_layout
        .as_deref()
        .map(|path| load_layout_from_path(path, n))
        .transpose()?;
    let mut targets = parse_metric_targets(&args.targets)?;
    let labels = args
        .labels
        .as_deref()
        .map(|path| load_labels_from_path(path, n))
        .transpose()?;
    let half_widths = args
        .sprawlter_half_widths
        .as_deref()
        .map(|path| load_positive_f64_series_from_path(path, n, "sprawlter_half_widths"))
        .transpose()?;
    let half_heights = args
        .sprawlter_half_heights
        .as_deref()
        .map(|path| load_positive_f64_series_from_path(path, n, "sprawlter_half_heights"))
        .transpose()?;

    let has_all = args
        .targets
        .split(',')
        .map(|t| t.trim().to_ascii_lowercase().replace('_', "-"))
        .any(|t| t == "all");
    if has_all && labels.is_some() && matches!(loaded_layout, LoadedLayout::Euclidean2d(_)) {
        push_unique_target(&mut targets, MetricTarget::AdjustedPatchNig);
        push_unique_target(&mut targets, MetricTarget::ClusterAreaFaithfulness);
    }
    if args.gcaf_detail {
        push_unique_target(&mut targets, MetricTarget::ClusterAreaFaithfulness);
    }

    let cfg = MetricConfig {
        sns_num_pivots: args.num_pivots,
        sns_seed: args.seed,
        crossing_num_samples: args.num_samples,
        crossing_seed: args.seed,
        ideal_edge_length: args.ideal_edge_length,
        cluster_area: ClusterAreaConfig {
            g_max: args.gcaf_g_max,
            sigma_cells: args.gcaf_sigma_cells,
            kappa: args.gcaf_kappa,
            tau_frac: args.gcaf_tau_frac,
            boundary_r95: args.gcaf_boundary_r95,
            stevens_gamma: args.gcaf_stevens_gamma,
        },
        cluster_baselines: ClusterBaselineConfig {
            seed: args.seed,
            sprawlter_node_radius: args.overlap_radius,
            ..ClusterBaselineConfig::default()
        },
        ..MetricConfig::default()
    };
    cfg.validate()?;

    let inputs = MetricInputs {
        labels: labels.as_deref(),
        node_weights: None,
        cluster_weights: None,
        original_layout: original_layout.as_ref(),
        sprawlter_half_widths: half_widths.as_deref(),
        sprawlter_half_heights: half_heights.as_deref(),
    };

    let want_degeneracy = targets.contains(&MetricTarget::LayoutDegeneracy);
    let t0 = Instant::now();
    let (report, degeneracy, gcaf_report, gcaf_detail) = match &loaded_layout {
        LoadedLayout::Euclidean2d(layout) => {
            let want_gcaf = targets.contains(&MetricTarget::ClusterAreaFaithfulness);
            let non_gcaf_targets: Vec<MetricTarget> = targets
                .iter()
                .copied()
                .filter(|t| *t != MetricTarget::ClusterAreaFaithfulness)
                .collect();
            let report =
                evaluate_metrics_with_inputs(&graph, layout, &non_gcaf_targets, &cfg, &inputs)?;
            let deg = if want_degeneracy {
                Some(layout_degeneracy(&graph, layout)?)
            } else {
                None
            };
            let (gcaf_report, gcaf_detail) = if want_gcaf {
                let lbls = labels.as_deref().ok_or_else(|| {
                    CoreError::invalid_input("ClusterAreaFaithfulness target requires labels")
                })?;
                if args.gcaf_detail {
                    let det = cluster_area_faithfulness_detail(layout, lbls, &cfg.cluster_area)?;
                    (Some(det.report), Some(det))
                } else {
                    let rep = cluster_area_faithfulness(layout, lbls, &cfg.cluster_area)?;
                    (Some(rep), None)
                }
            } else {
                (None, None)
            };
            (report, deg, gcaf_report, gcaf_detail)
        }
        LoadedLayout::Nd(nd_layout) => {
            let report =
                evaluate_metrics_nd_with_inputs(&graph, nd_layout, &targets, &cfg, &inputs)?;
            let deg = if want_degeneracy {
                Some(layout_degeneracy_nd(&graph, nd_layout)?)
            } else {
                None
            };
            (report, deg, None, None)
        }
        LoadedLayout::Manifold(m_layout) => {
            let report = evaluate_metrics_with_inputs(&graph, m_layout, &targets, &cfg, &inputs)?;
            let deg = if want_degeneracy {
                Some(layout_degeneracy_manifold(&graph, m_layout)?)
            } else {
                None
            };
            (report, deg, None, None)
        }
    };
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    Ok(MetricsOutput::from_report(
        source,
        graph.num_nodes(),
        graph.num_edges(),
        elapsed_ms,
        report,
        degeneracy,
        gcaf_report,
        gcaf_detail,
    ))
}
