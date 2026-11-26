use std::time::Instant;

use egraph::prelude::{
    KMeansConfig, LabelPropagationConfig, LeidenConfig, LeidenObjective, LouvainConfig,
    MetricConfig, MetricTarget, SpectralClusteringConfig, evaluate_metrics, kmeans_plus_plus,
    label_propagation, leiden, louvain, spectral_clustering,
};

use crate::args::{BenchArgs, ClusterArgs, ClusterMethod, InitMethod};
use crate::io::{CliResult, resolve_graph};
use crate::layout_methods::{build_initial_layout, run_layout_method};
use crate::output::{BenchOutput, ClusterOutput};

/// Executes the `cluster` subcommand and returns the structured JSON output.
pub fn execute_cluster(args: &ClusterArgs) -> CliResult<ClusterOutput> {
    let (graph, _, source) = resolve_graph(args.input.as_deref(), args.dataset.as_deref(), None)?;

    let t0 = Instant::now();
    let (labels, num_clusters, modularity, quality, inertia) = match args.method {
        ClusterMethod::Louvain => {
            let cfg = LouvainConfig {
                seed: args.seed,
                ..LouvainConfig::default()
            };
            let res = louvain(&graph, &cfg)?;
            (
                res.labels,
                res.num_communities,
                Some(res.modularity),
                None,
                None,
            )
        }
        ClusterMethod::LeidenMod => {
            let cfg = LeidenConfig {
                objective: LeidenObjective::Modularity,
                resolution: args.resolution,
                randomness: args.randomness,
                seed: args.seed,
                ..LeidenConfig::default()
            };
            let res = leiden(&graph, &cfg)?;
            (
                res.labels,
                res.num_communities,
                Some(res.modularity),
                Some(res.quality),
                None,
            )
        }
        ClusterMethod::LeidenCpm => {
            let cfg = LeidenConfig {
                objective: LeidenObjective::Cpm,
                resolution: args.resolution,
                randomness: args.randomness,
                seed: args.seed,
                ..LeidenConfig::default()
            };
            let res = leiden(&graph, &cfg)?;
            (
                res.labels,
                res.num_communities,
                Some(res.modularity),
                Some(res.quality),
                None,
            )
        }
        ClusterMethod::Spectral => {
            let cfg = SpectralClusteringConfig {
                k: args.k,
                seed: args.seed,
                ..SpectralClusteringConfig::default()
            };
            let res = spectral_clustering(&graph, &cfg)?;
            let count = count_distinct_labels(&res.labels);
            (res.labels, count, None, None, None)
        }
        ClusterMethod::LabelProp => {
            let cfg = LabelPropagationConfig {
                seed: args.seed,
                ..LabelPropagationConfig::default()
            };
            let res = label_propagation(&graph, &cfg)?;
            let count = count_distinct_labels(&res.labels);
            (res.labels, count, None, None, None)
        }
        ClusterMethod::Kmeans => {
            let layout = build_initial_layout(&graph, InitMethod::PivotMds, args.seed)?;
            let points = layout.to_interleaved();
            let cfg = KMeansConfig {
                k: args.k,
                seed: args.seed,
                ..KMeansConfig::default()
            };
            let res = kmeans_plus_plus(&points, graph.num_nodes(), 2, &cfg)?;
            let count = count_distinct_labels(&res.labels);
            (res.labels, count, None, None, Some(res.inertia))
        }
    };
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    Ok(ClusterOutput {
        source,
        method: args.method,
        seed: args.seed,
        num_nodes: graph.num_nodes(),
        num_edges: graph.num_edges(),
        num_clusters,
        modularity,
        quality,
        inertia,
        elapsed_ms,
        labels,
    })
}

fn count_distinct_labels(labels: &[u32]) -> usize {
    let mut sorted = labels.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    sorted.len()
}

/// Executes the `bench` subcommand and returns the structured benchmark output.
pub fn execute_bench(args: &BenchArgs) -> CliResult<BenchOutput> {
    let total_t0 = Instant::now();
    let (graph, _, source) = resolve_graph(args.input.as_deref(), args.dataset.as_deref(), None)?;

    let init_t0 = Instant::now();
    let mut layout = build_initial_layout(&graph, args.init, args.seed)?;
    let init_ms = init_t0.elapsed().as_secs_f64() * 1000.0;

    let layout_t0 = Instant::now();
    run_layout_method(&graph, &mut layout, args.method, args.seed)?;
    let layout_ms = layout_t0.elapsed().as_secs_f64() * 1000.0;

    let metrics_t0 = Instant::now();
    let mut targets = vec![
        MetricTarget::NeighborhoodPreservation1Hop,
        MetricTarget::NeighborhoodPreservation2Hop,
    ];
    if graph.num_nodes() <= 2000 {
        targets.push(MetricTarget::ScaleNormalizedStress);
        targets.push(MetricTarget::CrossingNumber);
    } else {
        targets.push(MetricTarget::SampledScaleNormalizedStress);
    }
    let m_cfg = MetricConfig {
        sns_seed: args.seed,
        crossing_seed: args.seed,
        ..MetricConfig::default()
    };
    let report = evaluate_metrics(&graph, &layout, &targets, &m_cfg)?;
    let metrics_ms = metrics_t0.elapsed().as_secs_f64() * 1000.0;
    let total_ms = total_t0.elapsed().as_secs_f64() * 1000.0;

    Ok(BenchOutput {
        source,
        method: args.method,
        init: args.init,
        seed: args.seed,
        num_nodes: graph.num_nodes(),
        num_edges: graph.num_edges(),
        init_ms,
        layout_ms,
        metrics_ms,
        total_ms,
        scale_normalized_stress: report.scale_normalized_stress,
        sampled_scale_normalized_stress: report.sampled_scale_normalized_stress,
        crossing_number: report.crossing_number,
        neighborhood_preservation_1hop: report.neighborhood_preservation_1hop,
        neighborhood_preservation_2hop: report.neighborhood_preservation_2hop,
    })
}
