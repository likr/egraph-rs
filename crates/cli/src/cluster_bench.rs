use std::time::Instant;

use egraph::clustering::{
    KMeansConfig, LabelPropagationConfig, LeidenConfig, LeidenObjective, LouvainConfig,
    SpectralClusteringConfig, kmeans_plus_plus, label_propagation, leiden, louvain,
    spectral_clustering,
};
use egraph::core::{CoreError, ManifoldKind};
use egraph::metrics::{MetricConfig, MetricTarget, evaluate_metrics};

use crate::args::{BenchArgs, ClusterArgs, ClusterMethod, InitMethod, ManifoldArg};
use crate::io::{CliError, CliResult, resolve_graph};
use crate::layout_methods::{build_initial_layout, run_layout_method};
use crate::layout_nd_manifold::{
    build_initial_3d_layout, build_initial_manifold_layout, build_initial_nd_layout,
    run_layout_method_3d, run_layout_method_manifold, run_layout_method_nd,
};
use crate::output::{BenchOutput, ClusterOutput};

struct ClusterRunStats {
    labels: Vec<u32>,
    num_clusters: usize,
    modularity: Option<f64>,
    quality: Option<f64>,
    inertia: Option<f64>,
    passes: Option<usize>,
    iterations: Option<usize>,
    converged: Option<bool>,
    eigen_iterations: Option<usize>,
}

/// Executes the `cluster` subcommand and returns the structured JSON output.
pub fn execute_cluster(args: &ClusterArgs) -> CliResult<ClusterOutput> {
    let (graph, _, source) = resolve_graph(args.input.as_deref(), args.dataset.as_deref(), None)?;

    let t0 = Instant::now();
    let stats = match args.method {
        ClusterMethod::Louvain => {
            let cfg = LouvainConfig::new().with_seed(args.seed);
            let res = louvain(&graph, &cfg)?;
            ClusterRunStats {
                labels: res.labels,
                num_clusters: res.num_communities,
                modularity: Some(res.modularity),
                quality: None,
                inertia: None,
                passes: Some(res.passes),
                iterations: None,
                converged: None,
                eigen_iterations: None,
            }
        }
        ClusterMethod::LeidenMod => {
            let cfg = LeidenConfig::new()
                .with_objective(LeidenObjective::Modularity)
                .with_resolution(args.resolution)
                .with_randomness(args.randomness)
                .with_seed(args.seed);
            let res = leiden(&graph, &cfg)?;
            ClusterRunStats {
                labels: res.labels,
                num_clusters: res.num_communities,
                modularity: Some(res.modularity),
                quality: Some(res.quality),
                inertia: None,
                passes: Some(res.passes),
                iterations: None,
                converged: None,
                eigen_iterations: None,
            }
        }
        ClusterMethod::LeidenCpm => {
            let cfg = LeidenConfig::new()
                .with_objective(LeidenObjective::Cpm)
                .with_resolution(args.resolution)
                .with_randomness(args.randomness)
                .with_seed(args.seed);
            let res = leiden(&graph, &cfg)?;
            ClusterRunStats {
                labels: res.labels,
                num_clusters: res.num_communities,
                modularity: Some(res.modularity),
                quality: Some(res.quality),
                inertia: None,
                passes: Some(res.passes),
                iterations: None,
                converged: None,
                eigen_iterations: None,
            }
        }
        ClusterMethod::Spectral => {
            let cfg = SpectralClusteringConfig::new()
                .with_k(args.k)
                .with_seed(args.seed);
            let res = spectral_clustering(&graph, &cfg)?;
            ClusterRunStats {
                labels: res.labels,
                num_clusters: res.num_clusters,
                modularity: None,
                quality: None,
                inertia: None,
                passes: None,
                iterations: None,
                converged: None,
                eigen_iterations: Some(res.eigen_iterations),
            }
        }
        ClusterMethod::LabelProp => {
            let cfg = LabelPropagationConfig::new().with_seed(args.seed);
            let res = label_propagation(&graph, &cfg)?;
            ClusterRunStats {
                labels: res.labels,
                num_clusters: res.num_communities,
                modularity: None,
                quality: None,
                inertia: None,
                passes: None,
                iterations: Some(res.iterations),
                converged: Some(res.converged),
                eigen_iterations: None,
            }
        }
        ClusterMethod::Kmeans => {
            let layout = build_initial_layout(&graph, InitMethod::PivotMds, args.seed)?;
            let points = layout.to_interleaved();
            let cfg = KMeansConfig::new().with_k(args.k).with_seed(args.seed);
            let res = kmeans_plus_plus(&points, graph.num_nodes(), 2, &cfg)?;
            let count = count_distinct_labels(&res.labels);
            ClusterRunStats {
                labels: res.labels,
                num_clusters: count,
                modularity: None,
                quality: None,
                inertia: Some(res.inertia),
                passes: None,
                iterations: Some(res.iterations),
                converged: Some(res.converged),
                eigen_iterations: None,
            }
        }
    };
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    Ok(ClusterOutput {
        source,
        method: args.method,
        seed: args.seed,
        num_nodes: graph.num_nodes(),
        num_edges: graph.num_edges(),
        num_clusters: stats.num_clusters,
        modularity: stats.modularity,
        quality: stats.quality,
        inertia: stats.inertia,
        passes: stats.passes,
        iterations: stats.iterations,
        converged: stats.converged,
        eigen_iterations: stats.eigen_iterations,
        elapsed_ms,
        labels: stats.labels,
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

    let total_t0 = Instant::now();
    let (graph, _, source) = resolve_graph(args.input.as_deref(), args.dataset.as_deref(), None)?;

    let m_cfg = MetricConfig::new()
        .with_sns_seed(args.seed)
        .with_crossing_seed(args.seed)
        .with_np_seed(args.seed);

    if args.dims == 3 {
        let init_t0 = Instant::now();
        let mut layout_3d = build_initial_3d_layout(&graph, args.init, args.seed)?;
        let init_ms = init_t0.elapsed().as_secs_f64() * 1000.0;

        let layout_t0 = Instant::now();
        run_layout_method_3d(&graph, &mut layout_3d, args.method, args.seed)?;
        let layout_ms = layout_t0.elapsed().as_secs_f64() * 1000.0;

        let metrics_t0 = Instant::now();
        let mut targets = vec![
            MetricTarget::NeighborhoodPreservation1Hop,
            MetricTarget::NeighborhoodPreservation2Hop,
        ];
        if graph.num_nodes() <= 2000 {
            targets.push(MetricTarget::ScaleNormalizedStress);
        } else {
            targets.push(MetricTarget::SampledScaleNormalizedStress);
        }
        let report = evaluate_metrics(&graph, &layout_3d, &targets, &m_cfg)?;
        let metrics_ms = metrics_t0.elapsed().as_secs_f64() * 1000.0;
        let total_ms = total_t0.elapsed().as_secs_f64() * 1000.0;

        return Ok(BenchOutput {
            source,
            method: args.method,
            init: args.init,
            dims: 3,
            manifold: None,
            seed: args.seed,
            num_nodes: graph.num_nodes(),
            num_edges: graph.num_edges(),
            init_ms,
            layout_ms,
            metrics_ms,
            total_ms,
            scale_normalized_stress: report.scale_normalized_stress,
            sampled_scale_normalized_stress: report.sampled_scale_normalized_stress,
            crossing_number: None,
            neighborhood_preservation_1hop: report.neighborhood_preservation_1hop,
            neighborhood_preservation_2hop: report.neighborhood_preservation_2hop,
        });
    }

    if args.dims != 2 {
        let init_t0 = Instant::now();
        let mut nd_layout = build_initial_nd_layout(&graph, args.init, args.dims, args.seed)?;
        let init_ms = init_t0.elapsed().as_secs_f64() * 1000.0;

        let layout_t0 = Instant::now();
        run_layout_method_nd(&graph, &mut nd_layout, args.method, args.seed)?;
        let layout_ms = layout_t0.elapsed().as_secs_f64() * 1000.0;

        let metrics_t0 = Instant::now();
        let mut targets = vec![
            MetricTarget::NeighborhoodPreservation1Hop,
            MetricTarget::NeighborhoodPreservation2Hop,
        ];
        if graph.num_nodes() <= 2000 {
            targets.push(MetricTarget::ScaleNormalizedStress);
        } else {
            targets.push(MetricTarget::SampledScaleNormalizedStress);
        }
        let report = evaluate_metrics(&graph, &nd_layout, &targets, &m_cfg)?;
        let metrics_ms = metrics_t0.elapsed().as_secs_f64() * 1000.0;
        let total_ms = total_t0.elapsed().as_secs_f64() * 1000.0;

        return Ok(BenchOutput {
            source,
            method: args.method,
            init: args.init,
            dims: args.dims,
            manifold: None,
            seed: args.seed,
            num_nodes: graph.num_nodes(),
            num_edges: graph.num_edges(),
            init_ms,
            layout_ms,
            metrics_ms,
            total_ms,
            scale_normalized_stress: report.scale_normalized_stress,
            sampled_scale_normalized_stress: report.sampled_scale_normalized_stress,
            crossing_number: None,
            neighborhood_preservation_1hop: report.neighborhood_preservation_1hop,
            neighborhood_preservation_2hop: report.neighborhood_preservation_2hop,
        });
    }

    if let Some(kind) = args.manifold.to_manifold_kind() {
        let init_t0 = Instant::now();
        let mut m_layout = build_initial_manifold_layout(
            &graph,
            kind,
            args.init,
            args.manifold_scale,
            args.torus_aspect,
            args.seed,
        )?;
        let init_ms = init_t0.elapsed().as_secs_f64() * 1000.0;

        let layout_t0 = Instant::now();
        run_layout_method_manifold(
            &graph,
            &mut m_layout,
            args.method,
            args.seed,
            None,
            None,
            args.manifold_scale,
            None,
        )?;
        if args.torus_auto_pan && kind == ManifoldKind::Torus2d {
            m_layout = m_layout.torus_auto_pan(&graph)?;
        }
        let layout_ms = layout_t0.elapsed().as_secs_f64() * 1000.0;

        let metrics_t0 = Instant::now();
        let mut targets = vec![
            MetricTarget::NeighborhoodPreservation1Hop,
            MetricTarget::NeighborhoodPreservation2Hop,
        ];
        if graph.num_nodes() <= 2000 {
            targets.push(MetricTarget::ScaleNormalizedStress);
        } else {
            targets.push(MetricTarget::SampledScaleNormalizedStress);
        }
        let report = evaluate_metrics(&graph, &m_layout, &targets, &m_cfg)?;
        let metrics_ms = metrics_t0.elapsed().as_secs_f64() * 1000.0;
        let total_ms = total_t0.elapsed().as_secs_f64() * 1000.0;

        return Ok(BenchOutput {
            source,
            method: args.method,
            init: args.init,
            dims: 2,
            manifold: Some(args.manifold),
            seed: args.seed,
            num_nodes: graph.num_nodes(),
            num_edges: graph.num_edges(),
            init_ms,
            layout_ms,
            metrics_ms,
            total_ms,
            scale_normalized_stress: report.scale_normalized_stress,
            sampled_scale_normalized_stress: report.sampled_scale_normalized_stress,
            crossing_number: None,
            neighborhood_preservation_1hop: report.neighborhood_preservation_1hop,
            neighborhood_preservation_2hop: report.neighborhood_preservation_2hop,
        });
    }

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
    let report = evaluate_metrics(&graph, &layout, &targets, &m_cfg)?;
    let metrics_ms = metrics_t0.elapsed().as_secs_f64() * 1000.0;
    let total_ms = total_t0.elapsed().as_secs_f64() * 1000.0;

    Ok(BenchOutput {
        source,
        method: args.method,
        init: args.init,
        dims: 2,
        manifold: None,
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
