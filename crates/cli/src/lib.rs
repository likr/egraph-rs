//! Modernized single-threaded command-line interface library for `egraph`.
//!
//! Exposes [`Cli`], [`Commands`], and execution entrypoints for `layout`,
//! `metrics`, `cluster`, and `bench` subcommands backed exclusively by the
//! `egraph` facade crate.

pub mod io;

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{Args, Parser, Subcommand, ValueEnum};
use egraph::prelude::{
    CoreError, CsrGraph, DeterministicRng, DogVaradhanConfig, FieldLayoutSolver, ForceAtlas2Config,
    ForceAtlas2Solver, GuMapConfig, GuMapSolver, IbUmapConfig, IterativeSolver, KMeansConfig,
    KamadaKawaiConfig, KamadaKawaiSolver, LabelPropagationConfig, Layout2d, LouvainConfig,
    MetricConfig, MetricReport, MetricTarget, PacmapGraphConfig, PivotMdsConfig, PrismConfig,
    PrismOverlapRemoval, SgdScheduleConfig, SgdSolver, SnapTFdpConfig, SparseBarrierConfig,
    SpatialIntruderRefiner, SpatialRefinerConfig, SpectralClusteringConfig,
    StressMajorizationConfig, StressMajorizationSolver, TFdpConfig, TsNetConfig, TsNetSolver,
    YifanHuConfig, YifanHuSolver, evaluate_metrics, evaluate_metrics_with_communities,
    kmeans_plus_plus, koren_spectral_init, label_propagation, louvain, normalize_layout_2d,
    pivot_mds, spectral_clustering,
};
use serde::Serialize;

use crate::io::{CliError, CliResult, load_layout_from_path, resolve_graph, write_json_output};

/// Top-level `egraph` CLI parser.
#[derive(Debug, Parser)]
#[command(
    name = "egraph",
    version,
    about = "Single-threaded pure-Rust graph drawing, quality metrics, clustering, and benchmarking CLI"
)]
pub struct Cli {
    /// Subcommand to execute.
    #[command(subcommand)]
    pub command: Commands,
}

/// Supported CLI subcommands.
#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Compute a 2D graph layout and output coordinates + execution stats as JSON.
    Layout(LayoutArgs),
    /// Evaluate quality metrics on a graph and an existing 2D JSON layout.
    Metrics(MetricsArgs),
    /// Cluster a graph using Louvain, spectral clustering, label propagation, or K-Means++.
    Cluster(ClusterArgs),
    /// Run a reproducible layout + metrics timing benchmark.
    Bench(BenchArgs),
}

/// Supported layout algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum LayoutMethod {
    /// Pure 1+2-Hop adaptive Fit-tsNET (`TsNetSolver::from_graph_pure_2hop`).
    #[value(name = "tsnet-pure2hop")]
    TsnetPure2hop,
    /// Regularized Chebyshev DoG + 3-tier Sparse Barrier SGD (`SgdSolver::sparse_barrier_dog`).
    #[value(name = "sgd-barrier-dog")]
    SgdBarrierDog,
    /// Full all-pairs shortest-path SGD (`SgdSolver::full`).
    #[value(name = "sgd-full")]
    SgdFull,
    /// SNAP-tFDP streaming edge-attached negative SGD (`SgdSolver::snap_t_fdp`).
    #[value(name = "sgd-snap")]
    SgdSnap,
    /// 3-tier PaCMAP-Graph SGD (`SgdSolver::pacmap_graph`).
    #[value(name = "sgd-pacmap")]
    SgdPacmap,
    /// Graph UMAP (`GuMapSolver::from_graph_dog`).
    #[value(name = "gumap")]
    Gumap,
    /// Interpolation-FFT bounded Student-t FDP (`FieldLayoutSolver::t_fdp`).
    #[value(name = "tfdp")]
    Tfdp,
    /// Interpolation-FFT batch Graph UMAP (`FieldLayoutSolver::ib_umap`).
    #[value(name = "ibumap")]
    Ibumap,
    /// Barnes-Hut ForceAtlas2 (`ForceAtlas2Solver`).
    #[value(name = "fa2")]
    Fa2,
    /// Multilevel Yifan Hu (`YifanHuSolver`).
    #[value(name = "yifanhu")]
    YifanHu,
    /// Stress Majorization (`StressMajorizationSolver`).
    #[value(name = "stress")]
    Stress,
    /// Kamada-Kawai (`KamadaKawaiSolver`).
    #[value(name = "kk")]
    Kk,
}

/// Supported layout initialization methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum InitMethod {
    /// Normalized Core-First MaxMin PivotMDS (`pivot_mds`).
    #[value(name = "pivot-mds")]
    PivotMds,
    /// Koren degree-normalized spectral initialization (`koren_spectral_init`).
    #[value(name = "koren")]
    Koren,
    /// Deterministic uniform/Gaussian random initialization (`DeterministicRng`).
    #[value(name = "random")]
    Random,
}

/// Supported clustering methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ClusterMethod {
    /// Two-phase multi-level Louvain modularity maximization.
    #[value(name = "louvain")]
    Louvain,
    /// Ng-Jordan-Weiss spectral clustering.
    #[value(name = "spectral")]
    Spectral,
    /// Asynchronous deterministic Label Propagation.
    #[value(name = "label-prop")]
    LabelProp,
    /// K-Means++ over normalized 2D PivotMDS coordinates.
    #[value(name = "kmeans")]
    Kmeans,
}

/// Arguments for `egraph layout`.
#[derive(Debug, Args)]
pub struct LayoutArgs {
    /// Input graph file (`.json`, `.mtx`, or `.csv`).
    #[arg(long, value_name = "FILE", conflicts_with = "dataset")]
    pub input: Option<PathBuf>,
    /// Embedded benchmark dataset name (`1138_bus`, `3_elt`, `dwt_1005`, `dwt_2680`, `poli`, `qh882`, `uspower_grid`, `karate`, `lesmis`).
    #[arg(long, value_name = "NAME", conflicts_with = "input")]
    pub dataset: Option<String>,
    /// Layout algorithm to execute.
    #[arg(long, value_enum, default_value_t = LayoutMethod::TsnetPure2hop)]
    pub method: LayoutMethod,
    /// Initial layout placement strategy.
    #[arg(long, value_enum, default_value_t = InitMethod::PivotMds)]
    pub init: InitMethod,
    /// Apply post-layout 2D spatial intruder refinement (`SpatialIntruderRefiner`).
    #[arg(long, default_value_t = false)]
    pub refine: bool,
    /// Apply post-layout PRISM overlap removal (`PrismOverlapRemoval`).
    #[arg(long, default_value_t = false)]
    pub prism: bool,
    /// Uniform node half-extent radius used when `--prism` is enabled.
    #[arg(long, default_value_t = 0.05)]
    pub prism_radius: f64,
    /// Deterministic RNG seed.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    /// Optional output JSON file path (writes to stdout if omitted).
    #[arg(long, value_name = "FILE")]
    pub output: Option<PathBuf>,
}

/// Arguments for `egraph metrics`.
#[derive(Debug, Args)]
pub struct MetricsArgs {
    /// Input graph file (`.json`, `.mtx`, or `.csv`).
    #[arg(long, value_name = "FILE", conflicts_with = "dataset")]
    pub input: Option<PathBuf>,
    /// Embedded benchmark dataset name.
    #[arg(long, value_name = "NAME", conflicts_with = "input")]
    pub dataset: Option<String>,
    /// Input 2D coordinates JSON file.
    #[arg(long, value_name = "FILE")]
    pub layout: PathBuf,
    /// Optional community labels file (JSON `{"labels": [...]}` / `{"community_labels": [...]}` / `[...]` or plain-text/CSV integer list), required for `cluster-area-faithfulness` (`gcaf`).
    #[arg(long, value_name = "FILE")]
    pub community_labels: Option<PathBuf>,
    /// Comma-separated metric targets (`sns,sampled-sns,crossings,crossing-angle,sampled-crossings,crosslessness,sampled-crosslessness,np1,np2,sampled-np1,sampled-np2,gabriel,node-resolution,drng,dgg,aspect-ratio,angular-resolution,minimum-angle,minimum-angle-a1,layout-degeneracy,ideal-edge-lengths,cluster-area-faithfulness,gcaf,rta,all`).
    #[arg(long, default_value = "sns,crossings,np1,np2")]
    pub targets: String,
    /// Number of BFS pivot sources for `sampled-sns`.
    #[arg(long, default_value_t = 64)]
    pub num_pivots: usize,
    /// Number of edge-pair / node samples for `sampled-crossings`, `sampled-np1`, and `sampled-np2`.
    #[arg(long, default_value_t = 4096)]
    pub num_samples: usize,
    /// Ideal edge length fallback for unweighted graphs.
    #[arg(long, default_value_t = 1.0)]
    pub ideal_edge_length: f64,
    /// Deterministic RNG seed for sampled metrics.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    /// Optional output JSON file path (writes to stdout if omitted).
    #[arg(long, value_name = "FILE")]
    pub output: Option<PathBuf>,
}

/// Arguments for `egraph cluster`.
#[derive(Debug, Args)]
pub struct ClusterArgs {
    /// Input graph file (`.json`, `.mtx`, or `.csv`).
    #[arg(long, value_name = "FILE", conflicts_with = "dataset")]
    pub input: Option<PathBuf>,
    /// Embedded benchmark dataset name.
    #[arg(long, value_name = "NAME", conflicts_with = "input")]
    pub dataset: Option<String>,
    /// Clustering algorithm to execute.
    #[arg(long, value_enum, default_value_t = ClusterMethod::Louvain)]
    pub method: ClusterMethod,
    /// Target number of clusters (used by `spectral` and `kmeans`).
    #[arg(long, default_value_t = 2)]
    pub k: usize,
    /// Deterministic RNG seed.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    /// Optional output JSON file path (writes to stdout if omitted).
    #[arg(long, value_name = "FILE")]
    pub output: Option<PathBuf>,
}

/// Arguments for `egraph bench`.
#[derive(Debug, Args)]
pub struct BenchArgs {
    /// Input graph file (`.json`, `.mtx`, or `.csv`).
    #[arg(long, value_name = "FILE", conflicts_with = "dataset")]
    pub input: Option<PathBuf>,
    /// Embedded benchmark dataset name.
    #[arg(long, value_name = "NAME", conflicts_with = "input")]
    pub dataset: Option<String>,
    /// Layout algorithm to benchmark.
    #[arg(long, value_enum, default_value_t = LayoutMethod::TsnetPure2hop)]
    pub method: LayoutMethod,
    /// Initial layout placement strategy.
    #[arg(long, value_enum, default_value_t = InitMethod::PivotMds)]
    pub init: InitMethod,
    /// Deterministic RNG seed.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    /// Optional output JSON file path (writes to stdout if omitted).
    #[arg(long, value_name = "FILE")]
    pub output: Option<PathBuf>,
}

/// JSON output payload for `egraph layout`.
#[derive(Debug, Serialize)]
pub struct LayoutOutput {
    pub source: String,
    pub method: LayoutMethod,
    pub init: InitMethod,
    pub seed: u64,
    pub num_nodes: usize,
    pub num_edges: usize,
    pub refined: bool,
    pub prism_applied: bool,
    pub prism_overlaps_remaining: Option<usize>,
    pub elapsed_ms: f64,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub coords: Vec<[f64; 2]>,
}

/// JSON output payload for `egraph metrics`.
#[derive(Debug, Serialize)]
pub struct MetricsOutput {
    pub source: String,
    pub num_nodes: usize,
    pub num_edges: usize,
    pub elapsed_ms: f64,
    pub scale_normalized_stress: Option<f64>,
    pub sampled_scale_normalized_stress: Option<f64>,
    pub crossing_number: Option<usize>,
    pub crossing_angle: Option<f64>,
    pub sampled_crossing_number: Option<f64>,
    pub crosslessness: Option<f64>,
    pub sampled_crosslessness: Option<f64>,
    pub neighborhood_preservation_1hop: Option<f64>,
    pub neighborhood_preservation_2hop: Option<f64>,
    pub sampled_np1: Option<f64>,
    pub sampled_np2: Option<f64>,
    pub gabriel_graph_property: Option<f64>,
    pub node_resolution: Option<f64>,
    pub relative_neighborhood_preservation: Option<f64>,
    pub gabriel_graph_preservation: Option<f64>,
    pub aspect_ratio: Option<f64>,
    pub angular_resolution: Option<f64>,
    pub minimum_angle: Option<f64>,
    pub minimum_angle_a1: Option<f64>,
    pub layout_degeneracy: Option<f64>,
    pub ideal_edge_lengths: Option<f64>,
    pub cluster_area_faithfulness: Option<f64>,
    pub landmark_stratified_rta: Option<f64>,
}

impl MetricsOutput {
    fn from_report(
        source: String,
        num_nodes: usize,
        num_edges: usize,
        elapsed_ms: f64,
        report: MetricReport,
    ) -> Self {
        Self {
            source,
            num_nodes,
            num_edges,
            elapsed_ms,
            scale_normalized_stress: report.scale_normalized_stress,
            sampled_scale_normalized_stress: report.sampled_scale_normalized_stress,
            crossing_number: report.crossing_number,
            crossing_angle: report.crossing_angle,
            sampled_crossing_number: report.sampled_crossing_number,
            crosslessness: report.crosslessness,
            sampled_crosslessness: report.sampled_crosslessness,
            neighborhood_preservation_1hop: report.neighborhood_preservation_1hop,
            neighborhood_preservation_2hop: report.neighborhood_preservation_2hop,
            sampled_np1: report.sampled_np1,
            sampled_np2: report.sampled_np2,
            gabriel_graph_property: report.gabriel_graph_property,
            node_resolution: report.node_resolution,
            relative_neighborhood_preservation: report.relative_neighborhood_preservation,
            gabriel_graph_preservation: report.gabriel_graph_preservation,
            aspect_ratio: report.aspect_ratio,
            angular_resolution: report.angular_resolution,
            minimum_angle: report.minimum_angle,
            minimum_angle_a1: report.minimum_angle_a1,
            layout_degeneracy: report.layout_degeneracy,
            ideal_edge_lengths: report.ideal_edge_lengths,
            cluster_area_faithfulness: report.cluster_area_faithfulness,
            landmark_stratified_rta: report.landmark_stratified_rta,
        }
    }
}

/// JSON output payload for `egraph cluster`.
#[derive(Debug, Serialize)]
pub struct ClusterOutput {
    pub source: String,
    pub method: ClusterMethod,
    pub seed: u64,
    pub num_nodes: usize,
    pub num_edges: usize,
    pub num_clusters: usize,
    pub modularity: Option<f64>,
    pub inertia: Option<f64>,
    pub elapsed_ms: f64,
    pub labels: Vec<u32>,
}

/// JSON output payload for `egraph bench`.
#[derive(Debug, Serialize)]
pub struct BenchOutput {
    pub source: String,
    pub method: LayoutMethod,
    pub init: InitMethod,
    pub seed: u64,
    pub num_nodes: usize,
    pub num_edges: usize,
    pub init_ms: f64,
    pub layout_ms: f64,
    pub metrics_ms: f64,
    pub total_ms: f64,
    pub scale_normalized_stress: Option<f64>,
    pub sampled_scale_normalized_stress: Option<f64>,
    pub crossing_number: Option<usize>,
    pub neighborhood_preservation_1hop: Option<f64>,
    pub neighborhood_preservation_2hop: Option<f64>,
}

/// Parses and executes the CLI from process arguments.
pub fn run_cli() -> CliResult<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Layout(args) => {
            let out = execute_layout(&args)?;
            write_json_output(&out, args.output.as_deref())
        }
        Commands::Metrics(args) => {
            let out = execute_metrics(&args)?;
            write_json_output(&out, args.output.as_deref())
        }
        Commands::Cluster(args) => {
            let out = execute_cluster(&args)?;
            write_json_output(&out, args.output.as_deref())
        }
        Commands::Bench(args) => {
            let out = execute_bench(&args)?;
            write_json_output(&out, args.output.as_deref())
        }
    }
}

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
    }
    Ok(())
}

/// Executes the `layout` subcommand and returns the structured JSON output.
pub fn execute_layout(args: &LayoutArgs) -> CliResult<LayoutOutput> {
    if args.prism && (!args.prism_radius.is_finite() || args.prism_radius <= 0.0) {
        return Err(CliError::Core(CoreError::invalid_input(
            "prism-radius must be finite and > 0.0",
        )));
    }
    let t0 = Instant::now();
    let (graph, embedded_layout, source) =
        resolve_graph(args.input.as_deref(), args.dataset.as_deref(), None)?;

    let mut layout = match embedded_layout {
        Some(l) => l,
        None => build_initial_layout(&graph, args.init, args.seed)?,
    };

    run_layout_method(&graph, &mut layout, args.method, args.seed)?;

    if args.refine {
        let ref_cfg = SpatialRefinerConfig {
            seed: args.seed,
            ..SpatialRefinerConfig::default()
        };
        let mut refiner = SpatialIntruderRefiner::new(&graph, ref_cfg)?;
        refiner.run(&mut layout)?;
    }

    let mut prism_overlaps_remaining = None;
    if args.prism {
        let prism_cfg = PrismConfig {
            seed: args.seed,
            ..PrismConfig::default()
        };
        let radii = vec![args.prism_radius; graph.num_nodes()];
        let mut prism = PrismOverlapRemoval::from_radii(radii, prism_cfg)?;
        prism.run(&mut layout)?;
        prism_overlaps_remaining = Some(prism.last_overlap_count());
    }

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
        prism_applied: args.prism,
        prism_overlaps_remaining,
        elapsed_ms,
        x,
        y,
        coords,
    })
}

const ALL_STANDARD_METRIC_TARGETS: [MetricTarget; 22] = [
    MetricTarget::ScaleNormalizedStress,
    MetricTarget::SampledScaleNormalizedStress,
    MetricTarget::CrossingNumber,
    MetricTarget::CrossingAngle,
    MetricTarget::SampledCrossingNumber,
    MetricTarget::Crosslessness,
    MetricTarget::SampledCrosslessness,
    MetricTarget::NeighborhoodPreservation1Hop,
    MetricTarget::NeighborhoodPreservation2Hop,
    MetricTarget::SampledNp1,
    MetricTarget::SampledNp2,
    MetricTarget::GabrielGraphProperty,
    MetricTarget::NodeResolution,
    MetricTarget::RelativeNeighborhoodPreservation,
    MetricTarget::GabrielGraphPreservation,
    MetricTarget::AspectRatio,
    MetricTarget::AngularResolution,
    MetricTarget::MinimumAngle,
    MetricTarget::MinimumAngleA1,
    MetricTarget::LayoutDegeneracy,
    MetricTarget::IdealEdgeLengths,
    MetricTarget::LandmarkStratifiedRta,
];

fn push_unique_target(targets: &mut Vec<MetricTarget>, target: MetricTarget) {
    if !targets.contains(&target) {
        targets.push(target);
    }
}

/// Parses comma-separated metric target names into [`MetricTarget`] values.
pub fn parse_metric_targets(spec: &str) -> CliResult<Vec<MetricTarget>> {
    let mut targets = Vec::new();
    for raw in spec.split(',') {
        let token = raw.trim().to_ascii_lowercase().replace('_', "-");
        if token.is_empty() {
            continue;
        }
        match token.as_str() {
            "all" => {
                // Expand `all` in place instead of returning early so combinations
                // like `all,gcaf` and trailing invalid tokens are both processed.
                for &target in &ALL_STANDARD_METRIC_TARGETS {
                    push_unique_target(&mut targets, target);
                }
            }
            "sns" | "scale-normalized-stress" => {
                push_unique_target(&mut targets, MetricTarget::ScaleNormalizedStress);
            }
            "sampled-sns" | "sampled-scale-normalized-stress" => {
                push_unique_target(&mut targets, MetricTarget::SampledScaleNormalizedStress);
            }
            "crossings" | "crossing-number" => {
                push_unique_target(&mut targets, MetricTarget::CrossingNumber);
            }
            "crossing-angle" => {
                push_unique_target(&mut targets, MetricTarget::CrossingAngle);
            }
            "sampled-crossings" | "sampled-crossing-number" => {
                push_unique_target(&mut targets, MetricTarget::SampledCrossingNumber);
            }
            "crosslessness" => {
                push_unique_target(&mut targets, MetricTarget::Crosslessness);
            }
            "sampled-crosslessness" => {
                push_unique_target(&mut targets, MetricTarget::SampledCrosslessness);
            }
            "np1" | "neighborhood-preservation-1hop" => {
                push_unique_target(&mut targets, MetricTarget::NeighborhoodPreservation1Hop);
            }
            "np2" | "neighborhood-preservation-2hop" => {
                push_unique_target(&mut targets, MetricTarget::NeighborhoodPreservation2Hop);
            }
            "sampled-np1" | "sampled-neighborhood-preservation-1hop" => {
                push_unique_target(&mut targets, MetricTarget::SampledNp1);
            }
            "sampled-np2" | "sampled-neighborhood-preservation-2hop" => {
                push_unique_target(&mut targets, MetricTarget::SampledNp2);
            }
            "gabriel" | "gabriel-graph-property" => {
                push_unique_target(&mut targets, MetricTarget::GabrielGraphProperty);
            }
            "node-resolution" => {
                push_unique_target(&mut targets, MetricTarget::NodeResolution);
            }
            "drng" | "relative-neighborhood-preservation" => {
                push_unique_target(&mut targets, MetricTarget::RelativeNeighborhoodPreservation);
            }
            "dgg" | "gabriel-graph-preservation" => {
                push_unique_target(&mut targets, MetricTarget::GabrielGraphPreservation);
            }
            "aspect-ratio" => {
                push_unique_target(&mut targets, MetricTarget::AspectRatio);
            }
            "angular-resolution" => {
                push_unique_target(&mut targets, MetricTarget::AngularResolution);
            }
            "minimum-angle" | "min-angle" | "minimum-angle-a2" | "min-angle-a2" => {
                push_unique_target(&mut targets, MetricTarget::MinimumAngle);
            }
            "minimum-angle-a1" | "min-angle-a1" => {
                push_unique_target(&mut targets, MetricTarget::MinimumAngleA1);
            }
            "layout-degeneracy" => {
                push_unique_target(&mut targets, MetricTarget::LayoutDegeneracy);
            }
            "ideal-edge-lengths" => {
                push_unique_target(&mut targets, MetricTarget::IdealEdgeLengths);
            }
            "cluster-area-faithfulness" | "cluster-area" | "gcaf" | "g-caf" => {
                push_unique_target(&mut targets, MetricTarget::ClusterAreaFaithfulness);
            }
            "rta" | "l-stratified-rta" | "landmark-stratified-rta" => {
                push_unique_target(&mut targets, MetricTarget::LandmarkStratifiedRta);
            }
            other => {
                return Err(CliError::InvalidInput(format!(
                    "unknown metric target '{other}'"
                )));
            }
        }
    }
    Ok(targets)
}

/// Loads community labels from a JSON file (`{"labels": [...]}` / `{"community_labels": [...]}` / `[...]`)
/// or a plain-text/CSV integer list, validating that all entries are `u32` integers and that
/// the label count matches `expected_nodes`.
pub fn load_community_labels_from_path(path: &Path, expected_nodes: usize) -> CliResult<Vec<u32>> {
    let content = fs::read_to_string(path)?;
    let trimmed = content.trim();
    let is_json = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("json"))
        || trimmed.starts_with('{')
        || trimmed.starts_with('[');

    let labels = if is_json {
        let val: serde_json::Value = serde_json::from_str(trimmed)?;
        let items = if let Some(arr) = val.as_array() {
            arr
        } else if let Some(field) = val.get("labels") {
            field.as_array().ok_or_else(|| {
                CliError::InvalidInput(format!(
                    "'labels' field in '{}' must be a JSON array of integers",
                    path.display()
                ))
            })?
        } else if let Some(field) = val.get("community_labels") {
            field.as_array().ok_or_else(|| {
                CliError::InvalidInput(format!(
                    "'community_labels' field in '{}' must be a JSON array of integers",
                    path.display()
                ))
            })?
        } else {
            return Err(CliError::InvalidInput(format!(
                "unrecognized community labels JSON schema in '{}'",
                path.display()
            )));
        };

        let mut parsed = Vec::with_capacity(items.len());
        for (idx, item) in items.iter().enumerate() {
            let raw = item.as_u64().ok_or_else(|| {
                CliError::InvalidInput(format!(
                    "community label at index {idx} in '{}' must be a non-negative integer",
                    path.display()
                ))
            })?;
            let lbl = u32::try_from(raw).map_err(|_| {
                CliError::InvalidInput(format!(
                    "community label {raw} at index {idx} in '{}' exceeds u32::MAX",
                    path.display()
                ))
            })?;
            parsed.push(lbl);
        }
        parsed
    } else {
        let mut parsed = Vec::new();
        for (line_idx, raw_line) in content.lines().enumerate() {
            let line = raw_line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            for token in line
                .split(|c: char| c.is_ascii_whitespace() || c == ',')
                .filter(|t| !t.is_empty())
            {
                let lbl = token.parse::<u32>().map_err(|_| {
                    CliError::InvalidInput(format!(
                        "invalid community label '{token}' on line {} of '{}'",
                        line_idx + 1,
                        path.display()
                    ))
                })?;
                parsed.push(lbl);
            }
        }
        parsed
    };

    if labels.len() != expected_nodes {
        return Err(CliError::Core(CoreError::dimension_mismatch(
            expected_nodes,
            labels.len(),
            "load_community_labels_from_path",
        )));
    }
    Ok(labels)
}

/// Executes the `metrics` subcommand and returns the structured JSON output.
pub fn execute_metrics(args: &MetricsArgs) -> CliResult<MetricsOutput> {
    let (graph, _, source) = resolve_graph(args.input.as_deref(), args.dataset.as_deref(), None)?;
    let layout = load_layout_from_path(&args.layout, graph.num_nodes())?;
    let targets = parse_metric_targets(&args.targets)?;
    let community_labels = args
        .community_labels
        .as_deref()
        .map(|path| load_community_labels_from_path(path, graph.num_nodes()))
        .transpose()?;
    let cfg = MetricConfig {
        sns_num_pivots: args.num_pivots,
        sns_seed: args.seed,
        crossing_num_samples: args.num_samples,
        crossing_seed: args.seed,
        ideal_edge_length: args.ideal_edge_length,
        ..MetricConfig::default()
    };
    cfg.validate()?;

    let t0 = Instant::now();
    let report = evaluate_metrics_with_communities(
        &graph,
        &layout,
        &targets,
        &cfg,
        community_labels.as_deref(),
    )?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    Ok(MetricsOutput::from_report(
        source,
        graph.num_nodes(),
        graph.num_edges(),
        elapsed_ms,
        report,
    ))
}

/// Executes the `cluster` subcommand and returns the structured JSON output.
pub fn execute_cluster(args: &ClusterArgs) -> CliResult<ClusterOutput> {
    let (graph, _, source) = resolve_graph(args.input.as_deref(), args.dataset.as_deref(), None)?;

    let t0 = Instant::now();
    let (labels, num_clusters, modularity, inertia) = match args.method {
        ClusterMethod::Louvain => {
            let cfg = LouvainConfig {
                seed: args.seed,
                ..LouvainConfig::default()
            };
            let res = louvain(&graph, &cfg)?;
            (res.labels, res.num_communities, Some(res.modularity), None)
        }
        ClusterMethod::Spectral => {
            let cfg = SpectralClusteringConfig {
                k: args.k,
                seed: args.seed,
                ..SpectralClusteringConfig::default()
            };
            let labels = spectral_clustering(&graph, &cfg)?;
            let count = count_distinct_labels(&labels);
            (labels, count, None, None)
        }
        ClusterMethod::LabelProp => {
            let cfg = LabelPropagationConfig {
                seed: args.seed,
                ..LabelPropagationConfig::default()
            };
            let labels = label_propagation(&graph, &cfg)?;
            let count = count_distinct_labels(&labels);
            (labels, count, None, None)
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
            (res.labels, count, None, Some(res.inertia))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_metric_targets_expands_all_with_dedup_and_validation() {
        let all_only = parse_metric_targets("all").expect("all");
        assert_eq!(all_only.len(), 22);
        assert!(!all_only.contains(&MetricTarget::ClusterAreaFaithfulness));

        let all_gcaf = parse_metric_targets("all,gcaf").expect("all,gcaf");
        assert_eq!(all_gcaf.len(), 23);
        assert_eq!(
            all_gcaf.last(),
            Some(&MetricTarget::ClusterAreaFaithfulness)
        );

        let gcaf_all = parse_metric_targets("gcaf,all").expect("gcaf,all");
        assert_eq!(gcaf_all.len(), 23);
        assert_eq!(
            gcaf_all.first(),
            Some(&MetricTarget::ClusterAreaFaithfulness)
        );

        let sns_all = parse_metric_targets("sns,all,all,gcaf,gcaf").expect("dedup");
        assert_eq!(sns_all.len(), 23);
        assert_eq!(sns_all.first(), Some(&MetricTarget::ScaleNormalizedStress));
        assert_eq!(sns_all.last(), Some(&MetricTarget::ClusterAreaFaithfulness));

        let err = parse_metric_targets("all,invalid_target").expect_err("trailing invalid target");
        assert!(matches!(err, CliError::InvalidInput(_)));
    }
}
