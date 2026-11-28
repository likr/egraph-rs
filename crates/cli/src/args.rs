use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};
use egraph::prelude::ManifoldKind;
use serde::Serialize;

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
    /// Compute a 2D, N-D, or manifold graph layout and output coordinates + execution stats as JSON.
    Layout(LayoutArgs),
    /// Evaluate quality metrics on a graph and an existing 2D, N-D, or manifold JSON layout.
    Metrics(MetricsArgs),
    /// Cluster a graph using Louvain, Leiden, spectral clustering, label propagation, or K-Means++.
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
    /// Row-normalized 1+2-hop adaptive tsNET (`TsNetSolver::from_graph_row_normalized_with_options` with `Auto` backend and `TwoStepRandomWalk`).
    #[value(name = "tsnet-rn")]
    TsnetRn,
    /// Regularized Chebyshev DoG + 3-tier Sparse Barrier SGD (`SgdSolver::sparse_barrier_dog`).
    #[value(name = "sgd-barrier-dog")]
    SgdBarrierDog,
    /// Full all-pairs shortest-path SGD (`SgdSolver::full`); `egraph-cli` exposes single-criteria SGD presets (`sgd-full`, `sgd-ortmann`, `sgd-barrier-dog`, etc.) and does not expose multi-criteria `(SGD)^2` (`sgd2`).
    #[value(name = "sgd-full")]
    SgdFull,
    /// SNAP-tFDP streaming edge-attached negative SGD (`SnapTFdpSolver`).
    #[value(name = "sgd-snap")]
    SgdSnap,
    /// 3-tier PaCMAP-Graph SGD (`PacmapGraphSolver`).
    #[value(name = "sgd-pacmap")]
    SgdPacmap,
    /// Graph UMAP (`GuMapSolver::from_graph_dog`).
    #[value(name = "gumap")]
    Gumap,
    /// Interpolation-FFT bounded Student-t FDP (`TFdpSolver`).
    #[value(name = "tfdp")]
    Tfdp,
    /// Interpolation-FFT batch Graph UMAP (`IbUmapSolver`).
    #[value(name = "ibumap")]
    Ibumap,
    /// Barnes-Hut ForceAtlas2 (`ForceAtlas2Solver`; its `linlog_mode` follows Jacomy et al. 2014 and differs from Noack `LinLogSolver`).
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
    /// Noack (2007/2009) true LinLog energy-model force-directed layout (`LinLogSolver`, distinct from `ForceAtlas2Config::linlog_mode`).
    #[value(name = "linlog")]
    Linlog,
    /// Fruchterman-Reingold force-directed layout (`FruchtermanReingoldSolver`).
    #[value(name = "fr")]
    Fr,
    /// Barnes-Hut ForceAtlas2 with overlap prevention (`ForceAtlas2Solver` with `prevent_overlap = true`).
    #[value(name = "fa2-overlap")]
    Fa2Overlap,
    /// Quadrilateral Simmelian Backbone layout (`BackboneSolver`).
    #[value(name = "backbone")]
    Backbone,
    /// Group-in-a-Box clustered layout (`GroupInABoxSolver`).
    #[value(name = "gib")]
    Gib,
    /// High-Dimensional Embedding (`Hde`, Harel & Koren 2004).
    #[value(name = "hde")]
    Hde,
    /// Ortmann et al. (2016) sparse stress SGD (`SgdSolver::ortmann_sparse_stress`).
    #[value(name = "sgd-ortmann")]
    SgdOrtmann,
    /// Maxent-Stress layout with Barnes-Hut repulsion (`MaxentStressSolver`).
    #[value(name = "maxent")]
    Maxent,
    /// LargeVis negative-sampling SGD layout (`LargeVisSolver`).
    #[value(name = "largevis")]
    Largevis,
    /// DRGraph multilevel negative-sampling SGD layout (`DrGraphSolver`).
    #[value(name = "drgraph")]
    Drgraph,
    /// Graph-CNE InfoNCE / Neg-tSNE contrastive SGD layout (`GraphCneSolver`).
    #[value(name = "graph-cne")]
    GraphCne,
    /// Multilevel GRIP graph drawing (`GripSolver`, Gajer & Kobourov 2002).
    #[value(name = "grip")]
    Grip,
    /// GEM adaptive temperature force-directed layout (`GemSolver`, Frick et al. 1995).
    #[value(name = "gem")]
    Gem,
    /// Structural Entropy Layout with adaptive edge-weight modulation (`SelLayoutSolver`, Xue et al. 2025).
    #[value(name = "sel")]
    Sel,
    /// Canonical community-ordered Hilbert/Morton space-filling curve baseline (`SfcLayout`, Muelder & Ma (2008) proxy; guarantees zero co-location).
    #[value(name = "sfc")]
    Sfc,
}

/// Supported layout initialization methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum InitMethod {
    /// Normalized Core-First MaxMin PivotMDS (`PivotMds`).
    #[value(name = "pivot-mds")]
    PivotMds,
    /// Koren degree-normalized spectral initialization (`KorenSpectral`).
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
    /// Leiden modularity maximization (`leiden` with `LeidenObjective::Modularity`).
    #[value(name = "leiden-mod")]
    LeidenMod,
    /// Leiden Constant Potts Model maximization (`leiden` with `LeidenObjective::Cpm`).
    #[value(name = "leiden-cpm")]
    LeidenCpm,
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

/// Post-layout cluster refinement strategy (`--cluster-refine`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ClusterRefineArg {
    /// No cluster refinement.
    #[default]
    #[value(name = "none")]
    None,
    /// K-Means cluster assignment refinement (`ClusterRefineMethod::Kmeans`).
    #[value(name = "kmeans")]
    Kmeans,
    /// Hierarchical Agglomerative Clustering refinement (`ClusterRefineMethod::Hac`).
    #[value(name = "hac")]
    Hac,
}

/// Post-layout node overlap removal strategy (`--overlap-removal`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum OverlapRemovalArg {
    /// No overlap removal.
    #[default]
    #[value(name = "none")]
    None,
    /// Proximity stress overlap removal (`PrismOverlapRemoval`).
    #[value(name = "prism")]
    Prism,
    /// Delaunay/KdTree GTree overlap removal (`GTreeOverlapRemoval`).
    #[value(name = "gtree")]
    Gtree,
    /// FORBID stress-based overlap removal (`ForbidVariant::Forbid`).
    #[value(name = "forbid")]
    Forbid,
    /// FORBID' stress-based overlap removal (`ForbidVariant::ForbidPrime`).
    #[value(name = "forbid-prime")]
    ForbidPrime,
    /// Sparse FORBID overlap removal (`ForbidVariant::SparseForbid`).
    #[value(name = "sparse-forbid")]
    SparseForbid,
    /// PFS' scanline node overlap removal (`PfsPrimeOverlapRemoval`).
    #[value(name = "pfs-prime")]
    PfsPrime,
    /// Spatial-hash and golden-angle Vogel spiral twin/near-twin decluttering (`TwinDeclutterer`).
    #[value(name = "twin-declutter")]
    TwinDeclutter,
}

/// Non-Euclidean manifold geometry (`--manifold`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ManifoldArg {
    /// Standard Euclidean layout (`Layout2d` / `LayoutNd`).
    #[default]
    #[value(name = "none")]
    None,
    /// Hyperbolic Poincaré open unit disk (`ManifoldKind::HyperbolicPoincare`).
    #[value(name = "hyperbolic")]
    Hyperbolic,
    /// Hyperbolic 2D Lorentz / Gans hyperboloid coordinates (`ManifoldKind::HyperbolicLorentz`).
    #[value(name = "hyperbolic-lorentz")]
    HyperbolicLorentz,
    /// Unit sphere $S^2$ (`ManifoldKind::Spherical`).
    #[value(name = "spherical")]
    Spherical,
    /// Flat 2D torus $T^2 = [0, 1)^2$ (`ManifoldKind::Torus2d`).
    #[value(name = "torus")]
    Torus,
}

impl ManifoldArg {
    /// Converts the CLI manifold option into [`ManifoldKind`], returning `None` for Euclidean mode.
    pub fn to_manifold_kind(self) -> Option<ManifoldKind> {
        match self {
            Self::None => None,
            Self::Hyperbolic => Some(ManifoldKind::HyperbolicPoincare),
            Self::HyperbolicLorentz => Some(ManifoldKind::HyperbolicLorentz),
            Self::Spherical => Some(ManifoldKind::Spherical),
            Self::Torus => Some(ManifoldKind::Torus2d),
        }
    }
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
    /// Embedding dimensionality (`2` for `Layout2d`, `3` for `Layout3d`, otherwise `LayoutNd`).
    #[arg(long, default_value_t = 2)]
    pub dims: usize,
    /// Non-Euclidean manifold geometry (`none`, `hyperbolic`, `hyperbolic-lorentz`, `spherical`, `torus`).
    #[arg(long, value_enum, default_value_t = ManifoldArg::None)]
    pub manifold: ManifoldArg,
    /// Physical geodesic scale factor for manifold layouts (`> 0.0`).
    #[arg(long, default_value_t = 1.0)]
    pub manifold_scale: f64,
    /// Physical rectangular torus aspect ratio `Lx / Ly` (`> 0.0`, used with `--manifold torus`).
    #[arg(long, default_value_t = 1.0)]
    pub torus_aspect: f64,
    /// Apply optimal torus boundary auto-panning after layout (`--manifold torus` only).
    #[arg(long, default_value_t = false)]
    pub torus_auto_pan: bool,
    /// Number of SGD schedule windows for multi-window / size-optimization execution.
    #[arg(long, value_name = "COUNT")]
    pub schedule_windows: Option<usize>,
    /// Iterations per SGD schedule window (used with `--schedule-windows` or standalone).
    #[arg(long, value_name = "ITERS")]
    pub schedule_window_iters: Option<usize>,
    /// Final geodesic scale applied via `ManifoldLayout2d::rescale_geodesic` (`--manifold != none` only).
    #[arg(long, value_name = "SCALE")]
    pub manifold_rescale_final: Option<f64>,
    /// Optional community/cluster labels file (JSON or plain-text/CSV integer list) for `gib` and `--cluster-refine` (auto-clustered via Leiden when omitted).
    #[arg(long, value_name = "FILE")]
    pub labels: Option<PathBuf>,
    /// Apply post-layout 2D spatial intruder refinement (`SpatialIntruderRefiner`).
    #[arg(long, default_value_t = false)]
    pub refine: bool,
    /// Post-layout cluster refinement method (`none`, `kmeans`, `hac`).
    #[arg(long, value_enum, default_value_t = ClusterRefineArg::None)]
    pub cluster_refine: ClusterRefineArg,
    /// Apply post-layout PRISM overlap removal (`PrismOverlapRemoval`, alias for `--overlap-removal prism`).
    #[arg(long, default_value_t = false, conflicts_with = "overlap_removal")]
    pub prism: bool,
    /// Post-layout node overlap removal method (`none`, `prism`, `gtree`, `forbid`, `forbid-prime`, `sparse-forbid`, `pfs-prime`, `twin-declutter`; default: `none`).
    #[arg(long, value_enum)]
    pub overlap_removal: Option<OverlapRemovalArg>,
    /// Uniform node half-extent radius used when overlap removal (`--overlap-removal` or `--prism`) is enabled.
    #[arg(long, visible_alias = "overlap-radius", default_value_t = 0.05)]
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
    /// Input 2D, N-D, or manifold coordinates JSON file.
    #[arg(long, value_name = "FILE")]
    pub layout: PathBuf,
    /// Optional reference 2D coordinates JSON file required for `prism-distortion` and `nor-metrics`.
    #[arg(long, value_name = "FILE")]
    pub original_layout: Option<PathBuf>,
    /// Optional labels file (JSON `{"labels": [...]}` / `{"community_labels": [...]}` / `[...]` or plain-text/CSV integer list), required for `cluster-area-faithfulness` (`gcaf`), `adjusted-patch-nig`, and label-dependent `eval-baselines` targets (`distance-consistency`, `distribution-consistency`, `class-density-measure`, `histogram-density-measure`, `sepme-*`, `neighborhood-hit`, `silhouette-score`, `calinski-harabasz-score`, `davies-bouldin-score`, `cq-*`, `pairwise-distance-overlap-error`; `sprawlter-nn` and `nor-metrics` do not require labels).
    #[arg(long, value_name = "FILE")]
    pub labels: Option<PathBuf>,
    /// Uniform node half-extent radius (`> 0.0`) used by `sprawlter-nn` and `nor-metrics` when per-node half-extents are omitted.
    #[arg(long, default_value_t = 1.0)]
    pub overlap_radius: f64,
    /// Optional per-node rectangle half-widths file (JSON array or plain-text/CSV float list) for `sprawlter-nn` and `nor-metrics`.
    #[arg(long, value_name = "FILE")]
    pub sprawlter_half_widths: Option<PathBuf>,
    /// Optional per-node rectangle half-heights file (JSON array or plain-text/CSV float list) for `sprawlter-nn` and `nor-metrics`.
    #[arg(long, value_name = "FILE")]
    pub sprawlter_half_heights: Option<PathBuf>,
    /// Comma-separated metric targets (`sns,sampled-sns,geodesic-distortion,crossings,crossing-edges,crossing-angle,sampled-crossings,crosslessness,sampled-crosslessness,torus-crossing-number,crossing-edges-torus,np1,np2,sampled-np1,sampled-np2,gabriel,node-resolution,drng,dgg,aspect-ratio,angular-resolution,minimum-angle,minimum-angle-a1,layout-degeneracy,ideal-edge-lengths,prism-distortion,adjusted-patch-nig,cluster-area-faithfulness,gcaf,rta,distance-consistency,distribution-consistency,class-density-measure,histogram-density-measure,sepme-knng,sepme-knng2,sepme-rng,sepme-gab,sepme-gong,neighborhood-hit,silhouette-score,calinski-harabasz-score,davies-bouldin-score,cq-kmeans-ari,cq-kmeans-ami,cq-hac-hom,cq-hac-ami,pairwise-distance-overlap-error,sprawlter-nn,nor-metrics,nor,all`; note `drng`/`dgg` compute Eades et al. 2017 unweighted `S_RNG`/`S_GG`, distinct from deferred Hong et al. 2022 `dRNG`/`dGG` and Fang et al. 2025 `dcRNG`/`dcGG`).
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
    /// Maximum G-CAF grid resolution along unit radius R95 (`>= 12`).
    #[arg(long, default_value_t = 128)]
    pub gcaf_g_max: usize,
    /// Gaussian splatting kernel standard deviation in grid cells for G-CAF.
    #[arg(long, default_value_t = 0.85)]
    pub gcaf_sigma_cells: f64,
    /// Over-compression / scatter-tolerance multiplier kappa for G-CAF.
    #[arg(long, default_value_t = 10.0)]
    pub gcaf_kappa: f64,
    /// Physical visibility threshold fraction tau_frac for G-CAF.
    #[arg(long, default_value_t = 0.18)]
    pub gcaf_tau_frac: f64,
    /// Radial evaluation boundary in units of R95 for G-CAF (`inf` allowed).
    #[arg(long, default_value_t = 2.0)]
    pub gcaf_boundary_r95: f64,
    /// Stevens power-law exponent gamma for G-CAF target cluster weights.
    #[arg(long, default_value_t = 1.0)]
    pub gcaf_stevens_gamma: f64,
    /// Include per-cluster G-CAF diagnostics (`gcaf_detail`) in JSON output.
    #[arg(long, default_value_t = false)]
    pub gcaf_detail: bool,
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
    /// Resolution parameter gamma for `leiden-mod` and `leiden-cpm`.
    #[arg(long, default_value_t = 1.0)]
    pub resolution: f64,
    /// Randomness (temperature) parameter for Leiden refinement (`> 0.0`).
    #[arg(long, default_value_t = 0.01)]
    pub randomness: f64,
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
    /// Embedding dimensionality (`2` for `Layout2d`, `3` for `Layout3d`, otherwise `LayoutNd`).
    #[arg(long, default_value_t = 2)]
    pub dims: usize,
    /// Non-Euclidean manifold geometry (`none`, `hyperbolic`, `hyperbolic-lorentz`, `spherical`, `torus`).
    #[arg(long, value_enum, default_value_t = ManifoldArg::None)]
    pub manifold: ManifoldArg,
    /// Physical geodesic scale factor for manifold layouts (`> 0.0`).
    #[arg(long, default_value_t = 1.0)]
    pub manifold_scale: f64,
    /// Physical rectangular torus aspect ratio `Lx / Ly` (`> 0.0`, used with `--manifold torus`).
    #[arg(long, default_value_t = 1.0)]
    pub torus_aspect: f64,
    /// Apply optimal torus boundary auto-panning after layout (`--manifold torus` only).
    #[arg(long, default_value_t = false)]
    pub torus_auto_pan: bool,
    /// Deterministic RNG seed.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    /// Optional output JSON file path (writes to stdout if omitted).
    #[arg(long, value_name = "FILE")]
    pub output: Option<PathBuf>,
}
