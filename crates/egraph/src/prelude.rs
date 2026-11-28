//! Curated foundational types, traits, solvers, and evaluators for `egraph`.

pub use egraph_core::{
    CoreError, CoreResult, CsrGraph, CsrGraphBuilder, EdgePolylines, EuclideanLayout,
    IterativeSolver, Layout2d, Layout3d, LayoutGenerator, LayoutNd, LayoutStorage, LayoutTransform,
    ManifoldKind, ManifoldLayout2d, ManifoldMetric, MetricSpace, NodeExtents2d, NodeShape2d,
    PinMask, RepulsionBackend, RiemannianLayout, SolverStatus,
};

#[cfg(feature = "spatial")]
pub use crate::spatial::{
    ArenaOctree3d, ArenaQuadtree2d, NearestNeighborIndex, SpatialIndex, TorusView,
};

#[cfg(feature = "edge-routing")]
pub use egraph_edge_routing::{FdebConfig, FdebSolver};

#[cfg(feature = "linalg")]
pub use crate::linalg::{
    AmgPcgConfig, AmgPcgSolver, ClassicalMds, ClassicalMdsConfig, Hde, HdeConfig, KorenSpectral,
    KorenSpectralConfig, LinearOperator, PivotMds, PivotMdsConfig, RdMds, RdMdsConfig,
};

#[cfg(feature = "neighborhood")]
pub use egraph_neighborhood::{KnnGraph, PerplexityConfig, Sparse12HopPairs, SparseProbabilities};

#[cfg(feature = "clustering")]
pub use egraph_clustering::{
    ClusterResult, Clusterer, CommunityDetector, KMeansConfig, KMeansPlusPlus, Leiden,
    LeidenConfig, Louvain, LouvainConfig,
};

#[cfg(feature = "layout")]
pub use crate::layout::{
    Auto2d, Auto3d, CoarseBackbone, CoarseBackboneConfig, ConstraintProjector, DefaultForceBackend,
    DefaultIbUmapBackend, DefaultStressStrategy, DefaultTFdpBackend, DefaultTsNetBackend,
    DrGraphConfig, DrGraphSolver, EuclideanPcg, FitFft, ForceAtlas2Config, ForceAtlas2Solver,
    ForceBackend, ForceRepulsion, GeodesicGradient, GraphCneConfig, GraphCneSolver, GuMapConfig,
    GuMapSolver, IbFft, IbUmapBackend2d, IbUmapConfig, IbUmapRepulsion, IbUmapSolver,
    KamadaKawaiConfig, KamadaKawaiSolver, LargeVisConfig, LargeVisSolver, LinearFft,
    PacmapGraphConfig, PacmapGraphSolver, PrismConfig, PrismOverlapRemoval, ScreenedPoissonRefiner,
    ScreenedPoissonRefinerConfig, SeparationConstraint, Sgd2Config, Sgd2Solver, SgdPairBuffer,
    SgdScheduleConfig, SgdSolver, SnapTFdpConfig, SnapTFdpSolver, SparseBarrierConfig,
    SpatialIntruderRefiner, SpatialRefinerConfig, StressMajorizationConfig,
    StressMajorizationSolver, StressMajorizationStrategy, TFdpBackend2d, TFdpBackend3d, TFdpConfig,
    TFdpRepulsion, TFdpSolver, TsNetBackend2d, TsNetBackend3d, TsNetConfig, TsNetRepulsion,
    TsNetSolver, TsNetStage, TwinDeclutterConfig, TwinDeclutterer, YifanHuConfig, YifanHuRepulsion,
    YifanHuSolver, pacmap_graph_with_backbone,
};

#[cfg(all(feature = "layout", feature = "baselines"))]
pub use crate::layout::{
    BackboneConfig, BackboneSolver, ClusterRefineConfig, ClusterRefineSolver, DefaultFrBackend,
    DefaultMaxentBackend, ForbidConfig, ForbidOverlapRemoval, FrBackend, FrRepulsion,
    FruchtermanReingoldConfig, FruchtermanReingoldSolver, GTreeConfig, GTreeOverlapRemoval,
    GemConfig, GemSolver, Grid, GripConfig, GripSolver, GroupInABoxConfig, GroupInABoxSolver,
    LinLogConfig, LinLogRepulsionBackend, LinLogSolver, MaxentRepulsion, MaxentStressConfig,
    MaxentStressSolver, PfsPrimeConfig, PfsPrimeOverlapRemoval, SelConfig, SelLayoutSolver,
    SfcConfig, SfcLayout,
};

#[cfg(feature = "metrics")]
pub use egraph_metrics::{
    ClusterBaselineConfig, ClusterBaselineReport, MetricConfig, MetricInputs, MetricReport,
    MetricTarget, evaluate_metrics, evaluate_metrics_with_communities,
    evaluate_metrics_with_community_weights, evaluate_metrics_with_inputs,
};

#[cfg(feature = "dataset")]
pub use egraph_dataset::{DatasetInfo, dataset_catalog, dataset_info, list_datasets};
