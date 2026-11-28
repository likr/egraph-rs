//! Single-threaded, pure-Rust graph layout solvers, constraints, and spatial refiners.
//!
//! Each domain crate is exposed as a submodule (`sgd`, `embedding`, `neighbor`, `stress`,
//! `force`, `constraints`, `multilevel`, `refine`, `composite`) alongside non-colliding
//! flat re-exports. Solvers with typed repulsion backends (`TsNetSolver`, `TFdpSolver`,
//! `IbUmapSolver`, `ForceAtlas2Solver`, `LinLogSolver`, `YifanHuSolver`,
//! `FruchtermanReingoldSolver`, `MaxentStressSolver`, `StressMajorizationSolver`) resolve their
//! default backend per space automatically and reject incompatible `(L, B)` pairings at compile
//! time:
//!
//! ```
//! use egraph::core::{CsrGraph, IterativeSolver, Layout3d, LayoutNd};
//! use egraph::layout::{
//!     ForceAtlas2Config, ForceAtlas2Solver, TsNetConfig, TsNetSolver,
//!     neighbor::BarnesHut as NeighborBarnesHut,
//! };
//!
//! let g = CsrGraph::<()>::from_unweighted_edges(3, &[(0, 1), (1, 2)], false, false).unwrap();
//! let mut fa2 =
//!     ForceAtlas2Solver::<LayoutNd<f64>>::new_unweighted(&g, ForceAtlas2Config::new()).unwrap();
//! let mut l_nd = LayoutNd::<f64>::init(3, 4, 42, 1.0).unwrap();
//! assert!(fa2.run(&mut l_nd).unwrap().is_finished());
//!
//! let mut tsnet = TsNetSolver::<Layout3d<f64>, NeighborBarnesHut>::new_with_backend(
//!     &g,
//!     TsNetConfig::pure_2hop(),
//!     NeighborBarnesHut::default(),
//! )
//! .unwrap();
//! let mut l_3d = Layout3d::<f64>::init(3, 42, 1.0).unwrap();
//! assert!(tsnet.run(&mut l_3d).unwrap().is_finished());
//! ```
//!
//! Incompatible `(L, B)` backend combinations are rejected at compile time:
//!
//! ```compile_fail
//! use egraph::core::{CsrGraph, LayoutNd};
//! use egraph::layout::{ForceAtlas2Config, ForceAtlas2Solver, force::BarnesHut};
//!
//! let g = CsrGraph::<()>::from_unweighted_edges(3, &[(0, 1), (1, 2)], false, false).unwrap();
//! let _solver = ForceAtlas2Solver::<LayoutNd<f64>, BarnesHut>::new_unweighted(
//!     &g,
//!     ForceAtlas2Config::new(),
//! );
//! ```
//!
//! ```compile_fail
//! use egraph::core::{CsrGraph, Layout3d};
//! use egraph::layout::{FitFft, TsNetConfig, TsNetSolver};
//!
//! let g = CsrGraph::<()>::from_unweighted_edges(3, &[(0, 1), (1, 2)], false, false).unwrap();
//! let _solver = TsNetSolver::<Layout3d<f64>, FitFft>::new(&g, TsNetConfig::pure_2hop());
//! ```

use egraph_core::{CoreResult, CsrGraph, Layout2d, LayoutGenerator, Scalar};

/// Composite multi-solver layout baselines (`BackboneSolver`, `SelLayoutSolver`, `GroupInABoxSolver`).
pub use egraph_layout_composite as composite;
/// Separation constraints, DAG layering, transitive reduction, and overlap removal.
pub use egraph_layout_constraints as constraints;
/// Sampling-based neighborhood graph embedding solvers (`PaCMAP-Graph`, `LargeVis`, `DRGraph`, `GraphCNE`, `SNAP-tFDP`).
pub use egraph_layout_embedding as embedding;
/// Force-directed graph layout solvers (`ForceAtlas2`, `YifanHu`, `LinLog`, `FruchtermanReingold`, `GEM`).
pub use egraph_layout_force as force;
/// Multilevel coarse backbone, `GRIP`, and space-filling curve layout generators.
pub use egraph_layout_multilevel as multilevel;
/// Neighborhood-preserving `tsNET`, `GuMap`, `t-FDP`, and `ibUMAP` solvers and typed repulsion backends.
pub use egraph_layout_neighbor as neighbor;
/// Post-layout spatial intruder, screened-Poisson, and cluster-faithful refinement solvers.
pub use egraph_layout_refine as refine;
/// Stochastic gradient descent stress and sparse barrier solvers (`SgdSolver`, `Sgd2Solver`, `SgdPairBuffer`).
pub use egraph_layout_sgd as sgd;
/// Stress majorization, Kamada-Kawai, and Maxent-Stress layout solvers.
pub use egraph_layout_stress as stress;

#[cfg(feature = "baselines")]
#[cfg_attr(docsrs, doc(cfg(feature = "baselines")))]
pub use egraph_layout_composite::{
    BackboneConfig, BackboneLayoutMode, BackboneResult, BackboneSolver, GibBox, GibBoxLayoutMode,
    GibIntraLayoutMode, GroupInABoxConfig, GroupInABoxSolver, SelConfig, SelLayoutBackend,
    SelLayoutSolver, SelWeightMapping, extract_quadrilateral_backbone,
    extract_quadrilateral_backbone_unweighted, squarified_treemap,
};
pub use egraph_layout_constraints::{
    ConstraintProjector, LayeringMethod, PrismConfig, PrismOverlapRemoval, SeparationConstraint,
    SparsePcgWorkspace, TwinDeclutterConfig, TwinDeclutterer, assign_layers, cycle_edges,
    declutter_twins_2d, generate_layered_constraints, project_clustered_rectangle_no_overlap_2d,
    project_rectangle_no_overlap_2d, remove_cycles, transitive_reduction,
    transitive_reduction_weighted,
};
#[cfg(feature = "baselines")]
#[cfg_attr(docsrs, doc(cfg(feature = "baselines")))]
pub use egraph_layout_constraints::{
    ForbidConfig, ForbidOverlapRemoval, ForbidVariant, GTreeConfig, GTreeOverlapRemoval,
    PfsPrimeConfig, PfsPrimeOverlapRemoval,
};
pub use egraph_layout_embedding::{
    DrGraphConfig, DrGraphSolver, GraphCneConfig, GraphCneObjective, GraphCneSolver,
    LargeVisConfig, LargeVisSolver, PacmapGraphConfig, PacmapGraphSolver, SnapTFdpConfig,
    SnapTFdpSolver,
};
pub use egraph_layout_force::{
    DefaultForceBackend, ForceAtlas2Config, ForceAtlas2Solver, ForceBackend, ForceRepulsion,
    ForceRepulsionParams, YifanHuConfig, YifanHuLevelCoords, YifanHuRepulsion, YifanHuSolver,
};
#[cfg(feature = "baselines")]
#[cfg_attr(docsrs, doc(cfg(feature = "baselines")))]
pub use egraph_layout_force::{
    DefaultFrBackend, FrBackend, FrRepulsion, FruchtermanReingoldConfig, FruchtermanReingoldSolver,
    GemConfig, GemSolver, Grid, LinLogConfig, LinLogRepulsion, LinLogRepulsionBackend,
    LinLogSolver,
};
pub use egraph_layout_multilevel::{
    CoarseBackbone, CoarseBackboneConfig, CoarseBackboneResult, CoarseBackboneStats,
    CoarseDistanceBackend, CoarseEdgeLengthMode,
};
#[cfg(feature = "baselines")]
#[cfg_attr(docsrs, doc(cfg(feature = "baselines")))]
pub use egraph_layout_multilevel::{GripConfig, GripSolver, SfcConfig, SfcCurveKind, SfcLayout};
pub use egraph_layout_neighbor::{
    Auto2d, Auto3d, DEFAULT_AUTO_EXACT_THRESHOLD, DEFAULT_IB_UMAP_REPULSIVE_EPS,
    DEFAULT_MAX_2HOP_PER_NODE, DefaultIbUmapBackend, DefaultTFdpBackend, DefaultTsNetBackend,
    FieldFftStats, FitFft, GuMapConfig, GuMapSolver, GuMapTarget, IbFft, IbUmapBackend2d,
    IbUmapConfig, IbUmapRepulsion, IbUmapSolver, LinearFft, RhoSchedule, TFdpBackend2d,
    TFdpBackend3d, TFdpConfig, TFdpRepulsion, TFdpSolver, TsNetBackend2d, TsNetBackend3d,
    TsNetConfig, TsNetRepulsion, TsNetSolver, TsNetStage,
};
#[cfg(feature = "baselines")]
#[cfg_attr(docsrs, doc(cfg(feature = "baselines")))]
pub use egraph_layout_refine::{
    ClusterRefineConfig, ClusterRefineMethod, ClusterRefineSolver, refine_cluster_layout,
};
pub use egraph_layout_refine::{
    ScreenedPoissonRefiner, ScreenedPoissonRefinerConfig, SpatialIntruderRefiner,
    SpatialRefinerConfig, SpatialRefinerRoundStats, SpatialRefinerStats,
};
pub use egraph_layout_sgd::{
    BarrierSymmetry, OrtmannPivotStrategy, OrtmannSparseProblem, OrtmannSparseStressConfig,
    Sgd2Config, Sgd2Solver, SgdPairBuffer, SgdPairKind, SgdPairRecord, SgdScheduleConfig,
    SgdScheduleKind, SgdSolver, SparseBarrierConfig, SparseBarrierStats, StreamingNegativeConfig,
    build_ortmann_sparse_problem,
};
#[cfg(feature = "baselines")]
#[cfg_attr(docsrs, doc(cfg(feature = "baselines")))]
pub use egraph_layout_stress::{
    DefaultMaxentBackend, MaxentRepulsion, MaxentStressConfig, MaxentStressSolver,
};
pub use egraph_layout_stress::{
    DefaultStressStrategy, EuclideanPcg, GeodesicGradient, KamadaKawaiConfig, KamadaKawaiSolver,
    PcgWorkspace, StressMajorizationConfig, StressMajorizationSolver, StressMajorizationStrategy,
};

/// Constructs a 3-tier `PaCMAP-Graph` solver ([`PacmapGraphSolver`]) coupled with a coarse
/// Louvain area-reserved backbone initialization ([`CoarseBackbone`]), returning
/// `(solver, initial_layout, backbone_result)`.
///
/// Validates `pacmap_cfg`, `backbone_cfg`, and graph weights before generating the coarse
/// backbone layout and building the `PaCMAP-Graph` tier pairs.
#[allow(clippy::type_complexity)]
pub fn pacmap_graph_with_backbone<S: Scalar, W: Scalar>(
    graph: &CsrGraph<W>,
    pacmap_cfg: PacmapGraphConfig,
    backbone_cfg: &CoarseBackboneConfig,
) -> CoreResult<(
    PacmapGraphSolver<Layout2d<S>>,
    Layout2d<S>,
    CoarseBackboneResult<S>,
)> {
    pacmap_cfg.validate()?;
    backbone_cfg.validate()?;
    let solver = PacmapGraphSolver::<Layout2d<S>>::new(graph, pacmap_cfg)?;
    let mut generator = CoarseBackbone::<S, W>::from_config(graph, backbone_cfg.clone());
    let backbone = generator.generate()?;
    let init_layout = backbone.fine_layout.clone();
    Ok((solver, init_layout, backbone))
}
