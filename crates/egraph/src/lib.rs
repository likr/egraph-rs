//! Unified single-threaded, pure-Rust facade for `egraph-rs`.
//!
//! `egraph` re-exports `egraph-core` unconditionally and exposes feature-gated
//! sub-crates (`spatial`, `linalg`, `neighborhood`, `layout`, `clustering`,
//! `metrics`, `dataset`, and optional `petgraph`) alongside a convenient
//! [`prelude`] module for Native Rust, Python, and WebAssembly consumers.
//!
//! By default (`default` features), `egraph` includes all pure-Rust graph
//! layout, clustering, metrics, and CSR dataset capabilities while remaining
//! free of `petgraph` and multi-threading dependencies. Enable the `petgraph`
//! or `full` feature when `petgraph` interoperability is required.

#![warn(missing_docs)]

pub use egraph_core as core;

#[cfg(feature = "spatial")]
pub use egraph_spatial as spatial;

#[cfg(feature = "linalg")]
pub use egraph_linalg as linalg;

#[cfg(feature = "neighborhood")]
pub use egraph_neighborhood as neighborhood;

#[cfg(feature = "layout")]
pub use egraph_layout as layout;

#[cfg(feature = "clustering")]
pub use egraph_clustering as clustering;

#[cfg(feature = "metrics")]
pub use egraph_metrics as metrics;

#[cfg(feature = "dataset")]
pub use egraph_dataset as dataset;

#[cfg(feature = "petgraph")]
pub use egraph_petgraph as petgraph;

/// Convenient re-exports of foundational types, solvers, clustering routines, and metrics.
pub mod prelude {
    pub use egraph_core::{
        CoreError, CoreResult, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, LayoutNd,
        ManifoldKind, ManifoldLayout2d, Scalar, SolverStatus, VisitStamp,
        poincare_einstein_midpoint, poincare_mobius_add,
    };

    #[cfg(feature = "delaunay")]
    pub use egraph_spatial::Delaunay2d;

    #[cfg(feature = "spatial")]
    pub use egraph_spatial::{
        ArenaQuadtree2d, Complex32, CrossingPair, CrossingSummary, DensityFieldKernel, FitField2d,
        FitFieldConfig, IbFft2d, IbFftConfig, InterpolationOrder, R2cFft2dWorkspaceF32, Rect2d,
        RectangleOverlapSweep, ScreenedPoissonConfig, ScreenedPoissonField2d,
        ScreenedPoissonOutput, ScreenedPoissonStats, StaticKdTree, SweepLineCrossings,
    };

    #[cfg(feature = "linalg")]
    pub use egraph_linalg::{
        AmgPcgBatchResult, AmgPcgConfig, AmgPcgSolveResult, AmgPcgSolver, AmgPcgWorkspace,
        AnisotropicLaplacianConfig, AnisotropicLaplacianResult, ChebyshevDogConfig,
        ChebyshevDogKernel, ChebyshevWorkspace, CsrMatrix, DEFAULT_SPECTRUM_LANCZOS_SAFETY_FACTOR,
        DEFAULT_SPECTRUM_LANCZOS_STEPS, DEFAULT_SPECTRUM_POWER_ITERS,
        DEFAULT_SPECTRUM_SAFETY_FACTOR, EffectiveResistanceConfig, EffectiveResistanceResult,
        LaplacianKind, PivotMdsConfig, RdMds, ResistanceCurvatureConfig,
        ResistanceCurvatureFlowResult, ResistanceCurvatureStep, SuggestedGeometry,
        adaptive_chebyshev_degree, adaptive_chebyshev_degree_scaled, anisotropic_laplacian,
        classical_mds_2d, classify_geometry_from_curvature, combinatorial_laplacian,
        compute_effective_resistances, koren_spectral_init, normalize_layout_2d, pivot_mds, rd_mds,
        representative_mds, resistance_curvature_flow, robust_normalize_layout_2d,
        select_maxmin_sp_pivots, symmetric_normalized_laplacian,
    };

    #[cfg(feature = "neighborhood")]
    pub use egraph_neighborhood::{
        DogVaradhanConfig, DogVaradhanLandmarks, DogVaradhanStats, HybridKnnConfig, KnnGraph,
        PerplexityConfig, RhghLandmarkConfig, RhghLandmarkStats, Sparse12HopPairs,
        SparseProbabilities, SymmetrizationMode, apply_relative_distance_floor,
        build_capped_12hop_knn, build_gumap_12hop_priority_csr, build_hybrid_adaptive_knn,
        build_pure_2hop_knn, build_rerank_knn, compute_diffusion_embedding,
        compute_fuzzy_simplicial_set, compute_pivoted_rhgh_landmarks,
        compute_regularized_dog_varadhan_landmarks,
        compute_regularized_dog_varadhan_landmarks_from_matrix, compute_resistance_embedding,
        extract_1hop_2hop_csr, extract_sparse_1hop_2hop_pairs,
        graph_12hop_row_normalized_probabilities,
        graph_12hop_row_normalized_probabilities_with_degree_decay,
        graph_row_normalized_probabilities, symmetrize_row_normalized,
    };

    #[cfg(feature = "layout")]
    pub use egraph_layout::{
        BarrierSymmetry, CoarseBackboneConfig, CoarseBackboneResult, CoarseBackboneStats,
        CoarseDistanceBackend, CoarseEdgeLengthMode, ConstraintProjector, ContinuousFdp,
        FdebConfig, FdebSolver, FieldLayout, FieldLayoutSolver, FieldRepulsionBackend,
        ForceAtlas2Config, ForceAtlas2Solver, GuMapConfig, GuMapSolver, IbUmapConfig,
        KamadaKawaiConfig, KamadaKawaiSolver, LayeringMethod, PacmapGraphConfig, PrismConfig,
        PrismOverlapRemoval, RhoSchedule, ScreenedPoissonRefiner, ScreenedPoissonRefinerConfig,
        SeparationConstraint, Sgd2Config, SgdScheduleConfig, SgdScheduleKind, SgdSolver,
        SnapTFdpConfig, SparseBarrierConfig, SparseBarrierStats, SpatialIntruderRefiner,
        SpatialRefinerConfig, StressMajorizationConfig, StressMajorizationSolver, TFdpConfig,
        TsNetBackend, TsNetConfig, TsNetSolver, TsNetStage, YifanHuConfig, YifanHuSolver,
        assign_layers, coarse_backbone_init, cycle_edges, generate_layered_constraints,
        project_clustered_rectangle_no_overlap_2d, project_rectangle_no_overlap_2d, remove_cycles,
    };

    #[cfg(all(feature = "layout", feature = "baselines"))]
    pub use egraph_layout::{
        BackboneConfig, BackboneLayoutMode, BackboneResult, BackboneSolver, ClusterRefineConfig,
        ClusterRefineMethod, ClusterRefineSolver, ForbidConfig, ForbidOverlapRemoval,
        ForbidVariant, FrBackend, FruchtermanReingoldConfig, FruchtermanReingoldSolver,
        GTreeConfig, GTreeOverlapRemoval, GibBox, GibBoxLayoutMode, GibIntraLayoutMode,
        GroupInABoxConfig, GroupInABoxSolver, LinLogConfig, LinLogRepulsion, LinLogSolver,
        extract_quadrilateral_backbone, refine_cluster_layout, squarified_treemap,
    };

    #[cfg(feature = "clustering")]
    pub use egraph_clustering::{
        CoarseLouvainConfig, CoarsenedGraph, CoarsenedHierarchy, CoarsenedHierarchyLevel,
        KMeansConfig, KMeansResult, LabelPropagationConfig, LeidenConfig, LeidenObjective,
        LeidenResult, LouvainConfig, LouvainResult, SpectralClusteringConfig,
        align_clustered_centroids, coarsen_graph, kmeans_plus_plus, label_propagation, leiden,
        louvain, louvain_coarsen, project_clustered_layout, project_clustered_layout_with_radii,
        spectral_clustering,
    };

    #[cfg(feature = "metrics")]
    pub use egraph_metrics::{
        ClusterAreaConfig, ClusterAreaDetail, ClusterAreaReport, LayoutDegeneracy, MetricConfig,
        MetricReport, MetricTarget, MinimumAngleSummary, RtaEvaluation, adjusted_patch_nig,
        adjusted_patch_nig_multiscale, all_pairs_bfs, bfs_single_source, cluster_area_faithfulness,
        cluster_area_faithfulness_detail, cluster_area_faithfulness_detail_with_weights,
        cluster_area_faithfulness_with_weights, crossing_edges, crossing_edges_torus,
        crossing_number, crossing_summary, crosslessness, crosslessness_from_count,
        evaluate_metrics, evaluate_metrics_with_communities,
        evaluate_metrics_with_community_weights, gabriel_graph_edges, ideal_edge_lengths,
        ideal_edge_lengths_with_unit, landmark_stratified_rta, landmark_stratified_rta_detail,
        landmark_stratified_rta_from_landmarks, layout_degeneracy, max_crossing_pairs,
        minimum_angle, minimum_angle_summary, neighborhood_preservation_1hop,
        neighborhood_preservation_2hop, prism_distortion_with_edges, relative_neighborhood_edges,
        sampled_crossing_number, sampled_crosslessness, sampled_neighborhood_preservation_1hop,
        sampled_neighborhood_preservation_2hop, sampled_scale_normalized_stress,
        scale_normalized_stress, scale_normalized_stress_from_distances,
    };

    #[cfg(all(feature = "metrics", feature = "eval-baselines"))]
    pub use egraph_metrics::{
        ClusterBaselineConfig, NorMetrics, PartitionAgreement, calinski_harabasz_score,
        class_density_measure, cq_hac, cq_kmeans, davies_bouldin_score, distance_consistency,
        distribution_consistency, neighborhood_hit, nor_metrics, nor_metrics_without_graph,
        pairwise_distance_overlap_error, partition_agreement, sepme_gab, sepme_gong, sepme_knng,
        sepme_rng, silhouette_score, sprawlter_nn, sprawlter_nn_radii,
    };

    #[cfg(feature = "dataset")]
    pub use egraph_dataset::parse_csr_csv;

    #[cfg(feature = "petgraph")]
    pub use egraph_petgraph::{
        from_petgraph_digraph, from_petgraph_ungraph, to_petgraph_digraph, to_petgraph_ungraph,
    };
}
