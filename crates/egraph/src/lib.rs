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
/// Single-threaded linear algebra, spectral diffusion, MDS, and AMG-PCG solvers.
pub mod linalg {
    pub use egraph_linalg::{
        AnisotropicLaplacianConfig, AnisotropicLaplacianResult, BfsWorkspace, ChebyshevDogConfig,
        ChebyshevDogKernel, ChebyshevWorkspace, CsrMatrix, DEFAULT_SPECTRUM_LANCZOS_SAFETY_FACTOR,
        DEFAULT_SPECTRUM_LANCZOS_STEPS, DEFAULT_SPECTRUM_POWER_ITERS,
        DEFAULT_SPECTRUM_SAFETY_FACTOR, DijkstraWorkspace, EigenResult, LanczosEigen,
        LaplacianKind, SpectralMomentStats, SubspaceEigen, adaptive_chebyshev_degree,
        adaptive_chebyshev_degree_scaled, all_pairs_dijkstra, all_pairs_dijkstra_into,
        anisotropic_laplacian, build_dog_terms, build_regularized_dog_terms, chebyshev,
        combinatorial_laplacian, compute_chebyshev_bandpass_coefficients,
        compute_chebyshev_sqrt_filter_coefficients, crate_version, csr_matrix,
        dijkstra_single_source, eigen, exp_scaled_bessel_i, exp_scaled_bessel_i_all, laplacian,
        project_psd_spectral_filter, random_walk_laplacian, select_maxmin_sp_pivots,
        shifted_normalized_laplacian, shortest_path, smallest_sym_laplacian_eigenpairs,
        spectral_moment_stats, symmetric_eigen_jacobi, symmetric_normalized_laplacian,
    };
    pub use egraph_linalg_amg::{
        AmgPcgBatchResult, AmgPcgConfig, AmgPcgSolveResult, AmgPcgSolver, AmgPcgWorkspace,
        EffectiveResistanceConfig, EffectiveResistanceResult, ResistanceCurvatureConfig,
        ResistanceCurvatureFlowResult, ResistanceCurvatureStep, SuggestedGeometry, amg_pcg,
        classify_geometry_from_curvature, classify_geometry_from_curvature_with_sampling,
        compute_effective_resistances, resistance_curvature, resistance_curvature_flow,
    };
    pub use egraph_linalg_mds::{
        HdeConfig, PivotMdsConfig, RdMds, classical_mds, classical_mds_2d,
        classical_mds_from_distances, hde, hde_nd, hde_nd_with_pivots, hde_with_pivots,
        koren_spectral_init, mds, normalize_layout_2d, pivot_mds, pivot_mds_from_distances,
        pivot_mds_with_pivots, rd_mds, representative_mds, robust_normalize_layout_2d,
    };
}

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
        ArenaOctree3d, ArenaQuadtree2d, Complex32, CrossingPair, CrossingSummary,
        DensityFieldKernel, DynamicKdTree, FitField2d, FitFieldConfig, IbFft2d, IbFftConfig,
        InterpolationOrder, R2cFft2dWorkspaceF32, Rect2d, RectangleOverlapSweep,
        ScreenedPoissonConfig, ScreenedPoissonField2d, ScreenedPoissonOutput, ScreenedPoissonStats,
        StaticKdTree, SweepLineCrossings,
    };

    #[cfg(feature = "linalg")]
    pub use egraph_linalg::{
        AnisotropicLaplacianConfig, AnisotropicLaplacianResult, BfsWorkspace, ChebyshevDogConfig,
        ChebyshevDogKernel, ChebyshevWorkspace, CsrMatrix, DEFAULT_SPECTRUM_LANCZOS_SAFETY_FACTOR,
        DEFAULT_SPECTRUM_LANCZOS_STEPS, DEFAULT_SPECTRUM_POWER_ITERS,
        DEFAULT_SPECTRUM_SAFETY_FACTOR, DijkstraWorkspace, EigenResult, LaplacianKind,
        SpectralMomentStats, adaptive_chebyshev_degree, adaptive_chebyshev_degree_scaled,
        all_pairs_dijkstra, all_pairs_dijkstra_into, anisotropic_laplacian,
        build_regularized_dog_terms, combinatorial_laplacian, dijkstra_single_source,
        random_walk_laplacian, select_maxmin_sp_pivots, shifted_normalized_laplacian,
        smallest_sym_laplacian_eigenpairs, spectral_moment_stats, symmetric_eigen_jacobi,
        symmetric_normalized_laplacian,
    };

    #[cfg(feature = "linalg")]
    pub use egraph_linalg_amg::{
        AmgPcgBatchResult, AmgPcgConfig, AmgPcgSolveResult, AmgPcgSolver, AmgPcgWorkspace,
        EffectiveResistanceConfig, EffectiveResistanceResult, ResistanceCurvatureConfig,
        ResistanceCurvatureFlowResult, ResistanceCurvatureStep, SuggestedGeometry,
        classify_geometry_from_curvature, classify_geometry_from_curvature_with_sampling,
        compute_effective_resistances, resistance_curvature_flow,
    };

    #[cfg(feature = "linalg")]
    pub use egraph_linalg_mds::{
        HdeConfig, PivotMdsConfig, RdMds, classical_mds, classical_mds_2d,
        classical_mds_from_distances, hde, hde_nd, hde_nd_with_pivots, hde_with_pivots,
        koren_spectral_init, normalize_layout_2d, pivot_mds, pivot_mds_from_distances,
        pivot_mds_with_pivots, rd_mds, representative_mds, robust_normalize_layout_2d,
    };

    #[cfg(feature = "neighborhood")]
    pub use egraph_neighborhood::{
        DogVaradhanConfig, DogVaradhanLandmarks, DogVaradhanStats, HybridKnnConfig, KnnGraph,
        PerplexityConfig, RhghLandmarkConfig, RhghLandmarkStats, RhghLandmarks, Sparse12HopPairs,
        SparseProbabilities, SymmetrizationMode, TwoHopSamplingMode, apply_relative_distance_floor,
        build_capped_12hop_knn, build_gumap_12hop_priority_csr, build_hybrid_adaptive_knn,
        build_pure_2hop_knn, build_rerank_knn, compute_diffusion_embedding,
        compute_fuzzy_simplicial_set, compute_pivoted_rhgh_landmarks,
        compute_pivoted_rhgh_landmarks_result, compute_regularized_dog_varadhan_landmarks,
        compute_regularized_dog_varadhan_landmarks_from_matrix, compute_resistance_embedding,
        extract_1hop_2hop_csr, extract_sparse_1hop_2hop_pairs,
        extract_sparse_1hop_2hop_pairs_with_mode, graph_12hop_row_normalized_probabilities,
        graph_12hop_row_normalized_probabilities_with_degree_decay,
        graph_12hop_row_normalized_probabilities_with_options, graph_row_normalized_probabilities,
        symmetrize_row_normalized,
    };

    #[cfg(feature = "layout")]
    pub use egraph_layout::{
        BarrierSymmetry, CoarseBackboneConfig, CoarseBackboneResult, CoarseBackboneStats,
        CoarseDistanceBackend, CoarseEdgeLengthMode, ConstraintProjector, ContinuousFdp,
        DEFAULT_MAX_2HOP_PER_NODE, DrGraphConfig, FdebConfig, FdebSolver, FieldFftStats,
        FieldLayout, FieldLayoutSolver, FieldRepulsionBackend, ForceAtlas2Config,
        ForceAtlas2Solver, GraphCneConfig, GraphCneObjective, GuMapConfig, GuMapSolver,
        IbUmapConfig, KamadaKawaiConfig, KamadaKawaiSolver, LargeVisConfig, LayeringMethod,
        OrtmannPivotStrategy, OrtmannSparseStressConfig, PacmapGraphConfig, PrismConfig,
        PrismOverlapRemoval, RhoSchedule, ScreenedPoissonRefiner, ScreenedPoissonRefinerConfig,
        SeparationConstraint, Sgd2Config, SgdScheduleConfig, SgdScheduleKind, SgdSolver,
        SnapTFdpConfig, SparseBarrierConfig, SparseBarrierStats, SpatialIntruderRefiner,
        SpatialRefinerConfig, StressMajorizationConfig, StressMajorizationSolver, TFdpConfig,
        TsNetBackend, TsNetConfig, TsNetSolver, TsNetStage, TwinDeclutterConfig, TwinDeclutterer,
        YifanHuConfig, YifanHuSolver, assign_layers, coarse_backbone_init, cycle_edges,
        declutter_twins_2d, generate_layered_constraints,
        project_clustered_rectangle_no_overlap_2d, project_rectangle_no_overlap_2d, remove_cycles,
        transitive_reduction,
    };

    #[cfg(all(feature = "layout", feature = "baselines"))]
    pub use egraph_layout::{
        BackboneConfig, BackboneLayoutMode, BackboneResult, BackboneSolver, ClusterRefineConfig,
        ClusterRefineMethod, ClusterRefineSolver, ForbidConfig, ForbidOverlapRemoval,
        ForbidVariant, FrBackend, FruchtermanReingoldConfig, FruchtermanReingoldSolver,
        GTreeConfig, GTreeOverlapRemoval, GemConfig, GemSolver, GibBox, GibBoxLayoutMode,
        GibIntraLayoutMode, GripConfig, GripSolver, GroupInABoxConfig, GroupInABoxSolver,
        LinLogConfig, LinLogRepulsion, LinLogSolver, MaxentStressConfig, MaxentStressSolver,
        PfsPrimeConfig, PfsPrimeOverlapRemoval, SelConfig, SelLayoutBackend, SelLayoutSolver,
        SelWeightMapping, SfcConfig, SfcCurveKind, SfcLayoutSolver, extract_quadrilateral_backbone,
        refine_cluster_layout, squarified_treemap,
    };

    #[cfg(feature = "clustering")]
    pub use egraph_clustering::{
        CoarseLouvainConfig, CoarsenedGraph, CoarsenedHierarchy, CoarsenedHierarchyLevel,
        KMeansConfig, KMeansResult, LabelPropagationConfig, LabelPropagationResult, LeidenConfig,
        LeidenObjective, LeidenResult, LouvainConfig, LouvainResult, SpectralClusteringConfig,
        SpectralClusteringResult, align_clustered_centroids, coarsen_graph, kmeans_plus_plus,
        label_propagation, leiden, louvain, louvain_coarsen, project_clustered_layout,
        project_clustered_layout_with_radii, spectral_clustering,
    };

    #[cfg(feature = "metrics")]
    pub use egraph_metrics::{
        AdjustedPatchNigDetail, BfsScratch, ClusterAreaConfig, ClusterAreaDetail,
        ClusterAreaReport, LayoutDegeneracy, MetricConfig, MetricInputs, MetricLayout,
        MetricReport, MetricTarget, MinimumAngleSummary, PrismDistortion, RtaEvaluation,
        StressEvaluation, TorusCrossing, adjusted_patch_nig, adjusted_patch_nig_detail,
        adjusted_patch_nig_multiscale, all_pairs_bfs, all_pairs_bfs_into, angular_resolution,
        aspect_ratio, bfs_single_source, cluster_area_faithfulness,
        cluster_area_faithfulness_detail, cluster_area_faithfulness_detail_with_weights,
        cluster_area_faithfulness_with_weights, crossing_angle, crossing_angle_from_pairs,
        crossing_edges, crossing_edges_torus, crossing_edges_torus_from_edges, crossing_number,
        crossing_number_from_pairs, crossing_summary, crosslessness, crosslessness_from_count,
        edge_angle, evaluate_metrics, evaluate_metrics_nd, evaluate_metrics_nd_with_communities,
        evaluate_metrics_nd_with_community_weights, evaluate_metrics_nd_with_inputs,
        evaluate_metrics_with_communities, evaluate_metrics_with_community_weights,
        evaluate_metrics_with_inputs, gabriel_graph_edges, gabriel_graph_edges_nd,
        gabriel_graph_preservation, gabriel_graph_preservation_nd, gabriel_graph_property,
        gabriel_graph_property_nd, ideal_edge_lengths, ideal_edge_lengths_from_distances,
        ideal_edge_lengths_from_distances_nd, ideal_edge_lengths_nd, ideal_edge_lengths_scaled,
        ideal_edge_lengths_scaled_nd, ideal_edge_lengths_with_unit,
        ideal_edge_lengths_with_unit_nd, landmark_stratified_rta, landmark_stratified_rta_detail,
        landmark_stratified_rta_detail_nd, landmark_stratified_rta_from_landmarks,
        landmark_stratified_rta_from_landmarks_f64, landmark_stratified_rta_from_landmarks_f64_nd,
        landmark_stratified_rta_from_landmarks_nd, landmark_stratified_rta_nd, layout_degeneracy,
        layout_degeneracy_nd, max_crossing_pairs, minimum_angle, minimum_angle_summary,
        neighborhood_preservation_1hop, neighborhood_preservation_1hop_nd,
        neighborhood_preservation_2hop, neighborhood_preservation_2hop_nd, node_resolution,
        node_resolution_nd, prism_distortion, prism_distortion_with_edges,
        relative_neighborhood_edges, relative_neighborhood_edges_nd,
        relative_neighborhood_preservation, relative_neighborhood_preservation_nd,
        sampled_crossing_number, sampled_crosslessness, sampled_neighborhood_preservation_1hop,
        sampled_neighborhood_preservation_1hop_nd, sampled_neighborhood_preservation_2hop,
        sampled_neighborhood_preservation_2hop_nd, sampled_scale_normalized_stress,
        sampled_scale_normalized_stress_detail, sampled_scale_normalized_stress_detail_nd,
        sampled_scale_normalized_stress_nd, scale_normalized_stress,
        scale_normalized_stress_detail, scale_normalized_stress_detail_nd,
        scale_normalized_stress_from_distances, scale_normalized_stress_from_distances_detail,
        scale_normalized_stress_from_distances_detail_nd,
        scale_normalized_stress_from_distances_nd, scale_normalized_stress_from_pairs,
        scale_normalized_stress_from_pairs_nd, scale_normalized_stress_nd,
    };

    #[cfg(all(feature = "metrics", feature = "eval-baselines"))]
    pub use egraph_metrics::{
        ClusterBaselineConfig, NorMetrics, PartitionAgreement, calinski_harabasz_score,
        calinski_harabasz_score_nd, class_density_measure, cq_hac, cq_hac_nd, cq_kmeans,
        cq_kmeans_nd, davies_bouldin_score, davies_bouldin_score_nd, distance_consistency,
        distance_consistency_nd, distribution_consistency, distribution_consistency_nd,
        histogram_density_measure, neighborhood_hit, neighborhood_hit_nd, nor_metrics,
        nor_metrics_without_graph, pairwise_distance_overlap_error,
        pairwise_distance_overlap_error_nd, partition_agreement, sepme_gab, sepme_gab_nd,
        sepme_gong, sepme_gong_nd, sepme_knng, sepme_knng_nd, sepme_rng, sepme_rng_nd,
        silhouette_score, silhouette_score_nd, sprawlter_nn, sprawlter_nn_radii,
    };

    #[cfg(feature = "dataset")]
    pub use egraph_dataset::{
        DatasetInfo, dataset_catalog, dataset_info, generate_hyperbolic_tree, generate_layered_dag,
        generate_sbm_graph, generate_spherical_grid, generate_spherical_icosphere,
        generate_torus_grid, generate_weighted_graph, list_datasets, parse_csr_csv,
    };

    #[cfg(feature = "petgraph")]
    pub use egraph_petgraph::{
        from_petgraph_digraph, from_petgraph_ungraph, to_petgraph_digraph, to_petgraph_ungraph,
    };
}
