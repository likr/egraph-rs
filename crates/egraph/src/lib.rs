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
    };

    #[cfg(feature = "delaunay")]
    pub use egraph_spatial::Delaunay2d;

    #[cfg(feature = "spatial")]
    pub use egraph_spatial::{
        ArenaQuadtree2d, CrossingPair, CrossingSummary, FitField2d, FitFieldConfig, IbFft2d,
        IbFftConfig, InterpolationOrder, Rect2d, RectangleOverlapSweep, StaticKdTree,
        SweepLineCrossings,
    };

    #[cfg(feature = "linalg")]
    pub use egraph_linalg::{
        ChebyshevDogConfig, ChebyshevDogKernel, ChebyshevWorkspace, CsrMatrix, LaplacianKind,
        PivotMdsConfig, RdMds, classical_mds_2d, koren_spectral_init, normalize_layout_2d,
        pivot_mds, rd_mds, representative_mds, select_maxmin_sp_pivots,
    };

    #[cfg(feature = "neighborhood")]
    pub use egraph_neighborhood::{
        DogVaradhanConfig, DogVaradhanLandmarks, DogVaradhanStats, HybridKnnConfig, KnnGraph,
        PerplexityConfig, Sparse12HopPairs, SparseProbabilities, SymmetrizationMode,
        apply_relative_distance_floor, build_gumap_12hop_priority_csr, build_hybrid_adaptive_knn,
        build_pure_2hop_knn, build_rerank_knn, compute_diffusion_embedding,
        compute_fuzzy_simplicial_set, compute_regularized_dog_varadhan_landmarks,
        compute_resistance_embedding, extract_1hop_2hop_csr, extract_sparse_1hop_2hop_pairs,
        graph_12hop_row_normalized_probabilities, graph_row_normalized_probabilities,
        symmetrize_row_normalized,
    };

    #[cfg(feature = "layout")]
    pub use egraph_layout::{
        BarrierSymmetry, ConstraintProjector, ContinuousFdp, FdebConfig, FdebSolver, FieldLayout,
        FieldLayoutSolver, FieldRepulsionBackend, ForceAtlas2Config, ForceAtlas2Solver,
        GuMapConfig, GuMapSolver, IbUmapConfig, KamadaKawaiConfig, KamadaKawaiSolver,
        LayeringMethod, PacmapGraphConfig, PrismConfig, PrismOverlapRemoval, RhoSchedule,
        SeparationConstraint, Sgd2Config, SgdScheduleConfig, SgdScheduleKind, SgdSolver,
        SnapTFdpConfig, SparseBarrierConfig, SparseBarrierStats, SpatialIntruderRefiner,
        SpatialRefinerConfig, StressMajorizationConfig, StressMajorizationSolver, TFdpConfig,
        TsNetBackend, TsNetConfig, TsNetSolver, TsNetStage, YifanHuConfig, YifanHuSolver,
        assign_layers, cycle_edges, generate_layered_constraints,
        project_clustered_rectangle_no_overlap_2d, project_rectangle_no_overlap_2d, remove_cycles,
    };

    #[cfg(feature = "clustering")]
    pub use egraph_clustering::{
        CoarsenedGraph, KMeansConfig, KMeansResult, LabelPropagationConfig, LouvainConfig,
        LouvainResult, SpectralClusteringConfig, coarsen_graph, kmeans_plus_plus,
        label_propagation, louvain, project_clustered_layout, spectral_clustering,
    };

    #[cfg(feature = "metrics")]
    pub use egraph_metrics::{
        MetricConfig, MetricReport, MetricTarget, all_pairs_bfs, bfs_single_source, crossing_edges,
        crossing_edges_torus, crossing_number, crossing_summary, evaluate_metrics,
        gabriel_graph_edges, ideal_edge_lengths, ideal_edge_lengths_with_unit,
        neighborhood_preservation_1hop, neighborhood_preservation_2hop,
        prism_distortion_with_edges, relative_neighborhood_edges,
        sampled_neighborhood_preservation_1hop, sampled_neighborhood_preservation_2hop,
        scale_normalized_stress, scale_normalized_stress_from_distances,
    };

    #[cfg(feature = "dataset")]
    pub use egraph_dataset::parse_csr_csv;

    #[cfg(feature = "petgraph")]
    pub use egraph_petgraph::{
        from_petgraph_digraph, from_petgraph_ungraph, to_petgraph_digraph, to_petgraph_ungraph,
    };
}
