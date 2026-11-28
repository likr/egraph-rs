//! Single-threaded linear algebra, spectral diffusion, MDS, and AMG-PCG solvers.

pub use egraph_linalg::{
    AnisotropicLaplacianConfig, AnisotropicLaplacianResult, BfsWorkspace, ChebyshevDogConfig,
    ChebyshevDogKernel, ChebyshevWorkspace, CsrMatrix, DEFAULT_SPECTRUM_LANCZOS_SAFETY_FACTOR,
    DEFAULT_SPECTRUM_LANCZOS_STEPS, DEFAULT_SPECTRUM_POWER_ITERS, DEFAULT_SPECTRUM_SAFETY_FACTOR,
    DijkstraWorkspace, EigenResult, FnOperator, LanczosEigen, LaplacianKind, LinearOperator,
    SpectralMomentStats, SubspaceEigen, adaptive_chebyshev_degree,
    adaptive_chebyshev_degree_scaled, all_pairs_dijkstra, all_pairs_dijkstra_into,
    anisotropic_laplacian, build_dog_terms, build_regularized_dog_terms, combinatorial_laplacian,
    compute_chebyshev_bandpass_coefficients, compute_chebyshev_sqrt_filter_coefficients,
    dijkstra_single_source, effective_chebyshev_len, estimate_lambda_max_operator,
    exp_scaled_bessel_i, exp_scaled_bessel_i_all, laplacian, project_psd_spectral_filter,
    random_walk_laplacian, select_maxmin_sp_pivots, shifted_normalized_laplacian,
    smallest_eigenpairs_from_inverse_operator, smallest_sym_laplacian_eigenpairs,
    smallest_sym_laplacian_eigenpairs_with_config, spectral_moment_stats, symmetric_eigen_jacobi,
    symmetric_normalized_laplacian,
};
pub use egraph_linalg_amg::{
    AmgPcgBatchResult, AmgPcgConfig, AmgPcgSolveResult, AmgPcgSolver, AmgPcgWorkspace,
    EffectiveResistanceConfig, EffectiveResistanceResult, ResistanceCurvatureConfig,
    ResistanceCurvatureFlowResult, ResistanceCurvatureStep, SuggestedGeometry,
    classify_geometry_from_curvature, classify_geometry_from_curvature_with_sampling,
    compute_effective_resistances, compute_effective_resistances_with_operator,
    resistance_curvature_flow,
};
pub use egraph_linalg_mds::{
    ClassicalMds, ClassicalMdsConfig, Hde, HdeConfig, KorenSpectral, KorenSpectralConfig, PivotMds,
    PivotMdsConfig, RdMds, RdMdsConfig, normalize_euclidean_layout, normalize_layout_2d,
    robust_normalize_euclidean_layout, robust_normalize_layout_2d,
};
