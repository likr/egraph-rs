use egraph::{core as egraph_core, layout as egraph_layout, spatial as egraph_spatial};
use egraph_core::SolverStatus;
use egraph_layout::{
    BackboneLayoutMode, BarrierSymmetry, ClusterRefineMethod, CoarseDistanceBackend,
    CoarseEdgeLengthMode, ForbidVariant, FrBackend, GibBoxLayoutMode, GibIntraLayoutMode,
    GraphCneObjective, LinLogRepulsion, OrtmannPivotStrategy, SelLayoutBackend, SelWeightMapping,
    SfcCurveKind, SgdScheduleKind,
};
use egraph_spatial::DensityFieldKernel;

use crate::ffi_convert::solver_status_str;

use super::backbone::parse_backbone_layout_mode;
use super::cluster_refine::parse_cluster_refine_method;
use super::coarse_backbone::{parse_coarse_distance_backend, parse_coarse_edge_length_mode};
use super::force_baselines::{parse_sel_layout_backend, parse_sel_weight_mapping};
use super::force_classic::parse_linlog_repulsion;
use super::force_fr::{FrBackendChoice, parse_fr_backend, parse_fr_backend_choice};
use super::graph_cne::parse_graph_cne_objective;
use super::group_in_a_box::{parse_gib_box_mode, parse_gib_intra_mode};
use super::overlap_tree_forbid::parse_forbid_variant;
use super::screened_poisson_field::parse_density_field_kernel;
use super::sfc::parse_sfc_curve_kind;
use super::sgd_barrier::parse_barrier_symmetry;
use super::sgd_baselines::parse_ortmann_pivot_strategy;
use super::sgd_core::{default_inv_sq_weights, parse_sgd_schedule_kind};
use super::tsnet_config::TsNetBackendChoice;

#[test]
fn test_tsnet_and_sgd_enum_parsers_and_weights() {
    assert_eq!(
        TsNetBackendChoice::parse_str("auto").unwrap(),
        TsNetBackendChoice::Auto
    );
    assert_eq!(
        TsNetBackendChoice::parse_str("auto2d").unwrap(),
        TsNetBackendChoice::Auto2d
    );
    assert_eq!(
        TsNetBackendChoice::parse_str("auto3d").unwrap(),
        TsNetBackendChoice::Auto3d
    );
    assert_eq!(
        TsNetBackendChoice::parse_str("EXACT").unwrap(),
        TsNetBackendChoice::Exact
    );
    assert_eq!(
        TsNetBackendChoice::parse_str("barnes-hut").unwrap(),
        TsNetBackendChoice::BarnesHut
    );
    assert_eq!(
        TsNetBackendChoice::parse_str("bh").unwrap(),
        TsNetBackendChoice::BarnesHut
    );
    assert_eq!(
        TsNetBackendChoice::parse_str("fft_interpolation").unwrap(),
        TsNetBackendChoice::FitFft
    );
    assert_eq!(
        TsNetBackendChoice::parse_str("FIt").unwrap(),
        TsNetBackendChoice::FitFft
    );
    assert_eq!(
        TsNetBackendChoice::parse_str("linear_fft").unwrap(),
        TsNetBackendChoice::LinearFft
    );
    assert_eq!(
        TsNetBackendChoice::parse_str("L").unwrap(),
        TsNetBackendChoice::LinearFft
    );
    assert!(TsNetBackendChoice::parse_str("unknown").is_err());

    assert_eq!(
        parse_sgd_schedule_kind("EXPONENTIAL").unwrap(),
        SgdScheduleKind::Exponential
    );
    assert_eq!(
        parse_sgd_schedule_kind("linear").unwrap(),
        SgdScheduleKind::Linear
    );
    assert_eq!(
        parse_sgd_schedule_kind("Quadratic").unwrap(),
        SgdScheduleKind::Quadratic
    );
    assert_eq!(
        parse_sgd_schedule_kind("reciprocal").unwrap(),
        SgdScheduleKind::Reciprocal
    );
    assert_eq!(
        parse_sgd_schedule_kind("constant").unwrap(),
        SgdScheduleKind::Constant
    );
    assert_eq!(
        parse_sgd_schedule_kind("squared").unwrap(),
        SgdScheduleKind::Squared
    );
    assert!(parse_sgd_schedule_kind("cosine").is_err());

    assert_eq!(
        parse_barrier_symmetry("sym_max").unwrap(),
        BarrierSymmetry::SymMax
    );
    assert_eq!(
        parse_barrier_symmetry("SymAvg").unwrap(),
        BarrierSymmetry::SymAvg
    );
    assert!(parse_barrier_symmetry("sym_min").is_err());

    assert_eq!(solver_status_str(SolverStatus::Initialized), "initialized");
    assert_eq!(solver_status_str(SolverStatus::Running), "running");
    assert_eq!(solver_status_str(SolverStatus::Converged), "converged");
    assert_eq!(solver_status_str(SolverStatus::Completed), "completed");
    assert_eq!(solver_status_str(SolverStatus::Failed), "failed");

    let w = default_inv_sq_weights(&[2.0, 0.0, f64::NAN]);
    assert!((w[0] - 0.25).abs() < 1e-12);
    assert_eq!(w[1], 0.0);
    assert_eq!(w[2], 0.0);
}

#[test]
fn test_field_repulsion_backend_parser() {
    assert_eq!(
        super::t_fdp::TFdpBackendChoice::parse_str("ibfft").unwrap(),
        super::t_fdp::TFdpBackendChoice::IbFft
    );
    assert_eq!(
        super::t_fdp::TFdpBackendChoice::parse_str("IB_FFT").unwrap(),
        super::t_fdp::TFdpBackendChoice::IbFft
    );
    assert_eq!(
        super::t_fdp::TFdpBackendChoice::parse_str("fft").unwrap(),
        super::t_fdp::TFdpBackendChoice::IbFft
    );
    assert_eq!(
        super::t_fdp::TFdpBackendChoice::parse_str("barnes_hut").unwrap(),
        super::t_fdp::TFdpBackendChoice::BarnesHut
    );
    assert_eq!(
        super::t_fdp::TFdpBackendChoice::parse_str("bh").unwrap(),
        super::t_fdp::TFdpBackendChoice::BarnesHut
    );
    assert_eq!(
        super::t_fdp::TFdpBackendChoice::parse_str("EXACT").unwrap(),
        super::t_fdp::TFdpBackendChoice::Exact
    );
    assert!(super::t_fdp::TFdpBackendChoice::parse_str("quadtree").is_err());
}

#[test]
fn test_coarse_backbone_and_density_field_parsers() {
    assert_eq!(
        parse_coarse_edge_length_mode("conductance_inverse").unwrap(),
        CoarseEdgeLengthMode::ConductanceInverse
    );
    assert_eq!(
        parse_coarse_edge_length_mode("rcf").unwrap(),
        CoarseEdgeLengthMode::ResistanceCurvatureFlow
    );
    assert!(parse_coarse_edge_length_mode("invalid").is_err());

    assert_eq!(
        parse_coarse_distance_backend("rhgh").unwrap(),
        CoarseDistanceBackend::Rhgh
    );
    assert_eq!(
        parse_coarse_distance_backend("chebyshev_dog").unwrap(),
        CoarseDistanceBackend::ChebyshevDog
    );
    assert_eq!(
        parse_coarse_distance_backend("sp").unwrap(),
        CoarseDistanceBackend::ShortestPath
    );
    assert!(parse_coarse_distance_backend("unknown").is_err());

    assert_eq!(
        parse_density_field_kernel("screened_poisson").unwrap(),
        DensityFieldKernel::ScreenedPoisson
    );
    assert_eq!(
        parse_density_field_kernel("single_gaussian").unwrap(),
        DensityFieldKernel::SingleGaussian
    );
    assert!(parse_density_field_kernel("invalid_kernel").is_err());
}

#[test]
fn test_force_backbone_cluster_and_overlap_parsers() {
    assert_eq!(
        parse_linlog_repulsion("edge_repulsion").unwrap(),
        LinLogRepulsion::EdgeRepulsion
    );
    assert_eq!(
        parse_linlog_repulsion("node").unwrap(),
        LinLogRepulsion::NodeRepulsion
    );
    assert!(parse_linlog_repulsion("invalid").is_err());

    assert_eq!(parse_fr_backend("exact").unwrap(), FrBackend::exact());
    assert_eq!(
        parse_fr_backend("barnes-hut").unwrap(),
        FrBackend::barnes_hut(1.2)
    );
    assert_eq!(parse_fr_backend("bh").unwrap(), FrBackend::barnes_hut(1.2));
    assert_eq!(parse_fr_backend("GRID").unwrap(), FrBackend::grid());
    assert_eq!(
        parse_fr_backend_choice("auto").unwrap(),
        FrBackendChoice::Auto
    );
    assert!(parse_fr_backend("fft").is_err());

    assert_eq!(
        parse_backbone_layout_mode("stress").unwrap(),
        BackboneLayoutMode::StressMajorization
    );
    assert_eq!(
        parse_backbone_layout_mode("stress_majorization").unwrap(),
        BackboneLayoutMode::StressMajorization
    );
    assert_eq!(
        parse_backbone_layout_mode("sgd").unwrap(),
        BackboneLayoutMode::Sgd
    );
    assert!(parse_backbone_layout_mode("fr").is_err());

    assert_eq!(
        parse_cluster_refine_method("k-means").unwrap(),
        ClusterRefineMethod::Kmeans
    );
    assert_eq!(
        parse_cluster_refine_method("HAC").unwrap(),
        ClusterRefineMethod::Hac
    );
    assert!(parse_cluster_refine_method("spectral").is_err());

    assert_eq!(
        parse_gib_box_mode("treemap").unwrap(),
        GibBoxLayoutMode::SquarifiedTreemap
    );
    assert_eq!(
        parse_gib_box_mode("tr-gib").unwrap(),
        GibBoxLayoutMode::SquarifiedTreemap
    );
    assert_eq!(
        parse_gib_box_mode("force_directed").unwrap(),
        GibBoxLayoutMode::ForceDirected
    );
    assert_eq!(
        parse_gib_box_mode("fd_gib").unwrap(),
        GibBoxLayoutMode::ForceDirected
    );
    assert!(parse_gib_box_mode("circle").is_err());

    assert_eq!(parse_gib_intra_mode("fr").unwrap(), GibIntraLayoutMode::Fr);
    assert_eq!(
        parse_gib_intra_mode("sgd_full").unwrap(),
        GibIntraLayoutMode::SgdFull
    );
    assert_eq!(
        parse_gib_intra_mode("yifan_hu").unwrap(),
        GibIntraLayoutMode::YifanHu
    );
    assert!(parse_gib_intra_mode("tsnet").is_err());

    assert_eq!(
        parse_forbid_variant("forbid").unwrap(),
        ForbidVariant::Forbid
    );
    assert_eq!(
        parse_forbid_variant("forbid_prime").unwrap(),
        ForbidVariant::ForbidPrime
    );
    assert_eq!(
        parse_forbid_variant("forbid'").unwrap(),
        ForbidVariant::ForbidPrime
    );
    assert_eq!(
        parse_forbid_variant("sparse_forbid").unwrap(),
        ForbidVariant::SparseForbid
    );
    assert!(parse_forbid_variant("prism").is_err());
}

#[test]
fn test_baseline_and_sfc_solver_parsers() {
    assert_eq!(
        parse_ortmann_pivot_strategy("kmeans_sp").unwrap(),
        OrtmannPivotStrategy::KMeansSp
    );
    assert_eq!(
        parse_ortmann_pivot_strategy("kmeans").unwrap(),
        OrtmannPivotStrategy::KMeansSp
    );
    assert_eq!(
        parse_ortmann_pivot_strategy("maxmin-random-sp").unwrap(),
        OrtmannPivotStrategy::MaxMinRandomSp
    );
    assert_eq!(
        parse_ortmann_pivot_strategy("MaxMin_SP").unwrap(),
        OrtmannPivotStrategy::MaxMinSp
    );
    assert!(parse_ortmann_pivot_strategy("random").is_err());

    assert_eq!(
        parse_graph_cne_objective("infonce").unwrap(),
        GraphCneObjective::InfoNce
    );
    assert_eq!(
        parse_graph_cne_objective("InfoNCE").unwrap(),
        GraphCneObjective::InfoNce
    );
    assert_eq!(
        parse_graph_cne_objective("neg_tsne").unwrap(),
        GraphCneObjective::NegTSne
    );
    assert_eq!(
        parse_graph_cne_objective("neg-tsne").unwrap(),
        GraphCneObjective::NegTSne
    );
    assert!(parse_graph_cne_objective("umap").is_err());

    assert_eq!(
        parse_sel_weight_mapping("sigmoid").unwrap(),
        SelWeightMapping::Sigmoid
    );
    assert_eq!(
        parse_sel_weight_mapping("linear").unwrap(),
        SelWeightMapping::Linear
    );
    assert_eq!(
        parse_sel_weight_mapping("TANH").unwrap(),
        SelWeightMapping::Tanh
    );
    assert_eq!(
        parse_sel_weight_mapping("custom_logistic").unwrap(),
        SelWeightMapping::CustomLogistic
    );
    assert_eq!(
        parse_sel_weight_mapping("logistic").unwrap(),
        SelWeightMapping::CustomLogistic
    );
    assert_eq!(
        parse_sel_weight_mapping("binary").unwrap(),
        SelWeightMapping::Binary
    );
    assert!(parse_sel_weight_mapping("relu").is_err());

    assert_eq!(
        parse_sel_layout_backend("fr").unwrap(),
        SelLayoutBackend::Fr
    );
    assert_eq!(
        parse_sel_layout_backend("fruchterman_reingold").unwrap(),
        SelLayoutBackend::Fr
    );
    assert_eq!(
        parse_sel_layout_backend("t_fdp").unwrap(),
        SelLayoutBackend::TFdp
    );
    assert_eq!(
        parse_sel_layout_backend("t-fdp").unwrap(),
        SelLayoutBackend::TFdp
    );
    assert!(parse_sel_layout_backend("fa2").is_err());

    assert_eq!(
        parse_sfc_curve_kind("hilbert").unwrap(),
        SfcCurveKind::Hilbert
    );
    assert_eq!(
        parse_sfc_curve_kind("Morton").unwrap(),
        SfcCurveKind::Morton
    );
    assert_eq!(
        parse_sfc_curve_kind("z_order").unwrap(),
        SfcCurveKind::Morton
    );
    assert!(parse_sfc_curve_kind("peano").is_err());
}
