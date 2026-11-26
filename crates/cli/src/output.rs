use egraph::prelude::{ClusterAreaDetail, ClusterAreaReport, MetricReport};
use serde::Serialize;

use crate::args::{ClusterMethod, ClusterRefineArg, InitMethod, LayoutMethod, OverlapRemovalArg};

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
    pub cluster_refine: ClusterRefineArg,
    pub prism_applied: bool,
    pub prism_overlaps_remaining: Option<usize>,
    pub overlap_removal: OverlapRemovalArg,
    pub overlaps_remaining: Option<usize>,
    pub elapsed_ms: f64,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub coords: Vec<[f64; 2]>,
}

/// Per-cluster G-CAF diagnostics serialized when `--gcaf-detail` is enabled.
#[derive(Debug, Serialize)]
pub struct GcafDetailOutput {
    pub cluster_ids: Vec<u32>,
    pub counts: Vec<usize>,
    pub p: Vec<f64>,
    pub q: Vec<f64>,
    pub purity: Vec<f64>,
    pub resolution: Vec<f64>,
    pub res_density: Vec<f64>,
    pub res_deconv: Vec<f64>,
    pub caf_per_cluster: Vec<f64>,
    pub coherence: Vec<f64>,
}

impl From<ClusterAreaDetail> for GcafDetailOutput {
    fn from(detail: ClusterAreaDetail) -> Self {
        Self {
            cluster_ids: detail.cluster_ids,
            counts: detail.counts,
            p: detail.p,
            q: detail.q,
            purity: detail.purity,
            resolution: detail.resolution,
            res_density: detail.res_density,
            res_deconv: detail.res_deconv,
            caf_per_cluster: detail.caf_per_cluster,
            coherence: detail.coherence,
        }
    }
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
    /// Eades et al. (2017) unweighted shape-based `S_RNG` (distinct from deferred Hong et al. 2022 `dRNG` and Fang et al. 2025 `dcRNG`).
    pub relative_neighborhood_preservation: Option<f64>,
    /// Eades et al. (2017) unweighted shape-based `S_GG` (distinct from deferred Hong et al. 2022 `dGG` and Fang et al. 2025 `dcGG`).
    pub gabriel_graph_preservation: Option<f64>,
    pub aspect_ratio: Option<f64>,
    pub angular_resolution: Option<f64>,
    pub minimum_angle: Option<f64>,
    pub minimum_angle_a1: Option<f64>,
    pub layout_degeneracy: Option<f64>,
    pub ideal_edge_lengths: Option<f64>,
    pub adjusted_patch_nig: Option<f64>,
    pub cluster_area_faithfulness: Option<f64>,
    pub gcaf_purity_adj: Option<f64>,
    pub gcaf_prop_vis: Option<f64>,
    pub gcaf_res_density: Option<f64>,
    pub gcaf_res_deconv: Option<f64>,
    pub gcaf_res_combined: Option<f64>,
    pub gcaf_spatial_coherence: Option<f64>,
    pub gcaf_in_frac: Option<f64>,
    pub gcaf_num_clusters: Option<usize>,
    pub effective_boundary_r95: Option<f64>,
    pub g_eff: Option<usize>,
    pub gcaf_detail: Option<GcafDetailOutput>,
    pub landmark_stratified_rta: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dsc: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dc: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cdm: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hdm: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sepme_knng: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sepme_rng: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sepme_gab: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sepme_gong: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neighborhood_hit: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub silhouette: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calinski_harabasz: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub davies_bouldin: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cq_kmeans_ari: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cq_kmeans_ami: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cq_hac_hom: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cq_hac_ami: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pairwise_distance_overlap_error: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sprawlter_nn: Option<f64>,
}

impl MetricsOutput {
    pub(crate) fn from_report(
        source: String,
        num_nodes: usize,
        num_edges: usize,
        elapsed_ms: f64,
        report: MetricReport,
        gcaf_report: Option<ClusterAreaReport>,
        gcaf_detail: Option<ClusterAreaDetail>,
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
            adjusted_patch_nig: report.adjusted_patch_nig,
            cluster_area_faithfulness: gcaf_report
                .map(|r| r.caf)
                .or(report.cluster_area_faithfulness),
            gcaf_purity_adj: gcaf_report.map(|r| r.purity_adj),
            gcaf_prop_vis: gcaf_report.map(|r| r.prop_vis),
            gcaf_res_density: gcaf_report.map(|r| r.res_density),
            gcaf_res_deconv: gcaf_report.map(|r| r.res_deconv),
            gcaf_res_combined: gcaf_report.map(|r| r.res_combined),
            gcaf_spatial_coherence: gcaf_report.map(|r| r.spatial_coherence),
            gcaf_in_frac: gcaf_report.map(|r| r.in_frac),
            gcaf_num_clusters: gcaf_report.map(|r| r.num_clusters),
            effective_boundary_r95: gcaf_report.map(|r| r.effective_boundary_r95),
            g_eff: gcaf_report.map(|r| r.g_eff),
            gcaf_detail: gcaf_detail.map(GcafDetailOutput::from),
            landmark_stratified_rta: report.landmark_stratified_rta,
            dsc: report.distance_consistency,
            dc: report.distribution_consistency,
            cdm: report.class_density_measure,
            hdm: report.histogram_density_measure,
            sepme_knng: report.sepme_knng,
            sepme_rng: report.sepme_rng,
            sepme_gab: report.sepme_gab,
            sepme_gong: report.sepme_gong,
            neighborhood_hit: report.neighborhood_hit,
            silhouette: report.silhouette_score,
            calinski_harabasz: report.calinski_harabasz_score,
            davies_bouldin: report.davies_bouldin_score,
            cq_kmeans_ari: report.cq_kmeans_ari,
            cq_kmeans_ami: report.cq_kmeans_ami,
            cq_hac_hom: report.cq_hac_hom,
            cq_hac_ami: report.cq_hac_ami,
            pairwise_distance_overlap_error: report.pairwise_distance_overlap_error,
            sprawlter_nn: report.sprawlter_nn,
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
    pub quality: Option<f64>,
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
