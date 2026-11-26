//! WebAssembly bindings for cluster separation, clustering validity indices (CVI),
//! Sprawlter-NN, and Node Overlap Removal (NOR) baseline metrics (`f64` accumulation).

use egraph::metrics::{
    ClusterBaselineConfig, NorMetrics, PartitionAgreement, calinski_harabasz_score,
    class_density_measure, cq_hac, cq_kmeans, davies_bouldin_score, distance_consistency,
    distribution_consistency, neighborhood_hit, nor_metrics, nor_metrics_without_graph,
    pairwise_distance_overlap_error, partition_agreement, sepme_gab, sepme_gong, sepme_knng,
    sepme_rng, silhouette_score, sprawlter_nn, sprawlter_nn_radii,
};
use wasm_bindgen::prelude::*;

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::{
    core_err_to_js, parse_optional_usize, parse_seed_u64, parse_u32_sequence, set_f64_prop,
};
use crate::layout_buffers::JsLayout2d;

/// Partition agreement metrics returned by `partitionAgreement`, `cqKmeans`, and `cqHac`.
#[wasm_bindgen(js_name = PartitionAgreementResult)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JsPartitionAgreementResult {
    inner: PartitionAgreement,
}

#[wasm_bindgen(js_class = PartitionAgreementResult)]
impl JsPartitionAgreementResult {
    #[wasm_bindgen(getter)]
    pub fn ari(&self) -> f64 {
        self.inner.ari
    }

    #[wasm_bindgen(getter)]
    pub fn ami(&self) -> f64 {
        self.inner.ami
    }

    #[wasm_bindgen(getter)]
    pub fn homogeneity(&self) -> f64 {
        self.inner.homogeneity
    }

    #[wasm_bindgen(getter)]
    pub fn completeness(&self) -> f64 {
        self.inner.completeness
    }

    #[wasm_bindgen(getter, js_name = vMeasure)]
    pub fn v_measure(&self) -> f64 {
        self.inner.v_measure
    }

    #[wasm_bindgen(getter)]
    pub fn fmi(&self) -> f64 {
        self.inner.fmi
    }

    #[wasm_bindgen(js_name = toRecord)]
    pub fn to_record(&self) -> Result<js_sys::Object, JsError> {
        let obj = js_sys::Object::new();
        set_f64_prop(&obj, "ari", self.inner.ari)?;
        set_f64_prop(&obj, "ami", self.inner.ami)?;
        set_f64_prop(&obj, "homogeneity", self.inner.homogeneity)?;
        set_f64_prop(&obj, "completeness", self.inner.completeness)?;
        set_f64_prop(&obj, "vMeasure", self.inner.v_measure)?;
        set_f64_prop(&obj, "fmi", self.inner.fmi)?;
        Ok(obj)
    }
}

/// Class Density Measure (`cdm`) and 2D Histogram Density Measure (`hdm`) returned by `classDensityMeasure`.
#[wasm_bindgen(js_name = ClassDensityResult)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JsClassDensityResult {
    cdm: f64,
    hdm: f64,
}

#[wasm_bindgen(js_class = ClassDensityResult)]
impl JsClassDensityResult {
    #[wasm_bindgen(getter)]
    pub fn cdm(&self) -> f64 {
        self.cdm
    }

    #[wasm_bindgen(getter)]
    pub fn hdm(&self) -> f64 {
        self.hdm
    }

    #[wasm_bindgen(js_name = toRecord)]
    pub fn to_record(&self) -> Result<js_sys::Object, JsError> {
        let obj = js_sys::Object::new();
        set_f64_prop(&obj, "cdm", self.cdm)?;
        set_f64_prop(&obj, "hdm", self.hdm)?;
        Ok(obj)
    }
}

/// Chen et al. (2020) Node Overlap Removal (NOR) 5-metric summary returned by `norMetrics` / `norMetricsWithoutGraph`.
#[wasm_bindgen(js_name = NorMetricsResult)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JsNorMetricsResult {
    inner: NorMetrics,
}

#[wasm_bindgen(js_class = NorMetricsResult)]
impl JsNorMetricsResult {
    #[wasm_bindgen(getter, js_name = ooNni)]
    pub fn oo_nni(&self) -> f64 {
        self.inner.oo_nni
    }

    #[wasm_bindgen(getter, js_name = spChA)]
    pub fn sp_ch_a(&self) -> f64 {
        self.inner.sp_ch_a
    }

    #[wasm_bindgen(getter, js_name = gsBbIar)]
    pub fn gs_bb_iar(&self) -> f64 {
        self.inner.gs_bb_iar
    }

    #[wasm_bindgen(getter, js_name = nmDmImse)]
    pub fn nm_dm_imse(&self) -> f64 {
        self.inner.nm_dm_imse
    }

    #[wasm_bindgen(getter, js_name = elRsd)]
    pub fn el_rsd(&self) -> f64 {
        self.inner.el_rsd
    }

    #[wasm_bindgen(js_name = toRecord)]
    pub fn to_record(&self) -> Result<js_sys::Object, JsError> {
        let obj = js_sys::Object::new();
        set_f64_prop(&obj, "ooNni", self.inner.oo_nni)?;
        set_f64_prop(&obj, "spChA", self.inner.sp_ch_a)?;
        set_f64_prop(&obj, "gsBbIar", self.inner.gs_bb_iar)?;
        set_f64_prop(&obj, "nmDmImse", self.inner.nm_dm_imse)?;
        set_f64_prop(&obj, "elRsd", self.inner.el_rsd)?;
        Ok(obj)
    }
}

/// Computes `ARI`, `AMI`, `homogeneity`, `completeness`, `vMeasure`, and `fmi` between two labelings.
#[wasm_bindgen(js_name = partitionAgreement)]
pub fn js_partition_agreement(
    labels_true: &JsValue,
    labels_pred: &JsValue,
) -> Result<JsPartitionAgreementResult, JsError> {
    let lt = parse_u32_sequence(labels_true, "labelsTrue")?;
    let lp = parse_u32_sequence(labels_pred, "labelsPred")?;
    let inner = partition_agreement(&lt, &lp).map_err(core_err_to_js)?;
    Ok(JsPartitionAgreementResult { inner })
}

/// Evaluates `k`-means Cluster Quality (`CQ-kmeans`) of `layout` against `communityLabels`.
#[wasm_bindgen(js_name = cqKmeans)]
pub fn js_cq_kmeans(
    layout: &JsLayout2d,
    community_labels: &JsValue,
    n_init: Option<f64>,
    max_iters: Option<f64>,
    seed: Option<f64>,
) -> Result<JsPartitionAgreementResult, JsError> {
    let defaults = ClusterBaselineConfig::default();
    let labels = parse_u32_sequence(community_labels, "communityLabels")?;
    let n_init_val = parse_optional_usize(n_init, defaults.kmeans_n_init, "nInit")?;
    let max_iters_val = parse_optional_usize(max_iters, defaults.kmeans_max_iters, "maxIters")?;
    let seed_u64 = parse_seed_u64(seed)?;
    let soa = layout.inner().to_layout2d();
    let inner =
        cq_kmeans(&soa, &labels, n_init_val, max_iters_val, seed_u64).map_err(core_err_to_js)?;
    Ok(JsPartitionAgreementResult { inner })
}

/// Evaluates average-linkage HAC Cluster Quality (`CQ-HAC`) of `layout` against `communityLabels`.
#[wasm_bindgen(js_name = cqHac)]
pub fn js_cq_hac(
    layout: &JsLayout2d,
    community_labels: &JsValue,
) -> Result<JsPartitionAgreementResult, JsError> {
    let labels = parse_u32_sequence(community_labels, "communityLabels")?;
    let soa = layout.inner().to_layout2d();
    let inner = cq_hac(&soa, &labels).map_err(core_err_to_js)?;
    Ok(JsPartitionAgreementResult { inner })
}

/// Computes Distance Consistency (`DSC`) in `[0.0, 1.0]`.
#[wasm_bindgen(js_name = distanceConsistency)]
pub fn js_distance_consistency(layout: &JsLayout2d, labels: &JsValue) -> Result<f64, JsError> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let soa = layout.inner().to_layout2d();
    distance_consistency(&soa, &parsed_labels).map_err(core_err_to_js)
}

/// Computes Distribution Consistency (`DC`) in `[0.0, 1.0]`.
#[wasm_bindgen(js_name = distributionConsistency)]
pub fn js_distribution_consistency(
    layout: &JsLayout2d,
    labels: &JsValue,
    grid_size: Option<f64>,
) -> Result<f64, JsError> {
    let defaults = ClusterBaselineConfig::default();
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let gs = parse_optional_usize(grid_size, defaults.dc_grid_size, "gridSize")?;
    let soa = layout.inner().to_layout2d();
    distribution_consistency(&soa, &parsed_labels, gs).map_err(core_err_to_js)
}

/// Computes Class Density Measure (`CDM`) and 2D Histogram Density Measure (`2D-HDM`).
#[wasm_bindgen(js_name = classDensityMeasure)]
pub fn js_class_density_measure(
    layout: &JsLayout2d,
    labels: &JsValue,
    grid_size: Option<f64>,
    bandwidth: Option<f64>,
) -> Result<JsClassDensityResult, JsError> {
    let defaults = ClusterBaselineConfig::default();
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let gs = parse_optional_usize(grid_size, defaults.cdm_grid_size, "gridSize")?;
    let soa = layout.inner().to_layout2d();
    let (cdm, hdm) =
        class_density_measure(&soa, &parsed_labels, gs, bandwidth).map_err(core_err_to_js)?;
    Ok(JsClassDensityResult { cdm, hdm })
}

/// Computes SepMe $k$-NN graph DIR CPT (`k = 2` default).
#[wasm_bindgen(js_name = sepmeKnng)]
pub fn js_sepme_knng(
    layout: &JsLayout2d,
    labels: &JsValue,
    k: Option<f64>,
) -> Result<f64, JsError> {
    let defaults = ClusterBaselineConfig::default();
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let k_val = parse_optional_usize(k, defaults.sepme_knng_k, "k")?;
    let soa = layout.inner().to_layout2d();
    sepme_knng(&soa, &parsed_labels, k_val).map_err(core_err_to_js)
}

/// Computes SepMe Relative Neighborhood Graph (`RNG`) DIR CPT.
#[wasm_bindgen(js_name = sepmeRng)]
pub fn js_sepme_rng(layout: &JsLayout2d, labels: &JsValue) -> Result<f64, JsError> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let soa = layout.inner().to_layout2d();
    sepme_rng(&soa, &parsed_labels).map_err(core_err_to_js)
}

/// Computes SepMe Gabriel Graph (`GAB`) DIR CPT.
#[wasm_bindgen(js_name = sepmeGab)]
pub fn js_sepme_gab(layout: &JsLayout2d, labels: &JsValue) -> Result<f64, JsError> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let soa = layout.inner().to_layout2d();
    sepme_gab(&soa, &parsed_labels).map_err(core_err_to_js)
}

/// Computes SepMe $\gamma$-Observable Neighbor Graph (`GONG`) DIR CPT (`gamma = 0.35` default).
#[wasm_bindgen(js_name = sepmeGong)]
pub fn js_sepme_gong(
    layout: &JsLayout2d,
    labels: &JsValue,
    gamma: Option<f64>,
) -> Result<f64, JsError> {
    let defaults = ClusterBaselineConfig::default();
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let g = gamma.unwrap_or(defaults.gong_gamma);
    let soa = layout.inner().to_layout2d();
    sepme_gong(&soa, &parsed_labels, g).map_err(core_err_to_js)
}

/// Computes Neighborhood Hit ($M_{\mathrm{NH}}$, `k = 7` default).
#[wasm_bindgen(js_name = neighborhoodHit)]
pub fn js_neighborhood_hit(
    layout: &JsLayout2d,
    labels: &JsValue,
    k: Option<f64>,
) -> Result<f64, JsError> {
    let defaults = ClusterBaselineConfig::default();
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let k_val = parse_optional_usize(k, defaults.neighborhood_hit_k, "k")?;
    let soa = layout.inner().to_layout2d();
    neighborhood_hit(&soa, &parsed_labels, k_val).map_err(core_err_to_js)
}

/// Computes the Euclidean Silhouette coefficient in `[-1.0, 1.0]`.
#[wasm_bindgen(js_name = silhouetteScore)]
pub fn js_silhouette_score(layout: &JsLayout2d, labels: &JsValue) -> Result<f64, JsError> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let soa = layout.inner().to_layout2d();
    silhouette_score(&soa, &parsed_labels).map_err(core_err_to_js)
}

/// Computes the Calinski-Harabasz Index.
#[wasm_bindgen(js_name = calinskiHarabaszScore)]
pub fn js_calinski_harabasz_score(layout: &JsLayout2d, labels: &JsValue) -> Result<f64, JsError> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let soa = layout.inner().to_layout2d();
    calinski_harabasz_score(&soa, &parsed_labels).map_err(core_err_to_js)
}

/// Computes the Davies-Bouldin Index.
#[wasm_bindgen(js_name = daviesBouldinScore)]
pub fn js_davies_bouldin_score(layout: &JsLayout2d, labels: &JsValue) -> Result<f64, JsError> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let soa = layout.inner().to_layout2d();
    davies_bouldin_score(&soa, &parsed_labels).map_err(core_err_to_js)
}

/// Computes Pairwise Distance-Overlap Error (`PDOE`, `numBins = 100` default).
#[wasm_bindgen(js_name = pairwiseDistanceOverlapError)]
pub fn js_pairwise_distance_overlap_error(
    layout: &JsLayout2d,
    labels: &JsValue,
    num_bins: Option<f64>,
) -> Result<f64, JsError> {
    let defaults = ClusterBaselineConfig::default();
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let bins = parse_optional_usize(num_bins, defaults.num_bins, "numBins")?;
    let soa = layout.inner().to_layout2d();
    pairwise_distance_overlap_error(&soa, &parsed_labels, bins).map_err(core_err_to_js)
}

/// Computes Sprawlter-NN composite overlap clutter measure from per-node `halfWidths` and `halfHeights`.
#[wasm_bindgen(js_name = sprawlterNn)]
pub fn js_sprawlter_nn(
    layout: &JsLayout2d,
    half_widths: &[f32],
    half_heights: &[f32],
) -> Result<f64, JsError> {
    let soa = layout.inner().to_layout2d();
    sprawlter_nn(&soa, half_widths, half_heights).map_err(core_err_to_js)
}

/// Computes Sprawlter-NN composite overlap clutter measure from uniform square `radii`.
#[wasm_bindgen(js_name = sprawlterNnRadii)]
pub fn js_sprawlter_nn_radii(layout: &JsLayout2d, radii: &[f32]) -> Result<f64, JsError> {
    let soa = layout.inner().to_layout2d();
    sprawlter_nn_radii(&soa, radii).map_err(core_err_to_js)
}

/// Computes all 5 Chen et al. (2020) NOR metrics including graph edge-length distortion (`elRsd`).
#[wasm_bindgen(js_name = norMetrics)]
pub fn js_nor_metrics(
    graph: &JsCsrGraph,
    orig_layout: &JsLayout2d,
    adjusted_layout: &JsLayout2d,
    half_widths: &[f32],
    half_heights: &[f32],
) -> Result<JsNorMetricsResult, JsError> {
    let orig_soa = orig_layout.inner().to_layout2d();
    let adj_soa = adjusted_layout.inner().to_layout2d();
    let inner = nor_metrics(
        graph.inner(),
        &orig_soa,
        &adj_soa,
        half_widths,
        half_heights,
    )
    .map_err(core_err_to_js)?;
    Ok(JsNorMetricsResult { inner })
}

/// Computes Chen et al. (2020) NOR metrics without a graph (`elRsd = 0.0`).
#[wasm_bindgen(js_name = norMetricsWithoutGraph)]
pub fn js_nor_metrics_without_graph(
    orig_layout: &JsLayout2d,
    adjusted_layout: &JsLayout2d,
    half_widths: &[f32],
    half_heights: &[f32],
) -> Result<JsNorMetricsResult, JsError> {
    let orig_soa = orig_layout.inner().to_layout2d();
    let adj_soa = adjusted_layout.inner().to_layout2d();
    let inner = nor_metrics_without_graph(&orig_soa, &adj_soa, half_widths, half_heights)
        .map_err(core_err_to_js)?;
    Ok(JsNorMetricsResult { inner })
}
