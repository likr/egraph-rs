//! WebAssembly report wrapper structs for selective quality metric evaluation.

use egraph::metrics::{MetricReport, MinimumAngleSummary};
use wasm_bindgen::prelude::*;

/// PRISM layout-preservation distortion metrics comparing an original and adjusted layout.
#[wasm_bindgen(js_name = PrismDistortion)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JsPrismDistortion {
    sigma_dist: f64,
    sigma_disp: f64,
}

impl JsPrismDistortion {
    pub(crate) fn new(sigma_dist: f64, sigma_disp: f64) -> Self {
        Self {
            sigma_dist,
            sigma_disp,
        }
    }
}

#[wasm_bindgen(js_class = PrismDistortion)]
impl JsPrismDistortion {
    /// Returns the proximity edge-length ratio coefficient of variation (`sigma_dist`).
    #[wasm_bindgen(getter, js_name = sigmaDist)]
    pub fn sigma_dist(&self) -> f64 {
        self.sigma_dist
    }

    /// Returns the scale-, translation-, and rotation-invariant Procrustes disparity (`sigma_disp`)
    /// lying in `[0.0, 1.0]`.
    #[wasm_bindgen(getter, js_name = sigmaDisp)]
    pub fn sigma_disp(&self) -> f64 {
        self.sigma_disp
    }
}

/// Selective quality metric report returned by `evaluateMetrics`.
///
/// Unrequested metrics return `undefined` without requiring `serde` in minimal builds.
#[wasm_bindgen(js_name = MetricReport)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsMetricReport {
    inner: MetricReport,
}

impl JsMetricReport {
    pub(crate) fn from_inner(inner: MetricReport) -> Self {
        Self { inner }
    }
}

#[wasm_bindgen(js_class = MetricReport)]
impl JsMetricReport {
    /// Returns closed-form Scale-Normalized Stress (`SNS`), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = scaleNormalizedStress)]
    pub fn scale_normalized_stress(&self) -> Option<f64> {
        self.inner.scale_normalized_stress
    }

    /// Returns pivot-sampled Scale-Normalized Stress (`SNS`), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = sampledScaleNormalizedStress)]
    pub fn sampled_scale_normalized_stress(&self) -> Option<f64> {
        self.inner.sampled_scale_normalized_stress
    }

    /// Returns exact sweep-line edge crossing count, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = crossingNumber)]
    pub fn crossing_number(&self) -> Option<usize> {
        self.inner.crossing_number
    }

    /// Returns exact sweep-line edge crossing-angle penalty sum, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = crossingAngle)]
    pub fn crossing_angle(&self) -> Option<f64> {
        self.inner.crossing_angle
    }

    /// Returns deterministic sampled crossing estimate, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = sampledCrossingNumber)]
    pub fn sampled_crossing_number(&self) -> Option<f64> {
        self.inner.sampled_crossing_number
    }

    /// Returns exact DRGraph Eq. (17) crosslessness, or `undefined` if not requested.
    #[wasm_bindgen(getter)]
    pub fn crosslessness(&self) -> Option<f64> {
        self.inner.crosslessness
    }

    /// Returns sampled DRGraph Eq. (17) crosslessness, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = sampledCrosslessness)]
    pub fn sampled_crosslessness(&self) -> Option<f64> {
        self.inner.sampled_crosslessness
    }

    /// Returns 1-hop neighborhood preservation (`NP1`), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = neighborhoodPreservation1Hop)]
    pub fn neighborhood_preservation_1hop(&self) -> Option<f64> {
        self.inner.neighborhood_preservation_1hop
    }

    /// Returns 2-hop neighborhood preservation (`NP2`), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = neighborhoodPreservation2Hop)]
    pub fn neighborhood_preservation_2hop(&self) -> Option<f64> {
        self.inner.neighborhood_preservation_2hop
    }

    /// Returns deterministic vertex-sampled 1-hop neighborhood preservation (`NP1`), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = sampledNp1)]
    pub fn sampled_np1(&self) -> Option<f64> {
        self.inner.sampled_np1
    }

    /// Returns deterministic vertex-sampled 2-hop neighborhood preservation (`NP2`), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = sampledNp2)]
    pub fn sampled_np2(&self) -> Option<f64> {
        self.inner.sampled_np2
    }

    /// Returns Gabriel graph property violation sum, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = gabrielGraphProperty)]
    pub fn gabriel_graph_property(&self) -> Option<f64> {
        self.inner.gabriel_graph_property
    }

    /// Returns one-sided hinge node-resolution violation sum, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = nodeResolution)]
    pub fn node_resolution(&self) -> Option<f64> {
        self.inner.node_resolution
    }

    /// Returns Eades et al. (2017) unweighted shape-based Relative Neighborhood Graph preservation
    /// (`S_RNG`, distinct from deferred Hong et al. 2022 `dRNG` and Fang et al. 2025 `dcRNG`), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = relativeNeighborhoodPreservation)]
    pub fn relative_neighborhood_preservation(&self) -> Option<f64> {
        self.inner.relative_neighborhood_preservation
    }

    /// Returns Eades et al. (2017) unweighted shape-based Gabriel Graph preservation
    /// (`S_GG`, distinct from deferred Hong et al. 2022 `dGG` and Fang et al. 2025 `dcGG`), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = gabrielGraphPreservation)]
    pub fn gabriel_graph_preservation(&self) -> Option<f64> {
        self.inner.gabriel_graph_preservation
    }

    /// Returns PCA aspect ratio, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = aspectRatio)]
    pub fn aspect_ratio(&self) -> Option<f64> {
        self.inner.aspect_ratio
    }

    /// Returns exponential angular-resolution penalty sum, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = angularResolution)]
    pub fn angular_resolution(&self) -> Option<f64> {
        self.inner.angular_resolution
    }

    /// Returns DRGraph Eq. (19) strict minimum-angle resolution (`A2`), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = minimumAngle)]
    pub fn minimum_angle(&self) -> Option<f64> {
        self.inner.minimum_angle
    }

    /// Returns DRGraph Eq. (19) minimum-angle resolution (`A1`), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = minimumAngleA1)]
    pub fn minimum_angle_a1(&self) -> Option<f64> {
        self.inner.minimum_angle_a1
    }

    /// Returns scalar layout-degeneracy summary, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = layoutDegeneracy)]
    pub fn layout_degeneracy(&self) -> Option<f64> {
        self.inner.layout_degeneracy
    }

    /// Returns ideal edge-length preservation error sum, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = idealEdgeLengths)]
    pub fn ideal_edge_lengths(&self) -> Option<f64> {
        self.inner.ideal_edge_lengths
    }

    /// Returns Adjusted Patch-NIG score, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = adjustedPatchNig)]
    pub fn adjusted_patch_nig(&self) -> Option<f64> {
        self.inner.adjusted_patch_nig
    }

    /// Returns Grid-based Cluster Area Faithfulness (`G-CAF`), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = clusterAreaFaithfulness)]
    pub fn cluster_area_faithfulness(&self) -> Option<f64> {
        self.inner.cluster_area_faithfulness
    }

    /// Returns Landmark-Stratified Random Triplet Accuracy (`L-Stratified-RTA`), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = landmarkStratifiedRta)]
    pub fn landmark_stratified_rta(&self) -> Option<f64> {
        self.inner.landmark_stratified_rta
    }

    /// Returns Distance Consistency (`DSC`), or `undefined` if not requested.
    #[wasm_bindgen(getter)]
    pub fn dsc(&self) -> Option<f64> {
        self.inner.distance_consistency
    }

    /// Returns Distribution Consistency (`DC`), or `undefined` if not requested.
    #[wasm_bindgen(getter)]
    pub fn dc(&self) -> Option<f64> {
        self.inner.distribution_consistency
    }

    /// Returns Class Density Measure (`CDM`), or `undefined` if not requested.
    #[wasm_bindgen(getter)]
    pub fn cdm(&self) -> Option<f64> {
        self.inner.class_density_measure
    }

    /// Returns 2D Histogram Density Measure (`2D-HDM`), or `undefined` if not requested.
    #[wasm_bindgen(getter)]
    pub fn hdm(&self) -> Option<f64> {
        self.inner.histogram_density_measure
    }

    /// Returns SepMe $k$-NN graph DIR CPT, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = sepmeKnng)]
    pub fn sepme_knng(&self) -> Option<f64> {
        self.inner.sepme_knng
    }

    /// Returns SepMe Relative Neighborhood Graph DIR CPT, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = sepmeRng)]
    pub fn sepme_rng(&self) -> Option<f64> {
        self.inner.sepme_rng
    }

    /// Returns SepMe Gabriel Graph DIR CPT, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = sepmeGab)]
    pub fn sepme_gab(&self) -> Option<f64> {
        self.inner.sepme_gab
    }

    /// Returns SepMe $\gamma$-Observable Neighbor Graph DIR CPT, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = sepmeGong)]
    pub fn sepme_gong(&self) -> Option<f64> {
        self.inner.sepme_gong
    }

    /// Returns Neighborhood Hit ($M_{\mathrm{NH}}$), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = neighborhoodHit)]
    pub fn neighborhood_hit(&self) -> Option<f64> {
        self.inner.neighborhood_hit
    }

    /// Returns Silhouette coefficient, or `undefined` if not requested.
    #[wasm_bindgen(getter)]
    pub fn silhouette(&self) -> Option<f64> {
        self.inner.silhouette_score
    }

    /// Returns Calinski-Harabasz Index, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = calinskiHarabasz)]
    pub fn calinski_harabasz(&self) -> Option<f64> {
        self.inner.calinski_harabasz_score
    }

    /// Returns Davies-Bouldin Index, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = daviesBouldin)]
    pub fn davies_bouldin(&self) -> Option<f64> {
        self.inner.davies_bouldin_score
    }

    /// Returns $k$-means Cluster Quality (`CQ-kmeans` ARI), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = cqKmeansAri)]
    pub fn cq_kmeans_ari(&self) -> Option<f64> {
        self.inner.cq_kmeans_ari
    }

    /// Returns $k$-means Cluster Quality (`CQ-kmeans` AMI), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = cqKmeansAmi)]
    pub fn cq_kmeans_ami(&self) -> Option<f64> {
        self.inner.cq_kmeans_ami
    }

    /// Returns Average-linkage HAC Cluster Quality (`CQ-HAC` Homogeneity), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = cqHacHom)]
    pub fn cq_hac_hom(&self) -> Option<f64> {
        self.inner.cq_hac_hom
    }

    /// Returns Average-linkage HAC Cluster Quality (`CQ-HAC` AMI), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = cqHacAmi)]
    pub fn cq_hac_ami(&self) -> Option<f64> {
        self.inner.cq_hac_ami
    }

    /// Returns Pairwise Distance-Overlap Error (`PDOE`), or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = pairwiseDistanceOverlapError)]
    pub fn pairwise_distance_overlap_error(&self) -> Option<f64> {
        self.inner.pairwise_distance_overlap_error
    }

    /// Returns Sprawlter-NN composite overlap clutter measure, or `undefined` if not requested.
    #[wasm_bindgen(getter, js_name = sprawlterNn)]
    pub fn sprawlter_nn(&self) -> Option<f64> {
        self.inner.sprawlter_nn
    }

    /// Converts requested metrics into a plain JavaScript object without requiring `serde`.
    #[wasm_bindgen(js_name = toRecord)]
    pub fn to_record(&self) -> Result<js_sys::Object, JsError> {
        let obj = js_sys::Object::new();
        let entries: [(&str, Option<f64>); 42] = [
            ("scaleNormalizedStress", self.inner.scale_normalized_stress),
            (
                "sampledScaleNormalizedStress",
                self.inner.sampled_scale_normalized_stress,
            ),
            (
                "crossingNumber",
                self.inner.crossing_number.map(|v| v as f64),
            ),
            ("crossingAngle", self.inner.crossing_angle),
            ("sampledCrossingNumber", self.inner.sampled_crossing_number),
            ("crosslessness", self.inner.crosslessness),
            ("sampledCrosslessness", self.inner.sampled_crosslessness),
            (
                "neighborhoodPreservation1Hop",
                self.inner.neighborhood_preservation_1hop,
            ),
            (
                "neighborhoodPreservation2Hop",
                self.inner.neighborhood_preservation_2hop,
            ),
            ("sampledNp1", self.inner.sampled_np1),
            ("sampledNp2", self.inner.sampled_np2),
            ("gabrielGraphProperty", self.inner.gabriel_graph_property),
            ("nodeResolution", self.inner.node_resolution),
            (
                "relativeNeighborhoodPreservation",
                self.inner.relative_neighborhood_preservation,
            ),
            (
                "gabrielGraphPreservation",
                self.inner.gabriel_graph_preservation,
            ),
            ("aspectRatio", self.inner.aspect_ratio),
            ("angularResolution", self.inner.angular_resolution),
            ("minimumAngle", self.inner.minimum_angle),
            ("minimumAngleA1", self.inner.minimum_angle_a1),
            ("layoutDegeneracy", self.inner.layout_degeneracy),
            ("idealEdgeLengths", self.inner.ideal_edge_lengths),
            ("adjustedPatchNig", self.inner.adjusted_patch_nig),
            (
                "clusterAreaFaithfulness",
                self.inner.cluster_area_faithfulness,
            ),
            ("landmarkStratifiedRta", self.inner.landmark_stratified_rta),
            ("dsc", self.inner.distance_consistency),
            ("dc", self.inner.distribution_consistency),
            ("cdm", self.inner.class_density_measure),
            ("hdm", self.inner.histogram_density_measure),
            ("sepmeKnng", self.inner.sepme_knng),
            ("sepmeRng", self.inner.sepme_rng),
            ("sepmeGab", self.inner.sepme_gab),
            ("sepmeGong", self.inner.sepme_gong),
            ("neighborhoodHit", self.inner.neighborhood_hit),
            ("silhouette", self.inner.silhouette_score),
            ("calinskiHarabasz", self.inner.calinski_harabasz_score),
            ("daviesBouldin", self.inner.davies_bouldin_score),
            ("cqKmeansAri", self.inner.cq_kmeans_ari),
            ("cqKmeansAmi", self.inner.cq_kmeans_ami),
            ("cqHacHom", self.inner.cq_hac_hom),
            ("cqHacAmi", self.inner.cq_hac_ami),
            (
                "pairwiseDistanceOverlapError",
                self.inner.pairwise_distance_overlap_error,
            ),
            ("sprawlterNn", self.inner.sprawlter_nn),
        ];
        for (key, maybe_val) in entries {
            if let Some(val) = maybe_val {
                js_sys::Reflect::set(&obj, &JsValue::from_str(key), &JsValue::from_f64(val))
                    .map_err(|err| {
                        JsError::new(&format!("failed to set MetricReport.{key}: {err:?}"))
                    })?;
            }
        }
        Ok(obj)
    }
}

/// DRGraph Eq. (19) minimum-angle resolution summary (`a1` and `a2`).
#[wasm_bindgen(js_name = MinimumAngleSummary)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JsMinimumAngleSummary {
    inner: MinimumAngleSummary,
}

impl JsMinimumAngleSummary {
    pub(crate) fn from_inner(inner: MinimumAngleSummary) -> Self {
        Self { inner }
    }
}

#[wasm_bindgen(js_class = MinimumAngleSummary)]
impl JsMinimumAngleSummary {
    #[wasm_bindgen(getter)]
    pub fn a1(&self) -> f64 {
        self.inner.a1
    }

    #[wasm_bindgen(getter)]
    pub fn a2(&self) -> f64 {
        self.inner.a2
    }

    #[wasm_bindgen(js_name = toRecord)]
    pub fn to_record(&self) -> Result<js_sys::Object, JsError> {
        let obj = js_sys::Object::new();
        js_sys::Reflect::set(
            &obj,
            &JsValue::from_str("a1"),
            &JsValue::from_f64(self.inner.a1),
        )
        .map_err(|e| JsError::new(&format!("failed to set a1: {e:?}")))?;
        js_sys::Reflect::set(
            &obj,
            &JsValue::from_str("a2"),
            &JsValue::from_f64(self.inner.a2),
        )
        .map_err(|e| JsError::new(&format!("failed to set a2: {e:?}")))?;
        Ok(obj)
    }
}
