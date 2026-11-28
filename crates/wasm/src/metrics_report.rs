//! WebAssembly report wrapper (`MetricReport`) for selective quality metric evaluation.

use egraph::metrics::MetricReport;
use wasm_bindgen::prelude::*;

use crate::js_convert::{set_f64_prop, set_js_prop};
use crate::metrics_crossings::JsCrossingPair;
use crate::metrics_non_euclidean_crossings::JsTorusCrossing;

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
    #[wasm_bindgen(getter, js_name = scaleNormalizedStress)]
    pub fn scale_normalized_stress(&self) -> Option<f64> {
        self.inner.scale_normalized_stress
    }

    #[wasm_bindgen(getter, js_name = sampledScaleNormalizedStress)]
    pub fn sampled_scale_normalized_stress(&self) -> Option<f64> {
        self.inner.sampled_scale_normalized_stress
    }

    #[wasm_bindgen(getter, js_name = geodesicDistortion)]
    pub fn geodesic_distortion(&self) -> Option<f64> {
        self.inner.geodesic_distortion
    }

    #[wasm_bindgen(getter, js_name = crossingNumber)]
    pub fn crossing_number(&self) -> Option<usize> {
        self.inner.crossing_number
    }

    #[wasm_bindgen(getter, js_name = crossingEdges)]
    pub fn crossing_edges(&self) -> Option<Vec<JsCrossingPair>> {
        self.inner.crossing_edges.as_ref().map(|pairs| {
            pairs
                .iter()
                .cloned()
                .map(JsCrossingPair::from_inner)
                .collect()
        })
    }

    #[wasm_bindgen(getter, js_name = crossingAngle)]
    pub fn crossing_angle(&self) -> Option<f64> {
        self.inner.crossing_angle
    }

    #[wasm_bindgen(getter, js_name = sampledCrossingNumber)]
    pub fn sampled_crossing_number(&self) -> Option<f64> {
        self.inner.sampled_crossing_number
    }

    #[wasm_bindgen(getter)]
    pub fn crosslessness(&self) -> Option<f64> {
        self.inner.crosslessness
    }

    #[wasm_bindgen(getter, js_name = sampledCrosslessness)]
    pub fn sampled_crosslessness(&self) -> Option<f64> {
        self.inner.sampled_crosslessness
    }

    #[wasm_bindgen(getter, js_name = torusCrossingNumber)]
    pub fn torus_crossing_number(&self) -> Option<usize> {
        self.inner.torus_crossing_number
    }

    #[wasm_bindgen(getter, js_name = crossingEdgesTorus)]
    pub fn crossing_edges_torus(&self) -> Option<Vec<JsTorusCrossing>> {
        self.inner.crossing_edges_torus.as_ref().map(|pairs| {
            pairs
                .iter()
                .cloned()
                .map(JsTorusCrossing::from_inner)
                .collect()
        })
    }

    #[wasm_bindgen(getter, js_name = neighborhoodPreservation1Hop)]
    pub fn neighborhood_preservation_1hop(&self) -> Option<f64> {
        self.inner.neighborhood_preservation_1hop
    }

    #[wasm_bindgen(getter, js_name = neighborhoodPreservation2Hop)]
    pub fn neighborhood_preservation_2hop(&self) -> Option<f64> {
        self.inner.neighborhood_preservation_2hop
    }

    #[wasm_bindgen(getter, js_name = sampledNp1)]
    pub fn sampled_np1(&self) -> Option<f64> {
        self.inner.sampled_np1
    }

    #[wasm_bindgen(getter, js_name = sampledNp2)]
    pub fn sampled_np2(&self) -> Option<f64> {
        self.inner.sampled_np2
    }

    #[wasm_bindgen(getter, js_name = gabrielGraphProperty)]
    pub fn gabriel_graph_property(&self) -> Option<f64> {
        self.inner.gabriel_graph_property
    }

    #[wasm_bindgen(getter, js_name = nodeResolution)]
    pub fn node_resolution(&self) -> Option<f64> {
        self.inner.node_resolution
    }

    #[wasm_bindgen(getter, js_name = relativeNeighborhoodPreservation)]
    pub fn relative_neighborhood_preservation(&self) -> Option<f64> {
        self.inner.relative_neighborhood_preservation
    }

    #[wasm_bindgen(getter, js_name = gabrielGraphPreservation)]
    pub fn gabriel_graph_preservation(&self) -> Option<f64> {
        self.inner.gabriel_graph_preservation
    }

    #[wasm_bindgen(getter, js_name = aspectRatio)]
    pub fn aspect_ratio(&self) -> Option<f64> {
        self.inner.aspect_ratio
    }

    #[wasm_bindgen(getter, js_name = angularResolution)]
    pub fn angular_resolution(&self) -> Option<f64> {
        self.inner.angular_resolution
    }

    #[wasm_bindgen(getter, js_name = minimumAngle)]
    pub fn minimum_angle(&self) -> Option<f64> {
        self.inner.minimum_angle
    }

    #[wasm_bindgen(getter, js_name = minimumAngleA1)]
    pub fn minimum_angle_a1(&self) -> Option<f64> {
        self.inner.minimum_angle_a1
    }

    #[wasm_bindgen(getter, js_name = layoutDegeneracy)]
    pub fn layout_degeneracy(&self) -> Option<f64> {
        self.inner.layout_degeneracy
    }

    #[wasm_bindgen(getter, js_name = idealEdgeLengths)]
    pub fn ideal_edge_lengths(&self) -> Option<f64> {
        self.inner.ideal_edge_lengths
    }

    #[wasm_bindgen(getter, js_name = prismSigmaDist)]
    pub fn prism_sigma_dist(&self) -> Option<f64> {
        self.inner.prism_sigma_dist
    }

    #[wasm_bindgen(getter, js_name = prismSigmaDisp)]
    pub fn prism_sigma_disp(&self) -> Option<f64> {
        self.inner.prism_sigma_disp
    }

    #[wasm_bindgen(getter, js_name = adjustedPatchNig)]
    pub fn adjusted_patch_nig(&self) -> Option<f64> {
        self.inner.adjusted_patch_nig
    }

    #[wasm_bindgen(getter, js_name = clusterAreaFaithfulness)]
    pub fn cluster_area_faithfulness(&self) -> Option<f64> {
        self.inner.cluster_area_faithfulness
    }

    #[wasm_bindgen(getter, js_name = landmarkStratifiedRta)]
    pub fn landmark_stratified_rta(&self) -> Option<f64> {
        self.inner.landmark_stratified_rta
    }

    #[wasm_bindgen(getter, js_name = distanceConsistency)]
    pub fn distance_consistency(&self) -> Option<f64> {
        self.inner
            .baseline_metrics()
            .and_then(|b| b.distance_consistency)
    }

    #[wasm_bindgen(getter, js_name = distributionConsistency)]
    pub fn distribution_consistency(&self) -> Option<f64> {
        self.inner
            .baseline_metrics()
            .and_then(|b| b.distribution_consistency)
    }

    #[wasm_bindgen(getter, js_name = classDensityMeasure)]
    pub fn class_density_measure(&self) -> Option<f64> {
        self.inner
            .baseline_metrics()
            .and_then(|b| b.class_density_measure)
    }

    #[wasm_bindgen(getter, js_name = histogramDensityMeasure)]
    pub fn histogram_density_measure(&self) -> Option<f64> {
        self.inner
            .baseline_metrics()
            .and_then(|b| b.histogram_density_measure)
    }

    #[wasm_bindgen(getter, js_name = sepmeKnng)]
    pub fn sepme_knng(&self) -> Option<f64> {
        self.inner.baseline_metrics().and_then(|b| b.sepme_knng)
    }

    #[wasm_bindgen(getter, js_name = sepmeRng)]
    pub fn sepme_rng(&self) -> Option<f64> {
        self.inner.baseline_metrics().and_then(|b| b.sepme_rng)
    }

    #[wasm_bindgen(getter, js_name = sepmeGab)]
    pub fn sepme_gab(&self) -> Option<f64> {
        self.inner.baseline_metrics().and_then(|b| b.sepme_gab)
    }

    #[wasm_bindgen(getter, js_name = sepmeGong)]
    pub fn sepme_gong(&self) -> Option<f64> {
        self.inner.baseline_metrics().and_then(|b| b.sepme_gong)
    }

    #[wasm_bindgen(getter, js_name = neighborhoodHit)]
    pub fn neighborhood_hit(&self) -> Option<f64> {
        self.inner
            .baseline_metrics()
            .and_then(|b| b.neighborhood_hit)
    }

    #[wasm_bindgen(getter, js_name = silhouetteScore)]
    pub fn silhouette_score(&self) -> Option<f64> {
        self.inner
            .baseline_metrics()
            .and_then(|b| b.silhouette_score)
    }

    #[wasm_bindgen(getter, js_name = calinskiHarabaszScore)]
    pub fn calinski_harabasz_score(&self) -> Option<f64> {
        self.inner
            .baseline_metrics()
            .and_then(|b| b.calinski_harabasz_score)
    }

    #[wasm_bindgen(getter, js_name = daviesBouldinScore)]
    pub fn davies_bouldin_score(&self) -> Option<f64> {
        self.inner
            .baseline_metrics()
            .and_then(|b| b.davies_bouldin_score)
    }

    #[wasm_bindgen(getter, js_name = cqKmeansAri)]
    pub fn cq_kmeans_ari(&self) -> Option<f64> {
        self.inner.baseline_metrics().and_then(|b| b.cq_kmeans_ari)
    }

    #[wasm_bindgen(getter, js_name = cqKmeansAmi)]
    pub fn cq_kmeans_ami(&self) -> Option<f64> {
        self.inner.baseline_metrics().and_then(|b| b.cq_kmeans_ami)
    }

    #[wasm_bindgen(getter, js_name = cqHacHom)]
    pub fn cq_hac_hom(&self) -> Option<f64> {
        self.inner.baseline_metrics().and_then(|b| b.cq_hac_hom)
    }

    #[wasm_bindgen(getter, js_name = cqHacAmi)]
    pub fn cq_hac_ami(&self) -> Option<f64> {
        self.inner.baseline_metrics().and_then(|b| b.cq_hac_ami)
    }

    #[wasm_bindgen(getter, js_name = pairwiseDistanceOverlapError)]
    pub fn pairwise_distance_overlap_error(&self) -> Option<f64> {
        self.inner
            .baseline_metrics()
            .and_then(|b| b.pairwise_distance_overlap_error)
    }

    #[wasm_bindgen(getter, js_name = sprawlterNn)]
    pub fn sprawlter_nn(&self) -> Option<f64> {
        self.inner.baseline_metrics().and_then(|b| b.sprawlter_nn)
    }

    #[wasm_bindgen(getter, js_name = norOoNni)]
    pub fn nor_oo_nni(&self) -> Option<f64> {
        self.inner.baseline_metrics().and_then(|b| b.nor_oo_nni)
    }

    #[wasm_bindgen(getter, js_name = norSpChA)]
    pub fn nor_sp_ch_a(&self) -> Option<f64> {
        self.inner.baseline_metrics().and_then(|b| b.nor_sp_ch_a)
    }

    #[wasm_bindgen(getter, js_name = norGsBbIar)]
    pub fn nor_gs_bb_iar(&self) -> Option<f64> {
        self.inner.baseline_metrics().and_then(|b| b.nor_gs_bb_iar)
    }

    #[wasm_bindgen(getter, js_name = norNmDmImse)]
    pub fn nor_nm_dm_imse(&self) -> Option<f64> {
        self.inner.baseline_metrics().and_then(|b| b.nor_nm_dm_imse)
    }

    #[wasm_bindgen(getter, js_name = norElRsd)]
    pub fn nor_el_rsd(&self) -> Option<f64> {
        self.inner.baseline_metrics().and_then(|b| b.nor_el_rsd)
    }

    /// Converts requested metrics into a plain JavaScript object without requiring `serde`.
    #[wasm_bindgen(js_name = toRecord)]
    pub fn to_record(&self) -> Result<js_sys::Object, JsValue> {
        let obj = js_sys::Object::new();
        let b = self.inner.baseline_metrics();
        let entries: [(&str, Option<f64>); 51] = [
            ("scaleNormalizedStress", self.inner.scale_normalized_stress),
            (
                "sampledScaleNormalizedStress",
                self.inner.sampled_scale_normalized_stress,
            ),
            ("geodesicDistortion", self.inner.geodesic_distortion),
            (
                "crossingNumber",
                self.inner.crossing_number.map(|v| v as f64),
            ),
            ("crossingAngle", self.inner.crossing_angle),
            ("sampledCrossingNumber", self.inner.sampled_crossing_number),
            ("crosslessness", self.inner.crosslessness),
            ("sampledCrosslessness", self.inner.sampled_crosslessness),
            (
                "torusCrossingNumber",
                self.inner.torus_crossing_number.map(|v| v as f64),
            ),
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
            ("prismSigmaDist", self.inner.prism_sigma_dist),
            ("prismSigmaDisp", self.inner.prism_sigma_disp),
            ("adjustedPatchNig", self.inner.adjusted_patch_nig),
            (
                "clusterAreaFaithfulness",
                self.inner.cluster_area_faithfulness,
            ),
            ("landmarkStratifiedRta", self.inner.landmark_stratified_rta),
            (
                "distanceConsistency",
                b.and_then(|r| r.distance_consistency),
            ),
            (
                "distributionConsistency",
                b.and_then(|r| r.distribution_consistency),
            ),
            (
                "classDensityMeasure",
                b.and_then(|r| r.class_density_measure),
            ),
            (
                "histogramDensityMeasure",
                b.and_then(|r| r.histogram_density_measure),
            ),
            ("sepmeKnng", b.and_then(|r| r.sepme_knng)),
            ("sepmeRng", b.and_then(|r| r.sepme_rng)),
            ("sepmeGab", b.and_then(|r| r.sepme_gab)),
            ("sepmeGong", b.and_then(|r| r.sepme_gong)),
            ("neighborhoodHit", b.and_then(|r| r.neighborhood_hit)),
            ("silhouetteScore", b.and_then(|r| r.silhouette_score)),
            (
                "calinskiHarabaszScore",
                b.and_then(|r| r.calinski_harabasz_score),
            ),
            ("daviesBouldinScore", b.and_then(|r| r.davies_bouldin_score)),
            ("cqKmeansAri", b.and_then(|r| r.cq_kmeans_ari)),
            ("cqKmeansAmi", b.and_then(|r| r.cq_kmeans_ami)),
            ("cqHacHom", b.and_then(|r| r.cq_hac_hom)),
            ("cqHacAmi", b.and_then(|r| r.cq_hac_ami)),
            (
                "pairwiseDistanceOverlapError",
                b.and_then(|r| r.pairwise_distance_overlap_error),
            ),
            ("sprawlterNn", b.and_then(|r| r.sprawlter_nn)),
            ("norOoNni", b.and_then(|r| r.nor_oo_nni)),
            ("norSpChA", b.and_then(|r| r.nor_sp_ch_a)),
            ("norGsBbIar", b.and_then(|r| r.nor_gs_bb_iar)),
            ("norNmDmImse", b.and_then(|r| r.nor_nm_dm_imse)),
            ("norElRsd", b.and_then(|r| r.nor_el_rsd)),
        ];
        for (key, maybe_val) in entries {
            if let Some(val) = maybe_val {
                set_f64_prop(&obj, key, val)?;
            }
        }
        if let Some(pairs) = self.inner.crossing_edges.as_ref() {
            let arr = js_sys::Array::new_with_length(pairs.len() as u32);
            for (idx, p) in pairs.iter().enumerate() {
                arr.set(
                    idx as u32,
                    JsCrossingPair::from_inner(p.clone()).to_record()?.into(),
                );
            }
            set_js_prop(&obj, "crossingEdges", &arr.into())?;
        }
        if let Some(pairs) = self.inner.crossing_edges_torus.as_ref() {
            let arr = js_sys::Array::new_with_length(pairs.len() as u32);
            for (idx, p) in pairs.iter().enumerate() {
                arr.set(
                    idx as u32,
                    JsTorusCrossing::from_inner(p.clone()).to_record()?.into(),
                );
            }
            set_js_prop(&obj, "crossingEdgesTorus", &arr.into())?;
        }
        Ok(obj)
    }
}
