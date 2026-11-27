//! WebAssembly bindings for Grid-based Cluster Area Faithfulness (`G-CAF`).

use crate::js_convert::{
    core_err_to_js, parse_non_negative_usize, parse_optional_f64_sequence, parse_u32_sequence,
    set_f64_prop, set_js_prop,
};
use crate::layout_buffers::JsLayout2d;
use egraph::metrics::{
    ClusterAreaConfig, ClusterAreaDetail, cluster_area_faithfulness_detail_with_weights,
    cluster_area_faithfulness_with_weights,
};
use wasm_bindgen::prelude::*;

/// Detailed Grid-based Cluster Area Faithfulness (`G-CAF`) report returned by
/// `clusterAreaFaithfulnessDetail`.
#[wasm_bindgen(js_name = ClusterAreaReport)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsClusterAreaReport {
    inner: ClusterAreaDetail,
}

#[wasm_bindgen(js_class = ClusterAreaReport)]
impl JsClusterAreaReport {
    #[wasm_bindgen(getter)]
    pub fn caf(&self) -> f64 {
        self.inner.report.caf
    }

    #[wasm_bindgen(getter, js_name = purityAdj)]
    pub fn purity_adj(&self) -> f64 {
        self.inner.report.purity_adj
    }

    #[wasm_bindgen(getter, js_name = propVis)]
    pub fn prop_vis(&self) -> f64 {
        self.inner.report.prop_vis
    }

    #[wasm_bindgen(getter, js_name = resDensity)]
    pub fn res_density(&self) -> f64 {
        self.inner.report.res_density
    }

    #[wasm_bindgen(getter, js_name = resDeconv)]
    pub fn res_deconv(&self) -> f64 {
        self.inner.report.res_deconv
    }

    #[wasm_bindgen(getter, js_name = resCombined)]
    pub fn res_combined(&self) -> f64 {
        self.inner.report.res_combined
    }

    #[wasm_bindgen(getter, js_name = spatialCoherence)]
    pub fn spatial_coherence(&self) -> f64 {
        self.inner.report.spatial_coherence
    }

    #[wasm_bindgen(getter, js_name = inFrac)]
    pub fn in_frac(&self) -> f64 {
        self.inner.report.in_frac
    }

    #[wasm_bindgen(getter, js_name = effectiveBoundaryR95)]
    pub fn effective_boundary_r95(&self) -> f64 {
        self.inner.report.effective_boundary_r95
    }

    #[wasm_bindgen(getter, js_name = gEff)]
    pub fn g_eff(&self) -> usize {
        self.inner.report.g_eff
    }

    #[wasm_bindgen(getter, js_name = numClusters)]
    pub fn num_clusters(&self) -> usize {
        self.inner.report.num_clusters
    }

    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.report.num_nodes
    }

    #[wasm_bindgen(js_name = clusterIds)]
    pub fn cluster_ids(&self) -> Vec<u32> {
        self.inner.cluster_ids.clone()
    }

    pub fn counts(&self) -> Vec<u32> {
        self.inner.counts.iter().map(|&c| c as u32).collect()
    }

    pub fn p(&self) -> Vec<f64> {
        self.inner.p.clone()
    }

    pub fn q(&self) -> Vec<f64> {
        self.inner.q.clone()
    }

    pub fn purity(&self) -> Vec<f64> {
        self.inner.purity.clone()
    }

    pub fn resolution(&self) -> Vec<f64> {
        self.inner.resolution.clone()
    }

    #[wasm_bindgen(js_name = resDensityPerCluster)]
    pub fn res_density_per_cluster(&self) -> Vec<f64> {
        self.inner.res_density.clone()
    }

    #[wasm_bindgen(js_name = resDeconvPerCluster)]
    pub fn res_deconv_per_cluster(&self) -> Vec<f64> {
        self.inner.res_deconv.clone()
    }

    #[wasm_bindgen(js_name = cafPerCluster)]
    pub fn caf_per_cluster(&self) -> Vec<f64> {
        self.inner.caf_per_cluster.clone()
    }

    pub fn coherence(&self) -> Vec<f64> {
        self.inner.coherence.clone()
    }

    /// Converts summary and per-cluster diagnostics into a plain JavaScript object without `serde`.
    #[wasm_bindgen(js_name = toRecord)]
    pub fn to_record(&self) -> Result<js_sys::Object, JsError> {
        let obj = js_sys::Object::new();
        let r = &self.inner.report;
        for (k, v) in [
            ("caf", r.caf),
            ("purityAdj", r.purity_adj),
            ("propVis", r.prop_vis),
            ("resDensity", r.res_density),
            ("resDeconv", r.res_deconv),
            ("resCombined", r.res_combined),
            ("spatialCoherence", r.spatial_coherence),
            ("inFrac", r.in_frac),
            ("effectiveBoundaryR95", r.effective_boundary_r95),
            ("gEff", r.g_eff as f64),
            ("numClusters", r.num_clusters as f64),
            ("numNodes", r.num_nodes as f64),
        ] {
            set_f64_prop(&obj, k, v)?;
        }
        let counts_u32 = self.counts();
        for (k, arr) in [
            ("uniqueLabels", self.inner.cluster_ids.as_slice()),
            ("counts", counts_u32.as_slice()),
        ] {
            set_js_prop(&obj, k, &js_sys::Uint32Array::from(arr).into())?;
        }
        for (k, arr) in [
            ("p", self.inner.p.as_slice()),
            ("q", self.inner.q.as_slice()),
            ("purity", self.inner.purity.as_slice()),
            ("resolution", self.inner.resolution.as_slice()),
            ("resDensityPerCluster", self.inner.res_density.as_slice()),
            ("resDeconvPerCluster", self.inner.res_deconv.as_slice()),
            ("cafPerCluster", self.inner.caf_per_cluster.as_slice()),
            ("coherence", self.inner.coherence.as_slice()),
        ] {
            set_js_prop(&obj, k, &js_sys::Float64Array::from(arr).into())?;
        }
        Ok(obj)
    }
}

pub(crate) fn parse_cluster_area_config(
    g_max: Option<f64>,
    sigma_cells: Option<f64>,
    kappa: Option<f64>,
    tau_frac: Option<f64>,
    boundary_r95: Option<f64>,
    stevens_gamma: Option<f64>,
) -> Result<ClusterAreaConfig, JsError> {
    let def = ClusterAreaConfig::default();
    Ok(ClusterAreaConfig {
        g_max: match g_max {
            None => def.g_max,
            Some(v) => parse_non_negative_usize(v, "gMax")?,
        },
        sigma_cells: sigma_cells.unwrap_or(def.sigma_cells),
        kappa: kappa.unwrap_or(def.kappa),
        tau_frac: tau_frac.unwrap_or(def.tau_frac),
        boundary_r95: boundary_r95.unwrap_or(def.boundary_r95),
        stevens_gamma: stevens_gamma.unwrap_or(def.stevens_gamma),
    })
}

/// Computes Grid-based Cluster Area Faithfulness (`G-CAF`) in `[0.0, 1.0]`.
#[wasm_bindgen(js_name = clusterAreaFaithfulness)]
#[allow(clippy::too_many_arguments)]
pub fn js_cluster_area_faithfulness(
    layout: &JsLayout2d,
    labels: &JsValue,
    g_max: Option<f64>,
    sigma_cells: Option<f64>,
    kappa: Option<f64>,
    tau_frac: Option<f64>,
    boundary_r95: Option<f64>,
    stevens_gamma: Option<f64>,
    node_weights: &JsValue,
    cluster_weights: &JsValue,
) -> Result<f64, JsError> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let nw = parse_optional_f64_sequence(node_weights, "nodeWeights")?;
    let cw = parse_optional_f64_sequence(cluster_weights, "clusterWeights")?;
    let cfg = parse_cluster_area_config(
        g_max,
        sigma_cells,
        kappa,
        tau_frac,
        boundary_r95,
        stevens_gamma,
    )?;
    let soa = layout.inner().to_layout2d();
    cluster_area_faithfulness_with_weights(&soa, &parsed_labels, &cfg, nw.as_deref(), cw.as_deref())
        .map(|r| r.caf)
        .map_err(core_err_to_js)
}

/// Computes Grid-based Cluster Area Faithfulness (`G-CAF`) with per-cluster diagnostics ([`ClusterAreaReport`]).
#[wasm_bindgen(js_name = clusterAreaFaithfulnessDetail)]
#[allow(clippy::too_many_arguments)]
pub fn js_cluster_area_faithfulness_detail(
    layout: &JsLayout2d,
    labels: &JsValue,
    g_max: Option<f64>,
    sigma_cells: Option<f64>,
    kappa: Option<f64>,
    tau_frac: Option<f64>,
    boundary_r95: Option<f64>,
    stevens_gamma: Option<f64>,
    node_weights: &JsValue,
    cluster_weights: &JsValue,
) -> Result<JsClusterAreaReport, JsError> {
    let parsed_labels = parse_u32_sequence(labels, "labels")?;
    let nw = parse_optional_f64_sequence(node_weights, "nodeWeights")?;
    let cw = parse_optional_f64_sequence(cluster_weights, "clusterWeights")?;
    let cfg = parse_cluster_area_config(
        g_max,
        sigma_cells,
        kappa,
        tau_frac,
        boundary_r95,
        stevens_gamma,
    )?;
    let soa = layout.inner().to_layout2d();
    let inner = cluster_area_faithfulness_detail_with_weights(
        &soa,
        &parsed_labels,
        &cfg,
        nw.as_deref(),
        cw.as_deref(),
    )
    .map_err(core_err_to_js)?;
    Ok(JsClusterAreaReport { inner })
}
