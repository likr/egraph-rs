//! WebAssembly bindings for regularized DoG Varadhan landmark tables.

use crate::csr_graph::JsCsrGraph;
use crate::js_convert::core_err_to_js;
use crate::js_options::{check_options_arg, opt_maybe_f64_prop, parse_dog_config_from_options};
use crate::linalg_matrix::JsCsrMatrix;
use egraph::neighborhood::{
    DogVaradhanLandmarks, DogVaradhanStats, RhghLandmarkStats,
    compute_regularized_dog_varadhan_landmarks,
    compute_regularized_dog_varadhan_landmarks_from_matrix,
};
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone, PartialEq)]
struct DogStatsSummary {
    t1: f64,
    t2: f64,
    alpha: f64,
    delta_psd: f64,
    beta_eff: f64,
    beta_auto: f64,
    degree: usize,
    used_coarse_scaffold: bool,
}

impl From<&DogVaradhanStats> for DogStatsSummary {
    fn from(s: &DogVaradhanStats) -> Self {
        Self {
            t1: s.t1,
            t2: s.t2,
            alpha: s.alpha,
            delta_psd: s.delta_psd,
            beta_eff: s.beta_eff,
            beta_auto: s.beta_auto,
            degree: s.degree,
            used_coarse_scaffold: s.used_coarse_scaffold,
        }
    }
}

/// Precomputed regularized DoG Varadhan landmark tables and R1/R2 regularization diagnostics.
#[wasm_bindgen(js_name = DogVaradhanLandmarks)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsDogVaradhanLandmarks {
    inner: DogVaradhanLandmarks,
    stats: DogStatsSummary,
}

impl JsDogVaradhanLandmarks {
    /// Returns a shared reference to the underlying `egraph_neighborhood::DogVaradhanLandmarks`.
    pub fn inner(&self) -> &DogVaradhanLandmarks {
        &self.inner
    }

    pub(crate) fn from_rhgh_parts(inner: DogVaradhanLandmarks, rhgh: &RhghLandmarkStats) -> Self {
        let stats = DogStatsSummary {
            t1: rhgh.gamma,
            t2: rhgh.gamma,
            alpha: rhgh.p_exp,
            delta_psd: 0.0,
            beta_eff: 0.0,
            beta_auto: 0.0,
            degree: 0,
            used_coarse_scaffold: false,
        };
        Self { inner, stats }
    }
}

#[wasm_bindgen(js_class = DogVaradhanLandmarks)]
impl JsDogVaradhanLandmarks {
    /// Returns the number of vertices in the graph.
    #[wasm_bindgen(getter, js_name = numNodes)]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Returns the number of landmark columns.
    #[wasm_bindgen(getter, js_name = numLandmarks)]
    pub fn num_landmarks(&self) -> usize {
        self.inner.num_landmarks()
    }

    /// Returns a copy of the selected landmark vertex IDs (`Uint32Array`).
    pub fn landmarks(&self) -> Vec<u32> {
        self.inner.landmarks().to_vec()
    }

    /// Returns a flat row-major `(numNodes, numLandmarks)` `Float64Array` Varadhan distance table.
    pub fn table(&self) -> Vec<f64> {
        self.inner.d_dog().to_vec()
    }

    /// Returns a flat row-major `(numNodes, numLandmarks)` coarse scaffold table, or `undefined`.
    #[wasm_bindgen(js_name = coarseTable)]
    pub fn coarse_table(&self) -> Option<Vec<f64>> {
        self.inner.d_coarse().map(|slice| slice.to_vec())
    }

    /// Returns the effective short-range diffusion time $t_1$.
    #[wasm_bindgen(getter)]
    pub fn t1(&self) -> f64 {
        self.stats.t1
    }

    /// Returns the effective long-range diffusion time $t_2$.
    #[wasm_bindgen(getter)]
    pub fn t2(&self) -> f64 {
        self.stats.t2
    }

    /// Returns the effective DoG attenuation $\alpha$.
    #[wasm_bindgen(getter)]
    pub fn alpha(&self) -> f64 {
        self.stats.alpha
    }

    /// Returns the R2 PSD projection shift $\delta_{\mathrm{psd}}$.
    #[wasm_bindgen(getter, js_name = deltaPsd)]
    pub fn delta_psd(&self) -> f64 {
        self.stats.delta_psd
    }

    /// Returns the effective rank-1 positivity floor `betaEff`.
    #[wasm_bindgen(getter, js_name = betaEff)]
    pub fn beta_eff(&self) -> f64 {
        self.stats.beta_eff
    }

    /// Returns the parameter-free automatic positivity floor `betaAuto`.
    #[wasm_bindgen(getter, js_name = betaAuto)]
    pub fn beta_auto(&self) -> f64 {
        self.stats.beta_auto
    }

    /// Returns the Chebyshev polynomial degree used for the DoG filter.
    #[wasm_bindgen(getter)]
    pub fn degree(&self) -> usize {
        self.stats.degree
    }

    /// Returns `true` if a coarse diffusive scaffold table was computed.
    #[wasm_bindgen(getter, js_name = usedCoarseScaffold)]
    pub fn used_coarse_scaffold(&self) -> bool {
        self.stats.used_coarse_scaffold
    }

    /// Evaluates landmark ALT lower-bound distances for pairs `(u[i], v[i])`.
    #[wasm_bindgen(js_name = altDistances)]
    pub fn alt_distances(&self, u: &[u32], v: &[u32]) -> Result<Vec<f64>, JsValue> {
        self.inner.alt_pair_distances(u, v).map_err(core_err_to_js)
    }
}

/// Computes regularized DoG Varadhan landmark tables and optional coarse heat scaffold.
#[wasm_bindgen(js_name = computeRegularizedDogVaradhanLandmarks)]
pub fn js_compute_regularized_dog_varadhan_landmarks(
    graph: &JsCsrGraph,
    options: &JsValue,
) -> Result<JsDogVaradhanLandmarks, JsValue> {
    let cfg = parse_dog_config_from_options(options, "computeRegularizedDogVaradhanLandmarks")?;
    let (inner, stats) =
        compute_regularized_dog_varadhan_landmarks(graph.inner(), &cfg).map_err(core_err_to_js)?;
    Ok(JsDogVaradhanLandmarks {
        inner,
        stats: DogStatsSummary::from(&stats),
    })
}

/// Computes regularized DoG Varadhan landmark tables from a custom symmetric PSD `CsrMatrix`
/// with dynamic `[0, lambdaMax]` spectrum scaling.
#[wasm_bindgen(js_name = computeRegularizedDogVaradhanLandmarksFromMatrix)]
pub fn js_compute_regularized_dog_varadhan_landmarks_from_matrix(
    graph: &JsCsrGraph,
    matrix: &JsCsrMatrix,
    options: &JsValue,
) -> Result<JsDogVaradhanLandmarks, JsValue> {
    let cfg =
        parse_dog_config_from_options(options, "computeRegularizedDogVaradhanLandmarksFromMatrix")?;
    let lambda_max =
        if check_options_arg(options, "computeRegularizedDogVaradhanLandmarksFromMatrix")? {
            opt_maybe_f64_prop(options, "lambdaMax")?
        } else {
            None
        };
    let (inner, stats) = compute_regularized_dog_varadhan_landmarks_from_matrix(
        graph.inner(),
        matrix.inner(),
        lambda_max,
        &cfg,
    )
    .map_err(core_err_to_js)?;
    Ok(JsDogVaradhanLandmarks {
        inner,
        stats: DogStatsSummary::from(&stats),
    })
}
