//! Diagnostics emitted by sparse barrier and DoG SGD constructors (`SparseBarrierStats`, `DogStats`).

use egraph::layout::SparseBarrierStats;
use egraph::neighborhood::DogVaradhanStats;
use wasm_bindgen::prelude::*;

use crate::linalg_matrix::JsSpectralMomentStats;

/// Diagnostics emitted by `SgdSolver.sparseBarrierFromLandmarks` and `SgdSolver.sparseBarrierDog`.
#[wasm_bindgen(js_name = SparseBarrierStats)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsSparseBarrierStats {
    inner: SparseBarrierStats,
}

impl JsSparseBarrierStats {
    pub(crate) fn from_inner(inner: SparseBarrierStats) -> Self {
        Self { inner }
    }
}

#[wasm_bindgen(js_class = SparseBarrierStats)]
impl JsSparseBarrierStats {
    /// Returns the count of Tier 1 1-hop pairs.
    #[wasm_bindgen(getter, js_name = nPairs1hop)]
    pub fn n_pairs_1hop(&self) -> usize {
        self.inner.n_pairs_1hop
    }

    /// Returns the count of Tier 2 2-hop pairs.
    #[wasm_bindgen(getter, js_name = nPairs2hop)]
    pub fn n_pairs_2hop(&self) -> usize {
        self.inner.n_pairs_2hop
    }

    /// Returns the count of Tier 3 non-2-hop barrier pairs.
    #[wasm_bindgen(getter, js_name = nPairsNon2)]
    pub fn n_pairs_non2(&self) -> usize {
        self.inner.n_pairs_non2
    }

    /// Returns the count of one-sided hinge pairs.
    #[wasm_bindgen(getter, js_name = nPairsHinge)]
    pub fn n_pairs_hinge(&self) -> usize {
        self.inner.n_pairs_hinge
    }

    /// Returns the count of two-sided spring pairs.
    #[wasm_bindgen(getter, js_name = nPairsTwoSided)]
    pub fn n_pairs_two_sided(&self) -> usize {
        self.inner.n_pairs_two_sided
    }

    /// Returns the adaptive or explicit perplexity value.
    #[wasm_bindgen(getter, js_name = perplexityVal)]
    pub fn perplexity_val(&self) -> f64 {
        self.inner.perplexity_val
    }

    /// Returns the per-node negative sample budget used.
    #[wasm_bindgen(getter, js_name = perNodeNegUsed)]
    pub fn per_node_neg_used(&self) -> usize {
        self.inner.per_node_neg_used
    }

    /// Returns `"adaptive2hop"` or `"explicit"`.
    #[wasm_bindgen(getter, js_name = perplexitySource)]
    pub fn perplexity_source(&self) -> String {
        match self.inner.perplexity_source {
            "adaptive_2hop" => "adaptive2hop".to_owned(),
            other => other.to_owned(),
        }
    }

    /// Returns the applied relative distance floor.
    #[wasm_bindgen(getter, js_name = dFloor)]
    pub fn d_floor(&self) -> f64 {
        self.inner.d_floor
    }

    /// Returns the effective non-2-hop repulsion multiplier $\gamma_{\mathrm{eff}}$.
    #[wasm_bindgen(getter, js_name = gammaEff)]
    pub fn gamma_eff(&self) -> f64 {
        self.inner.gamma_eff
    }
}

/// Regularized DoG Varadhan landmark diagnostics returned by `SgdSolver.dogStats()`.
#[wasm_bindgen(js_name = DogStats)]
#[derive(Debug, Clone, PartialEq)]
pub struct JsDogStats {
    inner: DogVaradhanStats,
}

impl JsDogStats {
    pub(crate) fn from_inner(inner: DogVaradhanStats) -> Self {
        Self { inner }
    }
}

#[wasm_bindgen(js_class = DogStats)]
impl JsDogStats {
    #[wasm_bindgen(getter, js_name = momentStats)]
    pub fn moment_stats(&self) -> JsSpectralMomentStats {
        JsSpectralMomentStats::from_inner(self.inner.moment_stats.clone())
    }

    #[wasm_bindgen(getter)]
    pub fn t1(&self) -> f64 {
        self.inner.t1
    }

    #[wasm_bindgen(getter)]
    pub fn t2(&self) -> f64 {
        self.inner.t2
    }

    #[wasm_bindgen(getter)]
    pub fn alpha(&self) -> f64 {
        self.inner.alpha
    }

    #[wasm_bindgen(getter)]
    pub fn degree(&self) -> usize {
        self.inner.degree
    }

    #[wasm_bindgen(getter, js_name = deltaPsd)]
    pub fn delta_psd(&self) -> f64 {
        self.inner.delta_psd
    }

    #[wasm_bindgen(getter, js_name = hMin)]
    pub fn h_min(&self) -> f64 {
        self.inner.h_min
    }

    #[wasm_bindgen(getter, js_name = kMin)]
    pub fn k_min(&self) -> f64 {
        self.inner.k_min
    }

    #[wasm_bindgen(getter, js_name = kMed)]
    pub fn k_med(&self) -> f64 {
        self.inner.k_med
    }

    #[wasm_bindgen(getter, js_name = betaAuto)]
    pub fn beta_auto(&self) -> f64 {
        self.inner.beta_auto
    }

    #[wasm_bindgen(getter, js_name = betaEff)]
    pub fn beta_eff(&self) -> f64 {
        self.inner.beta_eff
    }

    #[wasm_bindgen(getter, js_name = numLandmarks)]
    pub fn num_landmarks(&self) -> usize {
        self.inner.n_landmarks
    }

    #[wasm_bindgen(getter, js_name = tCoarse)]
    pub fn t_coarse(&self) -> f64 {
        self.inner.t_coarse
    }

    #[wasm_bindgen(getter, js_name = usedCoarseScaffold)]
    pub fn used_coarse_scaffold(&self) -> bool {
        self.inner.used_coarse_scaffold
    }
}
