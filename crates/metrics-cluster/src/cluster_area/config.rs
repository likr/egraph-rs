use egraph_core::{CoreError, CoreResult};

/// Maximum grid half-width (`S <= 2 * H_MAX + 1 = 2049`) to keep memory bounded
/// even when `boundary_r95` is `+inf` or very large.
pub(super) const H_MAX: usize = 1024;

/// Minimum cluster dominance ratio $w_k(c) / w_{\mathrm{tot}}(c)$ required for a grid
/// cell to participate in 8-connected spatial coherence (`C_coh`) without falling
/// into a foreign-cluster barrier. Set to `0.30` (above `0.25`) to avoid 4-way
/// tie fragility on checkerboard intersections.
pub(super) const COH_DOMINANCE_MIN: f64 = 0.30;

/// Configuration parameters for [`super::cluster_area_faithfulness`] and
/// [`super::cluster_area_faithfulness_detail`].
///
/// # Physical Ink Threshold & Stencil Coverage Limit
/// Each in-bounds vertex splats a discrete Gaussian kernel with effective footprint
/// area $\approx \pi (2\sigma)^2$ (truncated at radius $R = \lceil 3.5\sigma \rceil$
/// grid cells). The global physical visibility threshold
/// $\tau = \tau_{\mathrm{frac}} \, \rho_{\mathrm{canv}}$ is intentionally
/// label-independent so that $\sum_k A_k^{\mathrm{vis}} = \sum_c \Phi(c)$ preserves
/// partition of unity and isolated single-point noise does not saturate full
/// $(2R+1)^2$ stencils. Because cluster $k$'s realizable visible area cannot exceed
/// $\sim n_k \pi (2\sigma)^2$ (and strictly $\le n_k (2R+1)^2$), faithful target
/// area proportion matching requires $p_k A_{\mathrm{disk}} \lesssim n_k \pi (2\sigma)^2$
/// and $\rho_k / \rho_{\mathrm{canv}} \gtrsim \tau_{\mathrm{frac}}$, where
/// $\rho_k = n_k / (p_k A_{\mathrm{disk}})$.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClusterAreaConfig {
    /// Maximum grid resolution along the unit radius $R_{95}$ (must be `>= 12`; default `128`).
    pub g_max: usize,
    /// Gaussian splatting kernel standard deviation in grid cells (must be finite `> 0.0`; default `0.85`).
    pub sigma_cells: f64,
    /// Over-compression / scatter-tolerance multiplier $\kappa$ (must be finite `> 0.0`; default `10.0`).
    pub kappa: f64,
    /// Physical visibility saturation threshold fraction $\tau_{\mathrm{frac}}$ of
    /// $\rho_{\mathrm{canv}} = N / A_{\mathrm{disk}}$ (must be finite `> 0.0`; default `0.18`).
    /// Preserved globally across all clusters to maintain partition of unity and suppress
    /// isolated-point false positives.
    pub tau_frac: f64,
    /// Radial inclusion boundary in units of $R_{95}$ (must be `> 0.0` and non-NaN; `+inf`
    /// selects dense-grid resolution-protection mode bounded by `H_MAX` while preserving
    /// `g_eff >= 12`; default `2.0`).
    pub boundary_r95: f64,
    /// Stevens power-law exponent $\gamma$ for Step 4 target cluster weights
    /// $p_k \propto (p_k^{(w)})^\gamma$ (must be finite `> 0.0`; default `1.0`).
    /// Practical exponents $\gamma \in [0.7, 1.0]$ remain well within the discrete
    /// Gaussian stencil coverage envelope $p_k A_{\mathrm{disk}} \lesssim n_k \pi (2\sigma)^2$.
    pub stevens_gamma: f64,
}

impl Default for ClusterAreaConfig {
    fn default() -> Self {
        Self {
            g_max: 128,
            sigma_cells: 0.85,
            kappa: 10.0,
            tau_frac: 0.18,
            boundary_r95: 2.0,
            stevens_gamma: 1.0,
        }
    }
}

impl ClusterAreaConfig {
    /// Validates all configuration parameters.
    pub fn validate(&self) -> CoreResult<()> {
        if self.g_max < 12 {
            return Err(CoreError::invalid_input(
                "ClusterAreaConfig::g_max must be >= 12",
            ));
        }
        if !self.sigma_cells.is_finite() {
            return Err(CoreError::non_finite("ClusterAreaConfig::sigma_cells"));
        }
        if self.sigma_cells <= 0.0 {
            return Err(CoreError::invalid_input(
                "ClusterAreaConfig::sigma_cells must be strictly positive",
            ));
        }
        let r_stencil_f = (3.5 * self.sigma_cells).ceil();
        if !r_stencil_f.is_finite() || r_stencil_f > (H_MAX - 2) as f64 {
            return Err(CoreError::invalid_input(
                "ClusterAreaConfig::sigma_cells stencil radius exceeds maximum grid bound",
            ));
        }
        if !self.kappa.is_finite() {
            return Err(CoreError::non_finite("ClusterAreaConfig::kappa"));
        }
        if self.kappa <= 0.0 {
            return Err(CoreError::invalid_input(
                "ClusterAreaConfig::kappa must be strictly positive",
            ));
        }
        if !self.tau_frac.is_finite() {
            return Err(CoreError::non_finite("ClusterAreaConfig::tau_frac"));
        }
        if self.tau_frac <= 0.0 {
            return Err(CoreError::invalid_input(
                "ClusterAreaConfig::tau_frac must be strictly positive",
            ));
        }
        if self.boundary_r95.is_nan() {
            return Err(CoreError::non_finite("ClusterAreaConfig::boundary_r95"));
        }
        if self.boundary_r95 <= 0.0 {
            return Err(CoreError::invalid_input(
                "ClusterAreaConfig::boundary_r95 must be strictly positive",
            ));
        }
        if !self.stevens_gamma.is_finite() {
            return Err(CoreError::non_finite("ClusterAreaConfig::stevens_gamma"));
        }
        if self.stevens_gamma <= 0.0 {
            return Err(CoreError::invalid_input(
                "ClusterAreaConfig::stevens_gamma must be strictly positive",
            ));
        }
        Ok(())
    }
}

/// Summary report returned by [`super::cluster_area_faithfulness`].
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct ClusterAreaReport {
    /// Overall Cluster Area Faithfulness score in `[0.0, 1.0]`.
    pub caf: f64,
    /// Footprint-weighted adjusted purity $\sum_k q_k P_k$ in `[0.0, 1.0]`.
    pub purity_adj: f64,
    /// Visual area proportion agreement $1 - \frac{1}{2} \sum_k |p_k - q_k|$ in `[0.0, 1.0]`.
    pub prop_vis: f64,
    /// Target-weighted density resolution $\sum_k p_k E_k$ in `[0.0, 1.0]`.
    pub res_density: f64,
    /// Target-weighted deconvolution resolution $\sum_k p_k D_k$ in `[0.0, 1.0]`.
    pub res_deconv: f64,
    /// Target-weighted combined resolution $\sum_k p_k R_k$ in `[0.0, 1.0]`.
    pub res_combined: f64,
    /// Footprint-weighted 8-connected spatial coherence $C_{\mathrm{coh}}$ over active occupied cells in `[0.0, 1.0]`.
    pub spatial_coherence: f64,
    /// Fraction of vertices inside the effective radial evaluation disk `effective_boundary_r95 * R95`.
    pub in_frac: f64,
    /// Effective radial evaluation boundary in units of $R_{95}$ (`0.0` for degenerate early exits).
    pub effective_boundary_r95: f64,
    /// Effective grid resolution $g_{\mathrm{eff}}$ in cells per $R_{95}$ (`0` for degenerate early exits).
    pub g_eff: usize,
    /// Number of unique community labels $K$.
    pub num_clusters: usize,
    /// Total number of vertices $N$.
    pub num_nodes: usize,
}

/// Detailed report including per-cluster diagnostics returned by
/// [`super::cluster_area_faithfulness_detail`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ClusterAreaDetail {
    /// Aggregate summary metrics.
    pub report: ClusterAreaReport,
    /// Sorted unique original community IDs of length `K`.
    pub cluster_ids: Vec<u32>,
    /// Total vertex counts $n_k$ per cluster of length `K`.
    pub counts: Vec<usize>,
    /// Target cluster weights $p_k^{(\gamma)}$ summing to `1.0` (length `K`).
    pub p: Vec<f64>,
    /// Observed visible area shares $q_k = A_k^{\mathrm{vis}} / A_{\mathrm{tot}}$ (length `K`).
    pub q: Vec<f64>,
    /// Per-cluster adjusted spatial purity $P_k$ (length `K`).
    pub purity: Vec<f64>,
    /// Per-cluster combined resolution $R_k = \min(E_k, D_k)$ (length `K`).
    pub resolution: Vec<f64>,
    /// Per-cluster density resolution $E_k$ (length `K`).
    pub res_density: Vec<f64>,
    /// Per-cluster deconvolution resolution $D_k$ (length `K`).
    pub res_deconv: Vec<f64>,
    /// Per-cluster contribution to `caf` divided by $p_k$ (length `K`).
    pub caf_per_cluster: Vec<f64>,
    /// Per-cluster 8-connected spatial coherence over active occupied cells (length `K`).
    pub coherence: Vec<f64>,
}

#[inline]
pub(super) fn degenerate_detail(
    cluster_ids: Vec<u32>,
    counts: Vec<usize>,
    p: Vec<f64>,
    n: usize,
    include_detail: bool,
) -> ClusterAreaDetail {
    let k_count = cluster_ids.len();
    let report = ClusterAreaReport {
        num_clusters: k_count,
        num_nodes: n,
        ..ClusterAreaReport::default()
    };
    if !include_detail {
        return ClusterAreaDetail {
            report,
            ..ClusterAreaDetail::default()
        };
    }
    ClusterAreaDetail {
        report,
        cluster_ids,
        counts,
        p,
        q: vec![0.0; k_count],
        purity: vec![0.0; k_count],
        resolution: vec![0.0; k_count],
        res_density: vec![0.0; k_count],
        res_deconv: vec![0.0; k_count],
        caf_per_cluster: vec![0.0; k_count],
        coherence: vec![0.0; k_count],
    }
}

#[inline]
fn compute_stevens_weights(counts: &[usize], gamma: f64) -> Vec<f64> {
    let k_count = counts.len();
    if k_count == 0 {
        return Vec::new();
    }
    let max_c = counts.iter().copied().max().unwrap_or(1).max(1) as f64;
    let mut w_vec = Vec::with_capacity(k_count);
    let mut w_sum = 0.0_f64;
    for &c in counts {
        // Normalize by max_c before exponentiating so large gamma never overflows f64.
        let w = if (gamma - 1.0).abs() == 0.0 {
            (c as f64) / max_c
        } else {
            ((c as f64) / max_c).powf(gamma)
        };
        w_vec.push(w);
        w_sum += w;
    }
    if w_sum > 0.0 && w_sum.is_finite() {
        for w in &mut w_vec {
            *w = (*w / w_sum).clamp(0.0, 1.0);
        }
    }
    w_vec
}

#[inline]
fn apply_stevens_gamma(p_raw: &[f64], gamma: f64) -> Vec<f64> {
    let k_count = p_raw.len();
    if k_count == 0 {
        return Vec::new();
    }
    let max_p = p_raw.iter().copied().fold(0.0_f64, f64::max);
    if max_p <= 0.0 || !max_p.is_finite() {
        return vec![0.0; k_count];
    }
    let mut w_vec = Vec::with_capacity(k_count);
    let mut w_sum = 0.0_f64;
    for &pk in p_raw {
        let w = if (gamma - 1.0).abs() == 0.0 {
            pk / max_p
        } else {
            (pk / max_p).powf(gamma)
        };
        w_vec.push(w);
        w_sum += w;
    }
    if w_sum > 0.0 && w_sum.is_finite() {
        for w in &mut w_vec {
            *w = (*w / w_sum).clamp(0.0, 1.0);
        }
    }
    w_vec
}

pub(super) fn compute_target_weights(
    inv: &[u32],
    k_count: usize,
    stevens_gamma: f64,
    node_weights: Option<&[f64]>,
    cluster_weights: Option<&[f64]>,
) -> (Vec<usize>, Vec<f64>, bool) {
    let n = inv.len();
    let mut counts = vec![0usize; k_count];
    for &c in inv {
        counts[c as usize] += 1;
    }
    let n_f = n as f64;
    let has_custom_weights = node_weights.is_some() || cluster_weights.is_some();
    let p_raw: Vec<f64> = if let Some(nw) = node_weights {
        let mut raw_w = vec![0.0_f64; k_count];
        for i in 0..n {
            raw_w[inv[i] as usize] += nw[i];
        }
        let mut sum_raw: f64 = raw_w.iter().sum();
        if !sum_raw.is_finite() || sum_raw <= 0.0 {
            let max_nw = nw.iter().copied().fold(0.0_f64, f64::max);
            raw_w.fill(0.0);
            for i in 0..n {
                raw_w[inv[i] as usize] += nw[i] / max_nw;
            }
            sum_raw = raw_w.iter().sum();
        }
        raw_w
            .into_iter()
            .map(|w| (w / sum_raw).clamp(0.0, 1.0))
            .collect()
    } else if let Some(cw) = cluster_weights {
        let mut raw_w = cw.to_vec();
        let mut sum_raw: f64 = raw_w.iter().sum();
        if !sum_raw.is_finite() || sum_raw <= 0.0 {
            let max_cw = cw.iter().copied().fold(0.0_f64, f64::max);
            for (slot, &w) in raw_w.iter_mut().zip(cw.iter()) {
                *slot = w / max_cw;
            }
            sum_raw = raw_w.iter().sum();
        }
        raw_w
            .into_iter()
            .map(|w| (w / sum_raw).clamp(0.0, 1.0))
            .collect()
    } else {
        counts
            .iter()
            .map(|&c| ((c as f64) / n_f).clamp(0.0, 1.0))
            .collect()
    };

    let p = if (stevens_gamma - 1.0).abs() == 0.0 {
        p_raw
    } else if has_custom_weights {
        apply_stevens_gamma(&p_raw, stevens_gamma)
    } else {
        compute_stevens_weights(&counts, stevens_gamma)
    };

    (counts, p, has_custom_weights)
}
