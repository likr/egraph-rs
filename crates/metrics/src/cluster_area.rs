//! Grid-based Cluster Area Faithfulness (`G-CAF`) metric.
//!
//! Evaluates how faithfully a 2D graph layout preserves a discrete community
//! partition across five geometric and perceptual dimensions:
//! - **Density & deconvolution resolution** ($E_k, D_k, R_k$): whether each
//!   community occupies sufficient non-collapsed canvas area without over-compression
//!   or scatter fragmentation.
//! - **Adjusted spatial purity** ($P_k$): pair-purity excess over the random-mixing
//!   baseline weighted by the visible density field $\Phi(x, y)$.
//! - **Visual area proportion** ($q_k$ vs. $p_k^{(\gamma)}$): balanced-match
//!   alignment between each community's visible area share and its (optionally
//!   Stevens power-law generalized) target weight.
//! - **Spatial coherence** ($C_{\mathrm{coh}}$): fraction of each community's
//!   visible area concentrated in its largest 8-connected occupied grid component.
//!
//! # Algorithm Highlights
//! - **P0-1 ($O(N(2R+1)^2 + S^2)$, $K$-independent)**: Groups in-bounds vertices
//!   into CSR buckets via $O(N + K)$ counting sort and reuses a single `S x S` grid
//!   backed by [`VisitStamp`] epoch markers and a `touched` cell list.
//! - **P0-2 (Invariance & Hardened Welzl MEC)**: Pre-normalizes coordinates by
//!   coordinate median and $Q_{0.95}$ radial quantile before Weiszfeld refinement
//!   and two-stage Exact Minimum Enclosing Circle (MEC) centering; uses
//!   vertex-$a$-local circumcircle arithmetic, relative Akl-Toussaint 8-extreme
//!   filtering, Andrew's monotone chain convex hull, and fixed-seed SplitMix64
//!   shuffling for $O(h)$ expected MEC runtime and zero permutation variance.
//! - **P0-3 (Edge Semantics & Generalizations)**: Explicitly guards $n_k \le 1$ and
//!   $B_k \le 10^{-15}$ boundaries, supports Stevens $\gamma$ target weighting, and
//!   computes 8-connected cell-component spatial coherence.

use std::f64::consts::PI;

use egraph_core::{CoreError, CoreResult, Layout2d, Scalar, VisitStamp};

/// Maximum grid half-width (`S <= 2 * H_MAX + 1 = 2049`) to keep memory bounded
/// even when `boundary_r95` is `+inf` or very large.
const H_MAX: usize = 1024;

/// Configuration parameters for [`cluster_area_faithfulness`] and
/// [`cluster_area_faithfulness_detail`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClusterAreaConfig {
    /// Maximum grid resolution along the unit radius $R_{95}$ (must be `>= 12`; default `128`).
    pub g_max: usize,
    /// Gaussian splatting kernel standard deviation in grid cells (must be finite `> 0.0`; default `0.85`).
    pub sigma_cells: f64,
    /// Over-compression / scatter-tolerance multiplier $\kappa$ (must be finite `> 0.0`; default `10.0`).
    pub kappa: f64,
    /// Visibility saturation threshold fraction $\tau_{\mathrm{frac}}$ of $\rho_{\mathrm{canv}}$
    /// (must be finite `> 0.0`; default `0.18`).
    pub tau_frac: f64,
    /// Radial inclusion boundary in units of $R_{95}$ (must be `> 0.0` and non-NaN; `+inf` is
    /// accepted and safely bounded by the `H_MAX` grid limit; default `2.0`).
    pub boundary_r95: f64,
    /// Stevens power-law exponent $\gamma$ for Step 4 target cluster weights
    /// $p_k^{(\gamma)} \propto n_k^\gamma$ (must be finite `> 0.0`; default `1.0`).
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

/// Summary report returned by [`cluster_area_faithfulness`].
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
    /// Footprint-weighted 8-connected spatial coherence $C_{\mathrm{coh}} = \sum_k q_k \, \mathrm{coherence}_k$ in `[0.0, 1.0]`.
    pub spatial_coherence: f64,
    /// Fraction of vertices inside the radial evaluation disk `boundary_r95 * R95`.
    pub in_frac: f64,
    /// Number of unique community labels $K$.
    pub num_clusters: usize,
    /// Total number of vertices $N$.
    pub num_nodes: usize,
}

/// Detailed report including per-cluster diagnostics returned by
/// [`cluster_area_faithfulness_detail`].
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
    /// Per-cluster 8-connected spatial coherence (fraction of $A_k^{\mathrm{vis}}$ in largest
    /// 8-connected occupied grid component, length `K`).
    pub coherence: Vec<f64>,
}

/// Evaluates Grid-based Cluster Area Faithfulness (`G-CAF`) on `layout` and `community_labels`.
pub fn cluster_area_faithfulness<S: Scalar>(
    layout: &Layout2d<S>,
    community_labels: &[u32],
    config: &ClusterAreaConfig,
) -> CoreResult<ClusterAreaReport> {
    let detail = cluster_area_faithfulness_impl(layout, community_labels, config, false)?;
    Ok(detail.report)
}

/// Evaluates Grid-based Cluster Area Faithfulness (`G-CAF`) with full per-cluster diagnostics.
pub fn cluster_area_faithfulness_detail<S: Scalar>(
    layout: &Layout2d<S>,
    community_labels: &[u32],
    config: &ClusterAreaConfig,
) -> CoreResult<ClusterAreaDetail> {
    cluster_area_faithfulness_impl(layout, community_labels, config, true)
}

fn cluster_area_faithfulness_impl<S: Scalar>(
    layout: &Layout2d<S>,
    community_labels: &[u32],
    config: &ClusterAreaConfig,
    include_detail: bool,
) -> CoreResult<ClusterAreaDetail> {
    let n = layout.len();
    if community_labels.len() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            community_labels.len(),
            "cluster_area_faithfulness::community_labels",
        ));
    }
    layout.validate_finite()?;
    config.validate()?;

    if n == 0 {
        return Ok(ClusterAreaDetail::default());
    }

    // Copy coordinates once into f64 and canonicalize -0.0 to +0.0 so reflections
    // (e.g. y -> -y) cannot perturb f64::total_cmp tie-breaking.
    let xs_in = layout.x();
    let ys_in = layout.y();
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for i in 0..n {
        let x = xs_in[i].to_f64();
        let y = ys_in[i].to_f64();
        xs.push(if x == 0.0 { 0.0 } else { x });
        ys.push(if y == 0.0 { 0.0 } else { y });
    }

    cluster_area_core(&xs, &ys, community_labels, config, include_detail)
}

/// 8-pass 8-bit LSD radix sort on `u64` keys (falling back to `sort_unstable` for `N <= 1024`).
fn radix_sort_u64(keys: &mut Vec<u64>, scratch: &mut Vec<u64>) {
    let n = keys.len();
    if n <= 1024 {
        keys.sort_unstable();
        return;
    }
    scratch.resize(n, 0);
    for pass in 0..8usize {
        let shift = pass * 8;
        let mut counts = [0usize; 256];
        for &k in keys.iter() {
            counts[((k >> shift) & 0xFF) as usize] += 1;
        }
        let first_byte = ((keys[0] >> shift) & 0xFF) as usize;
        if counts[first_byte] == n {
            continue;
        }
        let mut offsets = [0usize; 256];
        let mut acc = 0usize;
        for b in 0..256 {
            offsets[b] = acc;
            acc += counts[b];
        }
        for &k in keys.iter() {
            let bucket = ((k >> shift) & 0xFF) as usize;
            let pos = offsets[bucket];
            scratch[pos] = k;
            offsets[bucket] = pos + 1;
        }
        std::mem::swap(keys, scratch);
    }
}

/// Maps arbitrary `u32` labels to sorted `cluster_ids` and dense `0..K` indices `inv`
/// without `HashMap`, matching `np.unique(labels, return_inverse=True)`.
fn remap_community_labels(labels: &[u32]) -> CoreResult<(Vec<u32>, Vec<u32>)> {
    let n = labels.len();
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let max_lbl = *labels.iter().max().unwrap_or(&0);
    let dense_limit = n.max(1024).saturating_mul(4).min(4_000_000);
    if (max_lbl as usize) <= dense_limit {
        let m = (max_lbl as usize) + 1;
        // u32::MAX is a safe sentinel because m <= 4_000_001 < u32::MAX.
        let mut map = vec![u32::MAX; m];
        for &lbl in labels {
            map[lbl as usize] = 0;
        }
        let mut cluster_ids = Vec::new();
        for (lbl, slot) in map.iter_mut().enumerate() {
            if *slot == 0 {
                let k = cluster_ids.len() as u32;
                *slot = k;
                cluster_ids.push(lbl as u32);
            }
        }
        let mut inv = Vec::with_capacity(n);
        for &lbl in labels {
            inv.push(map[lbl as usize]);
        }
        Ok((cluster_ids, inv))
    } else {
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_input(
                "community_labels length exceeds u32 index range",
            ));
        }
        let mut keys = Vec::with_capacity(n);
        for (i, &lbl) in labels.iter().enumerate() {
            keys.push((u64::from(lbl) << 32) | (i as u64));
        }
        let mut scratch = Vec::new();
        radix_sort_u64(&mut keys, &mut scratch);

        let mut cluster_ids = Vec::new();
        let mut inv = vec![0u32; n];
        let mut prev_lbl = 0u32;
        let mut has_prev = false;
        for &key in &keys {
            let lbl = (key >> 32) as u32;
            let idx = (key & 0xFFFF_FFFF) as usize;
            if !has_prev || lbl != prev_lbl {
                cluster_ids.push(lbl);
                prev_lbl = lbl;
                has_prev = true;
            }
            inv[idx] = (cluster_ids.len() - 1) as u32;
        }
        Ok((cluster_ids, inv))
    }
}

/// Linear-interpolation $O(N)$ quantile matching `np.quantile(buf, q)`.
fn quantile_linear(buf: &mut [f64], q: f64) -> f64 {
    let n = buf.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return buf[0];
    }
    let h = ((n - 1) as f64) * q;
    let lo = (h.floor() as usize).min(n - 1);
    let hi = (lo + 1).min(n - 1);
    let frac = h - (lo as f64);
    let (_, v_lo, right) = buf.select_nth_unstable_by(lo, f64::total_cmp);
    let v_lo = *v_lo;
    if hi == lo || frac == 0.0 {
        return v_lo;
    }
    let v_hi = right.iter().copied().min_by(f64::total_cmp).unwrap_or(v_lo);
    v_lo + frac * (v_hi - v_lo)
}

#[inline]
fn median_linear(buf: &mut [f64]) -> f64 {
    quantile_linear(buf, 0.5)
}

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[inline]
fn cross_2d(o: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

#[inline]
fn cmp_point(a: (f64, f64), b: (f64, f64)) -> std::cmp::Ordering {
    a.0.total_cmp(&b.0).then_with(|| a.1.total_cmp(&b.1))
}

#[inline]
fn c2(a: (f64, f64), b: (f64, f64)) -> ((f64, f64), f64) {
    let cx = 0.5 * (a.0 + b.0);
    let cy = 0.5 * (a.1 + b.1);
    let r = 0.5 * (a.0 - b.0).hypot(a.1 - b.1);
    ((cx, cy), r)
}

/// Vertex-`a`-local circumcircle of `(a, b, d)` avoiding catastrophic cancellation
/// under large coordinate offsets, with relative collinearity fallback.
#[inline]
fn c3(a: (f64, f64), b: (f64, f64), d: (f64, f64)) -> ((f64, f64), f64) {
    let bx = b.0 - a.0;
    let by = b.1 - a.1;
    let dx = d.0 - a.0;
    let dy = d.1 - a.1;
    let b2 = bx * bx + by * by;
    let d2 = dx * dx + dy * dy;
    let det = 2.0 * (bx * dy - by * dx);
    if det.abs() <= 1e-15 * b2.max(d2) {
        let (c_ab, r_ab) = c2(a, b);
        let (c_bd, r_bd) = c2(b, d);
        let (c_ad, r_ad) = c2(a, d);
        if r_ab >= r_bd && r_ab >= r_ad {
            (c_ab, r_ab)
        } else if r_bd >= r_ad {
            (c_bd, r_bd)
        } else {
            (c_ad, r_ad)
        }
    } else {
        let ux = (dy * b2 - by * d2) / det;
        let uy = (bx * d2 - dx * b2) / det;
        ((a.0 + ux, a.1 + uy), ux.hypot(uy))
    }
}

#[inline]
fn in_circle(c: (f64, f64), r: f64, p: (f64, f64)) -> bool {
    (p.0 - c.0).hypot(p.1 - c.1) <= r * (1.0 + 1e-12)
}

/// Akl-Toussaint 8-extreme interior point rejection with relative edge tolerance
/// and permutation-invariant tie-breaking.
fn akl_toussaint_filter(pts: &[(f64, f64)]) -> Vec<(f64, f64)> {
    if pts.len() <= 64 {
        return pts.to_vec();
    }
    let mut extremes = [pts[0]; 8];
    let mut keys = [
        pts[0].0,
        pts[0].0 + pts[0].1,
        pts[0].1,
        pts[0].0 - pts[0].1,
        pts[0].0,
        pts[0].0 + pts[0].1,
        pts[0].1,
        pts[0].0 - pts[0].1,
    ];
    for &p in &pts[1..] {
        let p_keys = [
            p.0,
            p.0 + p.1,
            p.1,
            p.0 - p.1,
            p.0,
            p.0 + p.1,
            p.1,
            p.0 - p.1,
        ];
        // Min indices: 0 (x), 3 (x-y), 5 (x+y), 6 (y)
        for &idx in &[0usize, 3, 5, 6] {
            let ord = p_keys[idx]
                .total_cmp(&keys[idx])
                .then_with(|| cmp_point(p, extremes[idx]));
            if ord == std::cmp::Ordering::Less {
                keys[idx] = p_keys[idx];
                extremes[idx] = p;
            }
        }
        // Max indices: 1 (x+y), 2 (y), 4 (x), 7 (x-y)
        for &idx in &[1usize, 2, 4, 7] {
            let ord = p_keys[idx]
                .total_cmp(&keys[idx])
                .then_with(|| cmp_point(p, extremes[idx]));
            if ord == std::cmp::Ordering::Greater {
                keys[idx] = p_keys[idx];
                extremes[idx] = p;
            }
        }
    }

    let mut poly: Vec<(f64, f64)> = extremes.to_vec();
    poly.sort_unstable_by(|&a, &b| cmp_point(a, b));
    poly.dedup();
    if poly.len() < 3 {
        return pts.to_vec();
    }

    let inv_len = 1.0 / (poly.len() as f64);
    let cx = poly.iter().map(|p| p.0).sum::<f64>() * inv_len;
    let cy = poly.iter().map(|p| p.1).sum::<f64>() * inv_len;
    poly.sort_unstable_by(|&a, &b| {
        (a.1 - cy)
            .atan2(a.0 - cx)
            .total_cmp(&(b.1 - cy).atan2(b.0 - cx))
            .then_with(|| cmp_point(a, b))
    });

    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for &(px, py) in &poly {
        min_x = min_x.min(px);
        max_x = max_x.max(px);
        min_y = min_y.min(py);
        max_y = max_y.max(py);
    }
    let poly_scale = (max_x - min_x).max(max_y - min_y).max(1e-300);

    let m = poly.len();
    let mut edges = Vec::with_capacity(m);
    for i in 0..m {
        let a = poly[i];
        let b = poly[(i + 1) % m];
        let ex = b.0 - a.0;
        let ey = b.1 - a.1;
        let tol = 1e-12 * ex.hypot(ey) * poly_scale;
        edges.push((a.0, a.1, ex, ey, tol));
    }

    let mut kept = Vec::with_capacity(pts.len().min(512));
    for &p in pts {
        let mut strictly_inside = true;
        for &(ax, ay, ex, ey, tol) in &edges {
            let c = ex * (p.1 - ay) - ey * (p.0 - ax);
            if c <= tol {
                strictly_inside = false;
                break;
            }
        }
        if !strictly_inside {
            kept.push(p);
        }
    }

    if kept.len() < 3 { pts.to_vec() } else { kept }
}

/// Computes the exact Minimum Enclosing Circle `(center, radius)` of `pts` via
/// Akl-Toussaint filtering, Andrew's monotone chain convex hull, lexicographical
/// canonicalization, fixed-seed SplitMix64 shuffling, and Welzl's incremental algorithm.
fn exact_mec(pts: &[(f64, f64)]) -> ((f64, f64), f64) {
    if pts.is_empty() {
        return ((0.0, 0.0), 0.0);
    }
    let mut cand = akl_toussaint_filter(pts);
    cand.sort_unstable_by(|&a, &b| cmp_point(a, b));
    cand.dedup();

    if cand.len() == 1 {
        return (cand[0], 0.0);
    }
    if cand.len() == 2 {
        return c2(cand[0], cand[1]);
    }

    let mut lower = Vec::with_capacity(cand.len());
    for &p in &cand {
        while lower.len() >= 2 && cross_2d(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0
        {
            lower.pop();
        }
        lower.push(p);
    }

    let mut upper = Vec::with_capacity(cand.len());
    for &p in cand.iter().rev() {
        while upper.len() >= 2 && cross_2d(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0
        {
            upper.pop();
        }
        upper.push(p);
    }

    lower.pop();
    upper.pop();
    let mut hull = lower;
    hull.extend_from_slice(&upper);

    if hull.is_empty() {
        return (cand[0], 0.0);
    }
    if hull.len() == 1 {
        return (hull[0], 0.0);
    }

    // Explicit lexicographical canonicalization guarantees identical shuffle order
    // regardless of input permutation or polygon orientation.
    hull.sort_unstable_by(|&a, &b| cmp_point(a, b));
    let mut rng_state = 42_u64;
    for i in (1..hull.len()).rev() {
        let j = (((splitmix64(&mut rng_state) as u128) * ((i + 1) as u128)) >> 64) as usize;
        hull.swap(i, j);
    }

    let mut c = hull[0];
    let mut r = 0.0_f64;
    for i in 1..hull.len() {
        let pi = hull[i];
        if !in_circle(c, r, pi) {
            c = pi;
            r = 0.0;
            for j in 0..i {
                let pj = hull[j];
                if !in_circle(c, r, pj) {
                    (c, r) = c2(pi, pj);
                    for &pk in &hull[0..j] {
                        if !in_circle(c, r, pk) {
                            (c, r) = c3(pi, pj, pk);
                        }
                    }
                }
            }
        }
    }
    (c, r)
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

/// Fills `buf` (`len == 2 * r_stencil + 1`) with normalized 1D Gaussian stencil
/// weights centered at fractional offset `frac` (`|frac| <= 0.5`).
///
/// Shifting squared exponents by `min_dx2 = frac * frac` prevents catastrophic
/// underflow and `0.0 * (-inf) = NaN` when `sigma_cells` is arbitrarily small,
/// smoothly converging to the exact Kronecker delta (or `[0.5, 0.5]` at half-cell
/// ties) as `sigma_cells -> 0+`.
#[inline]
fn fill_1d_gaussian(
    buf: &mut [f64],
    r_stencil: usize,
    frac: f64,
    sigma_cells: f64,
    inv_two_sig2: f64,
) {
    let _ = sigma_cells;
    let min_dx2 = frac * frac;
    let r_stencil_i = r_stencil as isize;
    let mut sum = 0.0_f64;
    for (t_idx, slot) in buf.iter_mut().enumerate() {
        let off = (t_idx as isize) - r_stencil_i;
        let dx = (off as f64) - frac;
        let delta2 = (dx * dx - min_dx2).max(0.0);
        let val = if delta2 == 0.0 {
            // Avoid 0.0 * (-inf) = NaN when inv_two_sig2 overflows for tiny sigma_cells.
            1.0
        } else if !inv_two_sig2.is_finite() {
            0.0
        } else {
            (inv_two_sig2 * delta2).exp()
        };
        *slot = val;
        sum += val;
    }
    let inv_sum = 1.0 / sum;
    for slot in buf.iter_mut() {
        *slot *= inv_sum;
    }
}

#[inline]
fn degenerate_detail(
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

/// Non-generic `f64` core implementation keeping WASM binary footprint minimal.
fn cluster_area_core(
    xs: &[f64],
    ys: &[f64],
    community_labels: &[u32],
    config: &ClusterAreaConfig,
    include_detail: bool,
) -> CoreResult<ClusterAreaDetail> {
    let n = xs.len();
    let (cluster_ids, inv) = remap_community_labels(community_labels)?;
    let k_count = cluster_ids.len();

    let mut counts = vec![0usize; k_count];
    for &c in &inv {
        counts[c as usize] += 1;
    }
    let n_f = n as f64;
    let p_raw: Vec<f64> = counts
        .iter()
        .map(|&c| ((c as f64) / n_f).clamp(0.0, 1.0))
        .collect();
    let p = if (config.stevens_gamma - 1.0).abs() == 0.0 {
        p_raw.clone()
    } else {
        compute_stevens_weights(&counts, config.stevens_gamma)
    };

    if n < 2 || k_count < 2 {
        return Ok(degenerate_detail(cluster_ids, counts, p, n, include_detail));
    }

    // Step 1a: Pre-normalize coordinates by median + Q_0.95 to guarantee full
    // scale invariance across 1e-100..1e100 and translation stability.
    let mut scratch = xs.to_vec();
    let m_pre_x = median_linear(&mut scratch);
    scratch.copy_from_slice(ys);
    let m_pre_y = median_linear(&mut scratch);

    for i in 0..n {
        scratch[i] = (xs[i] - m_pre_x).hypot(ys[i] - m_pre_y);
    }
    let r_pre = quantile_linear(&mut scratch, 0.95);
    if !r_pre.is_finite() || r_pre <= 0.0 || !(1.0 / r_pre).is_finite() {
        return Ok(degenerate_detail(cluster_ids, counts, p, n, include_detail));
    }

    let mut x_pre = Vec::with_capacity(n);
    let mut y_pre = Vec::with_capacity(n);
    for i in 0..n {
        x_pre.push((xs[i] - m_pre_x) / r_pre);
        y_pre.push((ys[i] - m_pre_y) / r_pre);
    }

    // Step 1b: 8-step Weiszfeld L1 median initialized at coordinate-wise median.
    scratch.copy_from_slice(&x_pre);
    let mut mx = median_linear(&mut scratch);
    scratch.copy_from_slice(&y_pre);
    let mut my = median_linear(&mut scratch);
    for _ in 0..8 {
        let mut w_sum = 0.0_f64;
        let mut wx_sum = 0.0_f64;
        let mut wy_sum = 0.0_f64;
        for i in 0..n {
            let d = (x_pre[i] - mx).hypot(y_pre[i] - my);
            let w = 1.0 / d.max(1e-12);
            w_sum += w;
            wx_sum += w * x_pre[i];
            wy_sum += w * y_pre[i];
        }
        if w_sum > 0.0 && w_sum.is_finite() {
            mx = wx_sum / w_sum;
            my = wy_sum / w_sum;
        }
    }

    let mut d_buf = vec![0.0_f64; n];
    for i in 0..n {
        d_buf[i] = (x_pre[i] - mx).hypot(y_pre[i] - my);
    }
    scratch.copy_from_slice(&d_buf);
    let r0 = quantile_linear(&mut scratch, 0.95);
    if !r0.is_finite() || r0 <= 1e-12 {
        return Ok(degenerate_detail(cluster_ids, counts, p, n, include_detail));
    }

    // Step 1c: Two-stage exact MEC centering.
    // Match `in_circle`'s 1e-12 relative tolerance across all hypot-derived radial
    // thresholds so 1-ULP platform `hypot` differences never split tied boundary points.
    let mut mec_pts = Vec::with_capacity(n);
    let r0_cut = (1.5 * r0) * (1.0 + 1e-12);
    for i in 0..n {
        if d_buf[i] <= r0_cut {
            mec_pts.push((x_pre[i], y_pre[i]));
        }
    }
    let (mu1, _) = exact_mec(&mec_pts);

    for i in 0..n {
        d_buf[i] = (x_pre[i] - mu1.0).hypot(y_pre[i] - mu1.1);
    }
    scratch.copy_from_slice(&d_buf);
    let r1 = quantile_linear(&mut scratch, 0.95);
    let r1_cut = r1 * (1.0 + 1e-12);

    mec_pts.clear();
    for i in 0..n {
        if d_buf[i] <= r1_cut {
            mec_pts.push((x_pre[i], y_pre[i]));
        }
    }
    let (mu, _) = exact_mec(&mec_pts);

    for i in 0..n {
        d_buf[i] = (x_pre[i] - mu.0).hypot(y_pre[i] - mu.1);
    }
    scratch.copy_from_slice(&d_buf);
    let r95 = quantile_linear(&mut scratch, 0.95);
    if !r95.is_finite() || r95 <= 1e-12 {
        return Ok(degenerate_detail(cluster_ids, counts, p, n, include_detail));
    }

    // Step 2: Grid sizing, frozen boundary_r95 OOM guard, and CSR counting sort.
    let denom_g = 0.55 * (PI / n_f).sqrt();
    let raw_g = (config.sigma_cells / denom_g).floor();
    let g_eff = if raw_g.is_finite() {
        (raw_g as usize).clamp(12, config.g_max)
    } else {
        config.g_max
    };
    let g_eff_f = g_eff as f64;
    let r_stencil = (3.5 * config.sigma_cells).ceil() as usize;
    let stencil_len = 2 * r_stencil + 1;
    let max_b_grid = ((H_MAX - r_stencil - 1) as f64) / g_eff_f;
    let effective_b = config.boundary_r95.min(max_b_grid);
    let r_cut = effective_b * r95 * (1.0 + 1e-12);

    let mut in_counts = vec![0usize; k_count];
    let mut n_in = 0usize;
    let mut max_abs_in = 0.0_f64;
    for i in 0..n {
        if d_buf[i] <= r_cut {
            in_counts[inv[i] as usize] += 1;
            n_in += 1;
            let xn = (x_pre[i] - mu.0) / r95;
            let yn = (y_pre[i] - mu.1) / r95;
            max_abs_in = max_abs_in.max(xn.abs()).max(yn.abs());
        }
    }
    let in_frac = ((n_in as f64) / n_f).clamp(0.0, 1.0);

    let h = if config.boundary_r95 <= max_b_grid {
        ((config.boundary_r95 * g_eff_f).ceil() as usize + r_stencil + 1).min(H_MAX)
    } else {
        ((max_abs_in * g_eff_f).ceil() as usize + r_stencil + 1).min(H_MAX)
    };
    let h_f = h as f64;
    let s = 2 * h + 1;
    let s_sq = s * s;

    // P0-1: O(N + K) CSR bucket grouping of in-bounds normalized coordinates.
    let mut offsets = vec![0usize; k_count + 1];
    for k in 0..k_count {
        offsets[k + 1] = offsets[k] + in_counts[k];
    }
    let mut cursor = offsets[..k_count].to_vec();
    let mut bucket_x = vec![0.0_f64; n_in];
    let mut bucket_y = vec![0.0_f64; n_in];
    for i in 0..n {
        if d_buf[i] <= r_cut {
            let k = inv[i] as usize;
            let pos = cursor[k];
            cursor[k] = pos + 1;
            bucket_x[pos] = (x_pre[i] - mu.0) / r95;
            bucket_y[pos] = (y_pre[i] - mu.1) / r95;
        }
    }

    // Reference constants.
    let mut g0 = vec![0.0_f64; stencil_len];
    let inv_two_sig2 = -0.5 / (config.sigma_cells * config.sigma_cells);
    fill_1d_gaussian(&mut g0, r_stencil, 0.0, config.sigma_cells, inv_two_sig2);
    let mut m0_entropy = 0.0_f64;
    for &val in &g0 {
        if val > 0.0 {
            m0_entropy -= val * val.max(1e-30).ln();
        }
    }
    let m0_disc = (2.0 * m0_entropy).exp();
    let a_disk = (PI / 0.95) * g_eff_f * g_eff_f;
    let rho_canv = n_f / a_disk;
    let tau = config.tau_frac * rho_canv;
    let cap_density = config.kappa * rho_canv;

    // Global density accumulation across all in-bounds points.
    // Track subtraction-free pairwise weight sum Q_m = 2 * sum_{i < j} w_i * w_j
    // via Q_{m+1} = Q_m + 2 * S_m * w_{m+1} to avoid catastrophic cancellation in
    // S^2 - sum(w^2) when one dominant Gaussian weight dwarfs a tail contribution.
    let mut w_tot = vec![0.0_f64; s_sq];
    let mut q_tot = vec![0.0_f64; s_sq];
    let mut wx = vec![0.0_f64; stencil_len];
    let mut wy = vec![0.0_f64; stencil_len];
    let r_stencil_i = r_stencil as isize;
    let s_i = s as isize;

    for m in 0..n_in {
        let u = bucket_x[m] * g_eff_f + h_f;
        let v = bucket_y[m] * g_eff_f + h_f;
        // round_ties_even matches np.rint() bit-for-bit.
        let iu = u.round_ties_even() as isize;
        let iv = v.round_ties_even() as isize;
        let fu = u - (iu as f64);
        let fv = v - (iv as f64);

        fill_1d_gaussian(&mut wx, r_stencil, fu, config.sigma_cells, inv_two_sig2);
        fill_1d_gaussian(&mut wy, r_stencil, fv, config.sigma_cells, inv_two_sig2);

        for (dy_idx, &wy_val) in wy.iter().enumerate() {
            let gy = iv + (dy_idx as isize - r_stencil_i);
            if gy < 0 || gy >= s_i {
                continue;
            }
            let row_off = (gy as usize) * s;
            for (dx_idx, &wx_val) in wx.iter().enumerate() {
                let gx = iu + (dx_idx as isize - r_stencil_i);
                if gx < 0 || gx >= s_i {
                    continue;
                }
                let w = wx_val * wy_val;
                let cell = row_off + (gx as usize);
                q_tot[cell] += 2.0 * w_tot[cell] * w;
                w_tot[cell] += w;
            }
        }
    }

    let mut phi_all = vec![0.0_f64; s_sq];
    for c in 0..s_sq {
        let wt = w_tot[c];
        if wt > 0.0 {
            // Compute (wt / tau)^2 before squaring and use exp_m1 to avoid both
            // 1 - exp(-x) subtraction cancellation and 0.0 * inf = NaN when wt and tau underflow.
            let wt_tau = wt / tau;
            phi_all[c] = -(-(wt_tau * wt_tau)).exp_m1();
        }
    }

    // Step 3: P0-1 single reusable S x S grid + VisitStamp + touched cells, plus P0-3 guards
    // and 8-connected spatial coherence BFS.
    let mut wk_grid = vec![0.0_f64; s_sq];
    let mut qk_grid = vec![0.0_f64; s_sq];
    let mut splat_stamp = VisitStamp::new(s_sq);
    let mut bfs_stamp = VisitStamp::new(s_sq);
    let mut touched = Vec::with_capacity(1024);
    let mut stack = Vec::with_capacity(256);

    let mut e_k = vec![0.0_f64; k_count];
    let mut d_k = vec![0.0_f64; k_count];
    let mut r_k = vec![0.0_f64; k_count];
    let mut purity = vec![0.0_f64; k_count];
    let mut a_vis = vec![0.0_f64; k_count];
    let mut coherence = vec![0.0_f64; k_count];

    for k in 0..k_count {
        let nk = counts[k];
        let st = offsets[k];
        let en = offsets[k + 1];
        if nk == 0 || st == en {
            if nk <= 1 {
                d_k[k] = 1.0;
            }
            continue;
        }

        splat_stamp.clear();
        touched.clear();
        for m in st..en {
            let u = bucket_x[m] * g_eff_f + h_f;
            let v = bucket_y[m] * g_eff_f + h_f;
            let iu = u.round_ties_even() as isize;
            let iv = v.round_ties_even() as isize;
            let fu = u - (iu as f64);
            let fv = v - (iv as f64);

            fill_1d_gaussian(&mut wx, r_stencil, fu, config.sigma_cells, inv_two_sig2);
            fill_1d_gaussian(&mut wy, r_stencil, fv, config.sigma_cells, inv_two_sig2);

            for (dy_idx, &wy_val) in wy.iter().enumerate() {
                let gy = iv + (dy_idx as isize - r_stencil_i);
                if gy < 0 || gy >= s_i {
                    continue;
                }
                let row_off = (gy as usize) * s;
                for (dx_idx, &wx_val) in wx.iter().enumerate() {
                    let gx = iu + (dx_idx as isize - r_stencil_i);
                    if gx < 0 || gx >= s_i {
                        continue;
                    }
                    let w = wx_val * wy_val;
                    let cell = row_off + (gx as usize);
                    if splat_stamp.visit(cell) {
                        // First touch in this cluster overwrites stale values from earlier clusters.
                        wk_grid[cell] = w;
                        qk_grid[cell] = 0.0;
                        touched.push(cell);
                    } else {
                        qk_grid[cell] += 2.0 * wk_grid[cell] * w;
                        wk_grid[cell] += w;
                    }
                }
            }
        }

        let mut wk_sum = 0.0_f64;
        let mut e_sum = 0.0_f64;
        let mut a_vis_k = 0.0_f64;
        let mut num_k = 0.0_f64;
        let mut b_k = 0.0_f64;

        for &cell in &touched {
            let wk = wk_grid[cell];
            let wt = w_tot[cell];
            if wk <= 1e-15 || wt <= 1e-15 {
                continue;
            }
            let qk = qk_grid[cell];
            let qt = q_tot[cell];
            let ph = phi_all[cell];
            let frac = wk / wt;
            let wt2 = wt * wt;
            wk_sum += wk;
            e_sum += frac * wt.min(cap_density);
            a_vis_k += frac * ph;
            num_k += (qk / wt2) * ph;
            b_k += frac * (qt / wt2) * ph;
        }

        e_k[k] = (e_sum / (nk as f64)).clamp(0.0, 1.0);
        if nk <= 1 {
            d_k[k] = 1.0;
        } else {
            let mut m_shan = 0.0_f64;
            if wk_sum > 0.0 {
                let mut h_shan = 0.0_f64;
                for &cell in &touched {
                    let wk = wk_grid[cell];
                    if wk > 1e-15 {
                        let uk = wk / wk_sum;
                        h_shan -= uk * uk.max(1e-30).ln();
                    }
                }
                m_shan = h_shan.exp();
            }
            let denom_d = p_raw[k] * a_disk / config.kappa;
            d_k[k] = if denom_d > 0.0 {
                ((m_shan - m0_disc).max(0.0) / denom_d).clamp(0.0, 1.0)
            } else {
                0.0
            };
        }
        r_k[k] = e_k[k].min(d_k[k]).clamp(0.0, 1.0);
        a_vis[k] = a_vis_k;

        let p_pair = ((nk as f64) - 1.0) / (n_f - 1.0);
        purity[k] = if b_k > 1e-15 && nk < n {
            ((num_k / b_k - p_pair) / (1.0 - p_pair)).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // 8-connected cell-component spatial coherence over occupied touched cells.
        if a_vis_k > 1e-15 {
            bfs_stamp.clear();
            let mut max_comp_mass = 0.0_f64;
            for &start_cell in &touched {
                if wk_grid[start_cell] <= 1e-15 || !bfs_stamp.visit(start_cell) {
                    continue;
                }
                stack.clear();
                stack.push(start_cell);
                let mut comp_mass = 0.0_f64;
                while let Some(cell) = stack.pop() {
                    let wk = wk_grid[cell];
                    let wt = w_tot[cell];
                    if wt > 1e-15 {
                        comp_mass += (wk / wt) * phi_all[cell];
                    }
                    let iy = (cell / s) as isize;
                    let ix = (cell % s) as isize;
                    for dy_nb in -1isize..=1 {
                        let ny = iy + dy_nb;
                        if ny < 0 || ny >= s_i {
                            continue;
                        }
                        let row_off = (ny as usize) * s;
                        for dx_nb in -1isize..=1 {
                            if dx_nb == 0 && dy_nb == 0 {
                                continue;
                            }
                            let nx = ix + dx_nb;
                            if nx < 0 || nx >= s_i {
                                continue;
                            }
                            let ncell = row_off + (nx as usize);
                            // Check splat_stamp before wk_grid to skip stale values from prior clusters.
                            if splat_stamp.is_visited(ncell)
                                && wk_grid[ncell] > 1e-15
                                && bfs_stamp.visit(ncell)
                            {
                                stack.push(ncell);
                            }
                        }
                    }
                }
                max_comp_mass = max_comp_mass.max(comp_mass);
            }
            coherence[k] = (max_comp_mass / a_vis_k).clamp(0.0, 1.0);
        }
    }

    // Step 4: Symmetric valid-subspace aggregation.
    // Clamp all [0.0, 1.0] quantities so 1-3 ULP floating-point summation drift
    // never violates the closed unit-interval output contract.
    let res_density: f64 = p
        .iter()
        .zip(e_k.iter())
        .map(|(&pk, &ek)| pk * ek)
        .sum::<f64>()
        .clamp(0.0, 1.0);
    let res_deconv: f64 = p
        .iter()
        .zip(d_k.iter())
        .map(|(&pk, &dk)| pk * dk)
        .sum::<f64>()
        .clamp(0.0, 1.0);
    let res_combined: f64 = p
        .iter()
        .zip(r_k.iter())
        .map(|(&pk, &rk)| pk * rk)
        .sum::<f64>()
        .clamp(0.0, 1.0);
    let a_tot: f64 = a_vis.iter().sum();

    if a_tot <= 1e-15 {
        let report = ClusterAreaReport {
            caf: 0.0,
            purity_adj: 0.0,
            prop_vis: 0.0,
            res_density,
            res_deconv,
            res_combined,
            spatial_coherence: 0.0,
            in_frac,
            num_clusters: k_count,
            num_nodes: n,
        };
        if !include_detail {
            return Ok(ClusterAreaDetail {
                report,
                ..ClusterAreaDetail::default()
            });
        }
        return Ok(ClusterAreaDetail {
            report,
            cluster_ids,
            counts,
            p,
            q: vec![0.0; k_count],
            purity,
            resolution: r_k,
            res_density: e_k,
            res_deconv: d_k,
            caf_per_cluster: vec![0.0; k_count],
            coherence: vec![0.0; k_count],
        });
    }

    let q: Vec<f64> = a_vis
        .iter()
        .map(|&ak| (ak / a_tot).clamp(0.0, 1.0))
        .collect();
    let purity_adj: f64 = purity
        .iter()
        .zip(q.iter())
        .map(|(&pk, &qk)| pk * qk)
        .sum::<f64>()
        .clamp(0.0, 1.0);
    let spatial_coherence: f64 = coherence
        .iter()
        .zip(q.iter())
        .map(|(&ck, &qk)| ck * qk)
        .sum::<f64>()
        .clamp(0.0, 1.0);
    let prop_vis = (1.0
        - 0.5
            * p.iter()
                .zip(q.iter())
                .map(|(&pk, &qk)| (pk - qk).abs())
                .sum::<f64>())
    .clamp(0.0, 1.0);

    let mut sp = 0.0_f64;
    let mut sq = 0.0_f64;
    for k in 0..k_count {
        let sk = r_k[k] * purity[k];
        sp += sk * p[k];
        sq += sk * q[k];
    }

    let mut caf = 0.0_f64;
    let mut caf_per_cluster = if include_detail {
        vec![0.0_f64; k_count]
    } else {
        Vec::new()
    };
    if sp > 1e-15 && sq > 1e-15 {
        let s_min = sp.min(sq);
        let scale_p = s_min / sp;
        let scale_q = s_min / sq;
        for k in 0..k_count {
            let sk = r_k[k] * purity[k];
            let term = sk * (scale_p * p[k]).min(scale_q * q[k]);
            caf += term;
            if include_detail && p[k] > 0.0 {
                caf_per_cluster[k] = (term / p[k]).clamp(0.0, 1.0);
            }
        }
    }
    let caf = caf.clamp(0.0, 1.0);

    let report = ClusterAreaReport {
        caf,
        purity_adj,
        prop_vis,
        res_density,
        res_deconv,
        res_combined,
        spatial_coherence,
        in_frac,
        num_clusters: k_count,
        num_nodes: n,
    };
    if !include_detail {
        return Ok(ClusterAreaDetail {
            report,
            ..ClusterAreaDetail::default()
        });
    }

    Ok(ClusterAreaDetail {
        report,
        cluster_ids,
        counts,
        p,
        q,
        purity,
        resolution: r_k,
        res_density: e_k,
        res_deconv: d_k,
        caf_per_cluster,
        coherence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use egraph_core::DeterministicRng;

    fn sample_disk_points(
        rng: &mut DeterministicRng,
        cx: f64,
        cy: f64,
        radius: f64,
        count: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut xs = Vec::with_capacity(count);
        let mut ys = Vec::with_capacity(count);
        for _ in 0..count {
            let r = radius * rng.next_f64().sqrt();
            let theta = 2.0 * PI * rng.next_f64();
            xs.push(cx + r * theta.cos());
            ys.push(cy + r * theta.sin());
        }
        (xs, ys)
    }

    fn make_ideal_10_disks(
        n_per_cluster: usize,
        radius: f64,
        spacing: f64,
    ) -> (Layout2d<f64>, Vec<u32>) {
        let mut rng = DeterministicRng::from_seed_u64(42);
        let mut xs = Vec::with_capacity(10 * n_per_cluster);
        let mut ys = Vec::with_capacity(10 * n_per_cluster);
        let mut labels = Vec::with_capacity(10 * n_per_cluster);
        for k in 0..10usize {
            let gx = (k % 5) as f64 * spacing;
            let gy = (k / 5) as f64 * spacing;
            let (dx, dy) = sample_disk_points(&mut rng, gx, gy, radius, n_per_cluster);
            xs.extend(dx);
            ys.extend(dy);
            labels.extend(std::iter::repeat_n(k as u32, n_per_cluster));
        }
        (Layout2d::from_xy(xs, ys).unwrap(), labels)
    }

    #[test]
    fn test_validation_and_degenerate() {
        let l = Layout2d::from_xy(vec![0.0, 1.0], vec![0.0, 1.0]).unwrap();
        let cfg = ClusterAreaConfig::default();

        // Dimension mismatch
        assert!(matches!(
            cluster_area_faithfulness(&l, &[0], &cfg),
            Err(CoreError::DimensionMismatch { .. })
        ));

        // Non-finite layout
        let l_nan = Layout2d::from_xy(vec![0.0, f64::NAN], vec![0.0, 1.0]).unwrap();
        assert!(matches!(
            cluster_area_faithfulness(&l_nan, &[0, 1], &cfg),
            Err(CoreError::NonFiniteValue { .. })
        ));
        let l_inf = Layout2d::from_xy(vec![0.0, f64::INFINITY], vec![0.0, 1.0]).unwrap();
        assert!(matches!(
            cluster_area_faithfulness(&l_inf, &[0, 1], &cfg),
            Err(CoreError::NonFiniteValue { .. })
        ));

        // Config validation
        assert!(matches!(
            cluster_area_faithfulness(&l, &[0, 1], &ClusterAreaConfig { g_max: 11, ..cfg }),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            cluster_area_faithfulness(
                &l,
                &[0, 1],
                &ClusterAreaConfig {
                    sigma_cells: 0.0,
                    ..cfg
                }
            ),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            cluster_area_faithfulness(
                &l,
                &[0, 1],
                &ClusterAreaConfig {
                    sigma_cells: -1.0,
                    ..cfg
                }
            ),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            cluster_area_faithfulness(
                &l,
                &[0, 1],
                &ClusterAreaConfig {
                    sigma_cells: f64::NAN,
                    ..cfg
                }
            ),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            cluster_area_faithfulness(&l, &[0, 1], &ClusterAreaConfig { kappa: 0.0, ..cfg }),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            cluster_area_faithfulness(
                &l,
                &[0, 1],
                &ClusterAreaConfig {
                    tau_frac: -0.1,
                    ..cfg
                }
            ),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            cluster_area_faithfulness(
                &l,
                &[0, 1],
                &ClusterAreaConfig {
                    boundary_r95: 0.0,
                    ..cfg
                }
            ),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            cluster_area_faithfulness(
                &l,
                &[0, 1],
                &ClusterAreaConfig {
                    boundary_r95: -1.0,
                    ..cfg
                }
            ),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            cluster_area_faithfulness(
                &l,
                &[0, 1],
                &ClusterAreaConfig {
                    boundary_r95: f64::NAN,
                    ..cfg
                }
            ),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            cluster_area_faithfulness(
                &l,
                &[0, 1],
                &ClusterAreaConfig {
                    stevens_gamma: 0.0,
                    ..cfg
                }
            ),
            Err(CoreError::InvalidInput(_))
        ));

        // N = 0
        let l0 = Layout2d::<f64>::from_xy(vec![], vec![]).unwrap();
        let d0 = cluster_area_faithfulness_detail(&l0, &[], &cfg).unwrap();
        assert_eq!(d0.report, ClusterAreaReport::default());
        assert!(d0.cluster_ids.is_empty());
        assert!(d0.p.is_empty());

        // N = 1
        let l1 = Layout2d::from_xy(vec![1.0], vec![2.0]).unwrap();
        let d1 = cluster_area_faithfulness_detail(&l1, &[7], &cfg).unwrap();
        assert_eq!(d1.report.num_nodes, 1);
        assert_eq!(d1.report.num_clusters, 1);
        assert_eq!(d1.report.caf, 0.0);
        assert_eq!(d1.cluster_ids, vec![7]);
        assert_eq!(d1.counts, vec![1]);
        assert_eq!(d1.p, vec![1.0]);
        assert_eq!(d1.purity, vec![0.0]);

        // K = 1
        let l_k1 = Layout2d::from_xy(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 0.0]).unwrap();
        let d_k1 = cluster_area_faithfulness_detail(&l_k1, &[5, 5, 5], &cfg).unwrap();
        assert_eq!(d_k1.report.num_nodes, 3);
        assert_eq!(d_k1.report.num_clusters, 1);
        assert_eq!(d_k1.report.caf, 0.0);
        assert_eq!(d_k1.cluster_ids, vec![5]);

        // All coincident
        let l_coin = Layout2d::from_xy(vec![3.0; 10], vec![-2.0; 10]).unwrap();
        let labs_coin = vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
        let d_coin = cluster_area_faithfulness_detail(&l_coin, &labs_coin, &cfg).unwrap();
        assert_eq!(d_coin.report.caf, 0.0);
        assert_eq!(d_coin.report.num_clusters, 2);
        assert_eq!(d_coin.report.num_nodes, 10);

        // 96% coincident -> r_pre == 0.0
        let mut xs_96 = vec![0.0; 100];
        let ys_96 = vec![0.0; 100];
        let mut labs_96 = vec![0u32; 100];
        for i in 96..100 {
            xs_96[i] = 10.0;
            labs_96[i] = 1;
        }
        let l_96 = Layout2d::from_xy(xs_96, ys_96).unwrap();
        let d_96 = cluster_area_faithfulness_detail(&l_96, &labs_96, &cfg).unwrap();
        assert_eq!(d_96.report.caf, 0.0);
        assert_eq!(d_96.report.in_frac, 0.0);
        assert_eq!(d_96.report.num_clusters, 2);
        assert_eq!(d_96.report.num_nodes, 100);
    }

    #[test]
    fn test_label_remapping_non_contiguous() {
        let cfg = ClusterAreaConfig::default();
        let l = Layout2d::from_xy(
            vec![0.0, 10.0, 20.0, 10.5, 0.5, 20.5],
            vec![0.0, 0.0, 0.0, 0.5, 0.5, 0.5],
        )
        .unwrap();
        let sparse = [u32::MAX, 42, 7, 42, u32::MAX, 7];
        let compact = [2, 1, 0, 1, 2, 0];

        let d_sparse = cluster_area_faithfulness_detail(&l, &sparse, &cfg).unwrap();
        let d_compact = cluster_area_faithfulness_detail(&l, &compact, &cfg).unwrap();
        assert_eq!(d_sparse.report, d_compact.report);
        assert_eq!(d_sparse.cluster_ids, vec![7, 42, u32::MAX]);
        assert_eq!(d_sparse.counts, vec![2, 2, 2]);
        assert_eq!(d_sparse.p, d_compact.p);
        assert_eq!(d_sparse.q, d_compact.q);
        assert_eq!(d_sparse.purity, d_compact.purity);
        assert_eq!(d_sparse.coherence, d_compact.coherence);

        // Also test dense non-contiguous path (> 1024 nodes for radix path too).
        let n = 1500usize;
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        let mut labs_dense = Vec::with_capacity(n);
        let mut labs_radix = Vec::with_capacity(n);
        let mut labs_zero = Vec::with_capacity(n);
        for i in 0..n {
            let c = i % 3;
            xs.push((c as f64) * 10.0 + ((i % 7) as f64) * 0.1);
            ys.push(((i / 7) % 7) as f64 * 0.1);
            labs_zero.push(c as u32);
            labs_dense.push([10u32, 500, 1000][c]);
            labs_radix.push([5u32, 10_000_000, u32::MAX][c]);
        }
        let l_big = Layout2d::from_xy(xs, ys).unwrap();
        let r_zero = cluster_area_faithfulness_detail(&l_big, &labs_zero, &cfg).unwrap();
        let r_dense = cluster_area_faithfulness_detail(&l_big, &labs_dense, &cfg).unwrap();
        let r_radix = cluster_area_faithfulness_detail(&l_big, &labs_radix, &cfg).unwrap();
        assert_eq!(r_zero.report, r_dense.report);
        assert_eq!(r_zero.report, r_radix.report);
        assert_eq!(r_dense.cluster_ids, vec![10, 500, 1000]);
        assert_eq!(r_radix.cluster_ids, vec![5, 10_000_000, u32::MAX]);
    }

    #[test]
    fn mec_100k_circle_timing() {
        let n = 100_000usize;
        let mut pts = Vec::with_capacity(n);
        for i in 0..n {
            let theta = (2.0 * PI * (i as f64)) / (n as f64);
            pts.push((theta.cos(), theta.sin()));
        }
        let t0 = std::time::Instant::now();
        let (c, r) = exact_mec(&pts);
        let elapsed = t0.elapsed();
        eprintln!(
            "mec_100k_circle_timing: {:?} (center={:?}, r={})",
            elapsed, c, r
        );
        assert!(c.0.hypot(c.1) < 1e-12, "center={:?}", c);
        assert!((r - 1.0).abs() < 1e-12, "r={}", r);
        // Release runs well under 10ms; debug builds allow 100ms headroom.
        #[cfg(not(debug_assertions))]
        assert!(elapsed.as_secs_f64() < 0.015, "elapsed={:?}", elapsed);
        #[cfg(debug_assertions)]
        assert!(elapsed.as_secs_f64() < 0.15, "elapsed={:?}", elapsed);
    }

    #[test]
    fn test_scale_1e_pm_100_and_translation() {
        let cfg = ClusterAreaConfig::default();
        let (base_layout, labels) = make_ideal_10_disks(1000, 3.0, 10.0);
        let base = cluster_area_faithfulness(&base_layout, &labels, &cfg).unwrap();
        assert!(base.caf > 0.90, "base={:?}", base);

        let bx = base_layout.x();
        let by = base_layout.y();
        for &scale in &[1e-100, 1e-10, 1e-3, 1e3, 1e10, 1e100] {
            let scaled = Layout2d::from_xy(
                bx.iter().map(|&v| v * scale).collect(),
                by.iter().map(|&v| v * scale).collect(),
            )
            .unwrap();
            let rep = cluster_area_faithfulness(&scaled, &labels, &cfg).unwrap();
            assert!(
                (rep.caf - base.caf).abs() < 1e-12,
                "scale={scale}, diff={}",
                (rep.caf - base.caf).abs()
            );
        }

        // Small/moderate translation achieves < 1e-12 invariance.
        let t1e5 = Layout2d::from_xy(
            bx.iter().map(|&v| v + 1e5).collect(),
            by.iter().map(|&v| v - 1e5).collect(),
        )
        .unwrap();
        let rep_1e5 = cluster_area_faithfulness(&t1e5, &labels, &cfg).unwrap();
        assert!((rep_1e5.caf - base.caf).abs() < 1e-12);

        // +1e8 and +1e10 avoid catastrophic c3 cancellation (bounded only by f64 input quantization).
        for &shift in &[1e8, 1e10] {
            let shifted = Layout2d::from_xy(
                bx.iter().map(|&v| v + shift).collect(),
                by.iter().map(|&v| v + shift).collect(),
            )
            .unwrap();
            let rep = cluster_area_faithfulness(&shifted, &labels, &cfg).unwrap();
            assert!(
                (rep.caf - base.caf).abs() < 1e-5,
                "shift={shift}, base={}, got={}",
                base.caf,
                rep.caf
            );
        }
    }

    #[test]
    fn test_isometry_and_permutation() {
        let cfg = ClusterAreaConfig::default();
        let mut rng = DeterministicRng::from_seed_u64(777);
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        let mut labels = Vec::new();
        // Skewed asymmetric layout (N ~ 20_000 so G_eff ~ 123): 50% cluster at (-20, 0),
        // smaller non-crushed lobes + outliers at ±60.
        let (x0, y0) = sample_disk_points(&mut rng, -20.0, 0.0, 5.0, 10_000);
        xs.extend(x0);
        ys.extend(y0);
        labels.extend(std::iter::repeat_n(0u32, 10_000));
        for k in 1..6usize {
            let cx = (k as f64) * 9.0 - 15.0;
            let cy = if k % 2 == 0 { 12.0 } else { -12.0 };
            let (xk, yk) = sample_disk_points(&mut rng, cx, cy, 3.5, 1998);
            xs.extend(xk);
            ys.extend(yk);
            labels.extend(std::iter::repeat_n(k as u32, 1998));
        }
        for i in 0..10usize {
            xs.push(if i % 2 == 0 { 60.0 } else { -60.0 });
            ys.push(if i < 5 { 60.0 } else { -60.0 });
            labels.push(((i % 5) + 1) as u32);
        }

        let base_layout = Layout2d::from_xy(xs.clone(), ys.clone()).unwrap();
        let base = cluster_area_faithfulness(&base_layout, &labels, &cfg).unwrap();

        // Reflections: (x, -y), (-x, y), (-x, -y)
        for &(sx, sy) in &[(1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
            let refl = Layout2d::from_xy(
                xs.iter().map(|&v| v * sx).collect(),
                ys.iter().map(|&v| v * sy).collect(),
            )
            .unwrap();
            let rep = cluster_area_faithfulness(&refl, &labels, &cfg).unwrap();
            assert!(
                (rep.caf - base.caf).abs() < 1e-12,
                "reflection ({sx},{sy}) diff={}",
                (rep.caf - base.caf).abs()
            );
        }

        // Permutation invariance
        let mut perm: Vec<usize> = (0..xs.len()).collect();
        rng.shuffle(&mut perm);
        let p_layout = Layout2d::from_xy(
            perm.iter().map(|&i| xs[i]).collect(),
            perm.iter().map(|&i| ys[i]).collect(),
        )
        .unwrap();
        let p_labels: Vec<u32> = perm.iter().map(|&i| labels[i]).collect();
        let p_rep = cluster_area_faithfulness(&p_layout, &p_labels, &cfg).unwrap();
        assert!((p_rep.caf - base.caf).abs() < 1e-12);

        // 12 rotations (30-degree steps)
        let mut min_caf = f64::INFINITY;
        let mut max_caf = f64::NEG_INFINITY;
        for step in 0..12 {
            let ang = (step as f64) * (PI / 6.0);
            let (sa, ca) = ang.sin_cos();
            let rot = Layout2d::from_xy(
                xs.iter()
                    .zip(ys.iter())
                    .map(|(&x, &y)| x * ca - y * sa)
                    .collect(),
                xs.iter()
                    .zip(ys.iter())
                    .map(|(&x, &y)| x * sa + y * ca)
                    .collect(),
            )
            .unwrap();
            let rep = cluster_area_faithfulness(&rot, &labels, &cfg).unwrap();
            min_caf = min_caf.min(rep.caf);
            max_caf = max_caf.max(rep.caf);
        }
        assert!(
            max_caf - min_caf < 3e-4,
            "rotation span={}",
            max_caf - min_caf
        );
    }

    #[test]
    fn test_large_k_10k_flat_scaling() {
        let cfg = ClusterAreaConfig::default();
        let n = 50_000usize;
        let mut rng = DeterministicRng::from_seed_u64(101);
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        let mut labs_10 = Vec::with_capacity(n);
        let mut labs_10k = Vec::with_capacity(n);
        for i in 0..n {
            let gx = (i % 100) as f64 * 2.0;
            let gy = (i / 100) as f64 * 2.0;
            xs.push(gx + rng.next_f64() * 0.4);
            ys.push(gy + rng.next_f64() * 0.4);
            labs_10.push((i % 10) as u32);
            labs_10k.push((i / 5) as u32);
        }
        let layout = Layout2d::from_xy(xs, ys).unwrap();

        let t1 = std::time::Instant::now();
        let rep10 = cluster_area_faithfulness(&layout, &labs_10, &cfg).unwrap();
        let d10 = t1.elapsed();

        let t2 = std::time::Instant::now();
        let rep10k = cluster_area_faithfulness(&layout, &labs_10k, &cfg).unwrap();
        let d10k = t2.elapsed();

        eprintln!(
            "large_k_10k_flat_scaling: K=10 {:?}, K=10000 {:?}",
            d10, d10k
        );
        assert_eq!(rep10.num_clusters, 10);
        assert_eq!(rep10k.num_clusters, 10_000);
        assert!(rep10k.caf.is_finite());
        // Must not scale as O(K * S^2): K=10k should remain well below 3x K=10 runtime.
        assert!(
            d10k.as_secs_f64() < (d10.as_secs_f64() * 5.0).max(0.5),
            "d10={:?}, d10k={:?}",
            d10,
            d10k
        );
    }

    #[test]
    fn test_split_4islands_coherence() {
        let cfg = ClusterAreaConfig::default();
        let mut rng = DeterministicRng::from_seed_u64(2026);
        let mut xs = Vec::with_capacity(4000);
        let mut ys = Vec::with_capacity(4000);
        let mut labels = Vec::with_capacity(4000);

        // Cluster 0 split into 4 equal distant islands; Cluster 1 split into 4 equal distant islands.
        let c0_centers = [(-10.0, -10.0), (10.0, -10.0), (-10.0, 10.0), (10.0, 10.0)];
        let c1_centers = [(-10.0, 0.0), (10.0, 0.0), (0.0, -10.0), (0.0, 10.0)];
        let (tmpl_x, tmpl_y) = sample_disk_points(&mut rng, 0.0, 0.0, 2.0, 500);
        for &(cx, cy) in &c0_centers {
            xs.extend(tmpl_x.iter().map(|&x| cx + x));
            ys.extend(tmpl_y.iter().map(|&y| cy + y));
            labels.extend(std::iter::repeat_n(0u32, 500));
        }
        for &(cx, cy) in &c1_centers {
            xs.extend(tmpl_x.iter().map(|&x| cx + x));
            ys.extend(tmpl_y.iter().map(|&y| cy + y));
            labels.extend(std::iter::repeat_n(1u32, 500));
        }

        let layout = Layout2d::from_xy(xs, ys).unwrap();
        let detail = cluster_area_faithfulness_detail(&layout, &labels, &cfg).unwrap();
        assert!(detail.report.caf > 0.90, "caf={}", detail.report.caf);
        assert!(detail.report.purity_adj > 0.99);
        assert!(
            (detail.coherence[0] - 0.25).abs() < 1e-3,
            "c0={}",
            detail.coherence[0]
        );
        assert!(
            (detail.coherence[1] - 0.25).abs() < 1e-3,
            "c1={}",
            detail.coherence[1]
        );
        assert!(
            (detail.report.spatial_coherence - 0.25).abs() < 1e-3,
            "coherence={}",
            detail.report.spatial_coherence
        );

        // Mixed: C0 = 4 islands (2000 pts), C1 = 1 disk of 4x area / equal visual weight (2000 pts)
        let mut xs_mix = Vec::with_capacity(4000);
        let mut ys_mix = Vec::with_capacity(4000);
        let mut labs_mix = Vec::with_capacity(4000);
        for &(cx, cy) in &c0_centers {
            xs_mix.extend(tmpl_x.iter().map(|&x| cx + x));
            ys_mix.extend(tmpl_y.iter().map(|&y| cy + y));
            labs_mix.extend(std::iter::repeat_n(0u32, 500));
        }
        // Radius 4.0 gives 4 * pi * 2.0^2 = pi * 4.0^2 equal area to the 4 islands combined.
        let (dx, dy) = sample_disk_points(&mut rng, 0.0, 0.0, 4.0, 2000);
        xs_mix.extend(dx);
        ys_mix.extend(dy);
        labs_mix.extend(std::iter::repeat_n(1u32, 2000));
        let mix_layout = Layout2d::from_xy(xs_mix, ys_mix).unwrap();
        let mix_detail = cluster_area_faithfulness_detail(&mix_layout, &labs_mix, &cfg).unwrap();
        assert!((mix_detail.coherence[0] - 0.25).abs() < 1e-3);
        assert!((mix_detail.coherence[1] - 1.0).abs() < 1e-6);
        let expected_mix_coh =
            mix_detail.q[0] * mix_detail.coherence[0] + mix_detail.q[1] * mix_detail.coherence[1];
        assert!((mix_detail.report.spatial_coherence - expected_mix_coh).abs() < 1e-12);
        assert!(
            (mix_detail.report.spatial_coherence - 0.625).abs() < 0.05,
            "mix spatial_coherence={}",
            mix_detail.report.spatial_coherence
        );
    }

    #[test]
    fn test_singleton_and_disjoint_stencil() {
        // (a) Singleton cluster with boundary_r95 = +inf
        let mut rng = DeterministicRng::from_seed_u64(55);
        let (mut xs, mut ys) = sample_disk_points(&mut rng, 0.0, 0.0, 1.0, 99);
        let mut labels = vec![0u32; 99];
        xs.push(5.0);
        ys.push(0.0);
        labels.push(1);
        let layout_s = Layout2d::from_xy(xs, ys).unwrap();
        let cfg_inf = ClusterAreaConfig {
            boundary_r95: f64::INFINITY,
            ..ClusterAreaConfig::default()
        };
        let det_s = cluster_area_faithfulness_detail(&layout_s, &labels, &cfg_inf).unwrap();
        assert_eq!(det_s.res_deconv[1], 1.0);
        assert_eq!(det_s.purity[1], 0.0);
        assert_eq!(det_s.caf_per_cluster[1], 0.0);
        assert!((det_s.coherence[1] - 1.0).abs() < 1e-12);

        // (b) DisjointStencil: 4 isolated points far apart -> B_k == 0 -> purity = 0 -> caf = 0
        let l_disj = Layout2d::from_xy(vec![0.0, 5.0, 0.0, 5.0], vec![0.0, 0.0, 5.0, 5.0]).unwrap();
        let det_d =
            cluster_area_faithfulness_detail(&l_disj, &[0, 0, 1, 1], &ClusterAreaConfig::default())
                .unwrap();
        assert_eq!(det_d.purity, vec![0.0, 0.0]);
        assert_eq!(det_d.report.caf, 0.0);
    }

    #[test]
    fn test_stevens_gamma_behavior() {
        let mut rng = DeterministicRng::from_seed_u64(314);
        let (x0, y0) = sample_disk_points(&mut rng, -5.0, 0.0, 1.0, 9000);
        let (x1, y1) = sample_disk_points(&mut rng, 5.0, 0.0, 1.0, 1000);
        let mut xs = x0;
        xs.extend(x1);
        let mut ys = y0;
        ys.extend(y1);
        let mut labels = vec![0u32; 9000];
        labels.extend(std::iter::repeat_n(1u32, 1000));
        let layout = Layout2d::from_xy(xs, ys).unwrap();

        let d1 = cluster_area_faithfulness_detail(&layout, &labels, &ClusterAreaConfig::default())
            .unwrap();
        let d07 = cluster_area_faithfulness_detail(
            &layout,
            &labels,
            &ClusterAreaConfig {
                stevens_gamma: 0.7,
                ..ClusterAreaConfig::default()
            },
        )
        .unwrap();

        assert!(d07.report.prop_vis > d1.report.prop_vis);
        assert!(d07.report.caf > d1.report.caf);
        // Step 3 physical fields & q-weighted scores must be bit-identical under gamma change.
        assert_eq!(d07.report.purity_adj, d1.report.purity_adj);
        assert_eq!(d07.report.spatial_coherence, d1.report.spatial_coherence);
        assert_eq!(d07.q, d1.q);
        assert_eq!(d07.purity, d1.purity);
        assert_eq!(d07.res_density, d1.res_density);
        assert_eq!(d07.res_deconv, d1.res_deconv);
        assert_eq!(d07.coherence, d1.coherence);
    }

    #[test]
    fn test_boundary_r95_contract() {
        let mut rng = DeterministicRng::from_seed_u64(991);
        let (mut xs, mut ys) = sample_disk_points(&mut rng, 0.0, 0.0, 99.0_f64.sqrt(), 9900);
        let (x1, y1) = sample_disk_points(&mut rng, 25.0, 0.0, 1.0, 100);
        xs.extend(x1);
        ys.extend(y1);
        let mut labels = vec![0u32; 9900];
        labels.extend(std::iter::repeat_n(1u32, 100));
        let layout = Layout2d::from_xy(xs, ys).unwrap();

        let r_b2 =
            cluster_area_faithfulness(&layout, &labels, &ClusterAreaConfig::default()).unwrap();
        assert!((r_b2.in_frac - 0.99).abs() < 1e-12);
        assert!((r_b2.caf - 0.99).abs() < 1e-2, "caf={}", r_b2.caf);
        assert!((r_b2.spatial_coherence - 1.0).abs() < 1e-12);

        let r_inf = cluster_area_faithfulness(
            &layout,
            &labels,
            &ClusterAreaConfig {
                boundary_r95: f64::INFINITY,
                ..ClusterAreaConfig::default()
            },
        )
        .unwrap();
        let r_1e9 = cluster_area_faithfulness(
            &layout,
            &labels,
            &ClusterAreaConfig {
                boundary_r95: 1e9,
                ..ClusterAreaConfig::default()
            },
        )
        .unwrap();
        assert!((r_inf.in_frac - 1.0).abs() < 1e-12);
        assert!(r_inf.caf > 0.98 && r_inf.caf.is_finite());
        assert_eq!(r_inf, r_1e9);
    }

    #[test]
    fn test_gcaf_summary_golden_values() {
        let cfg = ClusterAreaConfig::default();
        let mut rng = DeterministicRng::from_seed_u64(42);

        // 1. 20Dots monotonicity (r = 1e-4 .. 4.0)
        let mut prev_caf = -1.0_f64;
        for &r in &[1e-4, 0.05, 0.5, 2.0, 4.0] {
            let mut xs = Vec::with_capacity(10_000);
            let mut ys = Vec::with_capacity(10_000);
            let mut labs = Vec::with_capacity(10_000);
            for k in 0..20usize {
                let cx = (k % 5) as f64 * 10.0;
                let cy = (k / 5) as f64 * 10.0;
                let (dx, dy) = sample_disk_points(&mut rng, cx, cy, r, 500);
                xs.extend(dx);
                ys.extend(dy);
                labs.extend(std::iter::repeat_n(k as u32, 500));
            }
            let l = Layout2d::from_xy(xs, ys).unwrap();
            let rep = cluster_area_faithfulness(&l, &labs, &cfg).unwrap();
            assert!(
                rep.caf >= prev_caf,
                "20Dots non-monotonic at r={r}: {} < {}",
                rep.caf,
                prev_caf
            );
            prev_caf = rep.caf;
        }
        assert!(prev_caf > 0.95);

        // 2. B1a_WhiteNoise_K10 -> caf < 0.01
        let (wx, wy) = sample_disk_points(&mut rng, 0.0, 0.0, 10.0, 10_000);
        let w_labs: Vec<u32> = (0..10_000).map(|i| (i % 10) as u32).collect();
        let l_white = Layout2d::from_xy(wx, wy).unwrap();
        let rep_white = cluster_area_faithfulness(&l_white, &w_labs, &cfg).unwrap();
        assert!(rep_white.caf < 0.01, "white noise caf={}", rep_white.caf);

        // 3. D2_All10_95CrushHalo -> tight core + tiny outer halo -> caf < 0.03
        let mut dx_d2 = Vec::with_capacity(20_000);
        let mut dy_d2 = Vec::with_capacity(20_000);
        let mut labs_d2 = Vec::with_capacity(20_000);
        for k in 0..10usize {
            let ang = 2.0 * PI * (k as f64) / 10.0;
            let cx = 5.0 * ang.cos();
            let cy = 5.0 * ang.sin();
            let (cx0, cy0) = sample_disk_points(&mut rng, cx, cy, 1e-4, 1900);
            let (cx1, cy1) = sample_disk_points(&mut rng, cx, cy, 0.5, 100);
            dx_d2.extend(cx0);
            dx_d2.extend(cx1);
            dy_d2.extend(cy0);
            dy_d2.extend(cy1);
            labs_d2.extend(std::iter::repeat_n(k as u32, 2000));
        }
        let l_d2 = Layout2d::from_xy(dx_d2, dy_d2).unwrap();
        let rep_d2 = cluster_area_faithfulness(&l_d2, &labs_d2, &cfg).unwrap();
        assert!(rep_d2.caf < 0.03, "D2 caf={}", rep_d2.caf);

        // 4. G3a_Crush50_SurvEq vs G3b_Crush50_Surv5x -> G3a > G3b + 0.10
        let mut gx_a = Vec::with_capacity(10_000);
        let mut gy_a = Vec::with_capacity(10_000);
        let mut gx_b = Vec::with_capacity(10_000);
        let mut gy_b = Vec::with_capacity(10_000);
        let mut g_labs_3 = Vec::with_capacity(10_000);
        let g3_counts = [1667usize, 1667, 1666, 1667, 1667, 1666];
        let g3b_radii = [1e-4, 1e-4, 1e-4, 1.8, 1.8, 4.2];
        for k in 0..6usize {
            let ang = 2.0 * PI * (k as f64) / 6.0;
            let cx = 10.0 * ang.cos();
            let cy = 10.0 * ang.sin();
            let r_a = if k < 3 { 1e-4 } else { 2.5 };
            let (ax, ay) = sample_disk_points(&mut rng, cx, cy, r_a, g3_counts[k]);
            gx_a.extend(ax);
            gy_a.extend(ay);
            let (bx, by) = sample_disk_points(&mut rng, cx, cy, g3b_radii[k], g3_counts[k]);
            gx_b.extend(bx);
            gy_b.extend(by);
            g_labs_3.extend(std::iter::repeat_n(k as u32, g3_counts[k]));
        }
        let rep_g3a =
            cluster_area_faithfulness(&Layout2d::from_xy(gx_a, gy_a).unwrap(), &g_labs_3, &cfg)
                .unwrap();
        let rep_g3b =
            cluster_area_faithfulness(&Layout2d::from_xy(gx_b, gy_b).unwrap(), &g_labs_3, &cfg)
                .unwrap();
        assert!(
            rep_g3a.caf > rep_g3b.caf + 0.10,
            "G3a={} vs G3b={}",
            rep_g3a.caf,
            rep_g3b.caf
        );

        // 5. G4_ImpureSmudge90 (mixed 90% background + 10% cluster cores) -> caf < 0.35
        let (gx_base, gy_base) = sample_disk_points(&mut rng, 0.0, 0.0, 10.0, 9000);
        let mut gx = gx_base;
        let mut gy = gy_base;
        let mut g_labs: Vec<u32> = (0..9000).map(|i| (i % 10) as u32).collect();
        for k in 0..10usize {
            let ang = 2.0 * PI * (k as f64) / 10.0;
            let (cx, cy) = sample_disk_points(&mut rng, 5.0 * ang.cos(), 5.0 * ang.sin(), 0.5, 100);
            gx.extend(cx);
            gy.extend(cy);
            g_labs.extend(std::iter::repeat_n(k as u32, 100));
        }
        let l_g4 = Layout2d::from_xy(gx, gy).unwrap();
        let rep_g4 = cluster_area_faithfulness(&l_g4, &g_labs, &cfg).unwrap();
        assert!(rep_g4.caf < 0.35, "G4 caf={}", rep_g4.caf);

        // 5. f32 vs f64 parity (< 1e-5)
        let (l_ideal, labs_ideal) = make_ideal_10_disks(200, 1.0, 10.0);
        let l_ideal_32 = Layout2d::from_xy(
            l_ideal.x().iter().map(|&v| v as f32).collect(),
            l_ideal.y().iter().map(|&v| v as f32).collect(),
        )
        .unwrap();
        let rep64 = cluster_area_faithfulness(&l_ideal, &labs_ideal, &cfg).unwrap();
        let rep32 = cluster_area_faithfulness(&l_ideal_32, &labs_ideal, &cfg).unwrap();
        assert!(
            (rep64.caf - rep32.caf).abs() < 1e-5,
            "f32 vs f64 diff={}",
            (rep64.caf - rep32.caf).abs()
        );
    }

    fn assert_detail_unit_interval(detail: &ClusterAreaDetail, ctx: &str) {
        let r = &detail.report;
        for (name, val) in [
            ("caf", r.caf),
            ("purity_adj", r.purity_adj),
            ("prop_vis", r.prop_vis),
            ("res_density", r.res_density),
            ("res_deconv", r.res_deconv),
            ("res_combined", r.res_combined),
            ("spatial_coherence", r.spatial_coherence),
            ("in_frac", r.in_frac),
        ] {
            assert!(
                val.is_finite() && (0.0..=1.0).contains(&val),
                "{ctx}: report.{name}={val:?} out of [0, 1]"
            );
        }
        for (vec_name, vec) in [
            ("p", &detail.p),
            ("q", &detail.q),
            ("purity", &detail.purity),
            ("resolution", &detail.resolution),
            ("res_density", &detail.res_density),
            ("res_deconv", &detail.res_deconv),
            ("caf_per_cluster", &detail.caf_per_cluster),
            ("coherence", &detail.coherence),
        ] {
            for (k, &val) in vec.iter().enumerate() {
                assert!(
                    val.is_finite() && (0.0..=1.0).contains(&val),
                    "{ctx}: detail.{vec_name}[{k}]={val:?} out of [0, 1]"
                );
            }
        }
    }

    #[test]
    fn test_tiny_sigma_cells_kronecker_stability() {
        // Direct helper checks: parity at sigma=0.85 and Kronecker / half-tie limit as sigma -> 0+.
        let sigma_std = 0.85;
        let inv_std = -0.5 / (sigma_std * sigma_std);
        let r_std = (3.5 * sigma_std).ceil() as usize;
        let mut buf_std = vec![0.0_f64; 2 * r_std + 1];
        for &frac in &[-0.5, -0.37, 0.0, 0.25, 0.5] {
            fill_1d_gaussian(&mut buf_std, r_std, frac, sigma_std, inv_std);
            let mut raw = vec![0.0_f64; 2 * r_std + 1];
            let mut sum = 0.0_f64;
            for (i, slot) in raw.iter_mut().enumerate() {
                let dx = (i as isize - r_std as isize) as f64 - frac;
                let v = (inv_std * dx * dx).exp();
                *slot = v;
                sum += v;
            }
            for (a, b) in buf_std.iter().zip(raw.iter()) {
                assert!((*a - (*b / sum)).abs() < 1e-15);
            }
        }

        for &sigma in &[1e-160, 1e-200, f64::MIN_POSITIVE] {
            let inv = -0.5 / (sigma * sigma);
            let mut buf = vec![0.0_f64; 3];
            fill_1d_gaussian(&mut buf, 1, 0.13, sigma, inv);
            assert_eq!(buf, vec![0.0, 1.0, 0.0]);
            fill_1d_gaussian(&mut buf, 1, 0.5, sigma, inv);
            assert_eq!(buf, vec![0.0, 0.5, 0.5]);
            fill_1d_gaussian(&mut buf, 1, -0.5, sigma, inv);
            assert_eq!(buf, vec![0.5, 0.5, 0.0]);
        }

        // Non-grid-aligned 6-node layout across tiny sigma_cells values down to f64::MIN_POSITIVE.
        let layout = Layout2d::from_xy(
            vec![0.0, 0.37, 1.41, 2.53, 3.19, 4.27],
            vec![0.0, 0.49, 1.48, 2.13, 3.44, 4.49],
        )
        .unwrap();
        let labels = [0u32, 0, 0, 1, 1, 1];
        for &sigma in &[0.012, 0.01, 0.005, 1e-10, 1e-160, 1e-200, f64::MIN_POSITIVE] {
            let cfg = ClusterAreaConfig {
                sigma_cells: sigma,
                ..ClusterAreaConfig::default()
            };
            let detail = cluster_area_faithfulness_detail(&layout, &labels, &cfg).unwrap();
            assert_detail_unit_interval(&detail, &format!("sigma_cells={sigma:e}"));
            let rep = cluster_area_faithfulness(&layout, &labels, &cfg).unwrap();
            assert_eq!(rep, detail.report);
        }

        // Exact max_b_grid boundary clamping check.
        let max_b_grid = ((H_MAX - 4) as f64) / 128.0;
        let (l_big, labs_big) = make_ideal_10_disks(300, 1.0, 5.0);
        for &b in &[max_b_grid, max_b_grid + 1e-12] {
            let det = cluster_area_faithfulness_detail(
                &l_big,
                &labs_big,
                &ClusterAreaConfig {
                    boundary_r95: b,
                    ..ClusterAreaConfig::default()
                },
            )
            .unwrap();
            assert_detail_unit_interval(&det, &format!("max_b_grid={b}"));
        }
    }

    #[test]
    fn test_unit_interval_clamping_adversarial() {
        let cfg = ClusterAreaConfig::default();

        // 1. K = N singletons (N = 100) triggers ULP overshoot on sum(p * e_k), sum(p * d_k) without clamping.
        let mut xs_s = Vec::with_capacity(100);
        let mut ys_s = Vec::with_capacity(100);
        let labs_s: Vec<u32> = (0..100).collect();
        for i in 0..100usize {
            xs_s.push((i % 10) as f64 * 2.0);
            ys_s.push((i / 10) as f64 * 2.0);
        }
        let l_s = Layout2d::from_xy(xs_s, ys_s).unwrap();
        let det_s = cluster_area_faithfulness_detail(&l_s, &labs_s, &cfg).unwrap();
        assert_detail_unit_interval(&det_s, "K=N singletons (N=100)");

        // Also test a_tot <= 1e-15 early-return path with K = N singletons.
        let det_s_empty = cluster_area_faithfulness_detail(
            &l_s,
            &labs_s,
            &ClusterAreaConfig {
                boundary_r95: 1e-6,
                ..cfg
            },
        )
        .unwrap();
        assert_eq!(det_s_empty.report.in_frac, 0.0);
        assert_detail_unit_interval(&det_s_empty, "K=N singletons a_tot<=1e-15");

        // 2. Extreme scales 1e-50 / 1e50 and translation 1e8.
        let (base_layout, base_labels) = make_ideal_10_disks(500, 3.0, 10.0);
        let bx = base_layout.x();
        let by = base_layout.y();
        for &scale in &[1e-50, 1e50] {
            let l_sc = Layout2d::from_xy(
                bx.iter().map(|&v| v * scale).collect(),
                by.iter().map(|&v| v * scale).collect(),
            )
            .unwrap();
            let det = cluster_area_faithfulness_detail(&l_sc, &base_labels, &cfg).unwrap();
            assert_detail_unit_interval(&det, &format!("scale={scale:e}"));
        }
        let l_tr = Layout2d::from_xy(
            bx.iter().map(|&v| v + 1e8).collect(),
            by.iter().map(|&v| v + 1e8).collect(),
        )
        .unwrap();
        let det_tr = cluster_area_faithfulness_detail(&l_tr, &base_labels, &cfg).unwrap();
        assert_detail_unit_interval(&det_tr, "trans=1e8");

        // 3. Asymmetric aspect ratio 1e6 : 1 (horizontal and vertical).
        let mut rng = DeterministicRng::from_seed_u64(808);
        let mut ax = Vec::with_capacity(400);
        let mut ay = Vec::with_capacity(400);
        let mut alabs = Vec::with_capacity(400);
        for k in 0..4usize {
            let (dx, dy) = sample_disk_points(&mut rng, (k as f64) * 1e6, 0.0, 1.0, 100);
            ax.extend(dx);
            ay.extend(dy);
            alabs.extend(std::iter::repeat_n(k as u32, 100));
        }
        let l_ah = Layout2d::from_xy(ax.clone(), ay.clone()).unwrap();
        let det_ah = cluster_area_faithfulness_detail(&l_ah, &alabs, &cfg).unwrap();
        assert_detail_unit_interval(&det_ah, "aspect=1e6:1 horizontal");
        let l_av = Layout2d::from_xy(ay, ax).unwrap();
        let det_av = cluster_area_faithfulness_detail(&l_av, &alabs, &cfg).unwrap();
        assert_detail_unit_interval(&det_av, "aspect=1e6:1 vertical");

        // 4. Stevens gamma = 3.0.
        let det_g3 = cluster_area_faithfulness_detail(
            &base_layout,
            &base_labels,
            &ClusterAreaConfig {
                stevens_gamma: 3.0,
                ..cfg
            },
        )
        .unwrap();
        assert_detail_unit_interval(&det_g3, "stevens_gamma=3.0");
    }

    fn assert_report_close(a: &ClusterAreaReport, b: &ClusterAreaReport, tol: f64, ctx: &str) {
        assert_eq!(a.num_clusters, b.num_clusters, "{ctx}: num_clusters");
        assert_eq!(a.num_nodes, b.num_nodes, "{ctx}: num_nodes");
        for (name, va, vb) in [
            ("caf", a.caf, b.caf),
            ("purity_adj", a.purity_adj, b.purity_adj),
            ("prop_vis", a.prop_vis, b.prop_vis),
            ("res_density", a.res_density, b.res_density),
            ("res_deconv", a.res_deconv, b.res_deconv),
            ("res_combined", a.res_combined, b.res_combined),
            (
                "spatial_coherence",
                a.spatial_coherence,
                b.spatial_coherence,
            ),
            ("in_frac", a.in_frac, b.in_frac),
        ] {
            assert!(
                (va - vb).abs() < tol,
                "{ctx}: {name} diff={} >= {tol} (a={va}, b={vb})",
                (va - vb).abs()
            );
        }
    }

    #[test]
    fn test_small_n_mec_quantile_tie_stability() {
        // 1. Synthetic equidistant Stage-1/Stage-2 MEC tie (N = 30 and N = 41) with ±1e-15 jitter
        // on vertex 0 across both boundary_r95 = 2.0 (default) and boundary_r95 = 1.0.
        for &n in &[30usize, 41usize] {
            let mut xs = Vec::with_capacity(n);
            let mut ys = Vec::with_capacity(n);
            let mut labels = Vec::with_capacity(n);
            for i in 0..3 {
                let theta = 0.37 + 2.0 * PI * (i as f64) / 3.0;
                xs.push(theta.cos());
                ys.push(theta.sin());
                labels.push((i % 3) as u32);
            }
            let inner = n - 3;
            for j in 0..inner {
                let r = 0.65 * (((j + 1) as f64) / (inner as f64)).sqrt();
                let theta = 2.399_963_229_728_653 * ((j + 1) as f64);
                xs.push(r * theta.cos());
                ys.push(r * theta.sin());
                labels.push(((j + 3) % 3) as u32);
            }

            let base_l = Layout2d::from_xy(xs.clone(), ys.clone()).unwrap();
            let mut xs_p = xs.clone();
            let mut ys_p = ys.clone();
            xs_p[0] *= 1.0 + 1e-15;
            ys_p[0] *= 1.0 + 1e-15;
            let plus_l = Layout2d::from_xy(xs_p, ys_p).unwrap();

            let mut xs_m = xs;
            let mut ys_m = ys;
            xs_m[0] *= 1.0 - 1e-15;
            ys_m[0] *= 1.0 - 1e-15;
            let minus_l = Layout2d::from_xy(xs_m, ys_m).unwrap();

            for &b_r95 in &[2.0_f64, 1.0_f64] {
                let cfg = ClusterAreaConfig {
                    boundary_r95: b_r95,
                    ..ClusterAreaConfig::default()
                };
                let r_base = cluster_area_faithfulness_detail(&base_l, &labels, &cfg).unwrap();
                let r_plus = cluster_area_faithfulness(&plus_l, &labels, &cfg).unwrap();
                let r_minus = cluster_area_faithfulness(&minus_l, &labels, &cfg).unwrap();
                assert_detail_unit_interval(&r_base, &format!("synthetic_tie n={n} b={b_r95}"));
                assert_report_close(
                    &r_base.report,
                    &r_plus,
                    1e-12,
                    &format!("synthetic_tie +1e-15 n={n} b={b_r95}"),
                );
                assert_report_close(
                    &r_base.report,
                    &r_minus,
                    1e-12,
                    &format!("synthetic_tie -1e-15 n={n} b={b_r95}"),
                );
            }
        }

        // 2. Exact f32 N <= 40 cross-target parity regression fixtures (n30_k3, n39_k5, n40_b1, n6_b1).
        let n30_k3_coords: [f32; 60] = [
            -0.28154773,
            2.70013,
            1.5668566,
            0.07951676,
            -1.750706,
            3.2445662,
            -0.72710097,
            -1.1585857,
            -0.90493894,
            -0.016284715,
            0.64667726,
            0.38064554,
            -0.6412784,
            1.3978007,
            -0.4455091,
            -0.490506,
            2.5104165,
            1.2259845,
            0.4358604,
            -1.8759084,
            -2.0040073,
            1.1233397,
            1.072034,
            0.9364058,
            0.14748514,
            -0.58216316,
            -1.3126808,
            2.8923485,
            1.1164329,
            -2.4579682,
            -0.78497416,
            -4.539269,
            -1.1620724,
            -0.71214044,
            -0.56531346,
            -0.73464775,
            -2.099709,
            -0.093_136_556_446_552_28_f64 as f32,
            1.1614314,
            0.35444275,
            0.46526924,
            1.7718444,
            -0.44344476,
            1.4617308,
            -1.6758195,
            1.5955342,
            -3.2307613,
            -0.6647125,
            0.012851243,
            1.9697237,
            0.26684767,
            3.094563,
            -0.8386496,
            0.16661248,
            2.7859595,
            -2.0786989,
            -3.6580622,
            2.894068,
            4.2667637,
            0.91116905,
        ];
        let n30_k3_labels: [u32; 30] = [
            2, 1, 2, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0,
            1,
        ];

        let n39_k5_coords: [f32; 78] = [
            2.317929,
            -2.377_370_119_094_848_6_f64 as f32,
            -2.5181954,
            2.293665,
            1.5150898,
            -0.16263618,
            -0.541605,
            1.4072235,
            -2.6598365,
            1.2367346,
            0.23080106,
            2.890641,
            0.14955002,
            -2.7411644,
            2.9515316,
            0.5958319,
            -2.2528713,
            -1.2812335,
            -4.208866,
            -2.0573194,
            1.2402142,
            -0.64551497,
            -3.7162738,
            -1.5085694,
            0.5517861,
            0.43332744,
            -2.0856888,
            0.82401204,
            1.1661124,
            0.5285953,
            -1.7103713,
            0.91390324,
            -1.1898159,
            -2.2106411,
            2.106105,
            1.6581815,
            1.8313656,
            -5.0268283,
            0.159858,
            2.3602962,
            1.1456916,
            -0.6920323,
            -1.9657927,
            -1.7649826,
            -2.4530983,
            -2.5909653,
            0.39769387,
            0.16761969,
            -3.082718,
            -0.32256436,
            -3.2988117,
            0.354599,
            -2.2738812,
            0.9519124,
            -0.38830006,
            -1.5800346,
            -0.73172736,
            2.5496125,
            0.18937981,
            2.016829,
            -0.643776,
            -1.1126575,
            -1.6523632,
            -1.4371283,
            1.399753,
            -1.3744355,
            0.34608787,
            -0.20996541,
            0.8923608,
            -0.337803,
            0.48425642,
            -1.9588091,
            2.5424654,
            0.02001159,
            1.4568391,
            -3.30631,
            -2.2863953,
            -0.16748564,
        ];
        let n39_k5_labels: [u32; 39] = [
            3, 0, 0, 1, 4, 3, 0, 3, 1, 3, 3, 2, 2, 0, 0, 3, 4, 2, 0, 3, 0, 1, 1, 4, 1, 1, 3, 1, 0,
            3, 1, 2, 2, 3, 2, 3, 3, 2, 1,
        ];

        let n40_b1_coords: [f32; 80] = [
            -2.9233508,
            1.3258305,
            -1.2700397,
            0.055691067,
            -3.5743446,
            0.08754364,
            2.8365555,
            -1.2860936,
            0.67369485,
            2.1501174,
            0.027947975,
            -0.5190643,
            1.8806356,
            0.38272673,
            0.59430355,
            -5.774634,
            1.3010856,
            0.15634698,
            -0.19979255,
            2.8649082,
            -0.4877356,
            -0.35270914,
            1.160403,
            1.5458733,
            -3.2697394,
            -1.7934492,
            -1.9938316,
            0.7835421,
            2.9356518,
            -1.3010819,
            2.1025224,
            3.2613504,
            -3.110684,
            -3.0776134,
            4.413053,
            1.5375748,
            1.0798331,
            -0.56084615,
            -0.2496351,
            0.99739313,
            2.0195343,
            -1.7061172,
            -3.5770214,
            3.366582,
            1.310306,
            -1.3890607,
            0.76046354,
            0.39164633,
            1.9436023,
            1.0128682,
            -0.8362288,
            3.489198,
            -2.9951606,
            2.4799,
            -0.43555063,
            1.4823762,
            3.8454504,
            -0.4614333,
            -4.027053,
            -0.3833692,
            -2.1162024,
            2.3836763,
            2.3869655,
            -1.7980483,
            1.800023,
            1.4273919,
            -2.7043612,
            -2.191859,
            1.9361794,
            1.299877,
            -0.2770219,
            1.5509106,
            1.5726527,
            -3.354655,
            -0.31485745,
            2.2452877,
            1.2533443,
            -2.2307775,
            -0.2523287,
            0.99455553,
        ];
        let n40_b1_labels: [u32; 40] = [
            3, 1, 3, 0, 1, 2, 2, 2, 1, 0, 0, 0, 2, 3, 2, 1, 0, 1, 2, 1, 0, 3, 1, 1, 1, 0, 0, 2, 1,
            3, 0, 3, 2, 2, 0, 0, 1, 1, 0, 1,
        ];

        let n6_b1_coords: [f32; 12] = [
            0.09900733,
            3.4892879,
            -2.2565875,
            1.0310328,
            -0.25905266,
            0.7601389,
            0.6031524,
            1.156712,
            4.331787,
            2.1050458,
            -2.2524312,
            3.5577226,
        ];
        let n6_b1_labels: [u32; 6] = [3, 0, 5, 1, 3, 5];

        let cfg_n30_custom = ClusterAreaConfig {
            g_max: 12,
            sigma_cells: 0.15,
            boundary_r95: 0.8,
            ..ClusterAreaConfig::default()
        };
        let cfg_n39_custom = ClusterAreaConfig {
            g_max: 32,
            sigma_cells: 1.8,
            boundary_r95: 5.0,
            ..ClusterAreaConfig::default()
        };
        let cfg_b1 = ClusterAreaConfig {
            boundary_r95: 1.0,
            ..ClusterAreaConfig::default()
        };

        let cases: &[(&str, &[f32], &[u32], ClusterAreaConfig)] = &[
            (
                "n30_k3_default",
                &n30_k3_coords,
                &n30_k3_labels,
                ClusterAreaConfig::default(),
            ),
            (
                "n30_k3_custom",
                &n30_k3_coords,
                &n30_k3_labels,
                cfg_n30_custom,
            ),
            (
                "n39_k5_default",
                &n39_k5_coords,
                &n39_k5_labels,
                ClusterAreaConfig::default(),
            ),
            (
                "n39_k5_custom",
                &n39_k5_coords,
                &n39_k5_labels,
                cfg_n39_custom,
            ),
            ("n40_b1", &n40_b1_coords, &n40_b1_labels, cfg_b1),
            ("n6_b1", &n6_b1_coords, &n6_b1_labels, cfg_b1),
        ];

        for &(name, xy, labs, ref cfg) in cases {
            let xs32: Vec<f32> = xy.iter().step_by(2).copied().collect();
            let ys32: Vec<f32> = xy.iter().skip(1).step_by(2).copied().collect();
            let l32 = Layout2d::from_xy(xs32.clone(), ys32.clone()).unwrap();
            let det = cluster_area_faithfulness_detail(&l32, labs, cfg).unwrap();
            let rep = cluster_area_faithfulness(&l32, labs, cfg).unwrap();
            assert_eq!(rep, det.report, "{name}: report != detail.report");
            assert_detail_unit_interval(&det, name);

            let xs64: Vec<f64> = xs32.iter().map(|&v| v as f64).collect();
            let ys64: Vec<f64> = ys32.iter().map(|&v| v as f64).collect();
            let cx = xs64.iter().sum::<f64>() / (xs64.len() as f64);
            let cy = ys64.iter().sum::<f64>() / (ys64.len() as f64);
            let max_idx = (0..xs64.len())
                .max_by(|&a, &b| {
                    (xs64[a] - cx)
                        .hypot(ys64[a] - cy)
                        .total_cmp(&(xs64[b] - cx).hypot(ys64[b] - cy))
                })
                .unwrap();

            for &(tag, eps) in &[("+1e-15", 1.0 + 1e-15), ("-1e-15", 1.0 - 1e-15)] {
                let mut px = xs64.clone();
                let mut py = ys64.clone();
                px[max_idx] = cx + (px[max_idx] - cx) * eps;
                py[max_idx] = cy + (py[max_idx] - cy) * eps;
                let pl = Layout2d::from_xy(px, py).unwrap();
                let prep = cluster_area_faithfulness(&pl, labs, cfg).unwrap();
                assert_report_close(&rep, &prep, 1e-12, &format!("{name} {tag}"));
            }
        }
    }

    #[test]
    fn test_pair_purity_cancellation_regression() {
        // Regression fixtures from 500-case N in [6, 45] differential stress testing where
        // sub-cell sigma_cells = 0.15 produces a dominant Gaussian weight + tiny tail weight
        // (ratio ~1e13), causing catastrophic subtraction cancellation in `wk^2 - sum(w^2)`.
        let idx445_coords: [f32; 18] = [
            -0.93399155,
            -0.08828876,
            0.23824371,
            -1.6165369,
            0.3195233,
            -1.3285851,
            -0.2986154,
            1.6585639,
            -1.2817469,
            -0.23555677,
            0.9802638,
            -0.2675741,
            2.068217,
            0.88675106,
            0.44698393,
            1.5724736,
            0.40123367,
            -1.8822963,
        ];
        let idx445_labels: [u32; 9] = [1, 4, 4, 0, 3, 3, 3, 3, 4];
        let idx445_cfg = ClusterAreaConfig {
            g_max: 12,
            sigma_cells: 0.15,
            boundary_r95: 1.0,
            stevens_gamma: 0.3,
            ..ClusterAreaConfig::default()
        };

        let idx451_coords: [f32; 60] = [
            -0.2518732,
            0.07975522,
            1.3726488,
            0.8500951,
            -1.3431914,
            0.062369324,
            0.39769107,
            1.9115974,
            -0.9139043,
            -0.88805175,
            1.6514385,
            -0.22949933,
            -1.1391747,
            -0.951801,
            0.2050648,
            -1.6683543,
            -0.2511857,
            -0.68052495,
            0.300962,
            0.107651636,
            -0.111619756,
            -0.9943411,
            0.18235113,
            -0.6725447,
            -0.5753704,
            1.084559,
            -0.15376493,
            0.6685573,
            -0.7996044,
            -0.7258471,
            1.8715934,
            -0.4190787,
            -0.7483076,
            0.6918077,
            0.79846996,
            1.8965863,
            0.39656964,
            -0.7518983,
            -1.6543025,
            -0.009605116,
            -1.5369639,
            -0.49072692,
            -1.4884776,
            0.6713425,
            0.050311044,
            0.057186887,
            -0.53254545,
            -0.8697066,
            1.6423913,
            -1.789239,
            1.8272529,
            0.60150194,
            0.21556842,
            -0.25793338,
            1.4036045,
            0.28627896,
            -0.39399222,
            0.44464102,
            0.3066869,
            -0.12467769,
        ];
        let idx451_labels: [u32; 30] = [
            0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0,
            1,
        ];
        let idx451_cfg = ClusterAreaConfig {
            g_max: 32,
            sigma_cells: 0.15,
            boundary_r95: f64::INFINITY,
            stevens_gamma: 0.3,
            ..ClusterAreaConfig::default()
        };

        // idx=445: Cluster 4 (index 3) shares zero grid cells with any other cluster -> exact 1.0.
        let xs445: Vec<f32> = idx445_coords.iter().step_by(2).copied().collect();
        let ys445: Vec<f32> = idx445_coords.iter().skip(1).step_by(2).copied().collect();
        let l445 = Layout2d::from_xy(xs445.clone(), ys445.clone()).unwrap();
        let det445 = cluster_area_faithfulness_detail(&l445, &idx445_labels, &idx445_cfg).unwrap();
        assert_detail_unit_interval(&det445, "idx445_pure_tail");
        assert_eq!(det445.cluster_ids, vec![0, 1, 3, 4]);
        assert!(
            (det445.purity[3] - 1.0).abs() == 0.0,
            "idx445 cluster 4 must be exact 1.0, got {}",
            det445.purity[3]
        );

        let xs451: Vec<f32> = idx451_coords.iter().step_by(2).copied().collect();
        let ys451: Vec<f32> = idx451_coords.iter().skip(1).step_by(2).copied().collect();
        let l451 = Layout2d::from_xy(xs451.clone(), ys451.clone()).unwrap();
        let det451 = cluster_area_faithfulness_detail(&l451, &idx451_labels, &idx451_cfg).unwrap();
        assert_detail_unit_interval(&det451, "idx451_sparse_tail");

        // s500_292: sparse tail density phi_all[c] = 1 - exp(-x) cancellation regression (K=5 and K=6).
        let s500_292_coords: [f32; 74] = [
            -0.8589298,
            -0.8472999,
            -0.01037515,
            -1.7285855,
            0.00178981,
            2.9769006,
            -0.9761431,
            1.2507176,
            0.7966066,
            -0.43772125,
            -1.909174,
            0.88139206,
            -1.493889,
            0.055759937,
            -0.3533779,
            -1.4706259,
            0.6511914,
            -0.35015807,
            0.9030685,
            -1.5927824,
            -0.3389033,
            -1.0626435,
            -0.2578625,
            0.23115428,
            -1.4865725,
            0.38411444,
            0.55941355,
            -0.8532565,
            -1.2615016,
            -0.53420925,
            -0.9444083,
            0.5791151,
            -0.6351455,
            -0.9675821,
            -0.5154885,
            0.91004246,
            0.9587222,
            0.26757297,
            -0.87733287,
            -1.1966555,
            0.7086302,
            1.4324192,
            -1.5779736,
            -0.4205665,
            -0.9108668,
            0.9537005,
            0.76539826,
            0.99292314,
            -1.4883577,
            1.0644866,
            -0.7887485,
            0.22597103,
            -0.87951875,
            -0.9363758,
            0.8706658,
            0.7357092,
            0.95346594,
            -1.8786527,
            0.2655913,
            1.7928542,
            -0.10050357,
            -0.90063393,
            1.3498366,
            0.59151834,
            1.1290692,
            -0.5772221,
            1.0819533,
            -0.86151373,
            0.39588717,
            0.60193646,
            0.4208206,
            -1.8643262,
            0.8065901,
            0.9162831,
        ];
        let s500_292_k5_labels: [u32; 37] = [
            2, 4, 2, 4, 3, 0, 4, 1, 4, 1, 1, 0, 1, 0, 0, 3, 2, 0, 1, 3, 1, 3, 3, 0, 4, 4, 0, 1, 2,
            4, 4, 0, 3, 3, 4, 3, 4,
        ];
        let s500_292_k6_labels: [u32; 37] = [
            3, 5, 3, 4, 4, 0, 5, 1, 5, 1, 2, 0, 1, 0, 0, 4, 2, 0, 2, 3, 1, 4, 4, 1, 5, 5, 0, 2, 3,
            5, 5, 0, 4, 4, 5, 4, 5,
        ];
        let s500_292_cfg = ClusterAreaConfig {
            g_max: 12,
            sigma_cells: 0.15,
            boundary_r95: f64::INFINITY,
            stevens_gamma: 2.5,
            ..ClusterAreaConfig::default()
        };
        let xs292: Vec<f32> = s500_292_coords.iter().step_by(2).copied().collect();
        let ys292: Vec<f32> = s500_292_coords.iter().skip(1).step_by(2).copied().collect();
        let l292 = Layout2d::from_xy(xs292.clone(), ys292.clone()).unwrap();
        let rep292_k5 =
            cluster_area_faithfulness(&l292, &s500_292_k5_labels, &s500_292_cfg).unwrap();
        let det292_k5 =
            cluster_area_faithfulness_detail(&l292, &s500_292_k5_labels, &s500_292_cfg).unwrap();
        assert_eq!(
            rep292_k5, det292_k5.report,
            "s500_292_k5: report != detail.report"
        );
        assert_detail_unit_interval(&det292_k5, "s500_292_k5");

        let rep292_k6 =
            cluster_area_faithfulness(&l292, &s500_292_k6_labels, &s500_292_cfg).unwrap();
        let det292_k6 =
            cluster_area_faithfulness_detail(&l292, &s500_292_k6_labels, &s500_292_cfg).unwrap();
        assert_eq!(
            rep292_k6, det292_k6.report,
            "s500_292_k6: report != detail.report"
        );
        assert_detail_unit_interval(&det292_k6, "s500_292_k6");

        // Verify 1-ULP perturbation stability (< 1e-12) across all scalars and per-cluster purities.
        for (name, xs32, ys32, labs, cfg, base_det) in [
            (
                "idx445_pure_tail",
                &xs445,
                &ys445,
                &idx445_labels[..],
                &idx445_cfg,
                &det445,
            ),
            (
                "idx451_sparse_tail",
                &xs451,
                &ys451,
                &idx451_labels[..],
                &idx451_cfg,
                &det451,
            ),
            (
                "s500_292_k5",
                &xs292,
                &ys292,
                &s500_292_k5_labels[..],
                &s500_292_cfg,
                &det292_k5,
            ),
            (
                "s500_292_k6",
                &xs292,
                &ys292,
                &s500_292_k6_labels[..],
                &s500_292_cfg,
                &det292_k6,
            ),
        ] {
            let xs64_base: Vec<f64> = xs32.iter().map(|&v| v as f64).collect();
            let ys64_base: Vec<f64> = ys32.iter().map(|&v| v as f64).collect();
            for &(tag, eps) in &[("+1ulp", 1.0 + 2.22e-16), ("-1ulp", 1.0 - 2.22e-16)] {
                let mut px = xs64_base.clone();
                let mut py = ys64_base.clone();
                for i in 0..px.len() {
                    px[i] *= eps;
                    py[i] *= eps;
                }
                let pl = Layout2d::from_xy(px, py).unwrap();
                let pdet = cluster_area_faithfulness_detail(&pl, labs, cfg).unwrap();
                assert_detail_unit_interval(&pdet, &format!("{name} {tag}"));
                assert_report_close(
                    &base_det.report,
                    &pdet.report,
                    1e-12,
                    &format!("{name} {tag}"),
                );
                for (k, (&pa, &pb)) in base_det.purity.iter().zip(pdet.purity.iter()).enumerate() {
                    assert!(
                        (pa - pb).abs() < 1e-12,
                        "{name} {tag}: purity[{k}] diff={} >= 1e-12 (a={pa}, b={pb})",
                        (pa - pb).abs()
                    );
                }
            }
        }
    }

    #[test]
    fn test_phi_all_tiny_wt_tau_no_nan() {
        // Verify that tiny Gaussian tail weights (wt < 1e-162) combined with tiny tau (tau <= 1e-162)
        // never trigger 0.0 * inf = NaN in phi_all[c], and all outputs remain finite in [0.0, 1.0].
        let xs = vec![0.0_f64, 0.2, 0.4, 2.0, 2.2, 2.4];
        let ys = vec![0.0_f64, 0.1, -0.1, 0.0, 0.1, -0.1];
        let labels = [0_u32, 0, 0, 1, 1, 1];
        let layout = Layout2d::from_xy(xs, ys).unwrap();
        for &tau_frac in &[1e-200_f64, 1e-320_f64] {
            let cfg = ClusterAreaConfig {
                g_max: 24,
                sigma_cells: 0.04,
                tau_frac,
                ..ClusterAreaConfig::default()
            };
            let rep = cluster_area_faithfulness(&layout, &labels, &cfg).unwrap();
            let det = cluster_area_faithfulness_detail(&layout, &labels, &cfg).unwrap();
            assert_eq!(
                rep, det.report,
                "tiny tau_frac={tau_frac}: report != detail.report"
            );
            assert_detail_unit_interval(&det, &format!("tiny_wt_tau_{tau_frac:e}"));
        }
    }

    #[test]
    fn test_sigma_cells_grid_upper_bound_and_max_b_grid_clamp() {
        // Gap 1: Exact sigma_cells upper bound (ceil(3.5 * sigma_cells) <= H_MAX - 2 = 1022).
        // 3.5 * 292.0 = 1022.0 (passes); ceil(3.5 * 292.001) = 1023 (InvalidInput).
        let layout_sep = Layout2d::from_xy(
            vec![0.0, 1.0, 0.0, 1.0, 10.0, 11.0, 10.0, 11.0],
            vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        )
        .unwrap();
        let labels_sep = [0u32, 0, 0, 0, 1, 1, 1, 1];

        let ok_cfg = ClusterAreaConfig {
            sigma_cells: 292.0,
            ..ClusterAreaConfig::default()
        };
        let ok_det = cluster_area_faithfulness_detail(&layout_sep, &labels_sep, &ok_cfg).unwrap();
        assert_detail_unit_interval(&ok_det, "sigma_cells=292.0");

        let bad_cfg = ClusterAreaConfig {
            sigma_cells: 292.001,
            ..ClusterAreaConfig::default()
        };
        assert!(matches!(
            cluster_area_faithfulness(&layout_sep, &labels_sep, &bad_cfg),
            Err(CoreError::InvalidInput(_))
        ));

        // max_b_grid clamping: sigma_cells=100.0, boundary_r95=10.0, g_max=128
        // produces non-trivial caf > 0.4 with clamped grid extent.
        let clamp_cfg = ClusterAreaConfig {
            g_max: 128,
            sigma_cells: 100.0,
            boundary_r95: 10.0,
            ..ClusterAreaConfig::default()
        };
        let clamp_det =
            cluster_area_faithfulness_detail(&layout_sep, &labels_sep, &clamp_cfg).unwrap();
        assert_detail_unit_interval(&clamp_det, "max_b_grid_clamp");
        assert!(
            clamp_det.report.caf > 0.4,
            "expected caf > 0.4 under max_b_grid clamping, got {}",
            clamp_det.report.caf
        );
        assert_eq!(
            cluster_area_faithfulness(&layout_sep, &labels_sep, &clamp_cfg).unwrap(),
            clamp_det.report
        );
    }

    #[test]
    fn test_c3_near_collinear_slanted_collinear_and_8conn_diagonal_bridge() {
        // Gap 2a: 3 vertex permutations of obtuse triangles ([0,0], [2,0], [1,eps])
        // straddling det.abs() <= 1e-15 * b2.max(d2), paired with a second cluster.
        let perms: [[usize; 3]; 3] = [[0, 1, 2], [1, 2, 0], [2, 0, 1]];
        let cfg_tri = ClusterAreaConfig {
            g_max: 16,
            sigma_cells: 0.5,
            boundary_r95: f64::INFINITY,
            ..ClusterAreaConfig::default()
        };
        for &eps in &[0.0_f64, 4e-16, 1.2e-15, -1.2e-15, 1e-14] {
            let tri = [(0.0_f64, 0.0_f64), (2.0_f64, 0.0_f64), (1.0_f64, eps)];
            let mut base_rep: Option<ClusterAreaReport> = None;
            for perm in &perms {
                let (c_direct, r_direct) = c3(tri[perm[0]], tri[perm[1]], tri[perm[2]]);
                assert!(c_direct.0.is_finite() && c_direct.1.is_finite() && r_direct.is_finite());

                let xs = vec![
                    tri[perm[0]].0,
                    tri[perm[1]].0,
                    tri[perm[2]].0,
                    10.0,
                    12.0,
                    11.0,
                ];
                let ys = vec![
                    tri[perm[0]].1,
                    tri[perm[1]].1,
                    tri[perm[2]].1,
                    0.0,
                    0.0,
                    1.0,
                ];
                let labs = [0u32, 0, 0, 1, 1, 1];
                let layout = Layout2d::from_xy(xs, ys).unwrap();
                let det = cluster_area_faithfulness_detail(&layout, &labs, &cfg_tri).unwrap();
                assert_detail_unit_interval(&det, &format!("c3_obtuse eps={eps:e}"));
                if let Some(ref expected) = base_rep {
                    assert_report_close(
                        expected,
                        &det.report,
                        1e-12,
                        &format!("c3_obtuse perm eps={eps:e}"),
                    );
                } else {
                    base_rep = Some(det.report);
                }
            }
        }

        // Gap 2b: Slanted collinear N=80 clusters (t, 2t) & (t+10, 2t+10) triggering
        // akl_toussaint_filter poly.len() < 3 fallback (pts.len() > 64).
        let mut xs_sl = Vec::with_capacity(160);
        let mut ys_sl = Vec::with_capacity(160);
        let mut labs_sl = Vec::with_capacity(160);
        for t in 0..80usize {
            let tf = t as f64;
            xs_sl.push(tf);
            ys_sl.push(2.0 * tf);
            labs_sl.push(0);
        }
        for t in 0..80usize {
            let tf = t as f64;
            xs_sl.push(tf + 10.0);
            ys_sl.push(2.0 * tf + 10.0);
            labs_sl.push(1);
        }
        let l_slanted = Layout2d::from_xy(xs_sl, ys_sl).unwrap();
        let det_slanted =
            cluster_area_faithfulness_detail(&l_slanted, &labs_sl, &ClusterAreaConfig::default())
                .unwrap();
        assert_detail_unit_interval(&det_slanted, "slanted_collinear_n80_per_cluster");
        assert!(det_slanted.report.caf > 0.2);

        // Gap 2c: 8-connected diagonal-only bridge at g_max=12, sigma_cells=0.35, boundary_r95=inf.
        // Cluster 0 has subclusters at (0,0) and (1,1) while Cluster 1 sets median=(0,0) and r95=2.4,
        // placing the two subclusters at grid centers (h, h) and (h+5, h+5) with r_stencil=2
        // so their 5x5 stencils touch exclusively via a single diagonal corner.
        let mut xs_diag = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut ys_diag = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut labs_diag = vec![0u32; 6];
        xs_diag.extend(std::iter::repeat_n(-2.4_f64, 48));
        ys_diag.extend(std::iter::repeat_n(0.0_f64, 48));
        xs_diag.extend(std::iter::repeat_n(2.4_f64, 46));
        ys_diag.extend(std::iter::repeat_n(0.0_f64, 46));
        labs_diag.extend(std::iter::repeat_n(1u32, 94));
        let cfg_diag = ClusterAreaConfig {
            g_max: 12,
            sigma_cells: 0.35,
            boundary_r95: f64::INFINITY,
            ..ClusterAreaConfig::default()
        };
        let det_diag = cluster_area_faithfulness_detail(
            &Layout2d::from_xy(xs_diag.clone(), ys_diag.clone()).unwrap(),
            &labs_diag,
            &cfg_diag,
        )
        .unwrap();
        assert_detail_unit_interval(&det_diag, "diag_bridge_connected");
        assert!(
            (det_diag.coherence[0] - 1.0).abs() < 1e-12,
            "expected diagonal bridge coherence[0] == 1.0, got {}",
            det_diag.coherence[0]
        );

        // Shifting (1,1) slightly to (1.2,1.2) breaks the diagonal corner and drops coherence[0] to ~0.5.
        for i in 3..6 {
            xs_diag[i] = 1.2;
            ys_diag[i] = 1.2;
        }
        let det_broken = cluster_area_faithfulness_detail(
            &Layout2d::from_xy(xs_diag, ys_diag).unwrap(),
            &labs_diag,
            &cfg_diag,
        )
        .unwrap();
        assert!(
            det_broken.coherence[0] < 0.55,
            "expected broken bridge coherence[0] < 0.55, got {}",
            det_broken.coherence[0]
        );
    }

    #[test]
    fn test_stevens_gamma_numerical_extremes() {
        // Gap 3: Extreme stevens_gamma values on imbalanced 4-vs-3 clusters without NaN/Inf/-0.0.
        let layout = Layout2d::from_xy(
            vec![0.0, 0.2, 0.0, 0.2, 3.0, 3.2, 3.0],
            vec![0.0, 0.0, 0.2, 0.2, 0.0, 0.0, 0.2],
        )
        .unwrap();
        let labels = [0u32, 0, 0, 0, 1, 1, 1];
        for &gamma in &[1e-12_f64, 1e-300, 100.0, 1e4, 1e300] {
            let cfg = ClusterAreaConfig {
                stevens_gamma: gamma,
                ..ClusterAreaConfig::default()
            };
            let rep = cluster_area_faithfulness(&layout, &labels, &cfg).unwrap();
            let det = cluster_area_faithfulness_detail(&layout, &labels, &cfg).unwrap();
            assert_eq!(rep, det.report);
            assert_detail_unit_interval(&det, &format!("stevens_gamma={gamma:e}"));
            for v in [
                rep.caf,
                rep.purity_adj,
                rep.prop_vis,
                rep.res_density,
                rep.res_deconv,
                rep.res_combined,
                rep.spatial_coherence,
                rep.in_frac,
            ] {
                if v == 0.0 {
                    assert!(!v.is_sign_negative(), "negative zero for gamma={gamma:e}");
                }
            }
            for &pv in &det.p {
                if pv == 0.0 {
                    assert!(
                        !pv.is_sign_negative(),
                        "negative zero in p for gamma={gamma:e}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_remap_community_labels_dense_limit_boundary() {
        // Gap 5: Exact dense_limit = 4_000_000 vs 4_000_001 boundary at N=10 and N=1500.
        let cfg = ClusterAreaConfig::default();

        // N = 10
        let xs10: Vec<f64> = (0..10)
            .map(|i| (i % 3) as f64 * 5.0 + (i as f64) * 0.05)
            .collect();
        let ys10: Vec<f64> = (0..10).map(|i| (i / 3) as f64 * 0.1).collect();
        let l10 = Layout2d::from_xy(xs10, ys10).unwrap();
        let base10_labs: Vec<u32> = (0..10).map(|i| (i % 3) as u32).collect();
        let base10 = cluster_area_faithfulness_detail(&l10, &base10_labs, &cfg).unwrap();

        for &(lbls3, ctx) in &[
            ([0u32, 1, 4_000_000], "n10_dense_4M"),
            ([0u32, 1, 4_000_001], "n10_sparse_4M_plus_1"),
            ([7u32, 4_000_000, 4_000_001], "n10_mixed_4M_4M1"),
        ] {
            let labs: Vec<u32> = (0..10).map(|i| lbls3[i % 3]).collect();
            let det = cluster_area_faithfulness_detail(&l10, &labs, &cfg).unwrap();
            assert_eq!(det.report, base10.report, "{ctx}");
            assert_eq!(det.cluster_ids, lbls3, "{ctx}");
            assert_eq!(det.p, base10.p, "{ctx}");
            assert_eq!(det.q, base10.q, "{ctx}");
            assert_eq!(det.purity, base10.purity, "{ctx}");
            assert_eq!(det.coherence, base10.coherence, "{ctx}");
        }

        // N = 1500 (> 1024 triggers radix_sort_u64 in sparse path)
        let n = 1500usize;
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        let mut base_labs = Vec::with_capacity(n);
        for i in 0..n {
            let c = i % 3;
            xs.push((c as f64) * 10.0 + ((i % 7) as f64) * 0.1);
            ys.push(((i / 7) % 7) as f64 * 0.1);
            base_labs.push(c as u32);
        }
        let l1500 = Layout2d::from_xy(xs, ys).unwrap();
        let base1500 = cluster_area_faithfulness_detail(&l1500, &base_labs, &cfg).unwrap();

        for &(lbls3, ctx) in &[
            ([0u32, 1, 4_000_000], "n1500_dense_4M"),
            ([0u32, 1, 4_000_001], "n1500_radix_4M_plus_1"),
            ([7u32, 4_000_000, 4_000_001], "n1500_mixed_radix"),
        ] {
            let labs: Vec<u32> = (0..n).map(|i| lbls3[i % 3]).collect();
            let det = cluster_area_faithfulness_detail(&l1500, &labs, &cfg).unwrap();
            assert_eq!(det.report, base1500.report, "{ctx}");
            assert_eq!(det.cluster_ids, lbls3, "{ctx}");
            assert_eq!(det.p, base1500.p, "{ctx}");
            assert_eq!(det.q, base1500.q, "{ctx}");
            assert_eq!(det.purity, base1500.purity, "{ctx}");
            assert_eq!(det.coherence, base1500.coherence, "{ctx}");
        }
    }
}
