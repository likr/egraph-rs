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
//! - **$O(N(2R+1)^2 + S^2)$, $K$-independent**: Groups in-bounds vertices
//!   into CSR buckets via $O(N + K)$ counting sort and reuses a single `S x S` grid
//!   backed by [`egraph_core::VisitStamp`] epoch markers and a `touched` cell list.
//! - **Invariance & Hardened Welzl MEC**: Pre-normalizes coordinates by
//!   coordinate median and $Q_{0.95}$ radial quantile before Weiszfeld refinement
//!   and two-stage Exact Minimum Enclosing Circle (MEC) centering; uses
//!   vertex-$a$-local circumcircle arithmetic, relative Akl-Toussaint 8-extreme
//!   filtering, Andrew's monotone chain convex hull, and fixed-seed SplitMix64
//!   shuffling for $O(h)$ expected MEC runtime and zero permutation variance.
//! - **Edge Semantics & Generalizations**: Explicitly guards $n_k \le 1$ and
//!   $B_k \le 10^{-15}$ boundaries, supports Stevens $\gamma$ target weighting, and
//!   computes 8-connected cell-component spatial coherence.

mod coherence;
mod config;
mod grid;
mod mec;
#[cfg(test)]
mod tests;

use egraph_core::{CoreError, CoreResult, Layout2d, Scalar};

pub use config::{ClusterAreaConfig, ClusterAreaDetail, ClusterAreaReport};

use coherence::cluster_area_core;

/// Evaluates Grid-based Cluster Area Faithfulness (`G-CAF`) on `layout` and `labels`.
///
/// See [`ClusterAreaConfig`] for the physical ink threshold and stencil coverage envelope
/// ($p_k A_{\mathrm{disk}} \lesssim n_k \pi (2\sigma)^2$).
pub fn cluster_area_faithfulness<S: Scalar>(
    layout: &Layout2d<S>,
    labels: &[u32],
    config: &ClusterAreaConfig,
) -> CoreResult<ClusterAreaReport> {
    cluster_area_faithfulness_with_weights(layout, labels, config, None, None)
}

/// Evaluates Grid-based Cluster Area Faithfulness (`G-CAF`) with full per-cluster diagnostics.
///
/// See [`ClusterAreaConfig`] for the physical ink threshold and stencil coverage envelope
/// ($p_k A_{\mathrm{disk}} \lesssim n_k \pi (2\sigma)^2$).
pub fn cluster_area_faithfulness_detail<S: Scalar>(
    layout: &Layout2d<S>,
    labels: &[u32],
    config: &ClusterAreaConfig,
) -> CoreResult<ClusterAreaDetail> {
    cluster_area_faithfulness_detail_with_weights(layout, labels, config, None, None)
}

/// Evaluates Grid-based Cluster Area Faithfulness (`G-CAF`) with optional positive finite
/// target weights (`node_weights` or `cluster_weights`, mutually exclusive).
///
/// When `node_weights` is provided (`len == N`), weights are summed per community and
/// normalized before applying Stevens $\gamma$: $p_k \propto (p_k^{(w)})^\gamma$.
/// When `cluster_weights` is provided (`len == K`), weights correspond to sorted ascending
/// unique community IDs (matching [`ClusterAreaDetail::cluster_ids`]).
/// See [`ClusterAreaConfig`] for the stencil coverage limit ($p_k A_{\mathrm{disk}} \lesssim n_k \pi (2\sigma)^2$).
pub fn cluster_area_faithfulness_with_weights<S: Scalar>(
    layout: &Layout2d<S>,
    labels: &[u32],
    config: &ClusterAreaConfig,
    node_weights: Option<&[f64]>,
    cluster_weights: Option<&[f64]>,
) -> CoreResult<ClusterAreaReport> {
    let detail = cluster_area_faithfulness_impl(
        layout,
        labels,
        config,
        node_weights,
        cluster_weights,
        false,
    )?;
    Ok(detail.report)
}

/// Evaluates Grid-based Cluster Area Faithfulness (`G-CAF`) with full per-cluster diagnostics
/// and optional positive finite target weights (`node_weights` or `cluster_weights`, mutually exclusive).
///
/// `cluster_weights` corresponds to sorted ascending unique community IDs (matching
/// [`ClusterAreaDetail::cluster_ids`]).
/// See [`ClusterAreaConfig`] for the stencil coverage limit ($p_k A_{\mathrm{disk}} \lesssim n_k \pi (2\sigma)^2$).
pub fn cluster_area_faithfulness_detail_with_weights<S: Scalar>(
    layout: &Layout2d<S>,
    labels: &[u32],
    config: &ClusterAreaConfig,
    node_weights: Option<&[f64]>,
    cluster_weights: Option<&[f64]>,
) -> CoreResult<ClusterAreaDetail> {
    cluster_area_faithfulness_impl(layout, labels, config, node_weights, cluster_weights, true)
}

fn validate_positive_finite_weights(weights: &[f64], context: &'static str) -> CoreResult<()> {
    for &w in weights {
        if !w.is_finite() {
            return Err(CoreError::non_finite(context));
        }
        if w <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "{context} entries must be strictly positive"
            )));
        }
    }
    Ok(())
}

fn cluster_area_faithfulness_impl<S: Scalar>(
    layout: &Layout2d<S>,
    labels: &[u32],
    config: &ClusterAreaConfig,
    node_weights: Option<&[f64]>,
    cluster_weights: Option<&[f64]>,
    include_detail: bool,
) -> CoreResult<ClusterAreaDetail> {
    let n = layout.len();
    if labels.len() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            labels.len(),
            "cluster_area_faithfulness::labels",
        ));
    }
    if node_weights.is_some() && cluster_weights.is_some() {
        return Err(CoreError::invalid_input(
            "node_weights and cluster_weights are mutually exclusive",
        ));
    }
    layout.validate_finite()?;
    config.validate()?;

    if let Some(nw) = node_weights {
        if nw.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                nw.len(),
                "cluster_area_faithfulness::node_weights",
            ));
        }
        validate_positive_finite_weights(nw, "cluster_area_faithfulness::node_weights")?;
    }

    let (cluster_ids, inv) = remap_labels(labels)?;
    let k_count = cluster_ids.len();

    if let Some(cw) = cluster_weights {
        if cw.len() != k_count {
            return Err(CoreError::dimension_mismatch(
                k_count,
                cw.len(),
                "cluster_area_faithfulness::cluster_weights",
            ));
        }
        validate_positive_finite_weights(cw, "cluster_area_faithfulness::cluster_weights")?;
    }

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

    cluster_area_core(
        &xs,
        &ys,
        cluster_ids,
        inv,
        config,
        node_weights,
        cluster_weights,
        include_detail,
    )
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
fn remap_labels(labels: &[u32]) -> CoreResult<(Vec<u32>, Vec<u32>)> {
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
                "labels length exceeds u32 index range",
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
