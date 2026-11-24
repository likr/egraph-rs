//! Community spatial coherence metrics (`adjusted_patch_nig`).

use egraph_core::{CoreError, CoreResult, DeterministicRng, Layout2d, Scalar};

#[inline]
fn fit_viewport_equal_aspect<S: Scalar>(
    layout: &Layout2d<S>,
    canvas_size: usize,
    pad_frac: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = layout.len();
    let xs = layout.x();
    let ys = layout.y();
    let mut xmin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    let mut ymin = f64::INFINITY;
    let mut ymax = f64::NEG_INFINITY;
    for i in 0..n {
        let x = xs[i].to_f64();
        let y = ys[i].to_f64();
        xmin = xmin.min(x);
        xmax = xmax.max(x);
        ymin = ymin.min(y);
        ymax = ymax.max(y);
    }

    let xspan = (xmax - xmin).max(1e-9);
    let yspan = (ymax - ymin).max(1e-9);
    let cx_data = 0.5 * (xmin + xmax);
    let cy_data = 0.5 * (ymin + ymax);

    let c_sz = canvas_size as f64;
    let avail = (c_sz * (1.0 - 2.0 * pad_frac)).max(1.0);
    let scale = (avail / xspan).min(avail / yspan);
    let half = 0.5 * c_sz;

    let mut px = Vec::with_capacity(n);
    let mut py = Vec::with_capacity(n);
    for i in 0..n {
        px.push(half + (xs[i].to_f64() - cx_data) * scale);
        py.push(half - (ys[i].to_f64() - cy_data) * scale);
    }
    (px, py)
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
        // Skip pass when all keys share the same byte (e.g., upper bytes of small IDs).
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

/// Computes $H(C \mid \mathrm{Patch}) = \max(0, (\sum_p n_p \ln n_p - \sum_{p,c} n_{p,c} \ln n_{p,c}) / N)$
/// in a single linear scan over sorted 64-bit `(patch_id << 32) | comm_id` keys.
fn conditional_entropy_sorted_keys(keys: &[u64]) -> f64 {
    let n = keys.len();
    if n == 0 {
        return 0.0;
    }
    let mut sum_p_log = 0.0_f64;
    let mut sum_pc_log = 0.0_f64;

    let mut prev_key = keys[0];
    let mut prev_p = prev_key >> 32;
    let mut run_p = 1usize;
    let mut run_pc = 1usize;

    for &key in &keys[1..] {
        let p = key >> 32;
        if key == prev_key {
            run_pc += 1;
            run_p += 1;
        } else {
            if run_pc > 1 {
                let f = run_pc as f64;
                sum_pc_log += f * f.ln();
            }
            prev_key = key;
            run_pc = 1;
            if p == prev_p {
                run_p += 1;
            } else {
                if run_p > 1 {
                    let f = run_p as f64;
                    sum_p_log += f * f.ln();
                }
                prev_p = p;
                run_p = 1;
            }
        }
    }
    if run_pc > 1 {
        let f = run_pc as f64;
        sum_pc_log += f * f.ln();
    }
    if run_p > 1 {
        let f = run_p as f64;
        sum_p_log += f * f.ln();
    }

    ((sum_p_log - sum_pc_log) / (n as f64)).max(0.0)
}

/// Evaluates Adjusted Patch Normalized Information Gain (`Adjusted Patch-NIG`) across
/// multiple grid patch radii `[r0, r1, ...]` sharing viewport projection and permutation null.
///
/// Maps coordinates with equal-aspect fitting onto `[0, canvas_size]^2`, bins points into
/// square patches of side `2 * patch_radius`, and calibrates conditional entropy
/// $H(C \mid \mathrm{Patch})$ against a single deterministic `DeterministicRng` permutation null:
/// `(H_null - H_obs) / H_null` (`0.0` if `N < 2`, fewer than 2 unique labels, or `H_null <= 1e-12`).
pub fn adjusted_patch_nig_multiscale<S: Scalar>(
    layout: &Layout2d<S>,
    community_labels: &[u32],
    radii: &[usize],
    canvas_size: usize,
    pad_frac: f64,
    seed: u64,
) -> CoreResult<Vec<f64>> {
    layout.validate_finite()?;
    let n = layout.len();
    if community_labels.len() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            community_labels.len(),
            "adjusted_patch_nig::community_labels",
        ));
    }
    if canvas_size == 0 {
        return Err(CoreError::invalid_input("canvas_size must be >= 1"));
    }
    if !pad_frac.is_finite() {
        return Err(CoreError::non_finite("adjusted_patch_nig::pad_frac"));
    }
    if !(0.0..0.5).contains(&pad_frac) {
        return Err(CoreError::invalid_input("pad_frac must be in [0.0, 0.5)"));
    }
    for &r in radii {
        if r == 0 {
            return Err(CoreError::invalid_input("patch_radius must be >= 1"));
        }
    }
    if radii.is_empty() {
        return Ok(Vec::new());
    }

    if n < 2 {
        return Ok(vec![0.0; radii.len()]);
    }
    let first_label = community_labels[0];
    if community_labels.iter().all(|&lbl| lbl == first_label) {
        return Ok(vec![0.0; radii.len()]);
    }

    let c_sz = canvas_size as f64;
    let (px, py) = fit_viewport_equal_aspect(layout, canvas_size, pad_frac);

    let mut perm = community_labels.to_vec();
    let mut rng = DeterministicRng::from_seed_u64(seed);
    rng.shuffle(&mut perm);

    let mut keys = vec![0_u64; n];
    let mut scratch = Vec::new();
    let mut out = Vec::with_capacity(radii.len());

    for &r in radii {
        let side = (2 * r).max(1) as f64;
        let grid_dim = (c_sz / side).ceil().max(1.0) as usize;
        let max_patch = (grid_dim as u128)
            .checked_mul(grid_dim as u128)
            .and_then(|v| v.checked_sub(1))
            .ok_or_else(|| CoreError::invalid_input("patch grid dimensions overflow"))?;
        if u32::try_from(max_patch).is_err() {
            return Err(CoreError::invalid_input(
                "canvas_size / patch_radius grid exceeds u32 patch ID range",
            ));
        }
        let max_coord = (grid_dim - 1) as f64;

        for i in 0..n {
            let ix = (px[i] / side).floor().clamp(0.0, max_coord) as usize;
            let iy = (py[i] / side).floor().clamp(0.0, max_coord) as usize;
            let patch_id = (iy * grid_dim + ix) as u64;
            keys[i] = (patch_id << 32) | u64::from(community_labels[i]);
        }
        radix_sort_u64(&mut keys, &mut scratch);
        let h_obs = conditional_entropy_sorted_keys(&keys);

        for i in 0..n {
            let ix = (px[i] / side).floor().clamp(0.0, max_coord) as usize;
            let iy = (py[i] / side).floor().clamp(0.0, max_coord) as usize;
            let patch_id = (iy * grid_dim + ix) as u64;
            keys[i] = (patch_id << 32) | u64::from(perm[i]);
        }
        radix_sort_u64(&mut keys, &mut scratch);
        let h_null = conditional_entropy_sorted_keys(&keys);

        let nig = if h_null <= 1e-12 {
            0.0
        } else {
            ((h_null - h_obs) / h_null).min(1.0)
        };
        out.push(nig);
    }

    Ok(out)
}

/// Evaluates single-scale Adjusted Patch Normalized Information Gain (`Adjusted Patch-NIG`).
pub fn adjusted_patch_nig<S: Scalar>(
    layout: &Layout2d<S>,
    community_labels: &[u32],
    patch_radius: usize,
    canvas_size: usize,
    pad_frac: f64,
    seed: u64,
) -> CoreResult<f64> {
    let vals = adjusted_patch_nig_multiscale(
        layout,
        community_labels,
        &[patch_radius],
        canvas_size,
        pad_frac,
        seed,
    )?;
    Ok(vals[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn dense_contingency_entropy_oracle(patch_ids: &[u32], labels: &[u32]) -> f64 {
        let n = patch_ids.len();
        if n == 0 {
            return 0.0;
        }
        let mut p_counts = BTreeMap::<u32, usize>::new();
        let mut pc_counts = BTreeMap::<(u32, u32), usize>::new();
        for (&p, &c) in patch_ids.iter().zip(labels.iter()) {
            *p_counts.entry(p).or_insert(0) += 1;
            *pc_counts.entry((p, c)).or_insert(0) += 1;
        }
        let mut sum_p = 0.0_f64;
        for &cnt in p_counts.values() {
            if cnt > 1 {
                let f = cnt as f64;
                sum_p += f * f.ln();
            }
        }
        let mut sum_pc = 0.0_f64;
        for &cnt in pc_counts.values() {
            if cnt > 1 {
                let f = cnt as f64;
                sum_pc += f * f.ln();
            }
        }
        ((sum_p - sum_pc) / (n as f64)).max(0.0)
    }

    #[test]
    fn test_adjusted_patch_nig_dense_contingency_oracle_and_invariance() {
        // Test N > 1024 to exercise the 8-pass LSD radix sort path directly against dense oracle.
        let mut rng = DeterministicRng::from_seed_u64(12345);
        let n = 1500usize;
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        let mut labels = Vec::with_capacity(n);
        for i in 0..n {
            let comm = (i % 5) as u32;
            let cx = (comm as f64) * 5.0;
            let cy = if comm.is_multiple_of(2) { 0.0 } else { 5.0 };
            xs.push(cx + (rng.next_f64() - 0.5) * 0.8);
            ys.push(cy + (rng.next_f64() - 0.5) * 0.8);
            labels.push(comm);
        }
        let layout = Layout2d::from_xy(xs.clone(), ys.clone()).unwrap();

        // Separated communities should achieve near-1.0 Adjusted Patch-NIG.
        let nig_sep = adjusted_patch_nig(&layout, &labels, 16, 2048, 0.04, 42).unwrap();
        assert!(nig_sep > 0.95, "nig_sep={nig_sep}");

        // Scale and translation invariance.
        let scaled = Layout2d::from_xy(
            xs.iter().map(|&v| v * 7.5 - 100.0).collect(),
            ys.iter().map(|&v| v * 7.5 + 42.0).collect(),
        )
        .unwrap();
        let nig_scaled = adjusted_patch_nig(&scaled, &labels, 16, 2048, 0.04, 42).unwrap();
        assert!((nig_sep - nig_scaled).abs() < 1e-12);

        // Compare radix sort conditional entropy directly against dense contingency table oracle.
        let (px, py) = fit_viewport_equal_aspect(&layout, 2048, 0.04);
        let side = 32.0_f64;
        let grid_dim = (2048.0_f64 / side).ceil() as usize;
        let mut patch_ids = Vec::with_capacity(n);
        let mut keys = Vec::with_capacity(n);
        for i in 0..n {
            let ix = (px[i] / side).floor().clamp(0.0, (grid_dim - 1) as f64) as usize;
            let iy = (py[i] / side).floor().clamp(0.0, (grid_dim - 1) as f64) as usize;
            let pid = (iy * grid_dim + ix) as u32;
            patch_ids.push(pid);
            keys.push((u64::from(pid) << 32) | u64::from(labels[i]));
        }
        let mut scratch = Vec::new();
        radix_sort_u64(&mut keys, &mut scratch);
        let got_h = conditional_entropy_sorted_keys(&keys);
        let oracle_h = dense_contingency_entropy_oracle(&patch_ids, &labels);
        assert!(
            (got_h - oracle_h).abs() < 1e-12,
            "got={got_h}, oracle={oracle_h}"
        );

        // Multiscale parity with single-scale calls, plus f32 support.
        let ms =
            adjusted_patch_nig_multiscale(&layout, &labels, &[2, 8, 32], 2048, 0.04, 42).unwrap();
        for (idx, &r) in [2usize, 8, 32].iter().enumerate() {
            let single = adjusted_patch_nig(&layout, &labels, r, 2048, 0.04, 42).unwrap();
            assert!((ms[idx] - single).abs() < 1e-12);
        }

        let l32 = Layout2d::from_xy(
            xs.iter().map(|&v| v as f32).collect(),
            ys.iter().map(|&v| v as f32).collect(),
        )
        .unwrap();
        let nig32 = adjusted_patch_nig(&l32, &labels, 16, 2048, 0.04, 42).unwrap();
        assert!(nig32 > 0.95);

        // Random labels -> near 0.0.
        let mut rand_labels = labels.clone();
        let mut rng_lbl = DeterministicRng::from_seed_u64(999);
        rng_lbl.shuffle(&mut rand_labels);
        let nig_rand = adjusted_patch_nig(&layout, &rand_labels, 16, 2048, 0.04, 42).unwrap();
        assert!(nig_rand.abs() < 0.08, "nig_rand={nig_rand}");

        // Single community or N < 2 -> 0.0.
        let mono = vec![7u32; n];
        assert_eq!(
            adjusted_patch_nig(&layout, &mono, 8, 2048, 0.04, 42).unwrap(),
            0.0
        );
        let one = Layout2d::from_xy(vec![0.0], vec![0.0]).unwrap();
        assert_eq!(
            adjusted_patch_nig(&one, &[0], 8, 2048, 0.04, 42).unwrap(),
            0.0
        );
    }

    #[test]
    fn test_adjusted_patch_nig_validation_errors() {
        let l = Layout2d::from_xy(vec![0.0, 1.0], vec![0.0, 1.0]).unwrap();
        assert!(matches!(
            adjusted_patch_nig(&l, &[0], 8, 2048, 0.04, 42),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            adjusted_patch_nig(&l, &[0, 1], 0, 2048, 0.04, 42),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            adjusted_patch_nig(&l, &[0, 1], 8, 0, 0.04, 42),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            adjusted_patch_nig(&l, &[0, 1], 8, 2048, 0.5, 42),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            adjusted_patch_nig(&l, &[0, 1], 8, 2048, f64::NAN, 42),
            Err(CoreError::NonFiniteValue { .. })
        ));
    }
}
