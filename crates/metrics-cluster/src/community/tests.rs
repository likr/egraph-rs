use std::collections::BTreeMap;

use super::*;

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
    let ms = adjusted_patch_nig_multiscale(&layout, &labels, &[2, 8, 32], 2048, 0.04, 42).unwrap();
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

#[test]
fn test_adjusted_patch_nig_detail_diagnostics_and_degenerate_sparse() {
    // Small sparse 16-node layout on a 2048x2048 canvas with patch_radius=4 -> all singletons -> degenerate.
    let mut xs = Vec::with_capacity(16);
    let mut ys = Vec::with_capacity(16);
    let mut labels = Vec::with_capacity(16);
    for i in 0..16 {
        let c = (i / 4) as u32;
        xs.push((i % 4) as f64 * 10.0 + (c as f64) * 0.1);
        ys.push((i / 4) as f64 * 10.0);
        labels.push(c % 2);
    }
    let sparse = Layout2d::from_xy(xs, ys).unwrap();
    let deg = adjusted_patch_nig_detail(&sparse, &labels, 4, 2048, 0.04, 42).unwrap();
    assert!(deg.is_degenerate);
    assert_eq!(deg.nig, 0.0);
    assert_eq!(deg.num_nonempty_patches, 16);
    assert!((deg.singleton_patch_ratio - 1.0).abs() < 1e-12);
    assert!(deg.recommended_patch_radius > 4);

    // Re-evaluating at recommended_patch_radius resolves singleton collapse.
    let rec = adjusted_patch_nig_detail(
        &sparse,
        &labels,
        deg.recommended_patch_radius,
        2048,
        0.04,
        42,
    )
    .unwrap();
    assert!(!rec.is_degenerate);
    assert!(rec.h_null > 1e-12);
    assert!(rec.singleton_patch_ratio < 0.5);
    assert!(
        (rec.nig
            - adjusted_patch_nig(
                &sparse,
                &labels,
                deg.recommended_patch_radius,
                2048,
                0.04,
                42
            )
            .unwrap())
        .abs()
            < 1e-12
    );

    // N == 0 and N == 1 and uniform-label degeneracies.
    let empty = Layout2d::<f64>::zeros(0);
    let d0 = adjusted_patch_nig_detail(&empty, &[], 8, 2048, 0.04, 42).unwrap();
    assert!(d0.is_degenerate);
    assert_eq!(d0.num_nonempty_patches, 0);
    assert_eq!(d0.singleton_patch_ratio, 0.0);

    let one = Layout2d::from_xy(vec![0.0], vec![0.0]).unwrap();
    let d1 = adjusted_patch_nig_detail(&one, &[0], 8, 2048, 0.04, 42).unwrap();
    assert!(d1.is_degenerate);
    assert_eq!(d1.num_nonempty_patches, 1);
    assert_eq!(d1.singleton_patch_ratio, 1.0);

    let mono = vec![7u32; 16];
    let d_mono = adjusted_patch_nig_detail(&sparse, &mono, 256, 2048, 0.04, 42).unwrap();
    assert!(d_mono.is_degenerate);
    assert_eq!(d_mono.nig, 0.0);
    assert!(d_mono.num_nonempty_patches >= 1);
}
