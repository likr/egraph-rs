use egraph_core::{DeterministicRng, Layout2d, LayoutTransform, NodeExtents2d};
use egraph_spatial_trees::RectangleOverlapSweep;

use super::super::{PfsPrimeConfig, PfsPrimeOverlapRemoval};
use super::{
    assert_no_overlaps, assert_no_overlaps_all, brute_f64_exact_overlaps,
    count_orthogonal_inversions, count_orthogonal_inversions_scalar,
};

#[test]
fn test_pfs_prime_adversarial_ties_dynamic_range_and_sub_ulp_f32() {
    // 1. Same-X tie with interleaved Y node on a distinct X column:
    //    u0=(0,0), u1=(0,0.5), w=(10,1.0). Must not invert u1 past w.
    let xs_tie = vec![0.0_f64, 0.0, 10.0];
    let ys_tie = vec![0.0_f64, 0.5, 1.0];
    let hw_tie = vec![1.0_f64, 1.0, 0.5];
    let hh_tie = vec![2.0_f64, 2.0, 0.5];
    let mut s_tie = PfsPrimeOverlapRemoval::<f64>::new(
        NodeExtents2d::new(&hw_tie, &hh_tie).unwrap(),
        PfsPrimeConfig::default(),
    )
    .unwrap();
    let mut l_tie = Layout2d::from_xy(xs_tie.clone(), ys_tie.clone()).unwrap();
    s_tie.apply(&mut l_tie).unwrap();
    assert_eq!(s_tie.last_overlap_count(), 0);
    assert_no_overlaps(&l_tie, &hw_tie, &hh_tie);
    assert_eq!(
        count_orthogonal_inversions(&xs_tie, &ys_tie, l_tie.x(), l_tie.y()),
        0
    );

    // 2. Extreme dynamic range with coincident groups at both 0 and 1e10.
    let xs_dyn = vec![0.0_f64, 0.0, 1e-10, 1e10, 1e10];
    let ys_dyn = vec![0.0_f64, 0.0, 1e-10, 1e10, 1e10];
    let hw_dyn = vec![1.0_f64; 5];
    let hh_dyn = vec![1.0_f64; 5];
    let mut s_dyn = PfsPrimeOverlapRemoval::<f64>::new(
        NodeExtents2d::new(&hw_dyn, &hh_dyn).unwrap(),
        PfsPrimeConfig::default(),
    )
    .unwrap();
    let mut l_dyn = Layout2d::from_xy(xs_dyn.clone(), ys_dyn.clone()).unwrap();
    s_dyn.apply(&mut l_dyn).unwrap();
    assert_eq!(s_dyn.last_overlap_count(), 0);
    assert_no_overlaps(&l_dyn, &hw_dyn, &hh_dyn);
    assert_eq!(
        count_orthogonal_inversions(&xs_dyn, &ys_dyn, l_dyn.x(), l_dyn.y()),
        0
    );

    // 3. Sub-ULP f32 overlap at 10000.0 (initial S-sweep misses it, f64-exact catches it).
    //    10000.6_f32 == 10000.599609375, so dx = 0.599609375 < 2 * 0.3_f32 = 0.6000000238,
    //    whereas in f32 both 10000.0_f32 + 0.3_f32 and 10000.6_f32 - 0.3_f32 round to
    //    10000.2998046875, hiding the overlap from RectangleOverlapSweep.
    let hw_sub = vec![0.3_f32, 0.3_f32];
    let hh_sub = vec![0.5_f32, 0.5_f32];
    let xs_sub = vec![10000.0_f32, 10000.6_f32];
    let ys_sub = vec![10000.0_f32, 10000.0_f32];
    let mut l_sub = Layout2d::from_xy(xs_sub.clone(), ys_sub.clone()).unwrap();
    assert!(
        RectangleOverlapSweep::find_layout_overlaps(&l_sub, &hw_sub, &hh_sub)
            .unwrap()
            .is_empty()
    );
    assert!(!brute_f64_exact_overlaps(&l_sub, &hw_sub, &hh_sub, 0.0).is_empty());
    let mut s_sub = PfsPrimeOverlapRemoval::<f32>::new(
        NodeExtents2d::new(&hw_sub, &hh_sub).unwrap(),
        PfsPrimeConfig::default(),
    )
    .unwrap();
    s_sub.apply(&mut l_sub).unwrap();
    assert_eq!(s_sub.last_overlap_count(), 0);
    assert_no_overlaps(&l_sub, &hw_sub, &hh_sub);
    assert_eq!(
        count_orthogonal_inversions_scalar(&xs_sub, &ys_sub, l_sub.x(), l_sub.y()),
        0
    );

    // 4. Zero-extent boxes (hw=0, hh=0) with padding=0 remain untouched.
    let zero_ext = vec![0.0_f64; 3];
    let mut s_zero = PfsPrimeOverlapRemoval::<f64>::new(
        NodeExtents2d::new(&zero_ext, &zero_ext).unwrap(),
        PfsPrimeConfig::default(),
    )
    .unwrap();
    let mut l_zero = Layout2d::zeros(3);
    s_zero.apply(&mut l_zero).unwrap();
    assert_eq!(s_zero.last_overlap_count(), 0);
    assert_eq!(l_zero.x(), &[0.0, 0.0, 0.0]);
    assert_eq!(l_zero.y(), &[0.0, 0.0, 0.0]);
}

#[test]
fn test_pfs_prime_tight_cluster_outlier_and_multiscale_fuzz() {
    // Explicit multi-scale tight-cluster + distant-outlier layout with extreme extent ratios.
    let cluster_xs = vec![
        10000.0,
        10000.0 - 1e-6,
        10000.0 - 2.621e-9,
        10000.0 + 1.874e-10,
        10000.0 - 5e-9,
        10000.0 + 4e-9,
        10000.0 + 1.2,
        10000.0 - 0.8,
        -1e8,
        1e8,
        -1e10,
        1e10,
        10000.0,
        10000.0 + 3.0,
        10000.0 - 2.0,
        9999.999999997379,
        10000.000000000187,
        10000.0 + 5e-9,
        0.0,
    ];
    let cluster_ys = vec![
        0.0, 0.0, -1e8, 1e8, 0.2, -0.3, 0.0, 0.1, -1e8, 1e8, 0.0, 0.0, 0.5, -0.5, 0.0, -1e8, 1e8,
        0.0, 0.0,
    ];
    let cluster_hw = vec![
        40000.0, 1.0, 41.11e-6, 5.874e-9, 0.5, 2.0, 100.0, 50.0, 1.0, 1.0, 10.0, 10.0, 476.0, 25.0,
        12.0, 41.11e-6, 5.874e-9, 1e-7, 1.0,
    ];
    let cluster_hh = vec![
        1.0, 1.0, 1.0, 1e-7, 0.5, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1e-8, 1.0,
        1.0,
    ];
    let cfg_cluster = PfsPrimeConfig::new().with_padding(0.25).with_seed(42);
    let ext_cluster = NodeExtents2d::new(&cluster_hw, &cluster_hh).unwrap();
    let mut s_cluster_a =
        PfsPrimeOverlapRemoval::<f64>::new(ext_cluster, cfg_cluster.clone()).unwrap();
    let mut s_cluster_b = PfsPrimeOverlapRemoval::<f64>::new(ext_cluster, cfg_cluster).unwrap();
    let mut l_cluster_a = Layout2d::from_xy(cluster_xs.clone(), cluster_ys.clone()).unwrap();
    let mut l_cluster_b = l_cluster_a.clone();
    s_cluster_a.apply(&mut l_cluster_a).unwrap();
    s_cluster_b.apply(&mut l_cluster_b).unwrap();
    assert_eq!(s_cluster_a.last_overlap_count(), 0);
    assert_eq!(s_cluster_b.last_overlap_count(), 0);
    assert_no_overlaps_all(&l_cluster_a, &cluster_hw, &cluster_hh, 0.25);
    assert_eq!(
        count_orthogonal_inversions(&cluster_xs, &cluster_ys, l_cluster_a.x(), l_cluster_a.y()),
        0
    );
    assert_eq!(
        l_cluster_a
            .x()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        l_cluster_b
            .x()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        l_cluster_a
            .y()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        l_cluster_b
            .y()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );

    // 250 f64 multi-scale fuzz trials.
    let paddings = [0.0_f64, 0.25, 0.5, 1.0];
    let f64_coord_scales = [0.0_f64, 1e4, -1e4, 1e6, 1e8, -1e8, 1e10];
    let mut rng64 = DeterministicRng::from_seed_u64(0x5233_0064);
    for trial in 0..250 {
        let n = 6 + rng64.gen_range_usize(24);
        let padding = paddings[trial % paddings.len()];
        let base = f64_coord_scales[trial % f64_coord_scales.len()];
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        let mut hw = Vec::with_capacity(n);
        let mut hh = Vec::with_capacity(n);

        for i in 0..n {
            let kind = rng64.gen_range_usize(5);
            let (x, y) = match kind {
                0 => (
                    base + (rng64.next_f64() - 0.5) * 1e-8,
                    base + (rng64.next_f64() - 0.5) * 1e-8,
                ),
                1 => (
                    base + (rng64.next_f64() - 0.5) * 5.0,
                    base + (rng64.next_f64() - 0.5) * 5.0,
                ),
                2 => {
                    let s = f64_coord_scales[(i + trial) % f64_coord_scales.len()];
                    (s, base)
                }
                3 => (base, -base),
                _ => (
                    1e4 + (rng64.next_f64() - 0.5) * 1e-9,
                    (rng64.next_f64() - 0.5) * 1e8,
                ),
            };
            xs.push(x);
            ys.push(y);
            let exp_w = -9.0 + rng64.next_f64() * 12.0;
            let exp_h = -9.0 + rng64.next_f64() * 12.0;
            hw.push(10.0_f64.powf(exp_w).min(500.0));
            hh.push(10.0_f64.powf(exp_h).min(500.0));
        }

        let cfg = PfsPrimeConfig::new()
            .with_padding(padding)
            .with_seed(1000 + trial as u64);
        let ext = NodeExtents2d::new(&hw, &hh).unwrap();
        let mut s_step = PfsPrimeOverlapRemoval::<f64>::new(ext, cfg).unwrap();
        let mut l_step = Layout2d::from_xy(xs.clone(), ys.clone()).unwrap();

        s_step.apply(&mut l_step).unwrap();
        assert_eq!(s_step.last_overlap_count(), 0);
        assert_no_overlaps_all(&l_step, &hw, &hh, padding);
        assert_eq!(
            count_orthogonal_inversions(&xs, &ys, l_step.x(), l_step.y()),
            0,
            "f64 multiscale fuzz oo_nni > 0 at trial={trial}"
        );
    }

    // 250 f32 multi-scale fuzz trials.
    let f32_coord_scales = [0.0_f32, 1e3, -1e3, 1e4, 1e5, -1e5, 1e6];
    let mut rng32 = DeterministicRng::from_seed_u64(0x5233_0032);
    for trial in 0..250 {
        let n = 6 + rng32.gen_range_usize(24);
        let padding = paddings[trial % paddings.len()];
        let base = f32_coord_scales[trial % f32_coord_scales.len()];
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        let mut hw = Vec::with_capacity(n);
        let mut hh = Vec::with_capacity(n);

        for i in 0..n {
            let kind = rng32.gen_range_usize(4);
            let (x, y) = match kind {
                0 => (
                    base + ((rng32.next_f64() - 0.5) * 1e-3) as f32,
                    base + ((rng32.next_f64() - 0.5) * 1e-3) as f32,
                ),
                1 => (
                    base + ((rng32.next_f64() - 0.5) * 5.0) as f32,
                    base + ((rng32.next_f64() - 0.5) * 5.0) as f32,
                ),
                2 => {
                    let s = f32_coord_scales[(i + trial) % f32_coord_scales.len()];
                    (s, base)
                }
                _ => (
                    1e4_f32 + ((rng32.next_f64() - 0.5) * 1e-2) as f32,
                    ((rng32.next_f64() - 0.5) * 1e5) as f32,
                ),
            };
            xs.push(x);
            ys.push(y);
            let exp_w = -4.0 + rng32.next_f64() * 6.5;
            let exp_h = -4.0 + rng32.next_f64() * 6.5;
            hw.push((10.0_f64.powf(exp_w).min(300.0)) as f32);
            hh.push((10.0_f64.powf(exp_h).min(300.0)) as f32);
        }

        let cfg = PfsPrimeConfig::new()
            .with_padding(padding)
            .with_seed(2000 + trial as u64);
        let ext = NodeExtents2d::new(&hw, &hh).unwrap();
        let mut s_step = PfsPrimeOverlapRemoval::<f32>::new(ext, cfg).unwrap();
        let mut l_step = Layout2d::from_xy(xs.clone(), ys.clone()).unwrap();

        s_step.apply(&mut l_step).unwrap();
        assert_eq!(s_step.last_overlap_count(), 0);
        assert_no_overlaps_all(&l_step, &hw, &hh, padding);
        assert_eq!(
            count_orthogonal_inversions_scalar(&xs, &ys, l_step.x(), l_step.y()),
            0,
            "f32 multiscale fuzz oo_nni > 0 at trial={trial}"
        );
    }
}
