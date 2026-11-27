use egraph_core::{
    CoreError, CsrGraph, DeterministicRng, Layout2d, Layout3d, LayoutGenerator, LayoutNd,
};

use super::normalize::next_gaussian_pair;
use super::tests_hde_oracle::isotropic_std_nd;
use super::*;

fn grid_3x3_graph() -> CsrGraph<f64> {
    let mut edges = Vec::new();
    for r in 0..3 {
        for c in 0..3 {
            let u = (r * 3 + c) as u32;
            if c + 1 < 3 {
                edges.push((u, u + 1));
            }
            if r + 1 < 3 {
                edges.push((u, u + 3));
            }
        }
    }
    CsrGraph::<f64>::from_edges(9, &edges, None, false, false).unwrap()
}

fn isotropic_std_3d(layout: &Layout3d<f64>) -> f64 {
    let n = layout.len() as f64;
    let mx = layout.x().iter().sum::<f64>() / n;
    let my = layout.y().iter().sum::<f64>() / n;
    let mz = layout.z().iter().sum::<f64>() / n;
    let mut sum_sq = 0.0_f64;
    for i in 0..layout.len() {
        let dx = layout.x()[i] - mx;
        let dy = layout.y()[i] - my;
        let dz = layout.z()[i] - mz;
        sum_sq += dx * dx + dy * dy + dz * dz;
    }
    (sum_sq / (3.0 * n)).sqrt()
}

fn assert_layout3d_matches_nd3(l3: &Layout3d<f64>, lnd: &LayoutNd<f64>) {
    assert_eq!(lnd.dims(), 3);
    assert_eq!(l3.len(), lnd.num_nodes());
    for i in 0..l3.len() {
        let row = lnd.node(i).unwrap();
        assert_eq!(l3.x()[i].to_bits(), row[0].to_bits());
        assert_eq!(l3.y()[i].to_bits(), row[1].to_bits());
        assert_eq!(l3.z()[i].to_bits(), row[2].to_bits());
    }
}

#[test]
fn test_all_five_generators_layout2d_3d_nd_and_f32() {
    let g = grid_3x3_graph();

    // 1. ClassicalMds across 2D, 3D, ND (f64 + f32)
    let c2: Layout2d<f64> = ClassicalMds::new(&g).generate().unwrap();
    let c3: Layout3d<f64> = ClassicalMds::new(&g).generate().unwrap();
    let cnd3: LayoutNd<f64> = ClassicalMds::new(&g).with_dims(3).generate().unwrap();
    let cnd4: LayoutNd<f64> = ClassicalMds::new(&g).with_dims(4).generate().unwrap();
    let c3_f32: Layout3d<f32> = ClassicalMds::new(&g).generate().unwrap();
    assert!(c2.is_all_finite());
    assert!(c3.is_all_finite());
    assert_eq!(cnd4.dims(), 4);
    assert!(cnd4.is_all_finite());
    assert!(c3_f32.is_all_finite());
    assert_layout3d_matches_nd3(&c3, &cnd3);

    // 2. PivotMds across 2D, 3D, ND (f64 + f32) + from_distances
    let p2: Layout2d<f64> = PivotMds::new(&g).generate().unwrap();
    let p3: Layout3d<f64> = PivotMds::new(&g).generate().unwrap();
    let pnd3: LayoutNd<f64> = PivotMds::new(&g).with_dims(3).generate().unwrap();
    let pnd4: LayoutNd<f64> = PivotMds::new(&g).with_dims(4).generate().unwrap();
    let p2_f32: Layout2d<f32> = PivotMds::new(&g).generate().unwrap();
    let p3_f32: Layout3d<f32> = PivotMds::new(&g).generate().unwrap();
    let pnd_f32: LayoutNd<f32> = PivotMds::new(&g).with_dims(3).generate().unwrap();
    assert!(p2.is_all_finite());
    assert!((isotropic_std_3d(&p3) - 1.0).abs() < 1e-10);
    assert_eq!(pnd4.dims(), 4);
    assert!((isotropic_std_nd(&pnd4) - 1.0).abs() < 1e-10);
    assert!(p2_f32.is_all_finite());
    assert!(p3_f32.is_all_finite());
    assert!(pnd_f32.is_all_finite());
    assert_layout3d_matches_nd3(&p3, &pnd3);

    // PivotMds::from_distances (representative/landmark mode with target_std=None)
    let dists_nk = vec![
        0.0, 2.0, 4.0, 1.0, 1.0, 3.0, 2.0, 0.0, 2.0, 3.0, 1.0, 1.0, 4.0, 2.0, 0.0,
    ];
    let rep3: Layout3d<f64> = PivotMds::from_distances(&dists_nk, 5, 3)
        .with_target_std(None)
        .generate()
        .unwrap();
    let rep_nd: LayoutNd<f64> = PivotMds::from_distances(&dists_nk, 5, 3)
        .with_dims(3)
        .with_target_std(None)
        .generate()
        .unwrap();
    assert!(rep3.is_all_finite());
    assert_layout3d_matches_nd3(&rep3, &rep_nd);

    // 3. Hde across 2D, 3D, ND (f64 + f32)
    let h2: Layout2d<f64> = Hde::new(&g).generate().unwrap();
    let h3: Layout3d<f64> = Hde::new(&g).generate().unwrap();
    let hnd3: LayoutNd<f64> = Hde::new(&g).with_dims(3).generate().unwrap();
    let hnd4: LayoutNd<f64> = Hde::new(&g).with_output_dims(4).generate().unwrap();
    let h3_f32: Layout3d<f32> = Hde::new(&g).generate().unwrap();
    assert!(h2.is_all_finite());
    assert!((isotropic_std_3d(&h3) - 1.0).abs() < 1e-10);
    assert_eq!(hnd4.dims(), 4);
    assert!(h3_f32.is_all_finite());
    assert_layout3d_matches_nd3(&h3, &hnd3);

    // 4. RdMds across 2D, 3D, ND (f64 + f32)
    let r2: Layout2d<f64> = RdMds::new(&g).generate().unwrap();
    let r3: Layout3d<f64> = RdMds::new(&g).generate().unwrap();
    let rnd3: LayoutNd<f64> = RdMds::new(&g).with_dims(3).generate().unwrap();
    let rnd4: LayoutNd<f64> = RdMds::new(&g).with_dims(4).generate().unwrap();
    let r3_f32: Layout3d<f32> = RdMds::new(&g).generate().unwrap();
    let rnd_f32: LayoutNd<f32> = RdMds::new(&g).with_dims(3).generate().unwrap();
    assert!(r2.is_all_finite());
    assert!(r3.is_all_finite());
    assert_eq!(rnd4.dims(), 4);
    assert!(rnd4.is_all_finite());
    assert!(r3_f32.is_all_finite());
    assert!(rnd_f32.is_all_finite());
    assert_layout3d_matches_nd3(&r3, &rnd3);

    // 5. KorenSpectral across 2D, 3D, ND (f64 + f32)
    let k2: Layout2d<f64> = KorenSpectral::new(&g).generate().unwrap();
    let k3: Layout3d<f64> = KorenSpectral::new(&g).generate().unwrap();
    let knd3: LayoutNd<f64> = KorenSpectral::new(&g).with_dims(3).generate().unwrap();
    let knd4: LayoutNd<f64> = KorenSpectral::new(&g).with_dims(4).generate().unwrap();
    let k3_f32: Layout3d<f32> = KorenSpectral::new(&g).generate().unwrap();
    let knd_f32: LayoutNd<f32> = KorenSpectral::new(&g).with_dims(3).generate().unwrap();
    assert!(k2.is_all_finite());
    assert!((isotropic_std_3d(&k3) - 1.0).abs() < 1e-10);
    assert_eq!(knd4.dims(), 4);
    assert!((isotropic_std_nd(&knd4) - 1.0).abs() < 1e-10);
    assert!(k3_f32.is_all_finite());
    assert!(knd_f32.is_all_finite());
    assert_layout3d_matches_nd3(&k3, &knd3);
}

#[test]
fn test_generator_trait_object_and_dims_validation() {
    let g = grid_3x3_graph();

    // Dynamic trait dispatch via `&mut dyn LayoutGenerator`
    let mut gen2: Box<dyn LayoutGenerator<Output = Layout2d<f64>>> = Box::new(PivotMds::new(&g));
    let out2 = gen2.generate().unwrap();
    assert_eq!(out2.len(), 9);

    let mut gen3: Box<dyn LayoutGenerator<Output = Layout3d<f64>>> =
        Box::new(KorenSpectral::new(&g));
    let out3 = gen3.generate().unwrap();
    assert_eq!(out3.len(), 9);

    // Mismatched explicit dims fail fast for Layout2d / Layout3d, dims == 0 fails for LayoutNd
    assert!(matches!(
        ClassicalMds::<Layout2d<f64>>::new(&g)
            .with_dims(3)
            .generate(),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        ClassicalMds::<Layout3d<f64>>::new(&g)
            .with_dims(2)
            .generate(),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        ClassicalMds::<LayoutNd<f64>>::new(&g)
            .with_dims(0)
            .generate(),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        PivotMds::<Layout3d<f64>>::new(&g).with_dims(2).generate(),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        RdMds::<Layout2d<f64>>::new(&g).with_dims(3).generate(),
        Err(CoreError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        KorenSpectral::<Layout3d<f64>>::new(&g)
            .with_dims(4)
            .generate(),
        Err(CoreError::DimensionMismatch { .. })
    ));

    // PivotMds::from_distances + with_pivots is rejected
    let d = vec![0.0; 9];
    assert!(matches!(
        PivotMds::<Layout2d<f64>>::from_distances(&d, 3, 3)
            .with_pivots(&[0, 1])
            .generate(),
        Err(CoreError::InvalidInput(_))
    ));
}

#[test]
fn test_config_builders_and_euclidean_normalization_3d_nd() {
    let g = grid_3x3_graph();

    let c_cfg = ClassicalMdsConfig::new()
        .with_dims(Some(3))
        .with_eig_eps(1e-9)
        .with_tol(1e-11)
        .with_max_sweeps(100);
    let p_cfg = PivotMdsConfig::new()
        .with_dims(Some(3))
        .with_num_pivots(6)
        .with_min_degree(1)
        .with_seed(7)
        .with_target_std(Some(2.0));
    let h_cfg = HdeConfig::new()
        .with_dimensions(6)
        .with_output_dims(Some(3))
        .with_seed(7)
        .with_tol(1e-4)
        .with_max_sweeps(120)
        .with_normalize(true);
    let r_cfg = RdMdsConfig::new()
        .with_dims(Some(3))
        .with_eig_eps(1e-9)
        .with_tol(1e-6)
        .with_max_iters(200)
        .with_seed(7)
        .with_degree_normalize(false)
        .with_target_std(Some(1.5));
    let k_cfg = KorenSpectralConfig::new()
        .with_dims(Some(3))
        .with_target_std(Some(1.2))
        .with_tol(1e-6)
        .with_max_iters(200)
        .with_seed(7);

    let c_out: Layout3d<f64> = ClassicalMds::from_config(&g, c_cfg).generate().unwrap();
    let p_out: Layout3d<f64> = PivotMds::from_config(&g, p_cfg).generate().unwrap();
    let h_out: Layout3d<f64> = Hde::from_config(&g, h_cfg).generate().unwrap();
    let r_out: Layout3d<f64> = RdMds::from_config(&g, r_cfg).generate().unwrap();
    let k_out: Layout3d<f64> = KorenSpectral::from_config(&g, k_cfg).generate().unwrap();
    assert!(c_out.is_all_finite());
    assert!((isotropic_std_3d(&p_out) - 2.0).abs() < 1e-10);
    assert!((isotropic_std_3d(&h_out) - 1.0).abs() < 1e-10);
    assert!((isotropic_std_3d(&r_out) - 1.5).abs() < 1e-10);
    assert!((isotropic_std_3d(&k_out) - 1.2).abs() < 1e-10);

    // normalize_euclidean_layout on collapsed 3D + NaN sanitization matches 3D LayoutNd
    let mut l3 = Layout3d::from_xyz(
        vec![f64::NAN, 3.0, 3.0, 3.0],
        vec![5.0, f64::INFINITY, 5.0, 5.0],
        vec![-1.0, -1.0, f64::NEG_INFINITY, -1.0],
    )
    .unwrap();
    let mut lnd = LayoutNd::from_flat(
        4,
        3,
        &[
            f64::NAN,
            5.0,
            -1.0,
            3.0,
            f64::INFINITY,
            -1.0,
            3.0,
            5.0,
            f64::NEG_INFINITY,
            3.0,
            5.0,
            -1.0,
        ],
    )
    .unwrap();
    normalize_euclidean_layout(&mut l3, 1.0, 99).unwrap();
    normalize_euclidean_layout(&mut lnd, 1.0, 99).unwrap();
    assert!((isotropic_std_3d(&l3) - 1.0).abs() < 1e-12);
    assert_layout3d_matches_nd3(&l3, &lnd);

    // robust_normalize_euclidean_layout on 3D & 4D Gaussian + 4% extreme outliers
    let mut rng = DeterministicRng::from_seed_u64(321);
    let mut xs = Vec::with_capacity(100);
    let mut ys = Vec::with_capacity(100);
    let mut zs = Vec::with_capacity(100);
    for _ in 0..96 {
        let (gx, gy) = next_gaussian_pair(&mut rng);
        let (gz, _) = next_gaussian_pair(&mut rng);
        xs.push(gx);
        ys.push(gy);
        zs.push(gz);
    }
    for _ in 0..4 {
        xs.push(400.0);
        ys.push(-400.0);
        zs.push(400.0);
    }
    let mut rob3 = Layout3d::from_xyz(xs, ys, zs).unwrap();
    robust_normalize_euclidean_layout(&mut rob3, 1.0, 0.95, 42).unwrap();
    assert!(rob3.is_all_finite());
    let mut core_sq = 0.0_f64;
    for i in 0..96 {
        core_sq += rob3.x()[i].powi(2) + rob3.y()[i].powi(2) + rob3.z()[i].powi(2);
    }
    let core_std_3d = (core_sq / (3.0 * 96.0)).sqrt();
    assert!(
        (core_std_3d - 1.0).abs() < 0.35,
        "expected 3D core_std ~ 1.0, got {core_std_3d}"
    );

    let mut nd4_coords = Vec::with_capacity(100 * 4);
    for i in 0..100 {
        if i < 96 {
            let (a, b) = next_gaussian_pair(&mut rng);
            let (c, d) = next_gaussian_pair(&mut rng);
            nd4_coords.extend_from_slice(&[a, b, c, d]);
        } else {
            nd4_coords.extend_from_slice(&[500.0, 500.0, -500.0, 500.0]);
        }
    }
    let mut rob4 = LayoutNd::from_flat(100, 4, &nd4_coords).unwrap();
    robust_normalize_euclidean_layout(&mut rob4, 1.0, 0.95, 42).unwrap();
    assert!(rob4.is_all_finite());
    let mut core_sq4 = 0.0_f64;
    for i in 0..96 {
        for &v in rob4.node(i).unwrap() {
            core_sq4 += v * v;
        }
    }
    let core_std_4d = (core_sq4 / (4.0 * 96.0)).sqrt();
    assert!(
        (core_std_4d - 1.0).abs() < 0.35,
        "expected 4D core_std ~ 1.0, got {core_std_4d}"
    );
}

#[test]
fn test_from_distances_validation_and_disconnected_infinity() {
    // ClassicalMds::from_distances rejects NaN, -inf, and negative distances (including n == 1)
    for bad in [
        [0.0, f64::NAN, 1.0, 0.0],
        [0.0, f64::NEG_INFINITY, 1.0, 0.0],
    ] {
        assert!(matches!(
            ClassicalMds::<Layout2d<f64>>::from_distances(&bad, 2).generate(),
            Err(CoreError::NonFiniteValue { .. })
        ));
    }
    assert!(matches!(
        ClassicalMds::<Layout2d<f64>>::from_distances(&[f64::NAN], 1).generate(),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        ClassicalMds::<Layout2d<f64>>::from_distances(&[f64::NEG_INFINITY], 1).generate(),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        ClassicalMds::<Layout2d<f64>>::from_distances(&[0.0, -1.0, -1.0, 0.0], 2).generate(),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        ClassicalMds::<Layout2d<f64>>::from_distances(&[-1.0], 1).generate(),
        Err(CoreError::InvalidInput(_))
    ));

    // ClassicalMds::from_distances preserves +f64::INFINITY as disconnected-pair sentinel
    let cmds_inf: Layout2d<f64> = ClassicalMds::from_distances(
        &[
            0.0,
            1.0,
            f64::INFINITY,
            1.0,
            0.0,
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
            0.0,
        ],
        3,
    )
    .generate()
    .unwrap();
    assert_eq!(cmds_inf.len(), 3);
    assert!(cmds_inf.is_all_finite());

    // PivotMds::from_distances rejects NaN, -inf, negative distances, and num_pivots == 0
    for bad in [
        [0.0, f64::NAN, 1.0, 0.0],
        [0.0, f64::NEG_INFINITY, 1.0, 0.0],
    ] {
        assert!(matches!(
            PivotMds::<Layout2d<f64>>::from_distances(&bad, 2, 2).generate(),
            Err(CoreError::NonFiniteValue { .. })
        ));
    }
    assert!(matches!(
        PivotMds::<Layout2d<f64>>::from_distances(&[f64::NAN], 1, 1).generate(),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        PivotMds::<Layout2d<f64>>::from_distances(&[f64::NEG_INFINITY], 1, 1).generate(),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        PivotMds::<Layout2d<f64>>::from_distances(&[0.0, -1.0, -1.0, 0.0], 2, 2).generate(),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        PivotMds::<Layout2d<f64>>::from_distances(&[-1.0], 1, 1).generate(),
        Err(CoreError::InvalidInput(_))
    ));
    for n in [0, 1, 4] {
        assert!(matches!(
            PivotMds::<Layout2d<f64>>::from_distances(&[], n, 0).generate(),
            Err(CoreError::InvalidInput(_))
        ));
    }

    // PivotMds::from_distances preserves +f64::INFINITY as disconnected-pair sentinel
    let pmds_inf: Layout2d<f64> = PivotMds::from_distances(
        &[0.0, f64::INFINITY, 1.0, f64::INFINITY, f64::INFINITY, 0.0],
        3,
        2,
    )
    .generate()
    .unwrap();
    assert_eq!(pmds_inf.len(), 3);
    assert!(pmds_inf.is_all_finite());

    // PivotMds::with_pivots(&[]) rejected on non-empty graph; allowed on empty graph
    let g = grid_3x3_graph();
    assert!(matches!(
        PivotMds::<Layout2d<f64>>::new(&g)
            .with_pivots(&[])
            .generate(),
        Err(CoreError::InvalidInput(_))
    ));
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    assert_eq!(
        PivotMds::<Layout2d<f64>>::new(&empty)
            .with_pivots(&[])
            .generate()
            .unwrap()
            .len(),
        0
    );

    // eig_eps validation on ClassicalMds and RdMds distinguishes NonFiniteValue vs InvalidInput
    assert!(matches!(
        ClassicalMds::<Layout2d<f64>>::new(&g)
            .with_eig_eps(f64::NAN)
            .generate(),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        ClassicalMds::<Layout2d<f64>>::new(&g)
            .with_eig_eps(-1.0)
            .generate(),
        Err(CoreError::InvalidInput(_))
    ));
    assert!(matches!(
        RdMds::<Layout2d<f64>>::new(&g)
            .with_eig_eps(f64::INFINITY)
            .generate(),
        Err(CoreError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        RdMds::<Layout2d<f64>>::new(&g)
            .with_eig_eps(-1.0)
            .generate(),
        Err(CoreError::InvalidInput(_))
    ));
}
