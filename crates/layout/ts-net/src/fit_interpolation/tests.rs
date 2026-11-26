use super::*;

fn exact_kl_repulsion(points: &[[f64; 2]]) -> (Vec<[f64; 2]>, f64) {
    let n = points.len();
    let mut forces = vec![[0.0; 2]; n];
    let mut total_z = 0.0;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let dx = points[i][0] - points[j][0];
            let dy = points[i][1] - points[j][1];
            let w = 1.0 / (1.0 + dx * dx + dy * dy);
            total_z += w;
            let w2 = w * w;
            forces[i][0] += w2 * dx;
            forces[i][1] += w2 * dy;
        }
    }
    let denom = total_z.max(1e-12);
    for f in &mut forces {
        f[0] /= denom;
        f[1] /= denom;
    }
    (forces, total_z)
}

#[test]
fn test_compute_fit_kl_repulsion_matches_exact_small_and_large_span() {
    let small_points = [
        [-0.8, -0.4],
        [-0.5, 0.3],
        [-0.2, -0.6],
        [0.1, 0.5],
        [0.4, -0.1],
        [0.9, 0.7],
        [-1.1, 0.8],
        [1.2, -0.7],
    ];
    let large_points = [
        [-28.0, 11.0],
        [-19.5, -3.5],
        [-7.4, 6.8],
        [0.0, 0.0],
        [1.2, -14.1],
        [10.5, 13.7],
        [18.9, -12.0],
        [26.6, 9.7],
        [-27.5, 10.6],
        [26.1, 9.2],
    ];
    let medium_span_points = [
        [-100.0, -40.0],
        [-95.0, -39.0],
        [-20.0, 60.0],
        [0.0, 0.0],
        [1.5, 0.8],
        [45.0, -70.0],
        [98.0, 82.0],
        [100.0, 81.0],
    ];

    for (label, pts, max_force_err, max_z_rel_err) in [
        ("small_span", small_points.as_slice(), 5e-3, 0.05),
        ("large_span", large_points.as_slice(), 1e-2, 0.15),
        (
            "medium_span_adaptive",
            medium_span_points.as_slice(),
            2e-2,
            0.10,
        ),
    ] {
        let (exact_f, exact_z) = exact_kl_repulsion(pts);
        let (approx_f, approx_z) = compute_fit_kl_repulsion(pts, 25, 3);

        assert!(
            approx_z.is_finite() && approx_z > 0.0,
            "{label}: invalid total_z={approx_z}"
        );
        let z_rel_err = (approx_z - exact_z).abs() / exact_z;
        assert!(
            z_rel_err < max_z_rel_err,
            "{label}: total_z rel_err={z_rel_err} (approx={approx_z}, exact={exact_z})"
        );

        for (i, (af, ef)) in approx_f.iter().zip(exact_f.iter()).enumerate() {
            assert!(af[0].is_finite() && af[1].is_finite());
            let err = ((af[0] - ef[0]).powi(2) + (af[1] - ef[1]).powi(2)).sqrt();
            assert!(
                err < max_force_err,
                "{label}[{i}]: force err={err}, approx={af:?}, exact={ef:?}"
            );
        }
    }
}

#[test]
fn test_compute_fit_kl_repulsion_extreme_span_decays_without_saturating() {
    // Points spread across span = 2000 (> MAX_ADAPTIVE_GRID_DIM = 512) with pairs sharing
    // coarse grid cells (separation ~2.0 inside hx ~ 4.7 cells) must not saturate at 0.5.
    let mut pts = Vec::new();
    for k in 0..10 {
        let base = -1000.0 + k as f64 * 200.0;
        pts.push([base, base * 0.5]);
        pts.push([base + 1.8, base * 0.5 + 1.2]);
    }
    let grid = FitGridSpec::from_points(&pts, 25, 3);
    assert_eq!(grid.nx, 512);
    assert!(grid.use_bilinear && grid.softening_sq > 0.0);

    let (rep, total_z) = compute_fit_kl_repulsion(&pts, 25, 3);
    assert!(total_z.is_finite() && total_z > 0.0);
    for (i, f) in rep.iter().enumerate() {
        let norm = (f[0] * f[0] + f[1] * f[1]).sqrt();
        assert!(norm.is_finite(), "non-finite force at {i}");
        assert!(
            norm < 0.25,
            "extreme-span force[{i}] saturated instead of decaying: norm={norm}"
        );
    }
}

#[test]
fn test_compute_fit_entropy_gradient_matches_quadtree_and_exact_tsnet() {
    use crate::quadtree::Quadtree;

    let pts = [
        [-5.0, -3.0],
        [-2.5, 2.2],
        [-0.4, -0.8],
        [0.5, 4.1],
        [2.4, -2.9],
        [4.5, 1.3],
        [5.8, -1.2],
        [-4.2, 4.6],
    ];
    let n = pts.len();
    let n_sq = (n * n) as f64;
    let epsilon_r = 0.05;

    let fft_raw = compute_fit_entropy_gradient(&pts, 25, 3, epsilon_r);
    let tree = Quadtree::build(&pts);

    for (i, p) in pts.iter().enumerate() {
        let fft_grad = [fft_raw[i][0] / n_sq, fft_raw[i][1] / n_sq];

        let mut rep_x = 0.0;
        let mut rep_y = 0.0;
        tree.root
            .compute_entropy_repulsion(i, p[0], p[1], 0.0, epsilon_r, &mut rep_x, &mut rep_y);
        let bh_grad = [-rep_x / n_sq, -rep_y / n_sq];

        let mut exact_tsnet = [0.0; 2];
        let mut exact_fft_kernel = [0.0; 2];
        for (j, q) in pts.iter().enumerate() {
            if i == j {
                continue;
            }
            let dx = p[0] - q[0];
            let dy = p[1] - q[1];
            let d_sq = dx * dx + dy * dy;
            let d = d_sq.sqrt();
            assert!(d >= 2.0, "test fixture requires pairwise distances >= 2.0");

            let tsnet_factor = -1.0 / (n_sq * d * (d + epsilon_r));
            exact_tsnet[0] += tsnet_factor * dx;
            exact_tsnet[1] += tsnet_factor * dy;

            let kernel_factor = -1.0 / (n_sq * (epsilon_r + d_sq));
            exact_fft_kernel[0] += kernel_factor * dx;
            exact_fft_kernel[1] += kernel_factor * dy;
        }

        let norm_fft = (fft_grad[0].powi(2) + fft_grad[1].powi(2)).sqrt();
        let norm_bh = (bh_grad[0].powi(2) + bh_grad[1].powi(2)).sqrt();
        let norm_tsnet = (exact_tsnet[0].powi(2) + exact_tsnet[1].powi(2)).sqrt();
        let norm_kernel = (exact_fft_kernel[0].powi(2) + exact_fft_kernel[1].powi(2)).sqrt();

        let cos_bh = (fft_grad[0] * bh_grad[0] + fft_grad[1] * bh_grad[1]) / (norm_fft * norm_bh);
        let cos_tsnet =
            (fft_grad[0] * exact_tsnet[0] + fft_grad[1] * exact_tsnet[1]) / (norm_fft * norm_tsnet);
        assert!(
            cos_bh > 0.99 && cos_tsnet > 0.99,
            "point {i}: sign/direction mismatch cos_bh={cos_bh}, cos_tsnet={cos_tsnet}"
        );

        let rel_err_bh = ((fft_grad[0] - bh_grad[0]).powi(2) + (fft_grad[1] - bh_grad[1]).powi(2))
            .sqrt()
            / norm_bh;
        assert!(
            rel_err_bh < 0.05,
            "point {i}: relative error vs Quadtree too large: {rel_err_bh}"
        );

        let rel_err_kernel = ((fft_grad[0] - exact_fft_kernel[0]).powi(2)
            + (fft_grad[1] - exact_fft_kernel[1]).powi(2))
        .sqrt()
            / norm_kernel;
        assert!(
            rel_err_kernel < 1e-3,
            "point {i}: relative error vs exact 1/(eps+d^2) kernel: {rel_err_kernel}"
        );
    }
}
