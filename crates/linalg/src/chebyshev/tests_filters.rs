use std::f64::consts::PI;

use egraph_core::CsrGraph;

use crate::csr_matrix::symmetric_normalized_laplacian;

use super::*;

fn dense_expm_neg(a: &[f64], n: usize, t: f64) -> Vec<f64> {
    let mut scaled = vec![0.0_f64; n * n];
    let mut max_row_sum = 0.0_f64;
    for i in 0..n {
        let mut row_sum = 0.0_f64;
        for j in 0..n {
            let val = -t * a[i * n + j];
            scaled[i * n + j] = val;
            row_sum += val.abs();
        }
        if row_sum > max_row_sum {
            max_row_sum = row_sum;
        }
    }
    let s = if max_row_sum > 0.5 {
        (max_row_sum.log2().ceil().max(0.0) as u32) + 1
    } else {
        0
    };
    let pow_scale = 2.0_f64.powi(-(s as i32));
    for v in &mut scaled {
        *v *= pow_scale;
    }

    let mut res = vec![0.0_f64; n * n];
    let mut term = vec![0.0_f64; n * n];
    for i in 0..n {
        res[i * n + i] = 1.0;
        term[i * n + i] = 1.0;
    }
    let mut next = vec![0.0_f64; n * n];
    for k in 1..=35 {
        let inv_k = 1.0 / (k as f64);
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..n {
                    sum += term[i * n + p] * scaled[p * n + j];
                }
                next[i * n + j] = sum * inv_k;
            }
        }
        term.copy_from_slice(&next);
        for (dst, &val) in res.iter_mut().zip(term.iter()) {
            *dst += val;
        }
    }
    for _ in 0..s {
        next.fill(0.0);
        for i in 0..n {
            for p in 0..n {
                let rip = res[i * n + p];
                for j in 0..n {
                    next[i * n + j] += rip * res[p * n + j];
                }
            }
        }
        res.copy_from_slice(&next);
    }
    res
}

#[test]
fn test_bessel_miller_accuracy_to_200() {
    // Verify generating-function identities e^{t(cos(theta) - 1)} = e^{-t}I_0 + 2 sum e^{-t}I_k cos(k theta)
    // across t in [0, 200] including t=0, tiny t, and large t=200.
    for &t in &[0.0, 1e-12, 1e-5, 0.25, 1.0, 8.0, 32.0, 100.0, 200.0] {
        let max_k = adaptive_chebyshev_degree(t);
        let ive = exp_scaled_bessel_i_all(max_k, t).unwrap();
        for &theta in &[0.0, 0.7, PI * 0.5, 2.3, PI] {
            let exact = (t * (theta.cos() - 1.0)).exp();
            let mut approx = ive[0];
            for (k, &bk) in ive.iter().enumerate().skip(1) {
                approx += 2.0 * bk * ((k as f64) * theta).cos();
            }
            let err = (approx - exact).abs();
            assert!(
                err < 1e-14,
                "Bessel generating identity error too large for t={t}, theta={theta}: {err:.3e}"
            );
        }
    }
    // Single-element helper matches full vector
    let single = exp_scaled_bessel_i(5, 12.5).unwrap();
    let all = exp_scaled_bessel_i_all(5, 12.5).unwrap();
    assert!((single - all[5]).abs() < 1e-15);
}

#[test]
fn test_chebyshev_heat_and_bandpass_up_to_t32() {
    let g = CsrGraph::<f64>::from_edges(
        6,
        &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)],
        None,
        false,
        false,
    )
    .unwrap();
    let (l_sym, _) = symmetric_normalized_laplacian(&g);
    let l_dense = l_sym.to_dense().unwrap();
    let l_tilde = crate::csr_matrix::shifted_normalized_laplacian(&l_sym, 2.0).unwrap();
    let mut ws = ChebyshevWorkspace::with_capacity(6 * 6);

    let mut eye = vec![0.0_f64; 36];
    for i in 0..6 {
        eye[i * 6 + i] = 1.0;
    }
    let mut k_cheb = vec![0.0_f64; 36];

    for &t in &[0.0, 0.5, 4.0, 16.0, 32.0] {
        let terms = [(1.0, t)];
        let deg = adaptive_chebyshev_degree(t);
        let coeffs = compute_chebyshev_bandpass_coefficients(&terms, 2.0, deg).unwrap();
        ws.apply_batch(&l_tilde, &coeffs, &eye, &mut k_cheb, 6)
            .unwrap();

        let k_exact = dense_expm_neg(&l_dense, 6, t);
        for idx in 0..36 {
            let err = (k_cheb[idx] - k_exact[idx]).abs();
            assert!(
                err < 1e-10,
                "Chebyshev error at t={t}: got {} vs {} (err={err:.3e})",
                k_cheb[idx],
                k_exact[idx]
            );
        }
    }

    // Verify unshifted direct SpMM matches pre-shifted L_tilde SpMM on a mixed term
    let terms = [(1.2, 0.5), (-0.7, 3.2), (0.15, 0.0)];
    let coeffs = compute_chebyshev_bandpass_coefficients(&terms, 2.0, 60).unwrap();
    let v = vec![1.0, -0.5, 2.0, 0.3, -1.2, 0.8];
    let mut out_shifted = vec![0.0; 6];
    let mut out_direct = vec![0.0; 6];
    ws.apply_vec(&l_tilde, &coeffs, &v, &mut out_shifted)
        .unwrap();
    ws.apply_vec_with_lambda_max(&l_sym, 2.0, &coeffs, &v, &mut out_direct)
        .unwrap();
    for i in 0..6 {
        assert!((out_shifted[i] - out_direct[i]).abs() < 1e-12);
    }
}

#[test]
fn test_dct_sqrt_filter_and_psd_projection() {
    // Non-PSD DoG filter becomes strictly PSD after R2 projection
    let raw_dog = build_dog_terms(0.5, 2.0, 1.0, 0.0).unwrap();
    let (terms_psd, delta_psd, h_min) =
        project_psd_spectral_filter(&raw_dog, 2.0, 30, 1e-4).unwrap();
    assert!(h_min < 1e-4);
    assert!(delta_psd > 0.0);

    // Verify g(L)^2 == h_+(L) on a 6-node graph
    let g = CsrGraph::<f64>::from_edges(
        6,
        &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 4)],
        None,
        false,
        false,
    )
    .unwrap();
    let (l_sym, _) = symmetric_normalized_laplacian(&g);
    let mut ws = ChebyshevWorkspace::new();

    let deg = 140;
    let coeffs_h = compute_chebyshev_bandpass_coefficients(&terms_psd, 2.0, deg).unwrap();
    let coeffs_sqrt = compute_chebyshev_sqrt_filter_coefficients(&terms_psd, 2.0, deg).unwrap();

    let v = vec![1.0, -2.0, 0.5, 0.7, -0.3, 1.4];
    let mut hv = vec![0.0; 6];
    let mut sv = vec![0.0; 6];
    let mut ssv = vec![0.0; 6];
    ws.apply_vec_with_lambda_max(&l_sym, 2.0, &coeffs_h, &v, &mut hv)
        .unwrap();
    ws.apply_vec_with_lambda_max(&l_sym, 2.0, &coeffs_sqrt, &v, &mut sv)
        .unwrap();
    ws.apply_vec_with_lambda_max(&l_sym, 2.0, &coeffs_sqrt, &sv, &mut ssv)
        .unwrap();

    for i in 0..6 {
        assert!(
            (ssv[i] - hv[i]).abs() < 2e-5,
            "sqrt(h)^2 mismatch at {i}: {} vs {}",
            ssv[i],
            hv[i]
        );
    }

    // Smooth strictly positive filter achieves machine-level accuracy even at degree 45
    let smooth_terms = [(0.6, 0.5), (0.4, 2.0)];
    let coeffs_hs = compute_chebyshev_bandpass_coefficients(&smooth_terms, 2.0, 45).unwrap();
    let coeffs_ss = compute_chebyshev_sqrt_filter_coefficients(&smooth_terms, 2.0, 45).unwrap();
    ws.apply_vec_with_lambda_max(&l_sym, 2.0, &coeffs_hs, &v, &mut hv)
        .unwrap();
    ws.apply_vec_with_lambda_max(&l_sym, 2.0, &coeffs_ss, &v, &mut sv)
        .unwrap();
    ws.apply_vec_with_lambda_max(&l_sym, 2.0, &coeffs_ss, &sv, &mut ssv)
        .unwrap();
    for i in 0..6 {
        assert!(
            (ssv[i] - hv[i]).abs() < 1e-11,
            "smooth sqrt(h)^2 mismatch at {i}: {} vs {}",
            ssv[i],
            hv[i]
        );
    }

    // Strongly negative filter must be rejected by compute_chebyshev_sqrt_filter_coefficients
    let non_psd = [(-1.0, 0.5), (0.2, 2.0)];
    assert!(compute_chebyshev_sqrt_filter_coefficients(&non_psd, 2.0, 25).is_err());
}

#[test]
fn test_chebyshev_dog_kernel_regularized_and_edge_cases() {
    let edges: Vec<(u32, u32)> = (0..11)
        .map(|i| (i, i + 1))
        .chain([(0, 11), (0, 6), (3, 9)])
        .collect();
    let g = CsrGraph::<f64>::from_edges(12, &edges, None, false, false).unwrap();

    let cfg = ChebyshevDogConfig::default();
    let k1 = ChebyshevDogKernel::build(&g, &cfg).unwrap();
    let k2 = ChebyshevDogKernel::build(&g, &cfg).unwrap();
    assert_eq!(k1, k2);
    assert!(k1.beta_eff() >= 1e-4 && k1.beta_eff() <= 0.5);
    assert!(k1.diag_estimate().iter().all(|&d| d.is_finite() && d > 0.0));
    assert!(
        k1.landmark_kernel_matrix()
            .iter()
            .all(|&v| v.is_finite() && v > 0.0)
    );
    let varadhan = k1.varadhan_landmark_table(0.5, true, 1e-12).unwrap();
    assert_eq!(varadhan.len(), 12 * k1.num_landmarks());
    assert!(varadhan.iter().all(|&v| v.is_finite() && v >= 0.0));

    // Verify kernel apply_batch matches landmark_kernel columns on unit basis vectors
    let (l_sym, inv_sqrt_deg) = symmetric_normalized_laplacian(&g);
    let k_lm = k1.num_landmarks();
    let mut e_lm = vec![0.0_f64; 12 * k_lm];
    for (col, &lm) in k1.landmarks().iter().enumerate() {
        e_lm[(lm as usize) * k_lm + col] = 1.0;
    }
    let mut applied = vec![0.0_f64; 12 * k_lm];
    let mut ws = ChebyshevWorkspace::new();
    k1.apply_batch(&l_sym, &e_lm, &mut applied, k_lm, &mut ws)
        .unwrap();
    for (&a, &b) in applied.iter().zip(k1.landmark_kernel_matrix().iter()) {
        assert!((a - b).abs() < 1e-12);
    }

    // Verify JL embedding with inv_sqrt_deg scaling
    let jl_plain = ws
        .compute_kernel_jl_embedding(&l_sym, k1.terms_psd(), 2.0, k1.degree(), 16, 77, None)
        .unwrap();
    let jl_scaled = ws
        .compute_kernel_jl_embedding(
            &l_sym,
            k1.terms_psd(),
            2.0,
            k1.degree(),
            16,
            77,
            Some(&inv_sqrt_deg),
        )
        .unwrap();
    for i in 0..12 {
        for c in 0..16 {
            assert!((jl_scaled[i * 16 + c] - inv_sqrt_deg[i] * jl_plain[i * 16 + c]).abs() < 1e-12);
        }
    }

    // Empty and singleton graphs
    let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
    let ke = ChebyshevDogKernel::build(&empty, &cfg).unwrap();
    assert_eq!(ke.num_nodes(), 0);
    assert!(ChebyshevDogKernel::build_with_landmarks(&empty, &cfg, &[0]).is_err());

    let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
    let ks = ChebyshevDogKernel::build(&single, &cfg).unwrap();
    assert_eq!(ks.num_nodes(), 1);
    assert_eq!(ks.diag_estimate(), &[1.0]);
    assert!(ChebyshevDogKernel::build_with_landmarks(&single, &cfg, &[]).is_err());
    assert!(ChebyshevDogKernel::build_with_landmarks(&single, &cfg, &[1]).is_err());
    assert_eq!(
        ChebyshevDogKernel::build_with_landmarks(&single, &cfg, &[0])
            .unwrap()
            .landmarks(),
        &[0]
    );
}

#[test]
fn test_chebyshev_operator_paths_and_config_builders() {
    use crate::operator::FnOperator;

    let cfg = ChebyshevDogConfig::new()
        .with_t1(Some(0.5))
        .with_t2(Some(2.0))
        .with_alpha(Some(0.75))
        .with_beta_floor(Some(0.05))
        .with_degree(Some(24))
        .with_n_landmarks(8)
        .with_min_degree(1)
        .with_jl_dim(8)
        .with_eps_psd(1e-5)
        .with_lambda_max(2.0)
        .with_seed(7);
    assert_eq!(cfg.t1, Some(0.5));
    assert_eq!(cfg.t2, Some(2.0));
    assert_eq!(cfg.alpha, Some(0.75));
    assert_eq!(cfg.beta_floor, Some(0.05));
    assert_eq!(cfg.degree, Some(24));
    assert_eq!(cfg.n_landmarks, 8);
    assert_eq!(cfg.min_degree, 1);
    assert_eq!(cfg.jl_dim, 8);
    assert_eq!(cfg.eps_psd, 1e-5);
    assert_eq!(cfg.lambda_max, 2.0);
    assert_eq!(cfg.seed, 7);

    let edges: Vec<(u32, u32)> = (0..11)
        .map(|i| (i, i + 1))
        .chain([(0, 11), (0, 6), (3, 9)])
        .collect();
    let g = CsrGraph::<f64>::from_edges(12, &edges, None, false, false).unwrap();
    let (l_sym, _) = symmetric_normalized_laplacian(&g);
    let l_tilde = crate::csr_matrix::shifted_normalized_laplacian(&l_sym, 2.0).unwrap();
    let mut ws = ChebyshevWorkspace::new();

    let coeffs = compute_chebyshev_bandpass_coefficients(&[(1.0, 1.5)], 2.0, 20).unwrap();
    let mut v_batch = vec![0.0_f64; 12 * 3];
    for r in 0..12 {
        for c in 0..3 {
            v_batch[r * 3 + c] = ((r * 3 + c + 1) as f64 * 0.2).sin();
        }
    }
    let mut out_csr = vec![0.0_f64; 12 * 3];
    let mut out_op = vec![0.0_f64; 12 * 3];
    ws.apply_batch_with_lambda_max(&l_sym, 2.0, &coeffs, &v_batch, &mut out_csr, 3)
        .unwrap();
    let mut op_fn = FnOperator::new(12, |x: &[f64], y: &mut [f64]| l_sym.spmv(x, y));
    ws.apply_batch_operator_with_lambda_max(&mut op_fn, 2.0, &coeffs, &v_batch, &mut out_op, 3)
        .unwrap();
    for (&a, &b) in out_csr.iter().zip(out_op.iter()) {
        assert!((a - b).abs() < 1e-12);
    }

    let mut out_tilde_csr = vec![0.0_f64; 12];
    let mut out_tilde_op = vec![0.0_f64; 12];
    let v_single = &v_batch[..12];
    ws.apply_vec(&l_tilde, &coeffs, v_single, &mut out_tilde_csr)
        .unwrap();
    let mut op_tilde = &l_tilde;
    ws.apply_vec_operator(&mut op_tilde, &coeffs, v_single, &mut out_tilde_op)
        .unwrap();
    for (&a, &b) in out_tilde_csr.iter().zip(out_tilde_op.iter()) {
        assert!((a - b).abs() < 1e-12);
    }

    // Kernel apply_batch_operator matches apply_batch bit-for-bit up to 1e-12
    let kernel = ChebyshevDogKernel::build(&g, &cfg).unwrap();
    let mut k_csr = vec![0.0_f64; 12 * 3];
    let mut k_op = vec![0.0_f64; 12 * 3];
    kernel
        .apply_batch(&l_sym, &v_batch, &mut k_csr, 3, &mut ws)
        .unwrap();
    kernel
        .apply_batch_operator(&mut op_fn, &v_batch, &mut k_op, 3, &mut ws)
        .unwrap();
    for (&a, &b) in k_csr.iter().zip(k_op.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
}
