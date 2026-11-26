use egraph_core::CsrGraph;

use crate::csr_matrix::symmetric_normalized_laplacian;

use super::coefficients::CHEBYSHEV_TAIL_L1_TOL;
use super::*;

#[test]
fn test_tiled_apply_batch_dual_and_shifted_jl_norms() {
    let edges: Vec<(u32, u32)> = (0..15)
        .map(|i| (i, i + 1))
        .chain([(0, 15), (0, 8), (4, 12), (2, 10)])
        .collect();
    let g = CsrGraph::<f64>::from_edges(16, &edges, None, false, false).unwrap();
    let (l_sym, inv_sqrt_deg) = symmetric_normalized_laplacian(&g);
    let l_tilde = crate::csr_matrix::shifted_normalized_laplacian(&l_sym, 2.0).unwrap();

    let coeffs_a = compute_chebyshev_bandpass_coefficients(&[(1.0, 0.8)], 2.0, 25).unwrap();
    let coeffs_b = compute_chebyshev_bandpass_coefficients(&[(1.0, 12.0)], 2.0, 75).unwrap();
    let mut ws = ChebyshevWorkspace::new();

    // Check ncols in {1, 8, 13} covering scalar, full tile, and partial trailing tile.
    for ncols in [1usize, 8, 13] {
        let mut v = vec![0.0_f64; 16 * ncols];
        for r in 0..16 {
            for c in 0..ncols {
                v[r * ncols + c] = ((r * 7 + c * 3) as f64 * 0.37).sin();
            }
        }
        let mut out_a_dual = vec![0.0_f64; 16 * ncols];
        let mut out_b_dual = vec![0.0_f64; 16 * ncols];
        ws.apply_batch_dual(
            &l_tilde,
            &coeffs_a,
            Some(&coeffs_b),
            &v,
            &mut out_a_dual,
            Some(&mut out_b_dual),
            ncols,
        )
        .unwrap();

        let mut out_a_ref = vec![0.0_f64; 16 * ncols];
        let mut out_b_ref = vec![0.0_f64; 16 * ncols];
        ws.apply_batch_with_lambda_max(&l_sym, 2.0, &coeffs_a, &v, &mut out_a_ref, ncols)
            .unwrap();
        ws.apply_batch_with_lambda_max(&l_sym, 2.0, &coeffs_b, &v, &mut out_b_ref, ncols)
            .unwrap();

        for i in 0..(16 * ncols) {
            assert!((out_a_dual[i] - out_a_ref[i]).abs() < 1e-12);
            assert!((out_b_dual[i] - out_b_ref[i]).abs() < 1e-12);
        }
    }

    // Verify swapped polynomial lengths (coeffs_a longer than coeffs_b).
    let mut v13 = vec![0.5_f64; 16 * 13];
    v13[0] = 1.75;
    let mut out_long = vec![0.0_f64; 16 * 13];
    let mut out_short = vec![0.0_f64; 16 * 13];
    ws.apply_batch_dual(
        &l_tilde,
        &coeffs_b,
        Some(&coeffs_a),
        &v13,
        &mut out_long,
        Some(&mut out_short),
        13,
    )
    .unwrap();
    let mut ref_short = vec![0.0_f64; 16 * 13];
    ws.apply_batch(&l_tilde, &coeffs_a, &v13, &mut ref_short, 13)
        .unwrap();
    for (&got, &exp) in out_short.iter().zip(ref_short.iter()) {
        assert!((got - exp).abs() < 1e-12);
    }

    // Verify shifted JL embedding matches unshifted JL embedding across tile widths.
    let terms_a = [(1.0, 0.8)];
    let terms_b = [(1.0, 12.0)];
    for jl_dim in [1usize, 5, 8, 13, 32] {
        let emb_unshifted = ws
            .compute_kernel_jl_embedding(&l_sym, &terms_a, 2.0, 25, jl_dim, 99, Some(&inv_sqrt_deg))
            .unwrap();
        let emb_shifted = ws
            .compute_kernel_jl_embedding_shifted(
                &l_tilde,
                &terms_a,
                2.0,
                25,
                jl_dim,
                99,
                Some(&inv_sqrt_deg),
            )
            .unwrap();
        for (&a, &b) in emb_unshifted.iter().zip(emb_shifted.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    // Verify compute_dual_jl_squared_norms_shifted accuracy and single-vs-dual consistency.
    let (norms_a_only, none_b) = ws
        .compute_dual_jl_squared_norms_shifted(&l_tilde, &terms_a, 25, None, 2.0, 13, 42)
        .unwrap();
    assert!(none_b.is_none());
    let (norms_a_dual, norms_b_dual) = ws
        .compute_dual_jl_squared_norms_shifted(
            &l_tilde,
            &terms_a,
            25,
            Some((&terms_b, 75)),
            2.0,
            13,
            42,
        )
        .unwrap();
    let norms_b_dual = norms_b_dual.unwrap();
    let (norms_b_only, _) = ws
        .compute_dual_jl_squared_norms_shifted(&l_tilde, &terms_b, 75, None, 2.0, 13, 42)
        .unwrap();
    for i in 0..16 {
        assert!((norms_a_only[i] - norms_a_dual[i]).abs() < 1e-14);
        assert!((norms_b_only[i] - norms_b_dual[i]).abs() < 1e-14);
        assert!(norms_a_dual[i] > 0.0 && norms_b_dual[i] > 0.0);
    }
    // For jl_dim <= 8, tile order equals row-major order so norms match exact JL row norms.
    let emb_7 = ws
        .compute_kernel_jl_embedding_shifted(&l_tilde, &terms_a, 2.0, 25, 7, 42, None)
        .unwrap();
    let (norms_7, _) = ws
        .compute_dual_jl_squared_norms_shifted(&l_tilde, &terms_a, 25, None, 2.0, 7, 42)
        .unwrap();
    for i in 0..16 {
        let row_sq: f64 = emb_7[i * 7..(i + 1) * 7].iter().map(|&x| x * x).sum();
        assert!((norms_7[i] - row_sq).abs() < 1e-12);
    }

    // Mismatched coeffs_b / out_b must return an error cleanly.
    assert!(
        ws.apply_batch_dual(
            &l_tilde,
            &coeffs_a,
            Some(&coeffs_b),
            &v13,
            &mut out_long,
            None,
            13
        )
        .is_err()
    );
}

#[test]
fn test_effective_chebyshev_len_tail_truncation_and_16_col_tiling() {
    let coeffs_48 = compute_chebyshev_bandpass_coefficients(&[(1.0, 48.0)], 2.0, 75).unwrap();
    let eff_48 = effective_chebyshev_len(&coeffs_48, CHEBYSHEV_TAIL_L1_TOL);
    assert!(
        eff_48 <= 55,
        "expected effective_chebyshev_len(exp(-48 L)) <= 55, got {eff_48}"
    );

    let coeffs_sqrt_48 =
        compute_chebyshev_sqrt_filter_coefficients(&[(1.0, 48.0)], 2.0, 75).unwrap();
    let eff_sqrt_48 = effective_chebyshev_len(&coeffs_sqrt_48, CHEBYSHEV_TAIL_L1_TOL);
    assert!(
        eff_sqrt_48 <= 40,
        "expected effective_chebyshev_len(sqrt(exp(-48 L))) <= 40, got {eff_sqrt_48}"
    );

    // Verify truncated coefficients match untrimmed degree-75 Chebyshev evaluation < 1e-11
    let edges: Vec<(u32, u32)> = (0..19)
        .map(|i| (i, i + 1))
        .chain([(0, 19), (0, 10), (5, 15)])
        .collect();
    let g = CsrGraph::<f64>::from_edges(20, &edges, None, false, false).unwrap();
    let (l_sym, _) = symmetric_normalized_laplacian(&g);
    let l_tilde = crate::csr_matrix::shifted_normalized_laplacian(&l_sym, 2.0).unwrap();

    let ncols = 16;
    let mut v = vec![0.0_f64; 20 * ncols];
    for r in 0..20 {
        for c in 0..ncols {
            v[r * ncols + c] = ((r * 5 + c * 11) as f64 * 0.23).cos();
        }
    }
    let mut ws = ChebyshevWorkspace::new();
    let mut out_band_trunc = vec![0.0_f64; 20 * ncols];
    let mut out_sqrt_trunc = vec![0.0_f64; 20 * ncols];
    ws.apply_batch_dual(
        &l_tilde,
        &coeffs_48,
        Some(&coeffs_sqrt_48),
        &v,
        &mut out_band_trunc,
        Some(&mut out_sqrt_trunc),
        ncols,
    )
    .unwrap();

    let run_untrimmed = |coeffs: &[f64]| -> Vec<f64> {
        let mut t_prev = v.clone();
        let mut out = vec![0.0_f64; 20 * ncols];
        for (dst, &src) in out.iter_mut().zip(t_prev.iter()) {
            *dst = coeffs[0] * src;
        }
        let mut t_curr = vec![0.0_f64; 20 * ncols];
        l_tilde.spmm(&t_prev, &mut t_curr, ncols).unwrap();
        for (dst, &src) in out.iter_mut().zip(t_curr.iter()) {
            *dst += coeffs[1] * src;
        }
        let mut t_next = vec![0.0_f64; 20 * ncols];
        for &ck in &coeffs[2..] {
            l_tilde.spmm(&t_curr, &mut t_next, ncols).unwrap();
            for i in 0..(20 * ncols) {
                let nxt = 2.0 * t_next[i] - t_prev[i];
                t_next[i] = nxt;
                out[i] += ck * nxt;
            }
            std::mem::swap(&mut t_prev, &mut t_curr);
            std::mem::swap(&mut t_curr, &mut t_next);
        }
        out
    };

    let ref_band = run_untrimmed(&coeffs_48);
    let ref_sqrt = run_untrimmed(&coeffs_sqrt_48);
    for (&got, &exp) in out_band_trunc.iter().zip(ref_band.iter()) {
        assert!((got - exp).abs() < 1e-11);
    }
    for (&got, &exp) in out_sqrt_trunc.iter().zip(ref_sqrt.iter()) {
        assert!((got - exp).abs() < 1e-11);
    }

    // Check edge cases of effective_chebyshev_len
    assert_eq!(effective_chebyshev_len(&[], 1e-12), 0);
    assert_eq!(effective_chebyshev_len(&[1.0], 1e-12), 1);
    assert_eq!(effective_chebyshev_len(&[1.0, 0.5, 1e-13], 1e-12), 2);
}

#[test]
fn test_adaptive_degree_scaled_and_dynamic_spectrum_star_graph() {
    // Bit-exact match at lambda_max == 2.0
    for &t in &[0.0, 0.5, 1.0, 4.5, 16.0, 48.0] {
        assert_eq!(
            adaptive_chebyshev_degree_scaled(t, 2.0),
            adaptive_chebyshev_degree(t)
        );
    }
    assert!(adaptive_chebyshev_degree_scaled(2.0, 100.0) > adaptive_chebyshev_degree(2.0));

    // Zero-matrix fast path (edgeless graph under combinatorial Laplacian)
    let edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
    let l_zero = crate::csr_matrix::combinatorial_laplacian(&edgeless);
    let mut ws = ChebyshevWorkspace::new();
    let v_zero = vec![1.0, -2.0, 3.0, -4.0];
    let mut out_zero = vec![0.0; 4];
    let lmax_zero = ws
        .apply_heat_dynamic(
            &l_zero,
            &v_zero,
            &mut out_zero,
            1,
            2.5,
            None,
            None,
            16,
            1.05,
            42,
        )
        .unwrap();
    assert_eq!(lmax_zero, 0.0);
    assert_eq!(out_zero, v_zero);

    let terms_zero = build_dog_terms(1.0, 4.0, 0.5, 0.0).unwrap();
    let (emb_zero, lmax_emb_zero) = ws
        .compute_dog_jl_dynamic(
            &l_zero,
            &terms_zero,
            None,
            None,
            8,
            1e-4,
            16,
            1.05,
            42,
            None,
        )
        .unwrap();
    assert_eq!(lmax_emb_zero, 0.0);
    assert_eq!(emb_zero.len(), 32);
    assert!(emb_zero.iter().all(|v| v.is_finite() && v.abs() > 0.0));

    // Star graph K_{1, 50}: L_comb has lambda_max = 51.0 >> 2.0
    let n_star = 51usize;
    let edges: Vec<(u32, u32)> = (1..n_star).map(|i| (0, i as u32)).collect();
    let star = CsrGraph::<f64>::from_edges(n_star, &edges, None, false, false).unwrap();
    let l_star = crate::csr_matrix::combinatorial_laplacian(&star);
    let l_dense = l_star.to_dense().unwrap();
    let (evals, evecs) = crate::eigen::symmetric_eigen_jacobi(&l_dense, n_star, 64, 1e-12).unwrap();

    let t = 0.25;
    let v: Vec<f64> = (0..n_star).map(|i| ((i as f64) * 0.31).sin()).collect();
    let mut out = vec![0.0; n_star];
    let lmax_used = ws
        .apply_heat_dynamic(&l_star, &v, &mut out, 1, t, None, None, 16, 1.05, 42)
        .unwrap();
    assert!(lmax_used >= 51.0);

    // Exact exp(-t * L) v = V diag(exp(-t * lambda)) V^T v
    let mut proj = vec![0.0; n_star];
    for m in 0..n_star {
        let mut s = 0.0;
        for i in 0..n_star {
            s += evecs[i * n_star + m] * v[i];
        }
        proj[m] = s * (-t * evals[m]).exp();
    }
    let mut exact = vec![0.0; n_star];
    for i in 0..n_star {
        let mut s = 0.0;
        for m in 0..n_star {
            s += evecs[i * n_star + m] * proj[m];
        }
        exact[i] = s;
    }
    let max_err = out
        .iter()
        .zip(exact.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_err < 1e-8, "star heat dynamic max_err={max_err}");

    // Dynamic DoG JL embedding on star graph (lambda_max >> 2)
    let (dog_emb, lmax_dog) = ws
        .compute_dog_jl_dynamic(
            &l_star,
            &terms_zero,
            None,
            None,
            8,
            1e-4,
            16,
            1.05,
            42,
            None,
        )
        .unwrap();
    assert!(lmax_dog >= 51.0);
    assert_eq!(dog_emb.len(), n_star * 8);
    assert!(dog_emb.iter().all(|v| v.is_finite()));
}
