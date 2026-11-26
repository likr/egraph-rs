use crate::error::{CoreError, CoreResult};

/// Computes all eigenpairs `(eigenvalues_asc, row_major_eigenvectors)` of an $n \times n$
/// symmetric matrix using cyclic Jacobi rotations in `f64`.
pub(super) fn symmetric_eigen_jacobi(
    a_row_major: &[f64],
    n: usize,
    max_sweeps: usize,
    tol: f64,
) -> CoreResult<(Vec<f64>, Vec<f64>)> {
    let expected = n
        .checked_mul(n)
        .ok_or_else(|| CoreError::invalid_input("symmetric_eigen_jacobi size overflow"))?;
    if a_row_major.len() != expected {
        return Err(CoreError::dimension_mismatch(
            expected,
            a_row_major.len(),
            "symmetric_eigen_jacobi",
        ));
    }
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    if n == 1 {
        return Ok((vec![a_row_major[0]], vec![1.0]));
    }

    let mut a = vec![0.0_f64; expected];
    let mut frob_sq = 0.0_f64;
    for i in 0..n {
        let diag = a_row_major[i * n + i];
        a[i * n + i] = diag;
        frob_sq += diag * diag;
        for j in (i + 1)..n {
            let sym = 0.5 * (a_row_major[i * n + j] + a_row_major[j * n + i]);
            a[i * n + j] = sym;
            a[j * n + i] = sym;
            frob_sq += 2.0 * sym * sym;
        }
    }

    let mut v_t = vec![0.0_f64; expected];
    for i in 0..n {
        v_t[i * n + i] = 1.0;
    }

    let frob_norm = frob_sq.sqrt();
    let stop_thresh = tol * frob_norm.max(1e-300);

    if frob_norm > 0.0 {
        for _sweep in 0..max_sweeps {
            let mut max_off = 0.0_f64;
            for p in 0..n {
                for q in (p + 1)..n {
                    let abs_pq = a[p * n + q].abs();
                    if abs_pq > max_off {
                        max_off = abs_pq;
                    }
                }
            }
            if max_off <= stop_thresh {
                break;
            }

            let sweep_skip = 0.1 * stop_thresh;
            for p in 0..n {
                for q in (p + 1)..n {
                    let g = a[p * n + q];
                    if g.abs() <= sweep_skip {
                        continue;
                    }
                    let app = a[p * n + p];
                    let aqq = a[q * n + q];
                    let h = aqq - app;
                    let t = if g.abs() <= 1e-18 * h.abs() {
                        g / h
                    } else {
                        let theta = 0.5 * h / g;
                        let denom = theta.abs() + (1.0 + theta * theta).sqrt();
                        if theta < 0.0 {
                            -1.0 / denom
                        } else {
                            1.0 / denom
                        }
                    };
                    let c = 1.0 / (1.0 + t * t).sqrt();
                    let s = t * c;
                    let tau = s / (1.0 + c);

                    a[p * n + p] = app - t * g;
                    a[q * n + q] = aqq + t * g;
                    a[p * n + q] = 0.0;
                    a[q * n + p] = 0.0;

                    for r in 0..n {
                        if r == p || r == q {
                            continue;
                        }
                        let arp = a[r * n + p];
                        let arq = a[r * n + q];
                        let new_rp = arp - s * (arq + tau * arp);
                        let new_rq = arq + s * (arp - tau * arq);
                        a[r * n + p] = new_rp;
                        a[p * n + r] = new_rp;
                        a[r * n + q] = new_rq;
                        a[q * n + r] = new_rq;
                    }

                    let p_off = p * n;
                    let q_off = q * n;
                    for r in 0..n {
                        let vp = v_t[p_off + r];
                        let vq = v_t[q_off + r];
                        v_t[p_off + r] = vp - s * (vq + tau * vp);
                        v_t[q_off + r] = vq + s * (vp - tau * vq);
                    }
                }
            }
        }
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        a[i * n + i]
            .total_cmp(&a[j * n + j])
            .then_with(|| i.cmp(&j))
    });

    let mut lambdas = Vec::with_capacity(n);
    let mut vectors = vec![0.0_f64; expected];
    for (mode, &orig) in order.iter().enumerate() {
        lambdas.push(a[orig * n + orig]);
        let src = &v_t[orig * n..(orig + 1) * n];
        for row in 0..n {
            vectors[row * n + mode] = src[row];
        }
    }
    Ok((lambdas, vectors))
}
