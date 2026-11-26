use egraph_core::{CoreError, CoreResult};

/// Reusable, zero-per-solve-allocation Jacobi Preconditioned Conjugate Gradient (PCG) workspace.
#[derive(Debug, Clone, Default)]
pub struct PcgWorkspace {
    diag_inv: Vec<f64>,
    r: Vec<f64>,
    z: Vec<f64>,
    p: Vec<f64>,
    ap: Vec<f64>,
}

impl PcgWorkspace {
    /// Creates a [`PcgWorkspace`] pre-sized for `n` unknowns.
    pub fn new(n: usize) -> Self {
        Self {
            diag_inv: vec![0.0; n],
            r: vec![0.0; n],
            z: vec![0.0; n],
            p: vec![0.0; n],
            ap: vec![0.0; n],
        }
    }

    /// Ensures internal scratch buffers hold at least `n` elements.
    pub fn ensure_capacity(&mut self, n: usize) {
        if self.diag_inv.len() < n {
            self.diag_inv.resize(n, 0.0);
            self.r.resize(n, 0.0);
            self.z.resize(n, 0.0);
            self.p.resize(n, 0.0);
            self.ap.resize(n, 0.0);
        }
    }

    /// Solves the dense symmetric positive semi-definite system `A x = b` via Jacobi PCG.
    ///
    /// Safely exits early when the initial/updated residual is zero, when `denom = p^T A p <= 1e-30`,
    /// or when `r^T z <= 1e-30` (such as on singular or disconnected Laplacian components).
    /// Returns the number of CG iterations executed.
    pub fn solve_dense_symmetric(
        &mut self,
        a: &[f64],
        b: &[f64],
        x: &mut [f64],
        tol: f64,
        max_iters: usize,
    ) -> CoreResult<usize> {
        let n = b.len();
        if x.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                x.len(),
                "PcgWorkspace::solve_dense_symmetric::x",
            ));
        }
        let expected_a = n
            .checked_mul(n)
            .ok_or_else(|| CoreError::invalid_input("PCG matrix dimension overflow"))?;
        if a.len() != expected_a {
            return Err(CoreError::dimension_mismatch(
                expected_a,
                a.len(),
                "PcgWorkspace::solve_dense_symmetric::a",
            ));
        }
        if !tol.is_finite() || tol < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "PCG tolerance must be finite and >= 0.0, got {tol}"
            )));
        }
        if n == 0 || max_iters == 0 {
            return Ok(0);
        }

        self.ensure_capacity(n);

        for i in 0..n {
            let diag = a[i * n + i];
            self.diag_inv[i] = if diag.is_finite() && diag > 1e-30 {
                1.0 / diag
            } else {
                0.0
            };
        }

        let mut rr0 = 0.0_f64;
        for i in 0..n {
            let row = &a[i * n..(i + 1) * n];
            let mut ax = 0.0_f64;
            for j in 0..n {
                ax += row[j] * x[j];
            }
            let ri = b[i] - ax;
            self.r[i] = ri;
            rr0 += ri * ri;
        }

        if !rr0.is_finite() || rr0 == 0.0 {
            return Ok(0);
        }
        let stop_rr = (tol * tol) * rr0.max(1.0);
        if rr0 <= stop_rr {
            return Ok(0);
        }

        let mut rz_old = 0.0_f64;
        for i in 0..n {
            let zi = self.diag_inv[i] * self.r[i];
            self.z[i] = zi;
            self.p[i] = zi;
            rz_old += self.r[i] * zi;
        }
        if !rz_old.is_finite() || rz_old <= 1e-30 {
            return Ok(0);
        }

        for iter in 0..max_iters {
            let mut denom = 0.0_f64;
            for i in 0..n {
                let row = &a[i * n..(i + 1) * n];
                let mut sum = 0.0_f64;
                for j in 0..n {
                    sum += row[j] * self.p[j];
                }
                self.ap[i] = sum;
                denom += self.p[i] * sum;
            }
            if !denom.is_finite() || denom <= 1e-30 {
                return Ok(iter);
            }

            let alpha = rz_old / denom;
            if !alpha.is_finite() {
                return Ok(iter);
            }

            let mut rr = 0.0_f64;
            for i in 0..n {
                let cand_x = x[i] + alpha * self.p[i];
                if !cand_x.is_finite() {
                    return Ok(iter);
                }
                x[i] = cand_x;
                let ri = self.r[i] - alpha * self.ap[i];
                self.r[i] = ri;
                rr += ri * ri;
            }

            if !rr.is_finite() || rr == 0.0 || rr <= stop_rr {
                return Ok(iter + 1);
            }

            let mut rz_new = 0.0_f64;
            for i in 0..n {
                let zi = self.diag_inv[i] * self.r[i];
                self.z[i] = zi;
                rz_new += self.r[i] * zi;
            }
            if !rz_new.is_finite() || rz_new <= 1e-30 {
                return Ok(iter + 1);
            }

            let beta = rz_new / rz_old;
            if !beta.is_finite() {
                return Ok(iter + 1);
            }
            rz_old = rz_new;
            for i in 0..n {
                self.p[i] = self.z[i] + beta * self.p[i];
            }
        }

        Ok(max_iters)
    }
}
