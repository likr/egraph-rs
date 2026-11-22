//! Pure-Rust dense symmetric Jacobi eigensolver and SpMV-only block thick-restart Lanczos
//! with two-pass Modified Gram-Schmidt reorthogonalization.

use egraph_core::{CoreError, CoreResult, CsrGraph, DeterministicRng, Scalar};

use crate::csr_matrix::symmetric_normalized_laplacian;

/// Result of computing the smallest nontrivial eigenpairs of $L_{\mathrm{sym}}$.
#[derive(Debug, Clone, PartialEq)]
pub struct EigenResult {
    /// Smallest nontrivial eigenvalues sorted in ascending order.
    pub lambdas: Vec<f64>,
    /// Row-major orthonormal eigenvectors of shape `(num_nodes, num_modes)`.
    pub vectors: Vec<f64>,
    /// Inverse square-root degree vector $D^{-1/2}$ (`0.0` for isolated vertices).
    pub inv_sqrt_deg: Vec<f64>,
    /// Number of vertices in the graph.
    pub num_nodes: usize,
    /// Number of returned nontrivial modes (`min(n_modes, num_nodes - 1)` when `num_nodes >= 2`).
    pub num_modes: usize,
    /// Whether the dense Jacobi fallback path was used instead of iterative Lanczos.
    pub used_dense_fallback: bool,
}

/// Computes all eigenpairs `(eigenvalues, row_major_eigenvectors)` of an $n \times n$ symmetric
/// matrix using cyclic Jacobi rotations in `f64`, sorted in ascending eigenvalue order.
///
/// Safe on zero, diagonal, rank-deficient, and degenerate-eigenvalue matrices.
pub fn symmetric_eigen_jacobi(
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
            "symmetric_eigen_jacobi::a_row_major",
        ));
    }
    if !tol.is_finite() {
        return Err(CoreError::non_finite("symmetric_eigen_jacobi::tol"));
    }
    if tol <= 0.0 {
        return Err(CoreError::invalid_input("tol must be strictly positive"));
    }
    if a_row_major.iter().any(|v| !v.is_finite()) {
        return Err(CoreError::non_finite("symmetric_eigen_jacobi::a_row_major"));
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

    // Store eigenvectors transposed (rows = eigenvectors) so Jacobi vector updates are unit-stride.
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

#[inline]
fn next_gaussian_pair(rng: &mut DeterministicRng) -> (f64, f64) {
    let u1 = rng.next_f64().max(f64::MIN_POSITIVE);
    let u2 = rng.next_f64();
    let r = (-2.0 * u1.ln()).sqrt();
    let (s, c) = (std::f64::consts::TAU * u2).sin_cos();
    (r * c, r * s)
}

#[inline]
fn dot_slice(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

#[inline]
fn norm_slice(a: &[f64]) -> f64 {
    dot_slice(a, a).sqrt()
}

fn fill_random_unit_orthogonal_buf(
    basis: &[f64],
    num_existing: usize,
    extra_basis: &[f64],
    num_extra: usize,
    n: usize,
    dst: &mut [f64],
    rng: &mut DeterministicRng,
) -> bool {
    for _attempt in 0..8 {
        let mut i = 0;
        while i + 1 < n {
            let (g0, g1) = next_gaussian_pair(rng);
            dst[i] = g0;
            dst[i + 1] = g1;
            i += 2;
        }
        if i < n {
            dst[i] = next_gaussian_pair(rng).0;
        }

        for _pass in 0..2 {
            for prev in 0..num_existing {
                let q = &basis[prev * n..(prev + 1) * n];
                let d = dot_slice(q, dst);
                for (w, &qv) in dst.iter_mut().zip(q.iter()) {
                    *w -= d * qv;
                }
            }
            for prev in 0..num_extra {
                let q = &extra_basis[prev * n..(prev + 1) * n];
                let d = dot_slice(q, dst);
                for (w, &qv) in dst.iter_mut().zip(q.iter()) {
                    *w -= d * qv;
                }
            }
        }

        let nrm = norm_slice(dst);
        if nrm > 1e-12 {
            let inv = 1.0 / nrm;
            for w in dst.iter_mut() {
                *w *= inv;
            }
            return true;
        }
    }
    false
}

/// Reusable Krylov/Ritz subspace projection helper supporting block two-pass Modified
/// Gram-Schmidt orthogonalization and block Wu-Simon thick restarts.
#[derive(Debug, Clone)]
pub struct SubspaceEigen {
    dim: usize,
    m_sub: usize,
    block_size: usize,
    q_vecs: Vec<f64>,
    b_mat: Vec<f64>,
    w_block: Vec<f64>,
    r_block: Vec<f64>,
    y_buf: Vec<f64>,
}

impl SubspaceEigen {
    /// Allocates a subspace workspace for `dim`-dimensional vectors, Krylov capacity `m_sub`,
    /// and block width `block_size`.
    pub fn new(dim: usize, m_sub: usize, block_size: usize) -> Self {
        Self {
            dim,
            m_sub,
            block_size,
            q_vecs: vec![0.0; (m_sub + block_size) * dim],
            b_mat: vec![0.0; m_sub * m_sub],
            w_block: vec![0.0; block_size * dim],
            r_block: vec![0.0; block_size * block_size],
            y_buf: vec![0.0; m_sub * dim],
        }
    }

    fn orthogonalize_w_against_q(&mut self, j: usize, start_j: usize) {
        let n = self.dim;
        let b = self.block_size;
        let num_q = j + b;

        let mut diag_block = vec![0.0_f64; b * b];
        let mut prev_coupling = vec![0.0_f64; b * b];

        for pass in 0..2 {
            for i in 0..num_q {
                let q_i = &self.q_vecs[i * n..(i + 1) * n];
                for c in 0..b {
                    let w_c = &mut self.w_block[c * n..(c + 1) * n];
                    let h = dot_slice(q_i, w_c);
                    for (wv, &qv) in w_c.iter_mut().zip(q_i.iter()) {
                        *wv -= h * qv;
                    }
                    if i >= j {
                        diag_block[(i - j) * b + c] += h;
                    } else if pass == 0 && j > start_j && i + b >= j {
                        let prev_r = i - (j - b);
                        prev_coupling[prev_r * b + c] = h;
                    }
                }
            }
        }

        if j > start_j {
            for r in 0..b {
                for c in 0..b {
                    let prev_val = self.b_mat[(j - b + r) * self.m_sub + (j + c)];
                    let avg = 0.5 * (prev_val + prev_coupling[r * b + c]);
                    self.b_mat[(j - b + r) * self.m_sub + (j + c)] = avg;
                    self.b_mat[(j + c) * self.m_sub + (j - b + r)] = avg;
                }
            }
        }

        for r in 0..b {
            for c in r..b {
                let sym = 0.5 * (diag_block[r * b + c] + diag_block[c * b + r]);
                self.b_mat[(j + r) * self.m_sub + (j + c)] = sym;
                self.b_mat[(j + c) * self.m_sub + (j + r)] = sym;
            }
        }
    }

    fn qr_next_block(
        &mut self,
        num_q: usize,
        w_orig_norms: &[f64],
        dst_start: usize,
        rng: &mut DeterministicRng,
    ) -> usize {
        let n = self.dim;
        let b = self.block_size;
        self.r_block.fill(0.0);

        let mut rank_ok = 0usize;
        for c in 0..b {
            let (left_dst, right_dst) = self.w_block.split_at_mut(c * n);
            let w_c = &mut right_dst[..n];

            // Two-pass MGS QR against previously orthonormalized columns in this block.
            for _pass in 0..2 {
                for r in 0..c {
                    let q_r = &left_dst[r * n..(r + 1) * n];
                    let h = dot_slice(q_r, w_c);
                    self.r_block[r * b + c] += h;
                    for (wv, &qv) in w_c.iter_mut().zip(q_r.iter()) {
                        *wv -= h * qv;
                    }
                }
            }
            // Also sweep any residual overlap against Q[0..num_q].
            for i in 0..num_q {
                let q_i = &self.q_vecs[i * n..(i + 1) * n];
                let h = dot_slice(q_i, w_c);
                for (wv, &qv) in w_c.iter_mut().zip(q_i.iter()) {
                    *wv -= h * qv;
                }
            }

            let nrm = norm_slice(w_c);
            if nrm > 1e-13 * w_orig_norms[c].max(1.0) {
                self.r_block[c * b + c] = nrm;
                let inv = 1.0 / nrm;
                for wv in w_c.iter_mut() {
                    *wv *= inv;
                }
                rank_ok += 1;
            } else {
                // Invariant subspace reached on column c: keep R[c, c] = 0 and inject a fresh
                // orthogonal direction so degenerate eigenspaces remain spanned.
                self.r_block[c * b + c] = 0.0;
                if fill_random_unit_orthogonal_buf(&self.q_vecs, num_q, left_dst, c, n, w_c, rng) {
                    rank_ok += 1;
                } else {
                    return rank_ok;
                }
            }
        }

        self.q_vecs[dst_start * n..(dst_start + b) * n].copy_from_slice(&self.w_block);
        rank_ok
    }

    fn restart_with_ritz(
        &mut self,
        effective_m: usize,
        k_keep: usize,
        evals: &[f64],
        evecs: &[f64],
        rng: &mut DeterministicRng,
    ) {
        let n = self.dim;
        let b = self.block_size;

        for r in 0..k_keep {
            let eig_idx = effective_m - 1 - r;
            let y_row = &mut self.y_buf[r * n..(r + 1) * n];
            y_row.fill(0.0);
            for p in 0..effective_m {
                let s_pr = evecs[p * effective_m + eig_idx];
                let q_p = &self.q_vecs[p * n..(p + 1) * n];
                for k in 0..n {
                    y_row[k] += s_pr * q_p[k];
                }
            }
        }
        self.q_vecs[..k_keep * n].copy_from_slice(&self.y_buf[..k_keep * n]);

        for r in 0..k_keep {
            for _pass in 0..2 {
                for prev in 0..r {
                    let mut dot = 0.0_f64;
                    for k in 0..n {
                        dot += self.q_vecs[prev * n + k] * self.q_vecs[r * n + k];
                    }
                    for k in 0..n {
                        self.q_vecs[r * n + k] -= dot * self.q_vecs[prev * n + k];
                    }
                }
            }
            let r_off = r * n;
            let nrm = norm_slice(&self.q_vecs[r_off..r_off + n]);
            if nrm > 1e-14 {
                let inv = 1.0 / nrm;
                for k in 0..n {
                    self.q_vecs[r_off + k] *= inv;
                }
            }
        }

        // Move Q_next (effective_m .. effective_m + b) into slots k_keep .. k_keep + b and
        // preserve strict orthogonality against Q[0..k_keep].
        let (left, right) = self.q_vecs.split_at_mut(effective_m * n);
        left[k_keep * n..(k_keep + b) * n].copy_from_slice(&right[..b * n]);
        let mut col_ok = vec![false; b];
        for (c, ok) in col_ok.iter_mut().enumerate() {
            let col_off = (k_keep + c) * n;
            for _pass in 0..2 {
                for prev in 0..(k_keep + c) {
                    let mut dot = 0.0_f64;
                    for k in 0..n {
                        dot += self.q_vecs[prev * n + k] * self.q_vecs[col_off + k];
                    }
                    for k in 0..n {
                        self.q_vecs[col_off + k] -= dot * self.q_vecs[prev * n + k];
                    }
                }
            }
            let nrm = norm_slice(&self.q_vecs[col_off..col_off + n]);
            if nrm > 1e-12 {
                let inv = 1.0 / nrm;
                for k in 0..n {
                    self.q_vecs[col_off + k] *= inv;
                }
                *ok = true;
            }
        }

        self.b_mat.fill(0.0);
        let bot_start = effective_m - b;
        for p in 0..k_keep {
            let eig_idx = effective_m - 1 - p;
            self.b_mat[p * self.m_sub + p] = evals[eig_idx];
            for r in 0..b {
                if !col_ok[r] {
                    continue;
                }
                let mut rho_r = 0.0_f64;
                for c in r..b {
                    rho_r +=
                        self.r_block[r * b + c] * evecs[(bot_start + c) * effective_m + eig_idx];
                }
                self.b_mat[p * self.m_sub + (k_keep + r)] = rho_r;
                self.b_mat[(k_keep + r) * self.m_sub + p] = rho_r;
            }
        }

        for (c, &ok) in col_ok.iter().enumerate() {
            if !ok {
                let mut tmp = vec![0.0_f64; n];
                if fill_random_unit_orthogonal_buf(
                    &self.q_vecs,
                    k_keep + c,
                    &[],
                    0,
                    n,
                    &mut tmp,
                    rng,
                ) {
                    self.q_vecs[(k_keep + c) * n..(k_keep + c + 1) * n].copy_from_slice(&tmp);
                }
            }
        }
    }
}

/// Pure-Rust SpMV-only block thick-restart Lanczos eigensolver with two-pass MGS
/// reorthogonalization.
///
/// Uses block width $b = 4$ (when `k >= 2` and `n > 64`) so degenerate eigenvalues of
/// multiplicity up to 4 (e.g., 2D grids and periodic tori) are resolved deterministically
/// without relying on floating-point rounding drift.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LanczosEigen {
    /// Relative convergence tolerance (weighted by spectral gap for tight meshes).
    pub tol: f64,
    /// Maximum number of thick-restart cycles.
    pub max_iters: usize,
    /// Optional explicit Krylov subspace size override.
    pub subspace_dim: Option<usize>,
    /// Deterministic PRNG seed for initial block and invariant-subspace recovery.
    pub seed: u64,
}

impl Default for LanczosEigen {
    fn default() -> Self {
        Self {
            tol: 1e-6,
            max_iters: 500,
            subspace_dim: None,
            seed: 42,
        }
    }
}

impl LanczosEigen {
    /// Creates a new [`LanczosEigen`] solver configuration.
    pub fn new(tol: f64, max_iters: usize, seed: u64) -> Self {
        Self {
            tol,
            max_iters,
            subspace_dim: None,
            seed,
        }
    }

    /// Computes the `k` largest eigenpairs of an $n \times n$ symmetric linear operator `matvec`
    /// returning `(eigenvalues_desc, row_major_vectors)` with `vectors` shape `(n, k)`.
    pub fn largest_eigenpairs<F>(
        &self,
        n: usize,
        k: usize,
        mut matvec: F,
    ) -> CoreResult<(Vec<f64>, Vec<f64>)>
    where
        F: FnMut(&[f64], &mut [f64]) -> CoreResult<()>,
    {
        if k == 0 {
            return Err(CoreError::invalid_input("k must be >= 1"));
        }
        if k > n {
            return Err(CoreError::invalid_input(format!(
                "requested k={k} exceeds matrix dimension n={n}"
            )));
        }
        if !self.tol.is_finite() {
            return Err(CoreError::non_finite("LanczosEigen::tol"));
        }
        if self.tol <= 0.0 {
            return Err(CoreError::invalid_input("tol must be strictly positive"));
        }
        if self.max_iters == 0 {
            return Err(CoreError::invalid_input("max_iters must be >= 1"));
        }

        let b = if k == 1 || n <= 64 {
            1
        } else {
            4.min(n / 8).max(1)
        };
        let default_m = if b > 1 {
            (4 * k).max(64)
        } else {
            (3 * k).max(48)
        };
        let mut m_sub = self.subspace_dim.unwrap_or(default_m).max(k + 2 * b).min(n);
        if m_sub < n && b > 1 {
            m_sub = (m_sub / b).max(4) * b;
            m_sub = m_sub.min(n);
        }
        let eff_b = if m_sub == n { 1 } else { b };

        let mut ws = SubspaceEigen::new(n, m_sub, eff_b);
        let mut rng = DeterministicRng::from_seed_u64(self.seed);

        let mut tmp_vec = vec![0.0_f64; n];
        for c in 0..eff_b {
            if !fill_random_unit_orthogonal_buf(&ws.q_vecs, c, &[], 0, n, &mut tmp_vec, &mut rng) {
                return Err(CoreError::numerical_breakdown(
                    "failed to initialize orthogonal Lanczos starting block",
                ));
            }
            ws.q_vecs[c * n..(c + 1) * n].copy_from_slice(&tmp_vec);
        }

        let mut start_j = 0usize;
        let k_keep = if m_sub < n {
            (k + ((m_sub - k) / 3).max(2 * eff_b)).min(m_sub.saturating_sub(2 * eff_b))
        } else {
            k
        };

        let mut final_evals = Vec::new();
        let mut final_evecs = Vec::new();
        let mut final_m = m_sub;
        let mut w_orig_norms = vec![0.0_f64; eff_b];

        for iter in 0..self.max_iters {
            let mut j = start_j;
            let mut effective_m = m_sub;

            while j + eff_b <= m_sub {
                for c in 0..eff_b {
                    let q_jc = &ws.q_vecs[(j + c) * n..(j + c + 1) * n];
                    let w_dst = &mut ws.w_block[c * n..(c + 1) * n];
                    matvec(q_jc, w_dst)?;
                    w_orig_norms[c] = norm_slice(w_dst);
                }

                ws.orthogonalize_w_against_q(j, start_j);
                let dst_slot = j + eff_b;
                let rank_ok = ws.qr_next_block(j + eff_b, &w_orig_norms, dst_slot, &mut rng);

                if j + 2 * eff_b <= m_sub {
                    for r in 0..eff_b {
                        for c in 0..eff_b {
                            let val = ws.r_block[r * eff_b + c];
                            ws.b_mat[(j + eff_b + r) * m_sub + (j + c)] = val;
                            ws.b_mat[(j + c) * m_sub + (j + eff_b + r)] = val;
                        }
                    }
                    if rank_ok < eff_b {
                        effective_m = j + eff_b + rank_ok;
                        ws.r_block.fill(0.0);
                        break;
                    }
                } else {
                    effective_m = dst_slot;
                    break;
                }
                j += eff_b;
            }

            let mut b_sub = Vec::with_capacity(effective_m * effective_m);
            for r in 0..effective_m {
                b_sub.extend_from_slice(&ws.b_mat[r * m_sub..r * m_sub + effective_m]);
            }
            let (evals, evecs) = symmetric_eigen_jacobi(&b_sub, effective_m, 120, 1e-14)?;

            let gap_factor = if effective_m > k {
                (evals[effective_m - k] - evals[effective_m - k - 1])
                    .abs()
                    .clamp(1e-3, 1.0)
            } else {
                1.0
            };
            let eff_tol = (self.tol * gap_factor).max(1e-13);

            let bot_start = effective_m.saturating_sub(eff_b);
            let mut all_converged = true;
            for p in 0..k {
                let eig_idx = effective_m - 1 - p;
                let mut res_sq = 0.0_f64;
                for r in 0..(effective_m - bot_start) {
                    let mut rho_r = 0.0_f64;
                    for c in r..(effective_m - bot_start) {
                        rho_r += ws.r_block[r * eff_b + c]
                            * evecs[(bot_start + c) * effective_m + eig_idx];
                    }
                    res_sq += rho_r * rho_r;
                }
                let res_norm = res_sq.sqrt();
                let scale = evals[eig_idx].abs().max(1.0);
                if res_norm > eff_tol * scale {
                    all_converged = false;
                    break;
                }
            }

            if all_converged
                || effective_m == n
                || effective_m < k_keep + eff_b
                || iter + 1 == self.max_iters
            {
                final_evals = evals;
                final_evecs = evecs;
                final_m = effective_m;
                break;
            }

            ws.restart_with_ritz(effective_m, k_keep, &evals, &evecs, &mut rng);
            start_j = k_keep;
        }

        let mut lambdas_desc = Vec::with_capacity(k);
        let mut vectors = vec![0.0_f64; n * k];
        for r in 0..k {
            let eig_idx = final_m - 1 - r;
            lambdas_desc.push(final_evals[eig_idx]);
            for p in 0..final_m {
                let coeff = final_evecs[p * final_m + eig_idx];
                let q_p = &ws.q_vecs[p * n..(p + 1) * n];
                for row in 0..n {
                    vectors[row * k + r] += coeff * q_p[row];
                }
            }
            let mut nrm_sq = 0.0_f64;
            for row in 0..n {
                let val = vectors[row * k + r];
                nrm_sq += val * val;
            }
            let nrm = nrm_sq.sqrt();
            if nrm > 1e-14 {
                let inv = 1.0 / nrm;
                for row in 0..n {
                    vectors[row * k + r] *= inv;
                }
            }
        }

        Ok((lambdas_desc, vectors))
    }
}

/// Computes the smallest `n_modes` nontrivial eigenpairs of $L_{\mathrm{sym}} = I - D^{-1/2} A D^{-1/2}$,
/// skipping the trivial $\lambda_0 \approx 0$ mode cleanly.
///
/// Uses dense [`symmetric_eigen_jacobi`] fallback when `n <= max(48, 2 * actual_modes + 4)`,
/// and SpMV-only block thick-restart [`LanczosEigen`] on the shifted operator
/// $M v = 2.5 v - L_{\mathrm{sym}} v$ for larger graphs.
pub fn smallest_sym_laplacian_eigenpairs<W: Scalar>(
    graph: &CsrGraph<W>,
    n_modes: usize,
    tol: f64,
    max_iters: usize,
    seed: u64,
) -> CoreResult<EigenResult> {
    if n_modes == 0 {
        return Err(CoreError::invalid_input("n_modes must be >= 1"));
    }
    if !tol.is_finite() {
        return Err(CoreError::non_finite(
            "smallest_sym_laplacian_eigenpairs::tol",
        ));
    }
    if tol <= 0.0 {
        return Err(CoreError::invalid_input("tol must be strictly positive"));
    }
    if max_iters == 0 {
        return Err(CoreError::invalid_input("max_iters must be >= 1"));
    }

    let (l_sym, inv_sqrt_deg) = symmetric_normalized_laplacian(graph);
    let n = graph.num_nodes();
    if n <= 1 {
        return Ok(EigenResult {
            lambdas: Vec::new(),
            vectors: Vec::new(),
            inv_sqrt_deg,
            num_nodes: n,
            num_modes: 0,
            used_dense_fallback: true,
        });
    }

    let actual_modes = n_modes.min(n - 1);
    let use_dense = n <= 48usize.max(2 * actual_modes + 4);

    if use_dense {
        let dense = l_sym.to_dense()?;
        let (evals, evecs) = symmetric_eigen_jacobi(&dense, n, 120, tol.min(1e-12))?;
        let lambdas: Vec<f64> = evals[1..=actual_modes]
            .iter()
            .map(|&v| v.max(0.0))
            .collect();
        let mut vectors = vec![0.0_f64; n * actual_modes];
        for row in 0..n {
            for m in 0..actual_modes {
                vectors[row * actual_modes + m] = evecs[row * n + (1 + m)];
            }
        }
        return Ok(EigenResult {
            lambdas,
            vectors,
            inv_sqrt_deg,
            num_nodes: n,
            num_modes: actual_modes,
            used_dense_fallback: true,
        });
    }

    const SHIFT: f64 = 2.5;
    let k_total = actual_modes + 1;
    let lanczos = LanczosEigen::new(tol, max_iters, seed);
    let mut tmp = vec![0.0_f64; n];
    let (mu_desc, mu_vecs) = lanczos.largest_eigenpairs(n, k_total, |x, y| {
        l_sym.spmv(x, &mut tmp)?;
        for i in 0..n {
            y[i] = SHIFT * x[i] - tmp[i];
        }
        Ok(())
    })?;

    let lambdas: Vec<f64> = mu_desc[1..=actual_modes]
        .iter()
        .map(|&mu| (SHIFT - mu).max(0.0))
        .collect();
    let mut vectors = vec![0.0_f64; n * actual_modes];
    for row in 0..n {
        for m in 0..actual_modes {
            vectors[row * actual_modes + m] = mu_vecs[row * k_total + (1 + m)];
        }
    }

    Ok(EigenResult {
        lambdas,
        vectors,
        inv_sqrt_deg,
        num_nodes: n,
        num_modes: actual_modes,
        used_dense_fallback: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grid_2d(rows: usize, cols: usize, periodic: bool) -> CsrGraph<f64> {
        let n = rows * cols;
        let mut edges = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let u = (r * cols + c) as u32;
                if c + 1 < cols {
                    edges.push((u, u + 1));
                } else if periodic && cols > 2 {
                    edges.push((u, (r * cols) as u32));
                }
                if r + 1 < rows {
                    edges.push((u, u + cols as u32));
                } else if periodic && rows > 2 {
                    edges.push((u, c as u32));
                }
            }
        }
        CsrGraph::from_edges(n, &edges, None, false, true).unwrap()
    }

    /// Computes minimum principal-angle cosine between column spaces of two (n x k) row-major bases.
    fn min_principal_angle_cosine(u: &[f64], v: &[f64], n: usize, k: usize) -> f64 {
        let mut c = vec![0.0_f64; k * k];
        for r in 0..n {
            for a in 0..k {
                let ua = u[r * k + a];
                for b in 0..k {
                    c[a * k + b] += ua * v[r * k + b];
                }
            }
        }
        let mut ctc = vec![0.0_f64; k * k];
        for a in 0..k {
            for b in 0..k {
                let mut sum = 0.0_f64;
                for p in 0..k {
                    sum += c[p * k + a] * c[p * k + b];
                }
                ctc[a * k + b] = sum;
            }
        }
        let (evals, _) = symmetric_eigen_jacobi(&ctc, k, 120, 1e-13).unwrap();
        evals[0].max(0.0).sqrt().min(1.0)
    }

    #[test]
    fn test_jacobi_zero_degenerate_and_orthonormal() {
        let zero = vec![0.0; 9];
        let (evals, evecs) = symmetric_eigen_jacobi(&zero, 3, 50, 1e-12).unwrap();
        assert_eq!(evals, vec![0.0, 0.0, 0.0]);
        assert_eq!(evecs, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

        // Matrix with repeated eigenvalues: diag(3, 1, 3) rotated
        let a = vec![2.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 2.0];
        let (ev, vecs) = symmetric_eigen_jacobi(&a, 3, 80, 1e-13).unwrap();
        assert!((ev[0] - 1.0).abs() < 1e-12);
        assert!((ev[1] - 3.0).abs() < 1e-12);
        assert!((ev[2] - 3.0).abs() < 1e-12);

        // Verify A * V == V * diag(ev) and V^T * V == I
        for m in 0..3 {
            for i in 0..3 {
                let mut av = 0.0;
                for j in 0..3 {
                    av += a[i * 3 + j] * vecs[j * 3 + m];
                }
                assert!((av - ev[m] * vecs[i * 3 + m]).abs() < 1e-11);
            }
            for m2 in 0..3 {
                let dot: f64 = (0..3).map(|i| vecs[i * 3 + m] * vecs[i * 3 + m2]).sum();
                let expected = if m == m2 { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_lanczos_matches_jacobi_on_16x16_grid_principal_angle() {
        // 16x16 grid (n=256) exercises iterative Lanczos and allows fast exact dense Jacobi comparison.
        let grid = make_grid_2d(16, 16, false);
        let n = grid.num_nodes();
        let n_modes = 8; // Avoid splitting degenerate (p, q)/(q, p) pair at boundary.
        let res = smallest_sym_laplacian_eigenpairs(&grid, n_modes, 1e-5, 500, 42).unwrap();
        assert!(!res.used_dense_fallback);
        assert_eq!(res.num_nodes, n);
        assert_eq!(res.num_modes, n_modes);

        let (l_sym, _) = symmetric_normalized_laplacian(&grid);
        let dense = l_sym.to_dense().unwrap();
        let (evals_ref, evecs_ref) = symmetric_eigen_jacobi(&dense, n, 120, 1e-12).unwrap();

        for m in 0..n_modes {
            assert!(
                (res.lambdas[m] - evals_ref[1 + m]).abs() < 1e-7,
                "mode {m} eigenvalue mismatch: {} vs {}",
                res.lambdas[m],
                evals_ref[1 + m]
            );
        }

        let mut u_ref = vec![0.0_f64; n * n_modes];
        for row in 0..n {
            for m in 0..n_modes {
                u_ref[row * n_modes + m] = evecs_ref[row * n + (1 + m)];
            }
        }
        let cos_min = min_principal_angle_cosine(&res.vectors, &u_ref, n, n_modes);
        assert!(
            cos_min > 0.9999,
            "16x16 grid principal-angle cosine {cos_min:.6} <= 0.9999"
        );
    }

    #[test]
    fn test_lanczos_40x40_grid_tight_spectral_gaps() {
        // 40x40 2D grid (n=1600): validate residual accuracy, orthogonality, and subspace stability.
        let grid = make_grid_2d(40, 40, false);
        let n = grid.num_nodes();
        let n_modes = 16; // Modes 1..=16 end cleanly after (4,0)/(0,4) pair.
        let res = smallest_sym_laplacian_eigenpairs(&grid, n_modes, 1e-4, 500, 42).unwrap();
        let res_tight = smallest_sym_laplacian_eigenpairs(&grid, n_modes, 1e-8, 800, 99).unwrap();
        assert!(!res.used_dense_fallback);

        let (l_sym, _) = symmetric_normalized_laplacian(&grid);
        let mut x = vec![0.0_f64; n];
        let mut lx = vec![0.0_f64; n];
        for m in 0..n_modes {
            for i in 0..n {
                x[i] = res.vectors[i * n_modes + m];
            }
            l_sym.spmv(&x, &mut lx).unwrap();
            let res_norm: f64 = lx
                .iter()
                .zip(x.iter())
                .map(|(&lv, &xv)| {
                    let diff = lv - res.lambdas[m] * xv;
                    diff * diff
                })
                .sum::<f64>()
                .sqrt();
            assert!(
                res_norm < 1e-5,
                "40x40 mode {m} residual norm too large: {res_norm:.3e}"
            );
            assert!(
                (res.lambdas[m] - res_tight.lambdas[m]).abs() < 1e-7,
                "40x40 mode {m} eigenvalue mismatch vs tight ref"
            );
        }

        let cos_min = min_principal_angle_cosine(&res.vectors, &res_tight.vectors, n, n_modes);
        assert!(
            cos_min > 0.9999,
            "40x40 grid principal-angle cosine {cos_min:.6} <= 0.9999"
        );

        // Also validate exact closed-form eigenvalues with multiplicity 4 on a 40x40 periodic torus.
        let torus = make_grid_2d(40, 40, true);
        let torus_modes = 8; // (1,0),(0,1),(1,1) Fourier shells (4 + 4 = 8 modes cleanly separated).
        let res_torus =
            smallest_sym_laplacian_eigenpairs(&torus, torus_modes, 1e-4, 500, 7).unwrap();
        let expected_lam1 = 0.5 * (1.0 - (std::f64::consts::TAU / 40.0).cos());
        let expected_lam2 = 1.0 - (std::f64::consts::TAU / 40.0).cos();
        for (idx, &lam) in res_torus.lambdas[..4].iter().enumerate() {
            assert!(
                (lam - expected_lam1).abs() < 1e-7,
                "torus mode {idx}: got {lam}, expected {expected_lam1}, all={:?}",
                res_torus.lambdas
            );
        }
        for (idx, &lam) in res_torus.lambdas[4..8].iter().enumerate() {
            assert!(
                (lam - expected_lam2).abs() < 1e-7,
                "torus mode {}: got {lam}, expected {expected_lam2}, all={:?}",
                idx + 4,
                res_torus.lambdas
            );
        }
    }

    #[test]
    fn test_eigen_edge_cases_empty_singleton_edgeless_disconnected() {
        let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        let re = smallest_sym_laplacian_eigenpairs(&empty, 4, 1e-6, 100, 42).unwrap();
        assert_eq!(re.num_modes, 0);
        assert!(re.used_dense_fallback);

        let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
        let rs = smallest_sym_laplacian_eigenpairs(&single, 4, 1e-6, 100, 42).unwrap();
        assert_eq!(rs.num_modes, 0);

        // Large edgeless graph (triggers Lanczos invariant-subspace recovery for L_sym = I)
        let edgeless = CsrGraph::<f64>::from_edges(60, &[], None, false, false).unwrap();
        let r_iso = smallest_sym_laplacian_eigenpairs(&edgeless, 3, 1e-6, 100, 42).unwrap();
        assert!(!r_iso.used_dense_fallback);
        assert_eq!(r_iso.num_modes, 3);
        for &lam in &r_iso.lambdas {
            assert!((lam - 1.0).abs() < 1e-10);
        }

        // Disconnected graph: two 30-cycles (two zero eigenvalues)
        let mut edges = Vec::new();
        for i in 0..30 {
            edges.push((i, (i + 1) % 30));
            edges.push((30 + i, 30 + ((i + 1) % 30)));
        }
        let disc = CsrGraph::<f64>::from_edges(60, &edges, None, false, false).unwrap();
        let rd = smallest_sym_laplacian_eigenpairs(&disc, 3, 1e-6, 300, 42).unwrap();
        assert!(rd.lambdas[0] < 1e-10);
        assert!(rd.lambdas[1] > 1e-3);
    }
}
