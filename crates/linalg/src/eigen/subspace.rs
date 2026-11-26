use egraph_core::DeterministicRng;

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
pub(super) fn norm_slice(a: &[f64]) -> f64 {
    dot_slice(a, a).sqrt()
}

pub(super) fn fill_random_unit_orthogonal_buf(
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
    pub(super) q_vecs: Vec<f64>,
    pub(super) b_mat: Vec<f64>,
    pub(super) w_block: Vec<f64>,
    pub(super) r_block: Vec<f64>,
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

    pub(super) fn orthogonalize_w_against_q(&mut self, j: usize, start_j: usize) {
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

    pub(super) fn qr_next_block(
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

    pub(super) fn restart_with_ritz(
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
