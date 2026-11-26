use std::f64::consts::TAU;

use crate::csr::CsrGraph;
use crate::error::CoreResult;
use crate::layout::Layout2d;
use crate::scalar::Scalar;

use super::spectral_eigen::largest_eigenpairs_block_lanczos;
use super::spectral_jacobi::symmetric_eigen_jacobi;
use super::torus::wrap_torus_point;
use super::{ManifoldKind, ManifoldLayout2d, ManifoldMetric};

struct SymLaplacianCsr {
    n: usize,
    indptr: Vec<u32>,
    indices: Vec<u32>,
    values: Vec<f64>,
}

impl SymLaplacianCsr {
    fn from_graph<W: Scalar>(graph: &CsrGraph<W>) -> Self {
        let n = graph.num_nodes();
        if n == 0 {
            return Self {
                n: 0,
                indptr: vec![0],
                indices: Vec::new(),
                values: Vec::new(),
            };
        }

        let mut rows = vec![Vec::<(u32, f64)>::new(); n];
        let weights_opt = graph.weights();
        let directed = graph.is_directed();

        for u in 0..n {
            let start = graph.indptr()[u] as usize;
            let end = graph.indptr()[u + 1] as usize;
            for pos in start..end {
                let v = graph.indices()[pos] as usize;
                if v == u || v >= n {
                    continue;
                }
                let w = match weights_opt {
                    Some(ws) => {
                        let wf = ws[pos].to_f64();
                        if !wf.is_finite() || wf <= 0.0 {
                            continue;
                        }
                        wf
                    }
                    None => 1.0,
                };
                rows[u].push((v as u32, w));
                if directed {
                    rows[v].push((u as u32, w));
                }
            }
        }

        let mut degrees = vec![0.0_f64; n];
        for (u, row) in rows.iter_mut().enumerate() {
            row.sort_unstable_by_key(|&(v, _)| v);
            let mut write = 0usize;
            for read in 0..row.len() {
                let (v, w) = row[read];
                if write > 0 && row[write - 1].0 == v {
                    row[write - 1].1 = row[write - 1].1.max(w);
                } else {
                    row[write] = (v, w);
                    write += 1;
                }
            }
            row.truncate(write);
            degrees[u] = row.iter().map(|&(_, w)| w).sum();
        }

        let inv_sqrt_deg: Vec<f64> = degrees
            .iter()
            .map(|&d| if d > 0.0 { d.sqrt().recip() } else { 0.0 })
            .collect();

        let total_edges: usize = rows.iter().map(Vec::len).sum();
        let mut indptr = Vec::with_capacity(n + 1);
        indptr.push(0u32);
        let mut indices = Vec::with_capacity(total_edges + n);
        let mut values = Vec::with_capacity(total_edges + n);

        for (u, row) in rows.iter().enumerate() {
            let u_u32 = u as u32;
            let inv_u = inv_sqrt_deg[u];
            let mut diag_inserted = false;
            for &(v, w) in row {
                if !diag_inserted && v > u_u32 {
                    indices.push(u_u32);
                    values.push(1.0);
                    diag_inserted = true;
                }
                indices.push(v);
                values.push(-w * inv_u * inv_sqrt_deg[v as usize]);
            }
            if !diag_inserted {
                indices.push(u_u32);
                values.push(1.0);
            }
            indptr.push(indices.len() as u32);
        }

        Self {
            n,
            indptr,
            indices,
            values,
        }
    }

    fn to_dense(&self) -> Vec<f64> {
        let mut dense = vec![0.0_f64; self.n * self.n];
        for u in 0..self.n {
            let start = self.indptr[u] as usize;
            let end = self.indptr[u + 1] as usize;
            for pos in start..end {
                let v = self.indices[pos] as usize;
                dense[u * self.n + v] = self.values[pos];
            }
        }
        dense
    }

    fn spmv(&self, x: &[f64], y: &mut [f64]) {
        for (u, y_u) in y.iter_mut().enumerate() {
            let start = self.indptr[u] as usize;
            let end = self.indptr[u + 1] as usize;
            let mut acc = 0.0_f64;
            for pos in start..end {
                acc += self.values[pos] * x[self.indices[pos] as usize];
            }
            *y_u = acc;
        }
    }
}

impl<S: Scalar> ManifoldLayout2d<S> {
    /// Initializes a flat 2D torus layout via 4-mode spectral phase coordinates of the
    /// symmetric normalized graph Laplacian $L_{\mathrm{sym}} = I - D^{-1/2} A D^{-1/2}$.
    ///
    /// Computes the smallest 4 nontrivial eigenmodes `(v0, v1, v2, v3)` (zero-padded if fewer
    /// modes exist) using dense cyclic Jacobi for small graphs (`n <= 48`) and block thick-restart
    /// Lanczos for larger graphs, then maps each vertex to
    /// `((atan2(v1, v0) / TAU).rem_euclid(1.0), (atan2(v3, v2) / TAU).rem_euclid(1.0))` in `[0, 1)^2`.
    pub fn torus_spectral_phase_init<W: Scalar>(
        graph: &CsrGraph<W>,
        seed: u64,
    ) -> CoreResult<Self> {
        let n = graph.num_nodes();
        if n <= 1 {
            return Ok(Self::zeros(ManifoldKind::Torus2d, n));
        }

        let l_sym = SymLaplacianCsr::from_graph(graph);
        let actual_modes = 4.min(n - 1);
        let use_dense = n <= 48;

        let mut modes4 = vec![0.0_f64; n * 4];
        if use_dense {
            let dense = l_sym.to_dense();
            let (_evals, evecs) = symmetric_eigen_jacobi(&dense, n, 120, 1e-12)?;
            for row in 0..n {
                for m in 0..actual_modes {
                    modes4[row * 4 + m] = evecs[row * n + (1 + m)];
                }
            }
        } else {
            const SHIFT: f64 = 2.5;
            let k_total = actual_modes + 1;
            let mut tmp = vec![0.0_f64; n];
            let (_mu_desc, mu_vecs) =
                largest_eigenpairs_block_lanczos(n, k_total, 1e-6, 500, seed, |x, y| {
                    l_sym.spmv(x, &mut tmp);
                    for i in 0..n {
                        y[i] = SHIFT * x[i] - tmp[i];
                    }
                })?;
            for row in 0..n {
                for m in 0..actual_modes {
                    modes4[row * 4 + m] = mu_vecs[row * k_total + (1 + m)];
                }
            }
        }

        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for row in 0..n {
            let v0 = modes4[row * 4];
            let v1 = modes4[row * 4 + 1];
            let v2 = modes4[row * 4 + 2];
            let v3 = modes4[row * 4 + 3];
            let th1 = (v1.atan2(v0) / TAU).rem_euclid(1.0);
            let th2 = (v3.atan2(v2) / TAU).rem_euclid(1.0);
            let (px, py) = wrap_torus_point(S::from_f64(th1), S::from_f64(th2));
            xs.push(px);
            ys.push(py);
        }

        Ok(Self::from_parts_unchecked(
            ManifoldKind::Torus2d,
            Layout2d::from_xy(xs, ys)?,
            ManifoldMetric::default(),
        ))
    }
}
