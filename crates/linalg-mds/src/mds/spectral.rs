use egraph_core::{CoreError, CoreResult, CsrGraph, Layout2d, LayoutNd, Scalar};
use egraph_linalg::eigen::smallest_sym_laplacian_eigenpairs;

use super::normalize::normalize_layout_2d;

/// Computes Koren's degree-normalized 2D spectral layout $(D^{-1/2} v_2, D^{-1/2} v_3)$
/// from the 2nd and 3rd smallest eigenvectors of $L_{\mathrm{sym}}$, with optional isotropic
/// standard-deviation normalization.
pub fn koren_spectral_init<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    target_std: Option<f64>,
    tol: f64,
    max_iters: usize,
    seed: u64,
) -> CoreResult<Layout2d<S>> {
    if let Some(t_std) = target_std {
        if !t_std.is_finite() {
            return Err(CoreError::non_finite("koren_spectral_init::target_std"));
        }
        if t_std <= 0.0 {
            return Err(CoreError::invalid_input(
                "target_std must be strictly positive",
            ));
        }
    }
    let n = graph.num_nodes();
    if n == 0 {
        return Ok(Layout2d::zeros(0));
    }
    if n == 1 {
        return Ok(Layout2d::zeros(1));
    }

    let eig = smallest_sym_laplacian_eigenpairs(graph, 2, tol, max_iters, seed)?;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let m = eig.num_modes;
    for i in 0..n {
        let d_inv = eig.inv_sqrt_deg[i];
        let vx = if m >= 1 {
            d_inv * eig.vectors[i * m]
        } else {
            0.0
        };
        let vy = if m >= 2 {
            d_inv * eig.vectors[i * m + 1]
        } else {
            0.0
        };
        x.push(S::from_f64(vx));
        y.push(S::from_f64(vy));
    }

    let mut layout = Layout2d::from_xy(x, y)?;
    if let Some(t_std) = target_std {
        normalize_layout_2d(&mut layout, t_std, seed)?;
    }
    Ok(layout)
}

/// Zero-guarded Resistance-distance MDS (`RdMds`) spectral embedding builder.
///
/// Computes $d$-dimensional coordinates $D^{-1/2} U \Lambda^{-1/2}$ (or $U \Lambda^{-1/2}$
/// when `degree_normalize == false`) from the smallest nontrivial eigenpairs of $L_{\mathrm{sym}}$,
/// zeroing out modes with $\lambda \le \mathrm{eig\_eps}$ to prevent zero-division blowups on
/// disconnected or edgeless graphs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RdMds {
    /// Target spectral embedding dimension $d$.
    pub dims: usize,
    /// Minimum eigenvalue threshold; modes with $\lambda \le \mathrm{eig\_eps}$ map to zero.
    pub eig_eps: f64,
    /// Eigensolver convergence tolerance.
    pub tol: f64,
    /// Maximum number of thick-restart Lanczos cycles.
    pub max_iters: usize,
    /// Deterministic PRNG seed.
    pub seed: u64,
    /// Whether to apply $D^{-1/2}$ degree normalization (`true` for effective resistance).
    pub degree_normalize: bool,
    /// Optional target global standard deviation when producing 2D layouts via [`RdMds::embedding_2d`].
    pub target_std: Option<f64>,
}

impl Default for RdMds {
    fn default() -> Self {
        Self {
            dims: 2,
            eig_eps: 1e-10,
            tol: 1e-6,
            max_iters: 1000,
            seed: 42,
            degree_normalize: true,
            target_std: None,
        }
    }
}

impl RdMds {
    /// Creates a new [`RdMds`] configuration with default parameters (`dims = 2`).
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new [`RdMds`] configuration targeting `dims` spectral dimensions.
    pub fn with_dims(dims: usize) -> Self {
        Self {
            dims,
            ..Self::default()
        }
    }

    /// Sets the number of embedding dimensions.
    pub fn dims(mut self, dims: usize) -> Self {
        self.dims = dims;
        self
    }

    /// Sets the zero-eigenvalue guard threshold `eig_eps`.
    pub fn eig_eps(mut self, eig_eps: f64) -> Self {
        self.eig_eps = eig_eps;
        self
    }

    /// Sets the eigensolver convergence tolerance.
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Sets the maximum number of thick-restart Lanczos cycles.
    pub fn max_iters(mut self, max_iters: usize) -> Self {
        self.max_iters = max_iters;
        self
    }

    /// Sets the deterministic PRNG seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Configures whether $D^{-1/2}$ degree normalization is applied.
    pub fn degree_normalize(mut self, degree_normalize: bool) -> Self {
        self.degree_normalize = degree_normalize;
        self
    }

    /// Sets an optional target standard deviation for 2D layouts.
    pub fn target_std(mut self, target_std: Option<f64>) -> Self {
        self.target_std = target_std;
        self
    }

    /// Computes a $d$-dimensional zero-guarded resistance-distance spectral embedding.
    pub fn embedding<W: Scalar, S: Scalar>(&self, graph: &CsrGraph<W>) -> CoreResult<LayoutNd<S>> {
        if self.dims == 0 {
            return Err(CoreError::invalid_input("RdMds dims must be >= 1"));
        }
        if !self.eig_eps.is_finite() || self.eig_eps < 0.0 {
            return Err(CoreError::invalid_input(
                "RdMds eig_eps must be finite and >= 0",
            ));
        }

        let n = graph.num_nodes();
        let mut layout = LayoutNd::zeros(n, self.dims)?;
        if n <= 1 {
            return Ok(layout);
        }

        let eig = smallest_sym_laplacian_eigenpairs(
            graph,
            self.dims,
            self.tol,
            self.max_iters,
            self.seed,
        )?;
        let m_count = eig.num_modes;
        let mut inv_sqrt_lambda = vec![0.0_f64; m_count];
        for m in 0..m_count {
            let lam = eig.lambdas[m];
            inv_sqrt_lambda[m] = if lam > self.eig_eps {
                lam.sqrt().recip()
            } else {
                0.0
            };
        }

        let coords = layout.coords_mut();
        for i in 0..n {
            let row_scale = if self.degree_normalize {
                eig.inv_sqrt_deg[i]
            } else {
                1.0
            };
            let out_row = &mut coords[i * self.dims..i * self.dims + m_count];
            for m in 0..m_count {
                let val = row_scale * eig.vectors[i * m_count + m] * inv_sqrt_lambda[m];
                out_row[m] = S::from_f64(val);
            }
        }
        Ok(layout)
    }

    /// Computes a 2D [`Layout2d`] resistance-distance embedding with optional `target_std` normalization.
    pub fn embedding_2d<W: Scalar, S: Scalar>(
        &self,
        graph: &CsrGraph<W>,
    ) -> CoreResult<Layout2d<S>> {
        if let Some(t_std) = self.target_std {
            if !t_std.is_finite() {
                return Err(CoreError::non_finite("RdMds::target_std"));
            }
            if t_std <= 0.0 {
                return Err(CoreError::invalid_input(
                    "target_std must be strictly positive",
                ));
            }
        }
        let cfg = Self { dims: 2, ..*self };
        let nd = cfg.embedding::<W, f64>(graph)?;
        let n = graph.num_nodes();
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let row = nd.node(i)?;
            x.push(S::from_f64(row[0]));
            y.push(S::from_f64(row[1]));
        }
        let mut layout = Layout2d::from_xy(x, y)?;
        if let Some(t_std) = self.target_std {
            normalize_layout_2d(&mut layout, t_std, self.seed)?;
        }
        Ok(layout)
    }
}

/// Functional shorthand for computing a zero-guarded [`RdMds`] spectral embedding.
pub fn rd_mds<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    dims: usize,
    eig_eps: f64,
    tol: f64,
    max_iters: usize,
    seed: u64,
) -> CoreResult<LayoutNd<S>> {
    RdMds::with_dims(dims)
        .eig_eps(eig_eps)
        .tol(tol)
        .max_iters(max_iters)
        .seed(seed)
        .embedding(graph)
}
