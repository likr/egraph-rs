use std::marker::PhantomData;

use egraph_core::{CoreError, CoreResult, CsrGraph, EuclideanLayout, Layout2d, Scalar};
use egraph_linalg::smallest_sym_laplacian_eigenpairs;

use super::normalize::normalize_euclidean_layout;
use super::storage::{
    EuclideanLayoutAlloc, impl_euclidean_layout_generator, resolve_target_dims, set_storage_row,
};

/// Configuration for [`RdMds`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct RdMdsConfig {
    /// Optional explicit target dimensionality (`None` infers `2` for `Layout2d`, `3` for `Layout3d`, and `2` for `LayoutNd`).
    pub dims: Option<usize>,
    /// Minimum eigenvalue threshold; modes with $\lambda \le \mathrm{eig\_eps}$ map to zero (`default: 1e-10`).
    pub eig_eps: f64,
    /// Eigensolver convergence tolerance (`default: 1e-6`).
    pub tol: f64,
    /// Maximum number of thick-restart Lanczos cycles (`default: 1000`).
    pub max_iters: usize,
    /// Deterministic PRNG seed (`default: 42`).
    pub seed: u64,
    /// Whether to apply $D^{-1/2}$ degree normalization (`true` for effective resistance, `default: true`).
    pub degree_normalize: bool,
    /// Optional target global standard deviation enforced via [`normalize_euclidean_layout`] (`default: None`).
    pub target_std: Option<f64>,
}

impl Default for RdMdsConfig {
    fn default() -> Self {
        Self {
            dims: None,
            eig_eps: 1e-10,
            tol: 1e-6,
            max_iters: 1000,
            seed: 42,
            degree_normalize: true,
            target_std: None,
        }
    }
}

impl RdMdsConfig {
    /// Creates a new [`RdMdsConfig`] initialized to [`Self::default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the optional target dimensionality `dims`.
    pub fn with_dims(mut self, dims: Option<usize>) -> Self {
        self.dims = dims;
        self
    }

    /// Sets the zero-eigenvalue guard threshold `eig_eps`.
    pub fn with_eig_eps(mut self, eig_eps: f64) -> Self {
        self.eig_eps = eig_eps;
        self
    }

    /// Sets the eigensolver convergence tolerance `tol`.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Sets the maximum number of thick-restart Lanczos cycles `max_iters`.
    pub fn with_max_iters(mut self, max_iters: usize) -> Self {
        self.max_iters = max_iters;
        self
    }

    /// Sets the deterministic PRNG seed `seed`.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Configures whether $D^{-1/2}$ degree normalization is applied.
    pub fn with_degree_normalize(mut self, degree_normalize: bool) -> Self {
        self.degree_normalize = degree_normalize;
        self
    }

    /// Sets the optional target global standard deviation `target_std`.
    pub fn with_target_std(mut self, target_std: Option<f64>) -> Self {
        self.target_std = target_std;
        self
    }
}

/// Zero-guarded Resistance-distance MDS (`RdMds`) spectral layout generator implementing
/// [`LayoutGenerator`](egraph_core::LayoutGenerator) for `Layout2d`, `Layout3d`, and `LayoutNd`.
///
/// Computes $d$-dimensional coordinates $D^{-1/2} U \Lambda^{-1/2}$ (or $U \Lambda^{-1/2}$
/// when `degree_normalize == false`) from the smallest nontrivial eigenpairs of $L_{\mathrm{sym}}$,
/// zeroing out modes with $\lambda \le \mathrm{eig\_eps}$ to prevent zero-division blowups on
/// disconnected or edgeless graphs.
#[non_exhaustive]
#[derive(Debug)]
pub struct RdMds<'a, L: EuclideanLayout = Layout2d<f64>, W = f64> {
    graph: &'a CsrGraph<W>,
    config: RdMdsConfig,
    _layout: PhantomData<fn() -> L>,
}

impl<L: EuclideanLayout, W> Clone for RdMds<'_, L, W> {
    fn clone(&self) -> Self {
        Self {
            graph: self.graph,
            config: self.config.clone(),
            _layout: PhantomData,
        }
    }
}

impl<'a, L: EuclideanLayout, W: Scalar> RdMds<'a, L, W> {
    /// Creates a new [`RdMds`] generator with default parameters on `graph`.
    pub fn new(graph: &'a CsrGraph<W>) -> Self {
        Self {
            graph,
            config: RdMdsConfig::default(),
            _layout: PhantomData,
        }
    }

    /// Creates a new [`RdMds`] generator on `graph` with an explicit [`RdMdsConfig`].
    pub fn from_config(graph: &'a CsrGraph<W>, config: RdMdsConfig) -> Self {
        Self {
            graph,
            config,
            _layout: PhantomData,
        }
    }
}

impl<'a, L: EuclideanLayout, W> RdMds<'a, L, W> {
    /// Replaces the configuration on this generator.
    pub fn with_config(mut self, config: RdMdsConfig) -> Self {
        self.config = config;
        self
    }

    /// Sets the explicit target dimensionality `dims`.
    pub fn with_dims(mut self, dims: usize) -> Self {
        self.config.dims = Some(dims);
        self
    }

    /// Sets the zero-eigenvalue guard threshold `eig_eps`.
    pub fn with_eig_eps(mut self, eig_eps: f64) -> Self {
        self.config.eig_eps = eig_eps;
        self
    }

    /// Sets the eigensolver convergence tolerance `tol`.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.config.tol = tol;
        self
    }

    /// Sets the maximum number of thick-restart Lanczos cycles `max_iters`.
    pub fn with_max_iters(mut self, max_iters: usize) -> Self {
        self.config.max_iters = max_iters;
        self
    }

    /// Sets the deterministic PRNG seed `seed`.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
        self
    }

    /// Configures whether $D^{-1/2}$ degree normalization is applied.
    pub fn with_degree_normalize(mut self, degree_normalize: bool) -> Self {
        self.config.degree_normalize = degree_normalize;
        self
    }

    /// Sets an optional target global standard deviation `target_std`.
    pub fn with_target_std(mut self, target_std: Option<f64>) -> Self {
        self.config.target_std = target_std;
        self
    }

    /// Returns a reference to the current [`RdMdsConfig`].
    pub fn config(&self) -> &RdMdsConfig {
        &self.config
    }

    fn num_nodes(&self) -> usize {
        self.graph.num_nodes()
    }

    fn validate_config(&self) -> CoreResult<()> {
        if !self.config.eig_eps.is_finite() {
            return Err(CoreError::non_finite("RdMdsConfig::eig_eps"));
        }
        if self.config.eig_eps < 0.0 {
            return Err(CoreError::invalid_input(
                "RdMdsConfig::eig_eps must be >= 0",
            ));
        }
        if !self.config.tol.is_finite() {
            return Err(CoreError::non_finite("RdMdsConfig::tol"));
        }
        if self.config.tol <= 0.0 {
            return Err(CoreError::invalid_input(
                "RdMdsConfig::tol must be strictly positive",
            ));
        }
        if self.config.max_iters == 0 {
            return Err(CoreError::invalid_input(
                "RdMdsConfig::max_iters must be >= 1",
            ));
        }
        if let Some(t_std) = self.config.target_std {
            if !t_std.is_finite() || !L::Scalar::from_f64(t_std).is_finite() {
                return Err(CoreError::non_finite("RdMdsConfig::target_std"));
            }
            if t_std <= 0.0 {
                return Err(CoreError::invalid_input(
                    "target_std must be strictly positive",
                ));
            }
        }
        Ok(())
    }
}

impl<L: EuclideanLayoutAlloc, W> RdMds<'_, L, W> {
    fn resolve_dims(&self) -> CoreResult<usize> {
        resolve_target_dims::<L>(self.config.dims, "RdMdsConfig::dims")
    }
}

impl<L: EuclideanLayout, W: Scalar> RdMds<'_, L, W> {
    fn generate_into_layout(&self, layout: &mut L, dims: usize) -> CoreResult<()> {
        self.validate_config()?;
        let n = self.graph.num_nodes();
        if n <= 1 {
            return Ok(());
        }

        let eig = smallest_sym_laplacian_eigenpairs(
            self.graph,
            dims,
            self.config.tol,
            self.config.max_iters,
            self.config.seed,
        )?;
        let m_count = eig.num_modes;
        let used_dims = dims.min(m_count);
        let mut inv_sqrt_lambda = vec![0.0_f64; used_dims];
        for m in 0..used_dims {
            let lam = eig.lambdas[m];
            inv_sqrt_lambda[m] = if lam > self.config.eig_eps {
                lam.sqrt().recip()
            } else {
                0.0
            };
        }

        let mut row_buf = vec![0.0_f64; used_dims];
        let mut storage = layout.as_euclidean_storage_mut();
        for i in 0..n {
            let row_scale = if self.config.degree_normalize {
                eig.inv_sqrt_deg[i]
            } else {
                1.0
            };
            for m in 0..used_dims {
                row_buf[m] = row_scale * eig.vectors[i * m_count + m] * inv_sqrt_lambda[m];
            }
            set_storage_row(&mut storage, i, &row_buf);
        }

        if let Some(t_std) = self.config.target_std {
            normalize_euclidean_layout(layout, t_std, self.config.seed)?;
        }
        if layout.validate_finite().is_err() {
            return Err(CoreError::numerical_breakdown(
                "RdMds projection overflowed output scalar",
            ));
        }
        Ok(())
    }
}

impl_euclidean_layout_generator!(RdMds);
