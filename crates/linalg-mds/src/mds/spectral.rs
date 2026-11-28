use std::marker::PhantomData;

use egraph_core::{CoreError, CoreResult, CsrGraph, EdgeWeight, EuclideanLayout, Layout2d, Scalar};
use egraph_linalg::smallest_sym_laplacian_eigenpairs;

use super::normalize::normalize_euclidean_layout;
use super::storage::{
    EuclideanLayoutAlloc, impl_euclidean_layout_generator, resolve_target_dims, set_storage_row,
};

/// Configuration for [`KorenSpectral`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct KorenSpectralConfig {
    /// Optional explicit target dimensionality (`None` infers `2` for `Layout2d`, `3` for `Layout3d`, and `2` for `LayoutNd`).
    pub dims: Option<usize>,
    /// Optional target global standard deviation enforced via [`normalize_euclidean_layout`] (`default: Some(1.0)`).
    pub target_std: Option<f64>,
    /// Eigensolver convergence tolerance (`default: 1e-6`).
    pub tol: f64,
    /// Maximum number of thick-restart Lanczos cycles (`default: 300`).
    pub max_iters: usize,
    /// Deterministic PRNG seed (`default: 42`).
    pub seed: u64,
}

impl Default for KorenSpectralConfig {
    fn default() -> Self {
        Self {
            dims: None,
            target_std: Some(1.0),
            tol: 1e-6,
            max_iters: 300,
            seed: 42,
        }
    }
}

impl KorenSpectralConfig {
    /// Creates a new [`KorenSpectralConfig`] initialized to [`Self::default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the optional target dimensionality `dims`.
    pub fn with_dims(mut self, dims: Option<usize>) -> Self {
        self.dims = dims;
        self
    }

    /// Sets the optional target global standard deviation `target_std`.
    pub fn with_target_std(mut self, target_std: Option<f64>) -> Self {
        self.target_std = target_std;
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
}

/// Koren's degree-normalized spectral layout generator $(D^{-1/2} v_2, \dots, D^{-1/2} v_{d+1})$
/// implementing [`LayoutGenerator`](egraph_core::LayoutGenerator) for `Layout2d`, `Layout3d`, and `LayoutNd`.
///
/// Reference: Koren, Y. (2003/2005), *"On Spectral Graph Drawing"*, *COCOON 2003* /
/// *Computers & Mathematics with Applications* 49(11–12), 1867–1883.
#[non_exhaustive]
#[derive(Debug)]
pub struct KorenSpectral<'a, L: EuclideanLayout = Layout2d<f64>, W = f64> {
    graph: &'a CsrGraph<W>,
    config: KorenSpectralConfig,
    _layout: PhantomData<fn() -> L>,
}

impl<L: EuclideanLayout, W> Clone for KorenSpectral<'_, L, W> {
    fn clone(&self) -> Self {
        Self {
            graph: self.graph,
            config: self.config.clone(),
            _layout: PhantomData,
        }
    }
}

impl<'a, L: EuclideanLayout, W: EdgeWeight> KorenSpectral<'a, L, W> {
    /// Creates a [`KorenSpectral`] generator with default configuration on `graph`.
    pub fn new(graph: &'a CsrGraph<W>) -> Self {
        Self {
            graph,
            config: KorenSpectralConfig::default(),
            _layout: PhantomData,
        }
    }

    /// Creates a [`KorenSpectral`] generator on `graph` with an explicit [`KorenSpectralConfig`].
    pub fn from_config(graph: &'a CsrGraph<W>, config: KorenSpectralConfig) -> Self {
        Self {
            graph,
            config,
            _layout: PhantomData,
        }
    }
}

impl<'a, L: EuclideanLayout, W> KorenSpectral<'a, L, W> {
    /// Replaces the configuration on this generator.
    pub fn with_config(mut self, config: KorenSpectralConfig) -> Self {
        self.config = config;
        self
    }

    /// Sets the explicit target dimensionality `dims`.
    pub fn with_dims(mut self, dims: usize) -> Self {
        self.config.dims = Some(dims);
        self
    }

    /// Sets the optional target global standard deviation `target_std`.
    pub fn with_target_std(mut self, target_std: Option<f64>) -> Self {
        self.config.target_std = target_std;
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

    /// Returns a reference to the current [`KorenSpectralConfig`].
    pub fn config(&self) -> &KorenSpectralConfig {
        &self.config
    }

    fn num_nodes(&self) -> usize {
        self.graph.num_nodes()
    }

    fn validate_config(&self) -> CoreResult<()> {
        if let Some(t_std) = self.config.target_std {
            if !t_std.is_finite() || !L::Scalar::from_f64(t_std).is_finite() {
                return Err(CoreError::non_finite("KorenSpectralConfig::target_std"));
            }
            if t_std <= 0.0 {
                return Err(CoreError::invalid_input(
                    "target_std must be strictly positive",
                ));
            }
        }
        if !self.config.tol.is_finite() {
            return Err(CoreError::non_finite("KorenSpectralConfig::tol"));
        }
        if self.config.tol <= 0.0 {
            return Err(CoreError::invalid_input(
                "KorenSpectralConfig::tol must be strictly positive",
            ));
        }
        if self.config.max_iters == 0 {
            return Err(CoreError::invalid_input(
                "KorenSpectralConfig::max_iters must be >= 1",
            ));
        }
        Ok(())
    }
}

impl<L: EuclideanLayoutAlloc, W> KorenSpectral<'_, L, W> {
    fn resolve_dims(&self) -> CoreResult<usize> {
        resolve_target_dims::<L>(self.config.dims, "KorenSpectralConfig::dims")
    }
}

impl<L: EuclideanLayout, W: EdgeWeight> KorenSpectral<'_, L, W> {
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
        let mut row_buf = vec![0.0_f64; used_dims];
        let mut storage = layout.as_euclidean_storage_mut();
        for i in 0..n {
            let d_inv = eig.inv_sqrt_deg[i];
            for m in 0..used_dims {
                row_buf[m] = d_inv * eig.vectors[i * m_count + m];
            }
            set_storage_row(&mut storage, i, &row_buf);
        }

        if let Some(t_std) = self.config.target_std {
            normalize_euclidean_layout(layout, t_std, self.config.seed)?;
        }
        if layout.validate_finite().is_err() {
            return Err(CoreError::numerical_breakdown(
                "KorenSpectral projection overflowed output scalar",
            ));
        }
        Ok(())
    }
}

impl_euclidean_layout_generator!(KorenSpectral);
