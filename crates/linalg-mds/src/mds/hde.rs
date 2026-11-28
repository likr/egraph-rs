use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, EdgeWeight, EuclideanLayout, Layout2d, VisitStamp,
};
use egraph_linalg::{BfsWorkspace, DijkstraWorkspace, select_maxmin_sp_pivots};

use super::hde_pca::{hde_pca_project, prepare_unweighted_row_stats, validate_positive_weights};
use super::normalize::normalize_f64_buffer;
use super::storage::{
    EuclideanLayoutAlloc, impl_euclidean_layout_generator, resolve_target_dims,
    write_row_major_to_layout,
};

/// Configuration for Harel-Koren High-Dimensional Embedding ([`Hde`]).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct HdeConfig {
    /// Maximum number of MaxMin shortest-path pivot dimensions $m$ (`default: 50`).
    pub dimensions: usize,
    /// Optional explicit target PCA output dimensionality $k$ (`None` infers `2` for `Layout2d`, `3` for `Layout3d`, and `2` for `LayoutNd`).
    pub output_dims: Option<usize>,
    /// Deterministic PRNG seed for pivot selection and collapsed normalization fallback (`default: 42`).
    pub seed: u64,
    /// Convergence tolerance passed to [`symmetric_eigen_jacobi`](egraph_linalg::symmetric_eigen_jacobi) (`default: 1e-3`).
    pub tol: f64,
    /// Maximum Jacobi sweeps passed to [`symmetric_eigen_jacobi`](egraph_linalg::symmetric_eigen_jacobi) (`default: 100`).
    pub max_sweeps: usize,
    /// Whether to center and isotropically normalize the embedding to unit std (`default: true`).
    pub normalize: bool,
}

impl Default for HdeConfig {
    fn default() -> Self {
        Self {
            dimensions: 50,
            output_dims: None,
            seed: 42,
            tol: 1e-3,
            max_sweeps: 100,
            normalize: true,
        }
    }
}

impl HdeConfig {
    /// Creates a new [`HdeConfig`] initialized to [`Self::default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum number of MaxMin shortest-path pivot dimensions `dimensions`.
    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = dimensions;
        self
    }

    /// Sets the optional target PCA output dimensionality `output_dims`.
    pub fn with_output_dims(mut self, output_dims: Option<usize>) -> Self {
        self.output_dims = output_dims;
        self
    }

    /// Sets the deterministic PRNG seed `seed`.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sets the Jacobi eigensolver convergence tolerance `tol`.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Sets the maximum Jacobi sweeps `max_sweeps`.
    pub fn with_max_sweeps(mut self, max_sweeps: usize) -> Self {
        self.max_sweeps = max_sweeps;
        self
    }

    /// Sets whether the embedding is centered and isotropically normalized (`normalize`).
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
}

/// Harel-Koren High-Dimensional Embedding (`HDE`) layout generator implementing
/// [`LayoutGenerator`](egraph_core::LayoutGenerator) for `Layout2d`, `Layout3d`, and `LayoutNd`.
///
/// Reference: Harel, D. & Koren, Y. (2002), *"Graph Drawing by High-Dimensional Embedding"*,
/// *Graph Drawing (GD 2002)*, LNCS 2528, pp. 207–219.
#[non_exhaustive]
#[derive(Debug)]
pub struct Hde<'a, L: EuclideanLayout = Layout2d<f64>, W = f64> {
    graph: &'a CsrGraph<W>,
    pivots: Option<&'a [u32]>,
    config: HdeConfig,
    _layout: PhantomData<fn() -> L>,
}

impl<L: EuclideanLayout, W> Clone for Hde<'_, L, W> {
    fn clone(&self) -> Self {
        Self {
            graph: self.graph,
            pivots: self.pivots,
            config: self.config.clone(),
            _layout: PhantomData,
        }
    }
}

impl<'a, L: EuclideanLayout, W: EdgeWeight> Hde<'a, L, W> {
    /// Creates an [`Hde`] generator with default configuration on `graph`.
    pub fn new(graph: &'a CsrGraph<W>) -> Self {
        Self {
            graph,
            pivots: None,
            config: HdeConfig::default(),
            _layout: PhantomData,
        }
    }

    /// Creates an [`Hde`] generator on `graph` with an explicit [`HdeConfig`].
    pub fn from_config(graph: &'a CsrGraph<W>, config: HdeConfig) -> Self {
        Self {
            graph,
            pivots: None,
            config,
            _layout: PhantomData,
        }
    }
}

impl<'a, L: EuclideanLayout, W> Hde<'a, L, W> {
    /// Replaces the configuration on this generator.
    pub fn with_config(mut self, config: HdeConfig) -> Self {
        self.config = config;
        self
    }

    /// Uses an explicitly supplied slice of pivot vertex indices instead of MaxMin selection.
    pub fn with_pivots(mut self, pivots: &'a [u32]) -> Self {
        self.pivots = Some(pivots);
        self
    }

    /// Sets the maximum number of MaxMin shortest-path pivot dimensions `dimensions`.
    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.config.dimensions = dimensions;
        self
    }

    /// Sets the explicit target PCA output dimensionality `output_dims`.
    pub fn with_output_dims(mut self, output_dims: usize) -> Self {
        self.config.output_dims = Some(output_dims);
        self
    }

    /// Sets the explicit target dimensionality (alias for [`Self::with_output_dims`]).
    pub fn with_dims(mut self, dims: usize) -> Self {
        self.config.output_dims = Some(dims);
        self
    }

    /// Sets the deterministic PRNG seed `seed`.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
        self
    }

    /// Sets the Jacobi eigensolver convergence tolerance `tol`.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.config.tol = tol;
        self
    }

    /// Sets the maximum Jacobi sweeps `max_sweeps`.
    pub fn with_max_sweeps(mut self, max_sweeps: usize) -> Self {
        self.config.max_sweeps = max_sweeps;
        self
    }

    /// Sets whether the embedding is centered and isotropically normalized (`normalize`).
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.config.normalize = normalize;
        self
    }

    /// Returns a reference to the current [`HdeConfig`].
    pub fn config(&self) -> &HdeConfig {
        &self.config
    }

    fn num_nodes(&self) -> usize {
        self.graph.num_nodes()
    }

    fn validate_config(&self) -> CoreResult<()> {
        if self.config.dimensions == 0 {
            return Err(CoreError::invalid_input(
                "HdeConfig::dimensions must be >= 1",
            ));
        }
        if !self.config.tol.is_finite() {
            return Err(CoreError::non_finite("HdeConfig::tol"));
        }
        if self.config.tol <= 0.0 {
            return Err(CoreError::invalid_input(
                "HdeConfig::tol must be strictly positive",
            ));
        }
        if self.config.max_sweeps == 0 {
            return Err(CoreError::invalid_input(
                "HdeConfig::max_sweeps must be >= 1",
            ));
        }
        Ok(())
    }
}

impl<L: EuclideanLayoutAlloc, W> Hde<'_, L, W> {
    fn resolve_dims(&self) -> CoreResult<usize> {
        resolve_target_dims::<L>(self.config.output_dims, "HdeConfig::output_dims")
    }
}

impl<L: EuclideanLayout, W: EdgeWeight> Hde<'_, L, W> {
    fn generate_into_layout(&self, layout: &mut L, output_dims: usize) -> CoreResult<()> {
        self.validate_config()?;
        validate_positive_weights(self.graph)?;
        let n = self.graph.num_nodes();

        let owned_pivots;
        let pivots: &[u32] = match self.pivots {
            Some(piv) => piv,
            None => {
                if n <= 1 {
                    return Ok(());
                }
                owned_pivots = select_maxmin_sp_pivots(
                    self.graph,
                    self.config.dimensions,
                    self.config.seed,
                    1,
                )?;
                &owned_pivots
            }
        };

        for &p in pivots {
            if (p as usize) >= n {
                return Err(CoreError::index_out_of_bounds(p as usize, n, "Hde::pivots"));
            }
        }
        if n == 0 {
            return Ok(());
        }
        if pivots.is_empty() {
            return Err(CoreError::invalid_input("pivots must not be empty"));
        }

        let mut seen = VisitStamp::new(n);
        let mut unique_pivots = Vec::with_capacity(pivots.len().min(n));
        for &p in pivots {
            if seen.visit(p as usize) {
                unique_pivots.push(p);
            }
        }
        let m_eff = unique_pivots.len();

        if n == 1 {
            return Ok(());
        }

        let total = m_eff
            .checked_mul(n)
            .ok_or_else(|| CoreError::invalid_input("hde distance matrix size overflow"))?;
        m_eff
            .checked_mul(m_eff)
            .ok_or_else(|| CoreError::invalid_input("hde covariance matrix size overflow"))?;

        let mut out_coords = vec![0.0_f64; n * output_dims];
        compute_hde_projection(
            self.graph,
            &unique_pivots,
            total,
            output_dims,
            &self.config,
            &mut out_coords,
        )?;

        if self.config.normalize {
            normalize_f64_buffer(&mut out_coords, n, output_dims, 1.0, self.config.seed)?;
        }
        write_row_major_to_layout(
            layout,
            &out_coords,
            n,
            output_dims,
            "HDE projection overflowed output scalar",
        )
    }
}

impl_euclidean_layout_generator!(Hde);

fn compute_hde_projection<W: EdgeWeight>(
    graph: &CsrGraph<W>,
    unique_pivots: &[u32],
    total: usize,
    output_dims: usize,
    config: &HdeConfig,
    out_coords: &mut [f64],
) -> CoreResult<()> {
    let n = graph.num_nodes();
    let m_eff = unique_pivots.len();

    if graph.is_weighted() {
        let mut x_hat = vec![f64::INFINITY; total];
        let mut ws = DijkstraWorkspace::new(n);
        let mut max_abs_centered = 0.0_f64;
        let inv_n = 1.0 / (n as f64);

        for (i, &p) in unique_pivots.iter().enumerate() {
            let row = &mut x_hat[i * n..(i + 1) * n];
            ws.dijkstra_from(graph, p as usize, row)?;

            let mut max_finite = 0.0_f64;
            let mut has_finite = false;
            let mut has_nonfinite = false;
            for &d in row.iter() {
                if d.is_finite() {
                    has_finite = true;
                    if d > max_finite {
                        max_finite = d;
                    }
                } else {
                    has_nonfinite = true;
                }
            }
            if has_nonfinite {
                let fill = if has_finite { max_finite + 1.0 } else { 0.0 };
                if !fill.is_finite() {
                    return Err(CoreError::numerical_breakdown(
                        "HDE finite distance fallback overflowed f64",
                    ));
                }
                for d in row.iter_mut() {
                    if !d.is_finite() {
                        *d = fill;
                    }
                }
            }

            let mut mean = row.iter().sum::<f64>() * inv_n;
            if !mean.is_finite() {
                mean = row.iter().map(|&d| d * inv_n).sum::<f64>();
                if !mean.is_finite() {
                    return Err(CoreError::numerical_breakdown(
                        "HDE column mean overflowed f64",
                    ));
                }
            }
            for d in row.iter_mut() {
                *d -= mean;
                let abs_v = d.abs();
                if abs_v > max_abs_centered {
                    max_abs_centered = abs_v;
                }
            }
        }

        let zeros = vec![0.0_f64; m_eff];
        return hde_pca_project(
            &x_hat,
            n,
            m_eff,
            &zeros,
            &zeros,
            max_abs_centered,
            output_dims,
            config,
            out_coords,
        );
    }

    let mut ws = BfsWorkspace::new(n);
    let mut dist_col = vec![u32::MAX; n];
    let mut fills = vec![0.0_f64; m_eff];
    let mut means = vec![0.0_f64; m_eff];
    let mut max_abs_centered = 0.0_f64;

    if n <= u16::MAX as usize {
        let mut dists = vec![u16::MAX; total];
        for (i, &p) in unique_pivots.iter().enumerate() {
            ws.bfs_from_u32(graph, p as usize, &mut dist_col)?;
            let row = &mut dists[i * n..(i + 1) * n];
            for (dst, &d) in row.iter_mut().zip(dist_col.iter()) {
                *dst = if d == u32::MAX { u16::MAX } else { d as u16 };
            }
            let (fill, mean, row_max_abs) =
                prepare_unweighted_row_stats(row, u16::MAX, f64::from, n)?;
            fills[i] = fill;
            means[i] = mean;
            if row_max_abs > max_abs_centered {
                max_abs_centered = row_max_abs;
            }
        }
        hde_pca_project(
            &dists,
            n,
            m_eff,
            &fills,
            &means,
            max_abs_centered,
            output_dims,
            config,
            out_coords,
        )
    } else {
        let mut dists = vec![u32::MAX; total];
        for (i, &p) in unique_pivots.iter().enumerate() {
            ws.bfs_from_u32(graph, p as usize, &mut dist_col)?;
            let row = &mut dists[i * n..(i + 1) * n];
            row.copy_from_slice(&dist_col);
            let (fill, mean, row_max_abs) =
                prepare_unweighted_row_stats(row, u32::MAX, f64::from, n)?;
            fills[i] = fill;
            means[i] = mean;
            if row_max_abs > max_abs_centered {
                max_abs_centered = row_max_abs;
            }
        }
        hde_pca_project(
            &dists,
            n,
            m_eff,
            &fills,
            &means,
            max_abs_centered,
            output_dims,
            config,
            out_coords,
        )
    }
}
