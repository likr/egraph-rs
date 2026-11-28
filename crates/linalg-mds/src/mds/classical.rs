use std::marker::PhantomData;

use egraph_core::{CoreError, CoreResult, CsrGraph, EdgeWeight, EuclideanLayout, Layout2d};
use egraph_linalg::{BfsWorkspace, DijkstraWorkspace, symmetric_eigen_jacobi};

use super::storage::{
    EuclideanLayoutAlloc, impl_euclidean_layout_generator, resolve_target_dims, set_storage_row,
    validate_distance_slice,
};

/// Configuration for [`ClassicalMds`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct ClassicalMdsConfig {
    /// Optional explicit target dimensionality (`None` infers `2` for `Layout2d`, `3` for `Layout3d`, and `2` for `LayoutNd`).
    pub dims: Option<usize>,
    /// Minimum eigenvalue threshold; modes with $\lambda \le \mathrm{eig\_eps}$ map to zero (`default: 1e-10`).
    pub eig_eps: f64,
    /// Convergence tolerance passed to [`symmetric_eigen_jacobi`](egraph_linalg::symmetric_eigen_jacobi) (`default: 1e-12`).
    pub tol: f64,
    /// Maximum Jacobi sweeps passed to [`symmetric_eigen_jacobi`](egraph_linalg::symmetric_eigen_jacobi) (`default: 120`).
    pub max_sweeps: usize,
}

impl Default for ClassicalMdsConfig {
    fn default() -> Self {
        Self {
            dims: None,
            eig_eps: 1e-10,
            tol: 1e-12,
            max_sweeps: 120,
        }
    }
}

impl ClassicalMdsConfig {
    /// Creates a new [`ClassicalMdsConfig`] initialized to [`Self::default`].
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
}

#[derive(Debug)]
enum ClassicalMdsSource<'a, W> {
    Graph(&'a CsrGraph<W>),
    Distances {
        distances_nn: &'a [f64],
        num_nodes: usize,
    },
}

impl<W> Clone for ClassicalMdsSource<'_, W> {
    fn clone(&self) -> Self {
        match *self {
            Self::Graph(g) => Self::Graph(g),
            Self::Distances {
                distances_nn,
                num_nodes,
            } => Self::Distances {
                distances_nn,
                num_nodes,
            },
        }
    }
}

/// Zero-guarded Classical Multidimensional Scaling (MDS) layout generator implementing
/// [`LayoutGenerator`](egraph_core::LayoutGenerator) for `Layout2d`, `Layout3d`, and `LayoutNd`.
///
/// Reference: Torgerson, W. S. (1952), *"Multidimensional scaling: I. Theory and method"*,
/// *Psychometrika* 17(4), 401–419; Kruskal, J. B. & Wish, M. (1978), *"Multidimensional Scaling"*,
/// Sage Publications.
#[non_exhaustive]
#[derive(Debug)]
pub struct ClassicalMds<'a, L: EuclideanLayout = Layout2d<f64>, W = f64> {
    source: ClassicalMdsSource<'a, W>,
    config: ClassicalMdsConfig,
    _layout: PhantomData<fn() -> L>,
}

impl<L: EuclideanLayout, W> Clone for ClassicalMds<'_, L, W> {
    fn clone(&self) -> Self {
        Self {
            source: self.source.clone(),
            config: self.config.clone(),
            _layout: PhantomData,
        }
    }
}

impl<'a, L: EuclideanLayout, W: EdgeWeight> ClassicalMds<'a, L, W> {
    /// Creates a [`ClassicalMds`] generator computing all-pairs shortest paths on `graph`.
    pub fn new(graph: &'a CsrGraph<W>) -> Self {
        Self {
            source: ClassicalMdsSource::Graph(graph),
            config: ClassicalMdsConfig::default(),
            _layout: PhantomData,
        }
    }

    /// Creates a [`ClassicalMds`] generator on `graph` with an explicit [`ClassicalMdsConfig`].
    pub fn from_config(graph: &'a CsrGraph<W>, config: ClassicalMdsConfig) -> Self {
        Self {
            source: ClassicalMdsSource::Graph(graph),
            config,
            _layout: PhantomData,
        }
    }
}

impl<'a, L: EuclideanLayout> ClassicalMds<'a, L, f64> {
    /// Creates a [`ClassicalMds`] generator from a flat row-major `(num_nodes, num_nodes)`
    /// distance matrix.
    pub fn from_distances(distances_nn: &'a [f64], num_nodes: usize) -> Self {
        Self {
            source: ClassicalMdsSource::Distances {
                distances_nn,
                num_nodes,
            },
            config: ClassicalMdsConfig::default(),
            _layout: PhantomData,
        }
    }
}

impl<'a, L: EuclideanLayout, W> ClassicalMds<'a, L, W> {
    /// Replaces the configuration on this generator.
    pub fn with_config(mut self, config: ClassicalMdsConfig) -> Self {
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

    /// Returns a reference to the current [`ClassicalMdsConfig`].
    pub fn config(&self) -> &ClassicalMdsConfig {
        &self.config
    }

    fn num_nodes(&self) -> usize {
        match self.source {
            ClassicalMdsSource::Graph(g) => g.num_nodes(),
            ClassicalMdsSource::Distances { num_nodes, .. } => num_nodes,
        }
    }

    fn validate_config(&self) -> CoreResult<()> {
        if !self.config.eig_eps.is_finite() {
            return Err(CoreError::non_finite("ClassicalMdsConfig::eig_eps"));
        }
        if self.config.eig_eps < 0.0 {
            return Err(CoreError::invalid_input(
                "ClassicalMdsConfig::eig_eps must be >= 0",
            ));
        }
        if !self.config.tol.is_finite() {
            return Err(CoreError::non_finite("ClassicalMdsConfig::tol"));
        }
        if self.config.tol <= 0.0 {
            return Err(CoreError::invalid_input(
                "ClassicalMdsConfig::tol must be strictly positive",
            ));
        }
        if self.config.max_sweeps == 0 {
            return Err(CoreError::invalid_input(
                "ClassicalMdsConfig::max_sweeps must be >= 1",
            ));
        }
        Ok(())
    }
}

impl<L: EuclideanLayoutAlloc, W> ClassicalMds<'_, L, W> {
    fn resolve_dims(&self) -> CoreResult<usize> {
        resolve_target_dims::<L>(self.config.dims, "ClassicalMds::dims")
    }
}

impl<L: EuclideanLayout, W: EdgeWeight> ClassicalMds<'_, L, W> {
    fn generate_into_layout(&self, layout: &mut L, dims: usize) -> CoreResult<()> {
        self.validate_config()?;
        let num_nodes = self.num_nodes();
        let owned_dists;
        let distances_nn: &[f64] = match self.source {
            ClassicalMdsSource::Distances {
                distances_nn,
                num_nodes: n,
            } => {
                let expected = n.checked_mul(n).ok_or_else(|| {
                    CoreError::invalid_input("classical_mds matrix size overflow")
                })?;
                if distances_nn.len() != expected {
                    return Err(CoreError::dimension_mismatch(
                        expected,
                        distances_nn.len(),
                        "ClassicalMds::from_distances::distances_nn",
                    ));
                }
                validate_distance_slice(
                    distances_nn,
                    "ClassicalMds::from_distances::distances_nn",
                )?;
                distances_nn
            }
            ClassicalMdsSource::Graph(graph) => {
                if num_nodes <= 1 {
                    return Ok(());
                }
                let total = num_nodes
                    .checked_mul(num_nodes)
                    .ok_or_else(|| CoreError::invalid_input("classical_mds APSP size overflow"))?;
                let mut dists = vec![f64::INFINITY; total];
                if graph.is_weighted() {
                    let mut ws = DijkstraWorkspace::new(num_nodes);
                    for src in 0..num_nodes {
                        ws.dijkstra_from(
                            graph,
                            src,
                            &mut dists[src * num_nodes..(src + 1) * num_nodes],
                        )?;
                    }
                } else {
                    let mut ws = BfsWorkspace::new(num_nodes);
                    for src in 0..num_nodes {
                        ws.bfs_from(
                            graph,
                            src,
                            &mut dists[src * num_nodes..(src + 1) * num_nodes],
                        )?;
                    }
                }
                owned_dists = dists;
                &owned_dists
            }
        };

        if num_nodes <= 1 {
            return Ok(());
        }

        let b = double_center_distances(distances_nn, num_nodes, num_nodes)?;
        let (evals, evecs) =
            symmetric_eigen_jacobi(&b, num_nodes, self.config.max_sweeps, self.config.tol)?;
        let used_dims = dims.min(num_nodes);

        let mut weights = vec![0.0_f64; used_dims];
        for (m, w) in weights.iter_mut().enumerate() {
            let eig_val = evals[num_nodes - 1 - m];
            *w = if eig_val > self.config.eig_eps {
                eig_val.sqrt()
            } else {
                0.0
            };
        }

        let mut row_buf = vec![0.0_f64; used_dims];
        let mut storage = layout.as_euclidean_storage_mut();
        for i in 0..num_nodes {
            for m in 0..used_dims {
                let eig_idx = num_nodes - 1 - m;
                row_buf[m] = evecs[i * num_nodes + eig_idx] * weights[m];
            }
            set_storage_row(&mut storage, i, &row_buf);
        }
        if layout.validate_finite().is_err() {
            return Err(CoreError::numerical_breakdown(
                "ClassicalMds projection overflowed output scalar",
            ));
        }
        Ok(())
    }
}

impl_euclidean_layout_generator!(ClassicalMds);

/// Replaces `+f64::INFINITY` disconnected-pair entries (and overflowing `d * d` squares)
/// in a squared-distance buffer with a bounded finite fallback and applies double centering:
/// $C = -\frac{1}{2}(I - \frac{1}{n}\mathbf{1}\mathbf{1}^T)\Delta(I - \frac{1}{k}\mathbf{1}\mathbf{1}^T)$.
fn double_center_distances(distances: &[f64], nrows: usize, ncols: usize) -> CoreResult<Vec<f64>> {
    if nrows == 0 || ncols == 0 {
        return Ok(Vec::new());
    }
    let total = nrows * ncols;
    let mut delta = Vec::with_capacity(total);
    let mut max_finite_sq = 0.0_f64;
    let mut has_nonfinite = false;

    for &d in distances {
        if d.is_nan() || d == f64::NEG_INFINITY {
            return Err(CoreError::non_finite("double_center_distances::distances"));
        }
        if d < 0.0 {
            return Err(CoreError::invalid_input(
                "double_center_distances::distances must be >= 0",
            ));
        }
        if d.is_finite() {
            let sq = d * d;
            if sq.is_finite() {
                delta.push(sq);
                if sq > max_finite_sq {
                    max_finite_sq = sq;
                }
                continue;
            }
        }
        delta.push(f64::INFINITY);
        has_nonfinite = true;
    }

    if has_nonfinite {
        let denom = (total.max(1) * 4) as f64;
        let safe_cap = f64::MAX / denom;
        let fallback_sq = if max_finite_sq > 0.0 {
            (max_finite_sq * 4.0).min(safe_cap)
        } else {
            1.0
        };
        for val in &mut delta {
            if !val.is_finite() {
                *val = fallback_sq;
            }
        }
    }

    let mut row_means = vec![0.0_f64; nrows];
    let mut col_means = vec![0.0_f64; ncols];
    let inv_cols = 1.0 / (ncols as f64);
    let inv_rows = 1.0 / (nrows as f64);

    for i in 0..nrows {
        let row = &delta[i * ncols..(i + 1) * ncols];
        let mut r_sum = 0.0_f64;
        for (j, &val) in row.iter().enumerate() {
            r_sum += val;
            col_means[j] += val;
        }
        row_means[i] = r_sum * inv_cols;
    }
    for cm in &mut col_means {
        *cm *= inv_rows;
    }
    let grand_mean = row_means.iter().sum::<f64>() * inv_rows;

    for (i, &rm) in row_means.iter().enumerate() {
        let row_off = i * ncols;
        for j in 0..ncols {
            delta[row_off + j] = 0.5 * (rm + col_means[j] - delta[row_off + j] - grand_mean);
        }
    }
    Ok(delta)
}
