use std::marker::PhantomData;

use egraph_core::{CoreError, CoreResult, CsrGraph, EdgeWeight, EuclideanLayout, Layout2d, Scalar};
use egraph_linalg::{BfsWorkspace, DijkstraWorkspace, select_maxmin_sp_pivots};

use super::normalize::normalize_euclidean_layout;
use super::pivot_kernel::mds_from_row_major_distances;
use super::storage::{
    EuclideanLayoutAlloc, impl_euclidean_layout_generator, resolve_target_dims, set_storage_row,
    validate_distance_slice,
};

/// Configuration for [`PivotMds`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct PivotMdsConfig {
    /// Optional explicit target dimensionality (`None` infers `2` for `Layout2d`, `3` for `Layout3d`, and `2` for `LayoutNd`).
    pub dims: Option<usize>,
    /// Maximum number of Core-First MaxMin shortest-path pivots (`default: 50`).
    pub num_pivots: usize,
    /// Minimum degree prioritized during Core-First MaxMin pivot selection (`default: 2`).
    pub min_degree: usize,
    /// Deterministic PRNG seed controlling pivot selection and degenerate fallback (`default: 42`).
    pub seed: u64,
    /// Optional target global standard deviation enforced via [`normalize_euclidean_layout`] (`default: Some(1.0)`).
    pub target_std: Option<f64>,
}

impl Default for PivotMdsConfig {
    fn default() -> Self {
        Self {
            dims: None,
            num_pivots: 50,
            min_degree: 2,
            seed: 42,
            target_std: Some(1.0),
        }
    }
}

impl PivotMdsConfig {
    /// Creates a new [`PivotMdsConfig`] initialized to [`Self::default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the optional target dimensionality `dims`.
    pub fn with_dims(mut self, dims: Option<usize>) -> Self {
        self.dims = dims;
        self
    }

    /// Sets the maximum number of pivots `num_pivots`.
    pub fn with_num_pivots(mut self, num_pivots: usize) -> Self {
        self.num_pivots = num_pivots;
        self
    }

    /// Sets the minimum degree threshold `min_degree` for pivot selection.
    pub fn with_min_degree(mut self, min_degree: usize) -> Self {
        self.min_degree = min_degree;
        self
    }

    /// Sets the deterministic PRNG seed `seed`.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sets the optional target standard deviation `target_std`.
    pub fn with_target_std(mut self, target_std: Option<f64>) -> Self {
        self.target_std = target_std;
        self
    }
}

#[derive(Debug)]
enum PivotMdsSource<'a, W> {
    Graph(&'a CsrGraph<W>),
    Distances {
        distances_nk: &'a [f64],
        num_nodes: usize,
        num_pivots: usize,
    },
    PivotDistances {
        distances_kn: &'a [f64],
        num_nodes: usize,
        num_pivots: usize,
    },
}

impl<W> Clone for PivotMdsSource<'_, W> {
    fn clone(&self) -> Self {
        match *self {
            Self::Graph(g) => Self::Graph(g),
            Self::Distances {
                distances_nk,
                num_nodes,
                num_pivots,
            } => Self::Distances {
                distances_nk,
                num_nodes,
                num_pivots,
            },
            Self::PivotDistances {
                distances_kn,
                num_nodes,
                num_pivots,
            } => Self::PivotDistances {
                distances_kn,
                num_nodes,
                num_pivots,
            },
        }
    }
}

/// Core-First MaxMin Pivot / Landmark MDS layout generator implementing
/// [`LayoutGenerator`](egraph_core::LayoutGenerator) for `Layout2d`, `Layout3d`, and `LayoutNd`.
///
/// Reference: Brandes, U. & Pich, C. (2006), *"Eigensolver Methods for Progressive
/// Multidimensional Scaling of Large Data"*, *Graph Drawing (GD 2006)*, LNCS 4372, pp. 42–53.
#[non_exhaustive]
#[derive(Debug)]
pub struct PivotMds<'a, L: EuclideanLayout = Layout2d<f64>, W = f64> {
    source: PivotMdsSource<'a, W>,
    pivots: Option<&'a [u32]>,
    config: PivotMdsConfig,
    _layout: PhantomData<fn() -> L>,
}

impl<L: EuclideanLayout, W> Clone for PivotMds<'_, L, W> {
    fn clone(&self) -> Self {
        Self {
            source: self.source.clone(),
            pivots: self.pivots,
            config: self.config.clone(),
            _layout: PhantomData,
        }
    }
}

impl<'a, L: EuclideanLayout, W: EdgeWeight> PivotMds<'a, L, W> {
    /// Creates a [`PivotMds`] generator selecting MaxMin shortest-path pivots on `graph`.
    pub fn new(graph: &'a CsrGraph<W>) -> Self {
        Self {
            source: PivotMdsSource::Graph(graph),
            pivots: None,
            config: PivotMdsConfig::default(),
            _layout: PhantomData,
        }
    }

    /// Creates a [`PivotMds`] generator on `graph` with an explicit [`PivotMdsConfig`].
    pub fn from_config(graph: &'a CsrGraph<W>, config: PivotMdsConfig) -> Self {
        Self {
            source: PivotMdsSource::Graph(graph),
            pivots: None,
            config,
            _layout: PhantomData,
        }
    }
}

impl<'a, L: EuclideanLayout> PivotMds<'a, L, f64> {
    /// Creates a [`PivotMds`] generator from a precomputed row-major `(num_nodes, num_pivots)`
    /// distance matrix (also serving as Representative / Landmark MDS).
    pub fn from_distances(distances_nk: &'a [f64], num_nodes: usize, num_pivots: usize) -> Self {
        Self {
            source: PivotMdsSource::Distances {
                distances_nk,
                num_nodes,
                num_pivots,
            },
            pivots: None,
            config: PivotMdsConfig::default(),
            _layout: PhantomData,
        }
    }

    /// Creates a [`PivotMds`] generator from a precomputed pivot-major `(num_pivots, num_nodes)`
    /// distance matrix (`pivot_distances_kn[k * num_nodes + i]`), reading columns zero-copy
    /// without transposing into a temporary `(num_nodes, num_pivots)` buffer.
    pub fn from_pivot_distances(
        pivot_distances_kn: &'a [f64],
        num_nodes: usize,
        num_pivots: usize,
    ) -> Self {
        Self {
            source: PivotMdsSource::PivotDistances {
                distances_kn: pivot_distances_kn,
                num_nodes,
                num_pivots,
            },
            pivots: None,
            config: PivotMdsConfig::default(),
            _layout: PhantomData,
        }
    }
}

impl<'a, L: EuclideanLayout, W> PivotMds<'a, L, W> {
    /// Replaces the configuration on this generator.
    pub fn with_config(mut self, config: PivotMdsConfig) -> Self {
        self.config = config;
        self
    }

    /// Uses an explicitly supplied slice of pivot vertex indices instead of MaxMin selection.
    pub fn with_pivots(mut self, pivots: &'a [u32]) -> Self {
        self.pivots = Some(pivots);
        self
    }

    /// Sets the explicit target dimensionality `dims`.
    pub fn with_dims(mut self, dims: usize) -> Self {
        self.config.dims = Some(dims);
        self
    }

    /// Sets the maximum number of pivots `num_pivots`.
    pub fn with_num_pivots(mut self, num_pivots: usize) -> Self {
        self.config.num_pivots = num_pivots;
        self
    }

    /// Sets the minimum degree threshold `min_degree` for pivot selection.
    pub fn with_min_degree(mut self, min_degree: usize) -> Self {
        self.config.min_degree = min_degree;
        self
    }

    /// Sets the deterministic PRNG seed `seed`.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
        self
    }

    /// Sets the optional target global standard deviation `target_std`.
    pub fn with_target_std(mut self, target_std: Option<f64>) -> Self {
        self.config.target_std = target_std;
        self
    }

    /// Returns a reference to the current [`PivotMdsConfig`].
    pub fn config(&self) -> &PivotMdsConfig {
        &self.config
    }

    fn num_nodes(&self) -> usize {
        match self.source {
            PivotMdsSource::Graph(g) => g.num_nodes(),
            PivotMdsSource::Distances { num_nodes, .. }
            | PivotMdsSource::PivotDistances { num_nodes, .. } => num_nodes,
        }
    }

    fn validate_target_std(&self) -> CoreResult<()> {
        if let Some(t_std) = self.config.target_std {
            if !t_std.is_finite() || !L::Scalar::from_f64(t_std).is_finite() {
                return Err(CoreError::non_finite("PivotMdsConfig::target_std"));
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

impl<L: EuclideanLayoutAlloc, W> PivotMds<'_, L, W> {
    fn resolve_dims(&self) -> CoreResult<usize> {
        resolve_target_dims::<L>(self.config.dims, "PivotMds::dims")
    }
}

impl<L: EuclideanLayout, W: EdgeWeight> PivotMds<'_, L, W> {
    fn generate_into_layout(&self, layout: &mut L, dims: usize) -> CoreResult<()> {
        self.validate_target_std()?;
        match self.source {
            PivotMdsSource::Distances {
                distances_nk,
                num_nodes,
                num_pivots,
            } => {
                if self.pivots.is_some() {
                    return Err(CoreError::invalid_input(
                        "PivotMds::with_pivots cannot be combined with from_distances",
                    ));
                }
                if num_pivots == 0 {
                    return Err(CoreError::invalid_input("num_pivots must be >= 1"));
                }
                let expected = num_nodes.checked_mul(num_pivots).ok_or_else(|| {
                    CoreError::invalid_input("pivot_mds distance matrix size overflow")
                })?;
                if distances_nk.len() != expected {
                    return Err(CoreError::dimension_mismatch(
                        expected,
                        distances_nk.len(),
                        "PivotMds::from_distances::distances_nk",
                    ));
                }
                validate_distance_slice(distances_nk, "PivotMds::from_distances::distances_nk")?;
                if num_nodes > 1 {
                    let mut storage = layout.as_euclidean_storage_mut();
                    mds_from_row_major_distances(
                        num_nodes,
                        num_pivots,
                        dims,
                        |i, row_buf| {
                            row_buf.copy_from_slice(
                                &distances_nk[i * num_pivots..(i + 1) * num_pivots],
                            );
                        },
                        |i, vals| set_storage_row(&mut storage, i, vals),
                    )?;
                }
            }
            PivotMdsSource::PivotDistances {
                distances_kn,
                num_nodes,
                num_pivots,
            } => {
                if self.pivots.is_some() {
                    return Err(CoreError::invalid_input(
                        "PivotMds::with_pivots cannot be combined with from_pivot_distances",
                    ));
                }
                if num_pivots == 0 {
                    return Err(CoreError::invalid_input("num_pivots must be >= 1"));
                }
                let expected = num_pivots.checked_mul(num_nodes).ok_or_else(|| {
                    CoreError::invalid_input("pivot_mds distance matrix size overflow")
                })?;
                if distances_kn.len() != expected {
                    return Err(CoreError::dimension_mismatch(
                        expected,
                        distances_kn.len(),
                        "PivotMds::from_pivot_distances::pivot_distances_kn",
                    ));
                }
                validate_distance_slice(
                    distances_kn,
                    "PivotMds::from_pivot_distances::pivot_distances_kn",
                )?;
                if num_nodes > 1 {
                    let mut storage = layout.as_euclidean_storage_mut();
                    mds_from_row_major_distances(
                        num_nodes,
                        num_pivots,
                        dims,
                        |i, row_buf| {
                            for (k, dst) in row_buf.iter_mut().enumerate() {
                                *dst = distances_kn[k * num_nodes + i];
                            }
                        },
                        |i, vals| set_storage_row(&mut storage, i, vals),
                    )?;
                }
            }
            PivotMdsSource::Graph(graph) => {
                let n = graph.num_nodes();
                let owned_pivots;
                let pivots: &[u32] = match self.pivots {
                    Some(piv) => {
                        for &p in piv {
                            if (p as usize) >= n {
                                return Err(CoreError::index_out_of_bounds(
                                    p as usize,
                                    n,
                                    "PivotMds::pivots",
                                ));
                            }
                        }
                        if n > 0 && piv.is_empty() {
                            return Err(CoreError::invalid_input("pivots must not be empty"));
                        }
                        piv
                    }
                    None => {
                        if self.config.num_pivots == 0 {
                            return Err(CoreError::invalid_input("num_pivots must be >= 1"));
                        }
                        if n <= 1 {
                            return Ok(());
                        }
                        owned_pivots = select_maxmin_sp_pivots(
                            graph,
                            self.config.num_pivots,
                            self.config.seed,
                            self.config.min_degree,
                        )?;
                        &owned_pivots
                    }
                };

                if n <= 1 {
                    return Ok(());
                }
                let k = pivots.len();
                if k >= 2 {
                    self.compute_graph_with_pivots(graph, pivots, layout, dims)?;
                }
            }
        }

        if let Some(t_std) = self.config.target_std {
            normalize_euclidean_layout(layout, t_std, self.config.seed)?;
        }
        if layout.validate_finite().is_err() {
            return Err(CoreError::numerical_breakdown(
                "PivotMds projection overflowed output scalar",
            ));
        }
        Ok(())
    }

    fn compute_graph_with_pivots(
        &self,
        graph: &CsrGraph<W>,
        pivots: &[u32],
        layout: &mut L,
        dims: usize,
    ) -> CoreResult<()> {
        let n = graph.num_nodes();
        let k = pivots.len();
        let total = n
            .checked_mul(k)
            .ok_or_else(|| CoreError::invalid_input("pivot_mds distance matrix size overflow"))?;

        if graph.is_weighted() {
            let mut distances_nk = vec![f64::INFINITY; total];
            let mut dist_col = vec![f64::INFINITY; n];
            let mut ws = DijkstraWorkspace::new(n);
            for (col, &p) in pivots.iter().enumerate() {
                ws.dijkstra_from(graph, p as usize, &mut dist_col)?;
                for i in 0..n {
                    distances_nk[i * k + col] = dist_col[i];
                }
            }
            let mut storage = layout.as_euclidean_storage_mut();
            return mds_from_row_major_distances(
                n,
                k,
                dims,
                |i, row_buf| {
                    row_buf.copy_from_slice(&distances_nk[i * k..(i + 1) * k]);
                },
                |i, vals| set_storage_row(&mut storage, i, vals),
            );
        }

        let mut dist_col = vec![u32::MAX; n];
        let mut ws = BfsWorkspace::new(n);

        if n <= u16::MAX as usize {
            let mut distances_nk = vec![u16::MAX; total];
            for (col, &p) in pivots.iter().enumerate() {
                ws.bfs_from_u32(graph, p as usize, &mut dist_col)?;
                for i in 0..n {
                    let d = dist_col[i];
                    distances_nk[i * k + col] = if d == u32::MAX { u16::MAX } else { d as u16 };
                }
            }
            let mut storage = layout.as_euclidean_storage_mut();
            mds_from_row_major_distances(
                n,
                k,
                dims,
                |i, row_buf| {
                    let row = &distances_nk[i * k..(i + 1) * k];
                    for (dst, &d) in row_buf.iter_mut().zip(row.iter()) {
                        *dst = if d == u16::MAX {
                            f64::INFINITY
                        } else {
                            f64::from(d)
                        };
                    }
                },
                |i, vals| set_storage_row(&mut storage, i, vals),
            )
        } else {
            let mut distances_nk = vec![u32::MAX; total];
            for (col, &p) in pivots.iter().enumerate() {
                ws.bfs_from_u32(graph, p as usize, &mut dist_col)?;
                for i in 0..n {
                    distances_nk[i * k + col] = dist_col[i];
                }
            }
            let mut storage = layout.as_euclidean_storage_mut();
            mds_from_row_major_distances(
                n,
                k,
                dims,
                |i, row_buf| {
                    let row = &distances_nk[i * k..(i + 1) * k];
                    for (dst, &d) in row_buf.iter_mut().zip(row.iter()) {
                        *dst = if d == u32::MAX {
                            f64::INFINITY
                        } else {
                            f64::from(d)
                        };
                    }
                },
                |i, vals| set_storage_row(&mut storage, i, vals),
            )
        }
    }
}

impl_euclidean_layout_generator!(PivotMds);
