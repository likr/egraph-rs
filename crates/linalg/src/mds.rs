//! Multidimensional Scaling (`pivot_mds`, `representative_mds`, `classical_mds`, `RdMds`),
//! Koren degree-normalized spectral initialization, and isotropic 2D coordinate normalization.

use egraph_core::{CoreError, CoreResult, CsrGraph, DeterministicRng, Layout2d, LayoutNd, Scalar};

use crate::eigen::{smallest_sym_laplacian_eigenpairs, symmetric_eigen_jacobi};
use crate::shortest_path::{BfsWorkspace, DijkstraWorkspace, select_maxmin_sp_pivots};

#[inline]
fn next_gaussian_pair(rng: &mut DeterministicRng) -> (f64, f64) {
    let u1 = rng.next_f64().max(f64::MIN_POSITIVE);
    let u2 = rng.next_f64();
    let r = (-2.0 * u1.ln()).sqrt();
    let (s, c) = (std::f64::consts::TAU * u2).sin_cos();
    (r * c, r * s)
}

/// Sanitizes non-finite entries, centers coordinates per axis, and scales isotropically so
/// $\sqrt{\frac{1}{2N} \sum_i ((x_i - \bar{x})^2 + (y_i - \bar{y})^2)} = \mathrm{target\_std}$,
/// matching `numpy.std(coords)` on `(N, 2)` centered arrays.
///
/// Collapsed or degenerate layouts (`n >= 2`, `std < 1e-12`) fall back deterministically to
/// standard Gaussian coordinates drawn from `DeterministicRng::from_seed_u64(seed)`.
pub fn normalize_layout_2d<S: Scalar>(
    layout: &mut Layout2d<S>,
    target_std: f64,
    seed: u64,
) -> CoreResult<()> {
    if !target_std.is_finite() {
        return Err(CoreError::non_finite("normalize_layout_2d::target_std"));
    }
    if target_std <= 0.0 {
        return Err(CoreError::invalid_input(
            "target_std must be strictly positive",
        ));
    }
    let n = layout.len();
    if n == 0 {
        return Ok(());
    }

    let mut rng = DeterministicRng::from_seed_u64(seed);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for (&x, &y) in layout.x().iter().zip(layout.y().iter()) {
        let xf = x.to_f64();
        let yf = y.to_f64();
        let clean_x = if xf.is_finite() {
            xf
        } else {
            next_gaussian_pair(&mut rng).0
        };
        let clean_y = if yf.is_finite() {
            yf
        } else {
            next_gaussian_pair(&mut rng).0
        };
        xs.push(clean_x);
        ys.push(clean_y);
    }

    if n == 1 {
        layout.x_mut()[0] = S::ZERO;
        layout.y_mut()[0] = S::ZERO;
        return Ok(());
    }

    let n_f64 = n as f64;
    let center_and_std = |x_buf: &mut [f64], y_buf: &mut [f64]| -> f64 {
        let mx = x_buf.iter().sum::<f64>() / n_f64;
        let my = y_buf.iter().sum::<f64>() / n_f64;
        let mut sum_sq = 0.0_f64;
        for (x, y) in x_buf.iter_mut().zip(y_buf.iter_mut()) {
            *x -= mx;
            *y -= my;
            sum_sq += (*x) * (*x) + (*y) * (*y);
        }
        (sum_sq / (2.0 * n_f64)).sqrt()
    };

    let mut std_dev = center_and_std(&mut xs, &mut ys);
    if !std_dev.is_finite() || std_dev < 1e-12 {
        for (x, y) in xs.iter_mut().zip(ys.iter_mut()) {
            let (gx, gy) = next_gaussian_pair(&mut rng);
            *x = gx;
            *y = gy;
        }
        std_dev = center_and_std(&mut xs, &mut ys);
        if !std_dev.is_finite() || std_dev < 1e-12 {
            layout.x_mut().fill(S::ZERO);
            layout.y_mut().fill(S::ZERO);
            return Ok(());
        }
    }

    let scale = target_std / std_dev;
    let (dst_x, dst_y) = layout.xy_mut();
    for i in 0..n {
        dst_x[i] = S::from_f64(xs[i] * scale);
        dst_y[i] = S::from_f64(ys[i] * scale);
    }
    Ok(())
}

/// Replaces infinite/non-finite entries in a squared-distance buffer with a bounded finite
/// fallback and applies double centering $C = -\frac{1}{2}(I - \frac{1}{n}\mathbf{1}\mathbf{1}^T)\Delta(I - \frac{1}{k}\mathbf{1}\mathbf{1}^T)$.
fn double_center_distances(distances: &[f64], nrows: usize, ncols: usize) -> Vec<f64> {
    if nrows == 0 || ncols == 0 {
        return Vec::new();
    }
    let total = nrows * ncols;
    let mut delta = Vec::with_capacity(total);
    let mut max_finite_sq = 0.0_f64;
    let mut has_nonfinite = false;

    for &d in distances {
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

    let mut c = vec![0.0_f64; total];
    for (i, &rm) in row_means.iter().enumerate() {
        let row_off = i * ncols;
        for j in 0..ncols {
            c[row_off + j] = 0.5 * (rm + col_means[j] - delta[row_off + j] - grand_mean);
        }
    }
    c
}

/// Configuration for [`pivot_mds`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PivotMdsConfig {
    /// Maximum number of Core-First MaxMin shortest-path pivots (`default: 50`).
    pub num_pivots: usize,
    /// Minimum degree prioritized during Core-First MaxMin pivot selection (`default: 2`).
    pub min_degree: usize,
    /// Deterministic PRNG seed controlling pivot selection and degenerate fallback (`default: 42`).
    pub seed: u64,
    /// Optional target global standard deviation enforced via [`normalize_layout_2d`] (`default: Some(1.0)`).
    pub target_std: Option<f64>,
}

impl Default for PivotMdsConfig {
    fn default() -> Self {
        Self {
            num_pivots: 50,
            min_degree: 2,
            seed: 42,
            target_std: Some(1.0),
        }
    }
}

/// Computes $d$-dimensional Representative / Landmark MDS coordinates
/// $(n / k)^{1/4} C V \mathrm{diag}(e^{-1/4})$ from a precomputed row-major `(num_nodes, num_reps)`
/// distance matrix using [`symmetric_eigen_jacobi`] with `e > 1e-10` zero-eigenvalue guards.
pub fn representative_mds<S: Scalar>(
    distances_to_reps: &[f64],
    num_nodes: usize,
    num_reps: usize,
    dims: usize,
) -> CoreResult<LayoutNd<S>> {
    if dims == 0 {
        return Err(CoreError::invalid_input("dims must be >= 1"));
    }
    let expected = num_nodes
        .checked_mul(num_reps)
        .ok_or_else(|| CoreError::invalid_input("representative_mds matrix size overflow"))?;
    if distances_to_reps.len() != expected {
        return Err(CoreError::dimension_mismatch(
            expected,
            distances_to_reps.len(),
            "representative_mds::distances_to_reps",
        ));
    }

    let mut layout = LayoutNd::zeros(num_nodes, dims)?;
    if num_nodes <= 1 || num_reps == 0 {
        return Ok(layout);
    }

    let c = double_center_distances(distances_to_reps, num_nodes, num_reps);
    let mut ct_c = vec![0.0_f64; num_reps * num_reps];
    for row in 0..num_nodes {
        let c_row = &c[row * num_reps..(row + 1) * num_reps];
        for a in 0..num_reps {
            let ca = c_row[a];
            for b in a..num_reps {
                ct_c[a * num_reps + b] += ca * c_row[b];
            }
        }
    }
    for a in 0..num_reps {
        for b in 0..a {
            ct_c[a * num_reps + b] = ct_c[b * num_reps + a];
        }
    }

    let (evals, evecs) = symmetric_eigen_jacobi(&ct_c, num_reps, 120, 1e-12)?;
    let used_dims = dims.min(num_reps);
    let scale = ((num_nodes as f64) / (num_reps as f64)).powf(0.25);

    let mut proj = vec![0.0_f64; num_reps * used_dims];
    for m in 0..used_dims {
        let eig_idx = num_reps - 1 - m;
        let eig_val = evals[eig_idx];
        let weight = if eig_val > 1e-10 {
            scale / eig_val.sqrt().sqrt()
        } else {
            0.0
        };
        for j in 0..num_reps {
            proj[j * used_dims + m] = weight * evecs[j * num_reps + eig_idx];
        }
    }

    let coords = layout.coords_mut();
    for i in 0..num_nodes {
        let c_row = &c[i * num_reps..(i + 1) * num_reps];
        let out_row = &mut coords[i * dims..i * dims + used_dims];
        for j in 0..num_reps {
            let cij = c_row[j];
            let p_row = &proj[j * used_dims..(j + 1) * used_dims];
            for m in 0..used_dims {
                out_row[m] = S::from_f64(out_row[m].to_f64() + cij * p_row[m]);
            }
        }
    }
    Ok(layout)
}

/// Computes a 2D [`Layout2d`] via PivotMDS from a precomputed row-major `(num_nodes, num_pivots)`
/// distance matrix, optionally normalizing isotropic standard deviation to `target_std`.
pub fn pivot_mds_from_distances<S: Scalar>(
    distances_nk: &[f64],
    num_nodes: usize,
    num_pivots: usize,
    target_std: Option<f64>,
    seed: u64,
) -> CoreResult<Layout2d<S>> {
    if let Some(t_std) = target_std {
        if !t_std.is_finite() {
            return Err(CoreError::non_finite("pivot_mds::target_std"));
        }
        if t_std <= 0.0 {
            return Err(CoreError::invalid_input(
                "target_std must be strictly positive",
            ));
        }
    }
    let nd = representative_mds::<f64>(distances_nk, num_nodes, num_pivots, 2)?;
    let mut x = Vec::with_capacity(num_nodes);
    let mut y = Vec::with_capacity(num_nodes);
    for i in 0..num_nodes {
        let row = nd.node(i)?;
        x.push(S::from_f64(row[0]));
        y.push(S::from_f64(row[1]));
    }
    let mut layout = Layout2d::from_xy(x, y)?;
    if let Some(t_std) = target_std {
        normalize_layout_2d(&mut layout, t_std, seed)?;
    }
    Ok(layout)
}

/// Computes a 2D [`Layout2d`] using Core-First MaxMin PivotMDS with `(n/k)^{1/4} C V diag(e^{-1/4})`
/// scaling, disconnected finite distance fallback, and optional isotropic std normalization.
pub fn pivot_mds<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    config: &PivotMdsConfig,
) -> CoreResult<Layout2d<S>> {
    if config.num_pivots == 0 {
        return Err(CoreError::invalid_input("num_pivots must be >= 1"));
    }
    if let Some(t_std) = config.target_std {
        if !t_std.is_finite() {
            return Err(CoreError::non_finite("PivotMdsConfig::target_std"));
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

    let pivots = select_maxmin_sp_pivots(graph, config.num_pivots, config.seed, config.min_degree)?;
    pivot_mds_with_pivots(graph, &pivots, config.target_std, config.seed)
}

/// Computes PivotMDS using an explicitly supplied slice of pivot vertex indices.
pub fn pivot_mds_with_pivots<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    pivots: &[u32],
    target_std: Option<f64>,
    seed: u64,
) -> CoreResult<Layout2d<S>> {
    if let Some(t_std) = target_std {
        if !t_std.is_finite() {
            return Err(CoreError::non_finite("pivot_mds_with_pivots::target_std"));
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
    let k = pivots.len();
    if k < 2 {
        let mut layout = Layout2d::zeros(n);
        if let Some(t_std) = target_std {
            normalize_layout_2d(&mut layout, t_std, seed)?;
        }
        return Ok(layout);
    }
    for &p in pivots {
        if (p as usize) >= n {
            return Err(CoreError::index_out_of_bounds(
                p as usize,
                n,
                "pivot_mds_with_pivots::pivots",
            ));
        }
    }

    let mut distances_nk = vec![f64::INFINITY; n * k];
    let mut dist_col = vec![f64::INFINITY; n];
    if graph.is_weighted() {
        let mut ws = DijkstraWorkspace::new(n);
        for (col, &p) in pivots.iter().enumerate() {
            ws.dijkstra_from(graph, p as usize, &mut dist_col)?;
            for i in 0..n {
                distances_nk[i * k + col] = dist_col[i];
            }
        }
    } else {
        let mut ws = BfsWorkspace::new(n);
        for (col, &p) in pivots.iter().enumerate() {
            ws.bfs_from(graph, p as usize, &mut dist_col)?;
            for i in 0..n {
                distances_nk[i * k + col] = dist_col[i];
            }
        }
    }

    pivot_mds_from_distances(&distances_nk, n, k, target_std, seed)
}

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

/// Computes zero-guarded Classical MDS from a flat row-major `(num_nodes, num_nodes)`
/// distance matrix via [`symmetric_eigen_jacobi`].
pub fn classical_mds_from_distances<S: Scalar>(
    distances_nn: &[f64],
    num_nodes: usize,
    dims: usize,
) -> CoreResult<LayoutNd<S>> {
    if dims == 0 {
        return Err(CoreError::invalid_input("dims must be >= 1"));
    }
    let expected = num_nodes
        .checked_mul(num_nodes)
        .ok_or_else(|| CoreError::invalid_input("classical_mds matrix size overflow"))?;
    if distances_nn.len() != expected {
        return Err(CoreError::dimension_mismatch(
            expected,
            distances_nn.len(),
            "classical_mds_from_distances::distances_nn",
        ));
    }

    let mut layout = LayoutNd::zeros(num_nodes, dims)?;
    if num_nodes <= 1 {
        return Ok(layout);
    }

    let b = double_center_distances(distances_nn, num_nodes, num_nodes);
    let (evals, evecs) = symmetric_eigen_jacobi(&b, num_nodes, 120, 1e-12)?;
    let used_dims = dims.min(num_nodes);

    let mut weights = vec![0.0_f64; used_dims];
    for m in 0..used_dims {
        let eig_val = evals[num_nodes - 1 - m];
        weights[m] = if eig_val > 1e-10 { eig_val.sqrt() } else { 0.0 };
    }

    let coords = layout.coords_mut();
    for i in 0..num_nodes {
        let out_row = &mut coords[i * dims..i * dims + used_dims];
        for m in 0..used_dims {
            let eig_idx = num_nodes - 1 - m;
            out_row[m] = S::from_f64(evecs[i * num_nodes + eig_idx] * weights[m]);
        }
    }
    Ok(layout)
}

/// Computes zero-guarded Classical MDS on `graph` via all-pairs shortest paths and
/// [`symmetric_eigen_jacobi`].
pub fn classical_mds<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    dims: usize,
) -> CoreResult<LayoutNd<S>> {
    if dims == 0 {
        return Err(CoreError::invalid_input("dims must be >= 1"));
    }
    let n = graph.num_nodes();
    if n <= 1 {
        return LayoutNd::zeros(n, dims);
    }

    let total = n
        .checked_mul(n)
        .ok_or_else(|| CoreError::invalid_input("classical_mds APSP size overflow"))?;
    let mut dists = vec![f64::INFINITY; total];
    if graph.is_weighted() {
        let mut ws = DijkstraWorkspace::new(n);
        for src in 0..n {
            ws.dijkstra_from(graph, src, &mut dists[src * n..(src + 1) * n])?;
        }
    } else {
        let mut ws = BfsWorkspace::new(n);
        for src in 0..n {
            ws.bfs_from(graph, src, &mut dists[src * n..(src + 1) * n])?;
        }
    }
    classical_mds_from_distances(&dists, n, dims)
}

/// Computes a 2D [`Layout2d`] using zero-guarded Classical MDS via [`symmetric_eigen_jacobi`].
pub fn classical_mds_2d<W: Scalar, S: Scalar>(graph: &CsrGraph<W>) -> CoreResult<Layout2d<S>> {
    let nd = classical_mds::<W, f64>(graph, 2)?;
    let n = graph.num_nodes();
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let row = nd.node(i)?;
        x.push(S::from_f64(row[0]));
        y.push(S::from_f64(row[1]));
    }
    Layout2d::from_xy(x, y)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn isotropic_std(layout: &Layout2d<f64>) -> f64 {
        let n = layout.len() as f64;
        let mx = layout.x().iter().sum::<f64>() / n;
        let my = layout.y().iter().sum::<f64>() / n;
        let sum_sq: f64 = layout
            .x()
            .iter()
            .zip(layout.y().iter())
            .map(|(&x, &y)| (x - mx).powi(2) + (y - my).powi(2))
            .sum();
        (sum_sq / (2.0 * n)).sqrt()
    }

    #[test]
    fn test_normalize_layout_2d_nan_collapsed_singleton_empty() {
        // NaN / Inf sanitization + exact isotropic std == 1.0
        let mut layout = Layout2d::from_xy(
            vec![f64::NAN, 10.0, 20.0, f64::INFINITY],
            vec![-5.0, f64::NEG_INFINITY, 15.0, 25.0],
        )
        .unwrap();
        normalize_layout_2d(&mut layout, 1.0, 42).unwrap();
        assert!(layout.is_all_finite());
        assert!((isotropic_std(&layout) - 1.0).abs() < 1e-12);

        // Collapsed layout falls back deterministically to Gaussian with target_std
        let mut collapsed = Layout2d::from_xy(vec![3.0; 8], vec![7.0; 8]).unwrap();
        normalize_layout_2d(&mut collapsed, 2.5, 7).unwrap();
        assert!(collapsed.is_all_finite());
        assert!((isotropic_std(&collapsed) - 2.5).abs() < 1e-12);

        // Singleton and empty
        let mut single = Layout2d::from_xy(vec![f64::NAN], vec![5.0]).unwrap();
        normalize_layout_2d(&mut single, 1.0, 42).unwrap();
        assert_eq!(single.x(), &[0.0]);
        assert_eq!(single.y(), &[0.0]);

        let mut empty = Layout2d::<f64>::zeros(0);
        normalize_layout_2d(&mut empty, 1.0, 42).unwrap();
    }

    #[test]
    fn test_pivot_mds_std_and_disconnected_safety() {
        // 8x8 grid (64 nodes)
        let mut edges = Vec::new();
        for r in 0..8 {
            for c in 0..8 {
                let u = (r * 8 + c) as u32;
                if c + 1 < 8 {
                    edges.push((u, u + 1));
                }
                if r + 1 < 8 {
                    edges.push((u, u + 8));
                }
            }
        }
        let grid = CsrGraph::<f64>::from_edges(64, &edges, None, false, false).unwrap();
        let layout: Layout2d<f64> = pivot_mds(&grid, &PivotMdsConfig::default()).unwrap();
        assert_eq!(layout.len(), 64);
        assert!(layout.is_all_finite());
        let std_val = isotropic_std(&layout);
        assert!(
            (std_val - 1.0).abs() < 1e-5,
            "PivotMds std expected 1.0, got {std_val}"
        );

        // Disconnected graph: two 4-cycles + 2 isolated vertices
        let disc = CsrGraph::<f64>::from_edges(
            10,
            &[
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
            ],
            None,
            false,
            false,
        )
        .unwrap();
        let disc_layout: Layout2d<f64> = pivot_mds(&disc, &PivotMdsConfig::default()).unwrap();
        assert!(disc_layout.is_all_finite());
        assert!((isotropic_std(&disc_layout) - 1.0).abs() < 1e-5);

        // Empty and singleton
        let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        assert_eq!(
            pivot_mds::<f64, f64>(&empty, &PivotMdsConfig::default())
                .unwrap()
                .len(),
            0
        );
        let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
        let s_lay: Layout2d<f64> = pivot_mds(&single, &PivotMdsConfig::default()).unwrap();
        assert_eq!(s_lay.x(), &[0.0]);
        assert_eq!(s_lay.y(), &[0.0]);
    }

    #[test]
    fn test_koren_spectral_init_and_rd_and_classical_mds() {
        // Cycle C_20 embeds cleanly on a circle under Koren spectral init
        let edges: Vec<(u32, u32)> = (0..20).map(|i| (i, (i + 1) % 20)).collect();
        let cycle = CsrGraph::<f64>::from_edges(20, &edges, None, false, false).unwrap();

        let koren: Layout2d<f64> = koren_spectral_init(&cycle, Some(1.0), 1e-6, 300, 42).unwrap();
        assert!(koren.is_all_finite());
        assert!((isotropic_std(&koren) - 1.0).abs() < 1e-5);
        // Adjacent cycle vertices should have uniform chord length in 2D spectral init
        let mut step_dists = Vec::new();
        for i in 0..20 {
            let j = (i + 1) % 20;
            let dx = koren.x()[i] - koren.x()[j];
            let dy = koren.y()[i] - koren.y()[j];
            step_dists.push(dx.hypot(dy));
        }
        let mean_d = step_dists.iter().sum::<f64>() / 20.0;
        for &d in &step_dists {
            assert!((d - mean_d).abs() < 1e-4);
        }

        // RdMds on disconnected graph must remain finite (zero-eigenvalue guard)
        let disc = CsrGraph::<f64>::from_edges(
            6,
            &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)],
            None,
            false,
            false,
        )
        .unwrap();
        let rd: LayoutNd<f64> = rd_mds(&disc, 3, 1e-10, 1e-6, 200, 42).unwrap();
        assert!(rd.is_all_finite());
        // First mode corresponds to lambda_1 == 0 across components -> safely zeroed out
        for i in 0..6 {
            assert_eq!(rd.node(i).unwrap()[0], 0.0);
        }

        // Classical MDS recovers exact Euclidean coordinates of a 4-vertex unit square
        let mut d_mat = vec![0.0; 16];
        let pts = [(0.0_f64, 0.0_f64), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        for i in 0..4 {
            for j in 0..4 {
                d_mat[i * 4 + j] = (pts[i].0 - pts[j].0).hypot(pts[i].1 - pts[j].1);
            }
        }
        let cmds: LayoutNd<f64> = classical_mds_from_distances(&d_mat, 4, 2).unwrap();
        assert!(cmds.is_all_finite());
        for i in 0..4 {
            for j in (i + 1)..4 {
                let p_i = cmds.node(i).unwrap();
                let p_j = cmds.node(j).unwrap();
                let emb_d = (p_i[0] - p_j[0]).hypot(p_i[1] - p_j[1]);
                assert!(
                    (emb_d - d_mat[i * 4 + j]).abs() < 1e-10,
                    "Classical MDS distance mismatch ({i}, {j}): {emb_d} vs {}",
                    d_mat[i * 4 + j]
                );
            }
        }

        // Graph-based classical_mds on disconnected graph stays finite
        let cmds_disc: LayoutNd<f64> = classical_mds(&disc, 2).unwrap();
        assert!(cmds_disc.is_all_finite());

        // Weighted PivotMds + f32 Layout2d output + RdMds builder ergonomics
        let wg = CsrGraph::<f32>::from_edges(
            4,
            &[(0, 1), (1, 2), (2, 3), (3, 0)],
            Some(&[1.0_f32, 2.0, 1.0, 2.0]),
            false,
            false,
        )
        .unwrap();
        let lay_f32: Layout2d<f32> = pivot_mds(&wg, &PivotMdsConfig::default()).unwrap();
        assert!(lay_f32.is_all_finite());
        let rd_2d: Layout2d<f32> = RdMds::new()
            .dims(2)
            .eig_eps(1e-10)
            .tol(1e-6)
            .max_iters(200)
            .seed(42)
            .degree_normalize(true)
            .target_std(Some(1.0))
            .embedding_2d(&wg)
            .unwrap();
        assert!(rd_2d.is_all_finite());

        // Edgeless graph under koren_spectral_init with target_std=Some(1.0) hits finite fallback
        let iso = CsrGraph::<f64>::from_edges(8, &[], None, false, false).unwrap();
        let k_iso: Layout2d<f64> = koren_spectral_init(&iso, Some(1.0), 1e-6, 100, 42).unwrap();
        assert!(k_iso.is_all_finite());
        assert!((isotropic_std(&k_iso) - 1.0).abs() < 1e-5);
    }
}
