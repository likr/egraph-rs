use egraph_core::{CoreError, CoreResult, CsrGraph, Layout2d, LayoutNd, Scalar};
use egraph_linalg::eigen::symmetric_eigen_jacobi;
use egraph_linalg::shortest_path::{BfsWorkspace, DijkstraWorkspace};

/// Replaces infinite/non-finite entries in a squared-distance buffer with a bounded finite
/// fallback and applies double centering in-place:
/// $C = -\frac{1}{2}(I - \frac{1}{n}\mathbf{1}\mathbf{1}^T)\Delta(I - \frac{1}{k}\mathbf{1}\mathbf{1}^T)$.
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

    for (i, &rm) in row_means.iter().enumerate() {
        let row_off = i * ncols;
        for j in 0..ncols {
            delta[row_off + j] = 0.5 * (rm + col_means[j] - delta[row_off + j] - grand_mean);
        }
    }
    delta
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
