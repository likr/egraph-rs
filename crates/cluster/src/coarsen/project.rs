use egraph_core::{CoreError, CoreResult, DeterministicRng, Layout2d, Scalar};

/// Projects a 2D layout of coarse super-nodes onto the fine vertices specified by
/// `node_to_coarse`, adding deterministic disk offsets within `spread_radius` for
/// multi-member clusters to avoid exact coordinate overlap.
pub fn project_clustered_layout<S: Scalar>(
    coarse_layout: &Layout2d<S>,
    node_to_coarse: &[u32],
    spread_radius: S,
    seed: u64,
) -> CoreResult<Layout2d<S>> {
    let r = spread_radius.to_f64();
    if !r.is_finite() {
        return Err(CoreError::non_finite(
            "project_clustered_layout::spread_radius",
        ));
    }
    if r < 0.0 {
        return Err(CoreError::invalid_input(
            "spread_radius must be non-negative",
        ));
    }
    coarse_layout.validate_finite()?;

    let n = node_to_coarse.len();
    if n == 0 {
        return Ok(Layout2d::zeros(0));
    }

    let num_clusters = coarse_layout.len();
    let mut cluster_sizes = vec![0usize; num_clusters];
    for &c in node_to_coarse {
        let cid = c as usize;
        if cid >= num_clusters {
            return Err(CoreError::index_out_of_bounds(
                cid,
                num_clusters,
                "project_clustered_layout::node_to_coarse",
            ));
        }
        cluster_sizes[cid] += 1;
    }

    let mut rng = DeterministicRng::from_seed_u64(seed);
    let cx = coarse_layout.x();
    let cy = coarse_layout.y();
    let mut fine_x = Vec::with_capacity(n);
    let mut fine_y = Vec::with_capacity(n);

    for &c in node_to_coarse {
        let cid = c as usize;
        let base_x = cx[cid].to_f64();
        let base_y = cy[cid].to_f64();
        if r > 0.0 && cluster_sizes[cid] > 1 {
            let angle = std::f64::consts::TAU * rng.next_f64();
            let rad = r * rng.next_f64().sqrt();
            let (sin_a, cos_a) = angle.sin_cos();
            fine_x.push(S::from_f64(base_x + rad * cos_a));
            fine_y.push(S::from_f64(base_y + rad * sin_a));
        } else {
            fine_x.push(S::from_f64(base_x));
            fine_y.push(S::from_f64(base_y));
        }
    }

    Layout2d::from_xy(fine_x, fine_y)
}

/// Projects a 2D layout of coarse super-nodes onto fine vertices using per-cluster
/// reserved radii (`jitter_scale * cluster_radii[cid]`) for deterministic initial disk jitter.
pub fn project_clustered_layout_with_radii<S: Scalar>(
    coarse_layout: &Layout2d<S>,
    node_to_coarse: &[u32],
    cluster_radii: &[f64],
    jitter_scale: f64,
    seed: u64,
) -> CoreResult<Layout2d<S>> {
    if !jitter_scale.is_finite() {
        return Err(CoreError::non_finite(
            "project_clustered_layout_with_radii::jitter_scale",
        ));
    }
    if jitter_scale < 0.0 {
        return Err(CoreError::invalid_input(
            "jitter_scale must be non-negative",
        ));
    }
    coarse_layout.validate_finite()?;

    let num_clusters = coarse_layout.len();
    if cluster_radii.len() != num_clusters {
        return Err(CoreError::dimension_mismatch(
            num_clusters,
            cluster_radii.len(),
            "project_clustered_layout_with_radii::cluster_radii",
        ));
    }
    for &rad in cluster_radii {
        if !rad.is_finite() {
            return Err(CoreError::non_finite(
                "project_clustered_layout_with_radii::cluster_radii",
            ));
        }
        if rad < 0.0 {
            return Err(CoreError::invalid_input(
                "cluster_radii entries must be non-negative",
            ));
        }
    }

    let n = node_to_coarse.len();
    if n == 0 {
        return Ok(Layout2d::zeros(0));
    }

    let mut cluster_sizes = vec![0usize; num_clusters];
    for &c in node_to_coarse {
        let cid = c as usize;
        if cid >= num_clusters {
            return Err(CoreError::index_out_of_bounds(
                cid,
                num_clusters,
                "project_clustered_layout_with_radii::node_to_coarse",
            ));
        }
        cluster_sizes[cid] += 1;
    }

    let mut rng = DeterministicRng::from_seed_u64(seed);
    let cx = coarse_layout.x();
    let cy = coarse_layout.y();
    let mut fine_x = Vec::with_capacity(n);
    let mut fine_y = Vec::with_capacity(n);

    for &c in node_to_coarse {
        let cid = c as usize;
        let base_x = cx[cid].to_f64();
        let base_y = cy[cid].to_f64();
        let r = jitter_scale * cluster_radii[cid];
        if r > 0.0 && cluster_sizes[cid] > 1 {
            let angle = std::f64::consts::TAU * rng.next_f64();
            let rad = r * rng.next_f64().sqrt();
            let (sin_a, cos_a) = angle.sin_cos();
            fine_x.push(S::from_f64(base_x + rad * cos_a));
            fine_y.push(S::from_f64(base_y + rad * sin_a));
        } else {
            fine_x.push(S::from_f64(base_x));
            fine_y.push(S::from_f64(base_y));
        }
    }

    Layout2d::from_xy(fine_x, fine_y)
}

/// Aligns fine-level cluster centroids toward coarse super-node positions via rigid
/// per-cluster translation $x_i \leftarrow x_i + \gamma_{\mathrm{tether}} (x_C - \bar{x}_C)$
/// without rigid disk containment, preserving anisotropic intra-cluster geometry.
pub fn align_clustered_centroids<S: Scalar>(
    fine_layout: &mut Layout2d<S>,
    coarse_layout: &Layout2d<S>,
    node_to_coarse: &[u32],
    gamma_tether: f64,
) -> CoreResult<()> {
    if !gamma_tether.is_finite() {
        return Err(CoreError::non_finite(
            "align_clustered_centroids::gamma_tether",
        ));
    }
    if !(0.0..=1.0).contains(&gamma_tether) {
        return Err(CoreError::invalid_input(format!(
            "gamma_tether must be in [0.0, 1.0], got {gamma_tether}"
        )));
    }
    fine_layout.validate_finite()?;
    coarse_layout.validate_finite()?;

    let n = fine_layout.len();
    if node_to_coarse.len() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            node_to_coarse.len(),
            "align_clustered_centroids::node_to_coarse",
        ));
    }
    let num_clusters = coarse_layout.len();
    for &c in node_to_coarse {
        let cid = c as usize;
        if cid >= num_clusters {
            return Err(CoreError::index_out_of_bounds(
                cid,
                num_clusters,
                "align_clustered_centroids::node_to_coarse",
            ));
        }
    }
    if n == 0 || gamma_tether == 0.0 {
        return Ok(());
    }

    let mut sum_x = vec![0.0_f64; num_clusters];
    let mut sum_y = vec![0.0_f64; num_clusters];
    let mut counts = vec![0usize; num_clusters];

    {
        let fx = fine_layout.x();
        let fy = fine_layout.y();
        for i in 0..n {
            let cid = node_to_coarse[i] as usize;
            sum_x[cid] += fx[i].to_f64();
            sum_y[cid] += fy[i].to_f64();
            counts[cid] += 1;
        }
    }

    let cx = coarse_layout.x();
    let cy = coarse_layout.y();
    let mut shift_x = vec![0.0_f64; num_clusters];
    let mut shift_y = vec![0.0_f64; num_clusters];
    for cid in 0..num_clusters {
        if counts[cid] > 0 {
            let inv_cnt = 1.0 / (counts[cid] as f64);
            let mean_x = sum_x[cid] * inv_cnt;
            let mean_y = sum_y[cid] * inv_cnt;
            shift_x[cid] = gamma_tether * (cx[cid].to_f64() - mean_x);
            shift_y[cid] = gamma_tether * (cy[cid].to_f64() - mean_y);
        }
    }

    let (fx, fy) = fine_layout.xy_mut();
    for i in 0..n {
        let cid = node_to_coarse[i] as usize;
        fx[i] = S::from_f64(fx[i].to_f64() + shift_x[cid]);
        fy[i] = S::from_f64(fy[i].to_f64() + shift_y[cid]);
    }
    Ok(())
}
