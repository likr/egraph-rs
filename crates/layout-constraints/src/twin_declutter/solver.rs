use std::marker::PhantomData;

use egraph_core::{CoreResult, Layout2d, LayoutTransform, Scalar};
use egraph_spatial_trees::StaticKdTree;

use super::config::TwinDeclutterConfig;

/// Golden angle in radians: $\pi (3 - \sqrt{5})$.
const GOLDEN_ANGLE: f64 = 2.399_963_229_728_653_3;

/// Degenerate bounding-box diagonal threshold below which all nodes are treated as coincident.
const DEGENERATE_DIAG_EPS: f64 = 1e-12;

/// In-place 2D layout transform that resolves exact duplicate and near-twin node overlaps using a
/// 3x3 spatial hash, cluster-centroid `StaticKdTree` clearance queries, and golden-angle
/// Vogel spiral micro-offsets.
#[derive(Debug, Clone)]
pub struct TwinDeclutterer<S: Scalar = f64> {
    config: TwinDeclutterConfig,
    last_pass_count: usize,
    last_cluster_count: usize,
    last_moved_count: usize,
    _marker: PhantomData<S>,
}

impl<S: Scalar> TwinDeclutterer<S> {
    /// Creates a new [`TwinDeclutterer`] validated against `config`.
    pub fn new(config: TwinDeclutterConfig) -> CoreResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            last_pass_count: 0,
            last_cluster_count: 0,
            last_moved_count: 0,
            _marker: PhantomData,
        })
    }

    /// Returns the configuration used by this transform.
    #[inline]
    pub fn config(&self) -> &TwinDeclutterConfig {
        &self.config
    }

    /// Returns the number of decluttering passes executed in the most recent `apply` call.
    #[inline]
    pub fn last_pass_count(&self) -> usize {
        self.last_pass_count
    }

    /// Returns the number of twin clusters detected in the most recent pass.
    #[inline]
    pub fn last_cluster_count(&self) -> usize {
        self.last_cluster_count
    }

    /// Returns the number of vertices repositioned in the most recent pass.
    #[inline]
    pub fn last_moved_count(&self) -> usize {
        self.last_moved_count
    }
}

/// Runs twin/near-twin decluttering in-place on `layout`, returning the number of passes executed.
pub fn declutter_twins_2d<S: Scalar>(
    layout: &mut Layout2d<S>,
    config: &TwinDeclutterConfig,
) -> CoreResult<usize> {
    let mut declutterer = TwinDeclutterer::new(config.clone())?;
    declutterer.apply(layout)?;
    Ok(declutterer.last_pass_count())
}

struct UnionFind {
    parent: Vec<u32>,
    size: Vec<u32>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            size: vec![1; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] as usize != x {
            let p = self.parent[x] as usize;
            self.parent[x] = self.parent[p];
            x = p;
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.size[ra] < self.size[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra as u32;
        self.size[ra] += self.size[rb];
    }
}

#[inline]
fn canonical_f64<S: Scalar>(val: S) -> f64 {
    let f = val.to_f64();
    if f == 0.0 { 0.0 } else { f }
}

#[inline]
fn clamp_finite_scalar<S: Scalar>(val: f64) -> S {
    let max_abs = if std::mem::size_of::<S>() == 4 {
        f64::from(f32::MAX)
    } else {
        f64::MAX
    };
    S::from_f64(val.clamp(-max_abs, max_abs))
}

impl<S: Scalar> TwinDeclutterer<S> {
    fn run_single_pass(&mut self, layout: &mut Layout2d<S>) -> CoreResult<bool> {
        let n = layout.len();

        // Canonicalize `-0.0 -> +0.0` in f64 workspace and detect bounding box.
        let mut fx = Vec::with_capacity(n);
        let mut fy = Vec::with_capacity(n);
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        {
            let (xs_mut, ys_mut) = layout.xy_mut();
            for i in 0..n {
                if xs_mut[i].to_f64() == 0.0 && xs_mut[i].to_f64().is_sign_negative() {
                    xs_mut[i] = S::ZERO;
                }
                if ys_mut[i].to_f64() == 0.0 && ys_mut[i].to_f64().is_sign_negative() {
                    ys_mut[i] = S::ZERO;
                }
                let x = canonical_f64(xs_mut[i]);
                let y = canonical_f64(ys_mut[i]);
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
                fx.push(x);
                fy.push(y);
            }
        }

        let raw_diag = (max_x - min_x).hypot(max_y - min_y);
        // Scale down by 4.0 only when finite coordinates span across `[-1e308, +1e308]` so
        // `max_x - min_x` or `hypot` would overflow `f64::MAX`.
        let coord_scale = if raw_diag.is_finite() { 1.0 } else { 4.0 };
        if coord_scale != 1.0 {
            let inv_scale = 1.0 / coord_scale;
            for v in &mut fx {
                *v *= inv_scale;
            }
            for v in &mut fy {
                *v *= inv_scale;
            }
            min_x *= inv_scale;
            max_x *= inv_scale;
            min_y *= inv_scale;
            max_y *= inv_scale;
        }
        let diag = (max_x - min_x).hypot(max_y - min_y);
        if diag <= DEGENERATE_DIAG_EPS {
            // All nodes collapsed at a single point: fan out on a unit-radius Vogel spiral.
            // Divide before summing so large finite coordinates (e.g. `1e308`) never overflow.
            let inv_n = 1.0 / (n as f64);
            let cx_sum = fx.iter().map(|&v| v * inv_n).sum::<f64>();
            let cy_sum = fy.iter().map(|&v| v * inv_n).sum::<f64>();
            let cx = if cx_sum.is_finite() { cx_sum } else { fx[0] } * coord_scale;
            let cy = if cy_sum.is_finite() { cy_sum } else { fy[0] } * coord_scale;
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_unstable_by(|&a, &b| {
                fx[a]
                    .total_cmp(&fx[b])
                    .then_with(|| fy[a].total_cmp(&fy[b]))
                    .then_with(|| a.cmp(&b))
            });
            let (xs_mut, ys_mut) = layout.xy_mut();
            for (k, &vid) in order.iter().enumerate() {
                let theta = (k as f64) * GOLDEN_ANGLE;
                let r = (((k as f64) + 0.5) * inv_n).sqrt();
                xs_mut[vid] = clamp_finite_scalar(cx + r * theta.cos());
                ys_mut[vid] = clamp_finite_scalar(cy + r * theta.sin());
            }
            self.last_cluster_count = 1;
            self.last_moved_count = n;
            self.last_pass_count += 1;
            return Ok(true);
        }

        // Step 1: Lexicographic exact deduplication into unique coordinate points.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by(|&a, &b| {
            fx[a]
                .total_cmp(&fx[b])
                .then_with(|| fy[a].total_cmp(&fy[b]))
                .then_with(|| a.cmp(&b))
        });

        let mut ux = Vec::with_capacity(n);
        let mut uy = Vec::with_capacity(n);
        let mut vert_to_uniq = vec![0u32; n];
        for &vid in &order {
            let x = fx[vid];
            let y = fy[vid];
            if ux
                .last()
                .is_none_or(|&lx| lx != x || *uy.last().unwrap() != y)
            {
                ux.push(x);
                uy.push(y);
            }
            vert_to_uniq[vid] = (ux.len() - 1) as u32;
        }
        let n_uniq = ux.len();

        // Step 2: 3x3 spatial hash with 4 half-neighbors + Union-Find over unique points.
        let mut uf = UnionFind::new(n_uniq);
        let eps = self.config.eps_rel * diag;
        if eps > 0.0 && n_uniq >= 2 {
            let lo_bound = (i64::MIN + 2) as f64;
            let hi_bound = (i64::MAX - 2) as f64;
            let mut cells: Vec<(i64, i64, u32)> = Vec::with_capacity(n_uniq);
            for u in 0..n_uniq {
                let gx = ((ux[u] - min_x) / eps).floor().clamp(lo_bound, hi_bound) as i64;
                let gy = ((uy[u] - min_y) / eps).floor().clamp(lo_bound, hi_bound) as i64;
                cells.push((gy, gx, u as u32));
            }
            cells.sort_unstable();

            let mut runs: Vec<(i64, i64, usize, usize)> = Vec::new();
            let mut start = 0usize;
            while start < cells.len() {
                let (gy, gx, _) = cells[start];
                let mut end = start + 1;
                while end < cells.len() && cells[end].0 == gy && cells[end].1 == gx {
                    end += 1;
                }
                runs.push((gy, gx, start, end));
                start = end;
            }

            const HALF_NEIGHBORS: [(i64, i64); 4] = [(1, 0), (-1, 1), (0, 1), (1, 1)];
            for r_idx in 0..runs.len() {
                let (gy, gx, s0, e0) = runs[r_idx];
                // Intra-cell pairwise merges.
                for i in s0..e0 {
                    let ua = cells[i].2 as usize;
                    for &(_, _, ub_u32) in &cells[(i + 1)..e0] {
                        let ub = ub_u32 as usize;
                        if uf.find(ua) == uf.find(ub) {
                            continue;
                        }
                        if (ux[ua] - ux[ub]).hypot(uy[ua] - uy[ub]) <= eps {
                            uf.union(ua, ub);
                        }
                    }
                }
                // 4 forward half-neighbors in (gy, gx) lexicographic order.
                for (dx_off, dy_off) in HALF_NEIGHBORS {
                    let target = (gy + dy_off, gx + dx_off);
                    if let Ok(nbr_idx) =
                        runs.binary_search_by_key(&target, |&(ry, rx, _, _)| (ry, rx))
                    {
                        let (_, _, s1, e1) = runs[nbr_idx];
                        for &(_, _, ua_u32) in &cells[s0..e0] {
                            let ua = ua_u32 as usize;
                            for &(_, _, ub_u32) in &cells[s1..e1] {
                                let ub = ub_u32 as usize;
                                if uf.find(ua) == uf.find(ub) {
                                    continue;
                                }
                                if (ux[ua] - ux[ub]).hypot(uy[ua] - uy[ub]) <= eps {
                                    uf.union(ua, ub);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Step 3: Compact components into cluster IDs `0..n_clusters`.
        let mut root_to_cluster = vec![u32::MAX; n_uniq];
        let mut vert_cluster = vec![0u32; n];
        let mut cluster_sizes: Vec<usize> = Vec::new();
        for i in 0..n {
            let root = uf.find(vert_to_uniq[i] as usize);
            let cid = if root_to_cluster[root] == u32::MAX {
                let next_id = cluster_sizes.len() as u32;
                root_to_cluster[root] = next_id;
                cluster_sizes.push(0);
                next_id
            } else {
                root_to_cluster[root]
            };
            vert_cluster[i] = cid;
            cluster_sizes[cid as usize] += 1;
        }
        let n_clusters = cluster_sizes.len();
        self.last_cluster_count = n_clusters;

        let max_cluster_size = cluster_sizes.iter().copied().max().unwrap_or(0);
        if max_cluster_size < 2 {
            self.last_moved_count = 0;
            return Ok(true);
        }

        // Step 4: Group vertices by cluster and compute cluster centroids for StaticKdTree.
        // Divide before summing so large finite coordinates (e.g. `1e308`) never overflow.
        let mut cent_x = vec![0.0_f64; n_clusters];
        let mut cent_y = vec![0.0_f64; n_clusters];
        for i in 0..n {
            let c = vert_cluster[i] as usize;
            let inv = 1.0 / (cluster_sizes[c] as f64);
            cent_x[c] += fx[i] * inv;
            cent_y[c] += fy[i] * inv;
        }

        // Build flat CSR buckets over cluster members without nested Vec allocations.
        let mut offsets = vec![0usize; n_clusters + 1];
        for c in 0..n_clusters {
            offsets[c + 1] = offsets[c] + cluster_sizes[c];
        }
        let mut cursor = offsets[..n_clusters].to_vec();
        let mut members = vec![0usize; n];
        for (i, &c_u32) in vert_cluster.iter().enumerate() {
            let c = c_u32 as usize;
            members[cursor[c]] = i;
            cursor[c] += 1;
        }
        for c in 0..n_clusters {
            let rep = members[offsets[c]];
            if !cent_x[c].is_finite() {
                cent_x[c] = fx[rep];
            }
            if !cent_y[c].is_finite() {
                cent_y[c] = fy[rep];
            }
        }

        // Critical: StaticKdTree::nearest_excluding_self excludes only 1 index, so querying on
        // cluster centroids yields exact inter-cluster clearance `d_clear`.
        let centroids: Vec<[f64; 2]> = cent_x
            .iter()
            .zip(cent_y.iter())
            .map(|(&x, &y)| [x, y])
            .collect();
        let tree = StaticKdTree::<2, f64>::build(&centroids)?;

        let mut moved = 0usize;
        let (xs_mut, ys_mut) = layout.xy_mut();
        for c in 0..n_clusters {
            let start = offsets[c];
            let end = offsets[c + 1];
            let m = end - start;
            if m < 2 {
                continue;
            }
            let cx = cent_x[c];
            let cy = cent_y[c];
            let d_clear = match tree.nearest_excluding_self(c)? {
                Some((_, d2)) => d2.sqrt(),
                None => diag,
            };

            members[start..end].sort_unstable_by(|&a, &b| {
                let da_x = fx[a] - cx;
                let da_y = fy[a] - cy;
                let db_x = fx[b] - cx;
                let db_y = fy[b] - cy;
                let ang_a = da_y.atan2(da_x);
                let ang_b = db_y.atan2(db_x);
                let r2_a = da_x * da_x + da_y * da_y;
                let r2_b = db_x * db_x + db_y * db_y;
                ang_a
                    .total_cmp(&ang_b)
                    .then_with(|| r2_a.total_cmp(&r2_b))
                    .then_with(|| a.cmp(&b))
            });

            let r_need = self.config.min_pair_rel * diag * (m as f64).sqrt() / 1.5;
            let r_max =
                (self.config.r_cap_rel * diag).min((self.config.r_frac * d_clear).max(r_need));
            let inv_m = 1.0 / (m as f64);

            for (k, &vid) in members[start..end].iter().enumerate() {
                let theta = (k as f64) * GOLDEN_ANGLE;
                let r = r_max * (((k as f64) + 0.5) * inv_m).sqrt();
                xs_mut[vid] = clamp_finite_scalar((cx + r * theta.cos()) * coord_scale);
                ys_mut[vid] = clamp_finite_scalar((cy + r * theta.sin()) * coord_scale);
            }
            moved += m;
        }

        self.last_moved_count = moved;
        self.last_pass_count += 1;
        Ok(false)
    }
}

impl<S: Scalar> LayoutTransform for TwinDeclutterer<S> {
    type Target = Layout2d<S>;

    fn apply(&mut self, target: &mut Self::Target) -> CoreResult<()> {
        target.validate_finite()?;
        self.last_pass_count = 0;
        self.last_cluster_count = 0;
        self.last_moved_count = 0;

        let n = target.len();
        if n <= 1 {
            self.last_cluster_count = n;
            return Ok(());
        }

        for _ in 0..self.config.iterations {
            if self.run_single_pass(target)? {
                break;
            }
        }
        Ok(())
    }
}
