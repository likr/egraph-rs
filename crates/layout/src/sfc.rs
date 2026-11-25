//! Canonical community-ordered Space-Filling Curve (`SFC`) graph layout baseline.
//!
//! # Literature Note (`muelder2008_space_filling` `MISSING_PDF`)
//!
//! Muelder & Ma (*"Rapid Graph Layout Using Space Filling Curves"*, IEEE TVCG 2008) is currently
//! marked `MISSING_PDF` in the project literature archive, so exact clustering, ordering, and
//! post-refinement parameters from that paper cannot be verified or claimed. This module
//! implements a canonical **community-ordered Hilbert/Morton space-filling curve baseline**
//! powered by [`egraph_clustering::louvain_coarsen`] and strictly injective 2D curve mapping,
//! guaranteeing zero vertex co-location in both `f64` and `f32`.

use std::cmp::Ordering;
use std::collections::VecDeque;

use egraph_clustering::{CoarseLouvainConfig, louvain_coarsen};
use egraph_core::{
    CoreError, CoreResult, CsrGraph, IterativeSolver, Layout2d, Scalar, SolverStatus,
};

const MAX_SFC_ORDER: u32 = 15;

/// Space-filling curve mapping family for [`SfcLayoutSolver`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SfcCurveKind {
    /// Locality-preserving 2D Hilbert curve.
    #[default]
    Hilbert,
    /// Z-order (Morton) bit-interleaved 2D curve.
    Morton,
}

/// Hyperparameters for [`SfcLayoutSolver`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfcConfig {
    /// Space-filling curve family (`default: SfcCurveKind::Hilbert`).
    pub curve: SfcCurveKind,
    /// Target bounding square side length in `[-scale / 2, scale / 2]^2` (`default: 100.0`).
    pub scale: f64,
    /// Optional explicit curve resolution order $k \in [1, 15]$ yielding a $2^k \times 2^k$ grid
    /// (`default: None`, automatically choosing the smallest $k \ge 1$ with $2^{2k} \ge N$).
    pub resolution_order: Option<u32>,
    /// Deterministic PRNG seed passed to Louvain community coarsening (`default: 42`).
    pub seed: u64,
}

impl Default for SfcConfig {
    fn default() -> Self {
        Self {
            curve: SfcCurveKind::Hilbert,
            scale: 100.0,
            resolution_order: None,
            seed: 42,
        }
    }
}

impl SfcConfig {
    /// Validates all space-filling curve layout parameters.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.scale.is_finite() {
            return Err(CoreError::non_finite("SfcConfig::scale"));
        }
        if self.scale <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "SfcConfig::scale must be > 0.0, got {}",
                self.scale
            )));
        }
        if let Some(ord) = self.resolution_order
            && !(1..=MAX_SFC_ORDER).contains(&ord)
        {
            return Err(CoreError::invalid_input(format!(
                "SfcConfig::resolution_order must be in [1, {MAX_SFC_ORDER}], got {ord}"
            )));
        }
        Ok(())
    }
}

/// Cooperative community-ordered Space-Filling Curve (`SFC`) layout solver.
///
/// Precomputes a hierarchical Louvain community vertex ordering and maps ranks injectively onto
/// a $2^k \times 2^k$ Hilbert or Morton grid, writing the resulting coordinates in a single
/// [`IterativeSolver::step`] (`total_iterations() == 1`).
#[derive(Debug, Clone)]
pub struct SfcLayoutSolver<S: Scalar = f64> {
    config: SfcConfig,
    num_nodes: usize,
    vertex_order: Vec<u32>,
    coords_x: Vec<S>,
    coords_y: Vec<S>,
    iter: usize,
    status: SolverStatus,
}

impl<S: Scalar> SfcLayoutSolver<S> {
    /// Constructs an [`SfcLayoutSolver`] for `graph` using [`SfcConfig`].
    pub fn new<W: Scalar>(graph: &CsrGraph<W>, config: SfcConfig) -> CoreResult<Self> {
        config.validate()?;
        let n = graph.num_nodes();
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_csr(
                "SfcLayoutSolver node count exceeds u32::MAX",
            ));
        }
        if let Some(ws) = graph.weights() {
            for &w in ws {
                let wf = w.to_f64();
                if !wf.is_finite() {
                    return Err(CoreError::non_finite("SfcLayoutSolver::graph::weights"));
                }
                if wf < 0.0 {
                    return Err(CoreError::invalid_input(
                        "SfcLayoutSolver requires non-negative edge weights",
                    ));
                }
            }
        }

        let order = match config.resolution_order {
            Some(ord) => {
                let total_cells = 1u64 << (2 * ord);
                if (n as u64) > total_cells {
                    return Err(CoreError::invalid_input(format!(
                        "SfcConfig::resolution_order={ord} provides {total_cells} cells, insufficient for {n} vertices without co-location"
                    )));
                }
                ord
            }
            None => auto_order(n)?,
        };

        if n == 0 {
            return Ok(Self {
                config,
                num_nodes: 0,
                vertex_order: Vec::new(),
                coords_x: Vec::new(),
                coords_y: Vec::new(),
                iter: 0,
                status: SolverStatus::Initialized,
            });
        }
        if n == 1 {
            return Ok(Self {
                config,
                num_nodes: 1,
                vertex_order: vec![0],
                coords_x: vec![S::ZERO],
                coords_y: vec![S::ZERO],
                iter: 0,
                status: SolverStatus::Initialized,
            });
        }

        let undirected = canonical_undirected_graph(graph)?;
        let vertex_order = if undirected.num_edges() == 0 {
            (0..n as u32).collect()
        } else {
            compute_community_vertex_order(&undirected, config.seed)?
        };

        let side = 1u64 << order;
        let total_cells = side * side;
        let side_f = side as f64;
        let mut coords_x = vec![S::ZERO; n];
        let mut coords_y = vec![S::ZERO; n];

        for (rank, &v_u32) in vertex_order.iter().enumerate() {
            // Since total_cells >= n, d(rank) = floor(rank * total_cells / n) is strictly
            // increasing in rank, guaranteeing unique grid cells and zero co-location.
            let d = ((rank as u128 * (total_cells as u128)) / (n as u128)) as u64;
            let (ix, iy) = match config.curve {
                SfcCurveKind::Hilbert => hilbert_d2xy(order, d),
                SfcCurveKind::Morton => morton_d2xy(order, d),
            };
            let xf = (((ix as f64) + 0.5) / side_f - 0.5) * config.scale;
            let yf = (((iy as f64) + 0.5) / side_f - 0.5) * config.scale;
            let idx = v_u32 as usize;
            coords_x[idx] = S::from_f64(xf);
            coords_y[idx] = S::from_f64(yf);
        }

        Ok(Self {
            config,
            num_nodes: n,
            vertex_order,
            coords_x,
            coords_y,
            iter: 0,
            status: SolverStatus::Initialized,
        })
    }

    /// Returns the community-ordered permutation of vertex indices mapped along the curve.
    #[inline]
    pub fn vertex_order(&self) -> &[u32] {
        &self.vertex_order
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &SfcConfig {
        &self.config
    }

    /// Returns the number of vertices.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }
}

impl<S: Scalar> IterativeSolver<S> for SfcLayoutSolver<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "SfcLayoutSolver::step",
            ));
        }
        layout.validate_finite()?;

        if self.status.is_finished() {
            return Ok(self.status);
        }

        if self.num_nodes > 1 {
            let (xs_mut, ys_mut) = layout.xy_mut();
            xs_mut.copy_from_slice(&self.coords_x);
            ys_mut.copy_from_slice(&self.coords_y);
        }

        self.iter = 1;
        self.status = SolverStatus::Converged;
        Ok(self.status)
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        self.status
    }

    #[inline]
    fn iteration(&self) -> usize {
        self.iter
    }

    #[inline]
    fn total_iterations(&self) -> usize {
        1
    }
}

fn auto_order(n: usize) -> CoreResult<u32> {
    let target = (n.max(1)) as u64;
    for ord in 1..=MAX_SFC_ORDER {
        if (1u64 << (2 * ord)) >= target {
            return Ok(ord);
        }
    }
    Err(CoreError::invalid_input(format!(
        "graph node count {n} exceeds maximum SFC grid capacity 2^{}",
        2 * MAX_SFC_ORDER
    )))
}

#[inline]
fn hilbert_d2xy(order: u32, mut d: u64) -> (u64, u64) {
    let n = 1u64 << order;
    let mut x = 0u64;
    let mut y = 0u64;
    let mut s = 1u64;
    while s < n {
        let rx = (d >> 1) & 1;
        let ry = (d ^ rx) & 1;
        if ry == 0 {
            if rx == 1 {
                x = s - 1 - x;
                y = s - 1 - y;
            }
            std::mem::swap(&mut x, &mut y);
        }
        x += s * rx;
        y += s * ry;
        d >>= 2;
        s <<= 1;
    }
    (x, y)
}

#[inline]
fn morton_d2xy(order: u32, d: u64) -> (u64, u64) {
    let mut x = 0u64;
    let mut y = 0u64;
    for bit in 0..order {
        x |= ((d >> (2 * bit)) & 1) << bit;
        y |= ((d >> (2 * bit + 1)) & 1) << bit;
    }
    (x, y)
}

fn canonical_undirected_graph<W: Scalar>(graph: &CsrGraph<W>) -> CoreResult<CsrGraph<f64>> {
    let n = graph.num_nodes();
    let weights_opt = graph.weights();
    let mut raw: Vec<(u32, u32, f64)> = Vec::with_capacity(graph.nnz());

    for u in 0..n {
        let u32_u = u as u32;
        let start = graph.indptr()[u] as usize;
        let end = graph.indptr()[u + 1] as usize;
        for pos in start..end {
            let v = graph.indices()[pos];
            if u32_u == v {
                continue;
            }
            if !graph.is_directed() && u32_u > v {
                continue;
            }
            let w = weights_opt.map_or(1.0, |ws| ws[pos].to_f64());
            if w <= 0.0 {
                continue;
            }
            let (a, b) = if u32_u < v { (u32_u, v) } else { (v, u32_u) };
            raw.push((a, b, w));
        }
    }

    raw.sort_unstable_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)));

    let mut pairs = Vec::with_capacity(raw.len());
    let mut weights = Vec::with_capacity(raw.len());
    let mut idx = 0usize;
    while idx < raw.len() {
        let (a, b, first_w) = raw[idx];
        let mut sum_w = first_w;
        let mut cnt = 1usize;
        idx += 1;
        while idx < raw.len() && raw[idx].0 == a && raw[idx].1 == b {
            sum_w += raw[idx].2;
            cnt += 1;
            idx += 1;
        }
        pairs.push((a, b));
        weights.push(sum_w / (cnt as f64));
    }

    CsrGraph::from_edges(n, &pairs, Some(&weights), false, false)
}

fn compute_community_vertex_order(graph: &CsrGraph<f64>, seed: u64) -> CoreResult<Vec<u32>> {
    let n = graph.num_nodes();
    let hier = louvain_coarsen(
        graph,
        &CoarseLouvainConfig {
            seed,
            ..Default::default()
        },
    )?;

    let num_clusters = hier.coarsest.num_clusters;
    let coarse_graph = &hier.coarsest.graph;
    let coarse_sizes = &hier.coarsest.coarse_sizes;
    let node_to_coarsest = &hier.coarsest.node_to_coarse;
    let node_to_fine_cid = hier
        .levels
        .first()
        .map(|lvl| lvl.coarse.node_to_coarse.as_slice())
        .unwrap_or(node_to_coarsest.as_slice());

    // 1. Order coarsest clusters via BFS prioritizing largest clusters and strongest inter-cluster
    // edges so strongly connected communities sit adjacent along the space-filling curve.
    let mut cluster_seeds: Vec<usize> = (0..num_clusters).collect();
    cluster_seeds.sort_unstable_by(|&a, &b| {
        coarse_sizes[b]
            .cmp(&coarse_sizes[a])
            .then_with(|| a.cmp(&b))
    });

    let mut cluster_visited = vec![false; num_clusters];
    let mut coarse_rank = vec![0usize; num_clusters];
    let mut next_rank = 0usize;
    let mut c_queue = VecDeque::new();
    let mut c_nbrs: Vec<(u32, f64)> = Vec::new();
    let coarse_weights = coarse_graph.weights();

    for &seed_cid in &cluster_seeds {
        if cluster_visited[seed_cid] {
            continue;
        }
        cluster_visited[seed_cid] = true;
        c_queue.push_back(seed_cid);
        while let Some(c) = c_queue.pop_front() {
            coarse_rank[c] = next_rank;
            next_rank += 1;
            c_nbrs.clear();
            let start = coarse_graph.indptr()[c] as usize;
            let end = coarse_graph.indptr()[c + 1] as usize;
            for pos in start..end {
                let nc = coarse_graph.indices()[pos];
                if !cluster_visited[nc as usize] {
                    let w = coarse_weights.map_or(1.0, |ws| ws[pos]);
                    c_nbrs.push((nc, w));
                }
            }
            c_nbrs.sort_unstable_by(|a, b| {
                b.1.total_cmp(&a.1)
                    .then_with(|| coarse_sizes[b.0 as usize].cmp(&coarse_sizes[a.0 as usize]))
                    .then_with(|| a.0.cmp(&b.0))
            });
            for &(nc, _) in &c_nbrs {
                let nci = nc as usize;
                if !cluster_visited[nci] {
                    cluster_visited[nci] = true;
                    c_queue.push_back(nci);
                }
            }
        }
    }

    // 2. Group vertices by (coarse_rank[coarsest_cid], fine_cid), then order within each
    // coarsest community via intra-community BFS starting from the highest-degree node.
    let mut grouped: Vec<(usize, u32, usize, u32)> = (0..n)
        .map(|u| {
            let c_cid = node_to_coarsest[u] as usize;
            let f_cid = node_to_fine_cid[u];
            (coarse_rank[c_cid], f_cid, graph.degree(u), u as u32)
        })
        .collect();
    grouped.sort_unstable_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| b.2.cmp(&a.2))
            .then_with(|| a.3.cmp(&b.3))
    });

    let mut order = Vec::with_capacity(n);
    let mut visited = vec![false; n];
    let mut v_queue = VecDeque::new();
    let mut v_nbrs: Vec<(u32, f64)> = Vec::new();
    let fine_weights = graph.weights();

    let mut start = 0usize;
    while start < n {
        let cr = grouped[start].0;
        let mut end = start + 1;
        while end < n && grouped[end].0 == cr {
            end += 1;
        }
        for &(_, _, _, seed_u) in &grouped[start..end] {
            let su = seed_u as usize;
            if visited[su] {
                continue;
            }
            visited[su] = true;
            v_queue.push_back(su);
            while let Some(u) = v_queue.pop_front() {
                order.push(u as u32);
                let cu = node_to_coarsest[u];
                v_nbrs.clear();
                let s = graph.indptr()[u] as usize;
                let e = graph.indptr()[u + 1] as usize;
                for pos in s..e {
                    let v = graph.indices()[pos];
                    let vi = v as usize;
                    if !visited[vi] && node_to_coarsest[vi] == cu {
                        let w = fine_weights.map_or(1.0, |ws| ws[pos]);
                        v_nbrs.push((v, w));
                    }
                }
                v_nbrs.sort_unstable_by(|a, b| match b.1.total_cmp(&a.1) {
                    Ordering::Equal => graph
                        .degree(b.0 as usize)
                        .cmp(&graph.degree(a.0 as usize))
                        .then_with(|| a.0.cmp(&b.0)),
                    other => other,
                });
                for &(v, _) in &v_nbrs {
                    let vi = v as usize;
                    if !visited[vi] {
                        visited[vi] = true;
                        v_queue.push_back(vi);
                    }
                }
            }
        }
        start = end;
    }

    Ok(order)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_strictly_distinct<S: Scalar>(layout: &Layout2d<S>) {
        let n = layout.len();
        let mut pts: Vec<(u64, u64)> = (0..n)
            .map(|i| {
                (
                    layout.x()[i].to_f64().to_bits(),
                    layout.y()[i].to_f64().to_bits(),
                )
            })
            .collect();
        pts.sort_unstable();
        for w in pts.windows(2) {
            assert_ne!(w[0], w[1], "co-located points found in SFC layout");
        }
    }

    #[test]
    fn test_sfc_config_validation_and_degenerate_inputs() {
        assert!(SfcConfig::default().validate().is_ok());
        assert!(matches!(
            SfcConfig {
                scale: 0.0,
                ..Default::default()
            }
            .validate(),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            SfcConfig {
                scale: f64::NAN,
                ..Default::default()
            }
            .validate(),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            SfcConfig {
                resolution_order: Some(0),
                ..Default::default()
            }
            .validate(),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            SfcConfig {
                resolution_order: Some(16),
                ..Default::default()
            }
            .validate(),
            Err(CoreError::InvalidInput(_))
        ));

        // Explicit resolution_order too small for N -> error.
        let g5 = CsrGraph::<f64>::from_edges(5, &[(0, 1)], None, false, false).unwrap();
        assert!(matches!(
            SfcLayoutSolver::<f64>::new(
                &g5,
                SfcConfig {
                    resolution_order: Some(1), // 4 cells < 5 nodes
                    ..Default::default()
                },
            ),
            Err(CoreError::InvalidInput(_))
        ));

        // N = 0 and N = 1.
        let g0 = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        let mut s0 = SfcLayoutSolver::<f64>::new(&g0, SfcConfig::default()).unwrap();
        let mut l0 = Layout2d::zeros(0);
        assert_eq!(s0.step(&mut l0).unwrap(), SolverStatus::Converged);

        let g1 = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
        let mut s1 = SfcLayoutSolver::<f64>::new(&g1, SfcConfig::default()).unwrap();
        let mut l1 = Layout2d::zeros(1);
        assert_eq!(s1.step(&mut l1).unwrap(), SolverStatus::Converged);
        assert_eq!(s1.vertex_order(), &[0]);

        // Edgeless graph (N = 10).
        let g_edgeless = CsrGraph::<f64>::from_edges(10, &[], None, false, false).unwrap();
        let mut s_edgeless =
            SfcLayoutSolver::<f64>::new(&g_edgeless, SfcConfig::default()).unwrap();
        let mut l_edgeless = Layout2d::zeros(10);
        assert_eq!(
            s_edgeless.run(&mut l_edgeless).unwrap(),
            SolverStatus::Converged
        );
        assert!(l_edgeless.is_all_finite());
        assert_strictly_distinct(&l_edgeless);
    }

    #[test]
    fn test_sfc_hilbert_morton_zero_colocation_and_step_run_parity() {
        // Two cliques connected by a weak bridge + disconnected isolated node.
        let mut edges = Vec::new();
        let mut weights = Vec::new();
        for i in 0..5 {
            for j in (i + 1)..5 {
                edges.push((i as u32, j as u32));
                weights.push(3.0);
            }
        }
        for i in 5..10 {
            for j in (i + 1)..10 {
                edges.push((i as u32, j as u32));
                weights.push(2.5);
            }
        }
        edges.push((4, 5));
        weights.push(0.1);
        // N = 12 (nodes 10, 11 isolated).
        let g = CsrGraph::from_edges(12, &edges, Some(&weights), false, false).unwrap();

        for &curve in &[SfcCurveKind::Hilbert, SfcCurveKind::Morton] {
            let cfg = SfcConfig {
                curve,
                scale: 50.0,
                resolution_order: None,
                seed: 42,
            };
            // f64 step() == run()
            let mut s1 = SfcLayoutSolver::<f64>::new(&g, cfg).unwrap();
            let mut s2 = SfcLayoutSolver::<f64>::new(&g, cfg).unwrap();
            let mut l1 = Layout2d::zeros(12);
            let mut l2 = Layout2d::zeros(12);
            assert_eq!(s1.step(&mut l1).unwrap(), SolverStatus::Converged);
            assert_eq!(s2.run(&mut l2).unwrap(), SolverStatus::Converged);
            assert_strictly_distinct(&l1);
            assert_eq!(
                l1.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                l2.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(
                l1.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                l2.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );

            // Permutation check on vertex_order.
            let mut sorted_order = s1.vertex_order().to_vec();
            sorted_order.sort_unstable();
            assert_eq!(sorted_order, (0..12).collect::<Vec<u32>>());

            // f32 step() == run() & zero co-location.
            let mut s32_a = SfcLayoutSolver::<f32>::new(&g, cfg).unwrap();
            let mut s32_b = SfcLayoutSolver::<f32>::new(&g, cfg).unwrap();
            let mut l32_a = Layout2d::<f32>::zeros(12);
            let mut l32_b = Layout2d::<f32>::zeros(12);
            assert_eq!(s32_a.step(&mut l32_a).unwrap(), SolverStatus::Converged);
            assert_eq!(s32_b.run(&mut l32_b).unwrap(), SolverStatus::Converged);
            assert_strictly_distinct(&l32_a);
            assert_eq!(
                l32_a.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                l32_b.x().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(
                l32_a.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                l32_b.y().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }

        // Directed graph with reciprocal edges and zero-weight edge.
        let g_dir = CsrGraph::from_edges(
            6,
            &[(0, 1), (1, 0), (1, 2), (2, 0), (3, 4), (4, 5), (0, 3)],
            Some(&[1.0, 2.0, 1.5, 1.0, 2.0, 2.0, 0.0]),
            true,
            false,
        )
        .unwrap();
        let mut s_dir = SfcLayoutSolver::<f64>::new(&g_dir, SfcConfig::default()).unwrap();
        let mut l_dir = Layout2d::zeros(6);
        assert_eq!(s_dir.run(&mut l_dir).unwrap(), SolverStatus::Converged);
        assert!(l_dir.is_all_finite());
        assert_strictly_distinct(&l_dir);
    }
}
