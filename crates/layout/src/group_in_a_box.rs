//! Group-in-a-Box (GIB) clustered graph layout (`TR-GIB` and `FD-GIB`, Rodrigues et al., 2011)
//! and Bruls–Huizing–van Wijk (2000) Squarified Treemap partitioning.
//!
//! Partitions the 2D canvas into disjoint rectangular regions—one per community—using either
//! Bruls et al. (2000) squarified treemaps (`GibBoxLayoutMode::SquarifiedTreemap`) or force-directed
//! inter-cluster box placement with strict overlap removal (`GibBoxLayoutMode::ForceDirected`), then
//! cooperatively lays out each cluster's induced subgraph inside its padded [`GibBox`] with uniform
//! aspect-preserving scaling.

use egraph_core::{
    CoreError, CoreResult, CsrGraph, IterativeSolver, Layout2d, Scalar, SolverStatus,
};
use egraph_spatial::{Rect2d, RectangleOverlapSweep};

use crate::constraints::project_rectangle_no_overlap_2d;
use crate::continuous_fdp::{YifanHuConfig, YifanHuSolver};
use crate::fr::{FruchtermanReingoldConfig, FruchtermanReingoldSolver};
use crate::sgd::{SgdScheduleConfig, SgdSolver};

const GEOM_EPS: f64 = 1e-12;
const GOLDEN_ANGLE: f64 = 2.399_963_229_728_653;

/// Box placement strategy for [`GroupInABoxSolver`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GibBoxLayoutMode {
    /// Rodrigues et al. (2011) `TR-GIB`: Bruls et al. (2000) squarified treemap partitioning.
    #[default]
    SquarifiedTreemap,
    /// `FD-GIB`: force-directed inter-cluster box placement followed by strict overlap removal.
    ForceDirected,
}

/// Intra-cluster subgraph layout solver used inside each [`GibBox`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GibIntraLayoutMode {
    /// Fruchterman–Reingold force-directed layout (`FruchtermanReingoldSolver`).
    #[default]
    Fr,
    /// Full-APSP Stochastic Gradient Descent (`SgdSolver::full`).
    SgdFull,
    /// Multilevel Yifan Hu force-directed layout (`YifanHuSolver`).
    YifanHu,
}

/// Configuration for [`GroupInABoxSolver`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroupInABoxConfig {
    /// Cluster box placement strategy (`default: GibBoxLayoutMode::SquarifiedTreemap`).
    pub box_mode: GibBoxLayoutMode,
    /// Intra-cluster subgraph layout solver (`default: GibIntraLayoutMode::Fr`).
    pub intra_mode: GibIntraLayoutMode,
    /// Canvas width (`> 0.0`, `default: 100.0`).
    pub total_width: f64,
    /// Canvas height (`> 0.0`, `default: 100.0`).
    pub total_height: f64,
    /// Relative inner padding fraction per side in `[0.0, 0.5)` (`default: 0.05`).
    pub box_padding: f64,
    /// Maximum cooperative intra-box layout iterations (`default: 100`).
    pub intra_iterations: usize,
    /// Number of inter-cluster force-directed iterations in `ForceDirected` mode (`default: 100`).
    pub fd_iterations: usize,
    /// Deterministic explicit RNG seed (`default: 42`).
    pub seed: u64,
}

impl Default for GroupInABoxConfig {
    fn default() -> Self {
        Self {
            box_mode: GibBoxLayoutMode::SquarifiedTreemap,
            intra_mode: GibIntraLayoutMode::Fr,
            total_width: 100.0,
            total_height: 100.0,
            box_padding: 0.05,
            intra_iterations: 100,
            fd_iterations: 100,
            seed: 42,
        }
    }
}

impl GroupInABoxConfig {
    /// Validates all Group-in-a-Box hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.total_width.is_finite() {
            return Err(CoreError::non_finite("GroupInABoxConfig::total_width"));
        }
        if self.total_width <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "total_width must be > 0.0, got {}",
                self.total_width
            )));
        }
        if !self.total_height.is_finite() {
            return Err(CoreError::non_finite("GroupInABoxConfig::total_height"));
        }
        if self.total_height <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "total_height must be > 0.0, got {}",
                self.total_height
            )));
        }
        if !self.box_padding.is_finite() {
            return Err(CoreError::non_finite("GroupInABoxConfig::box_padding"));
        }
        if !(0.0..0.5).contains(&self.box_padding) {
            return Err(CoreError::invalid_input(format!(
                "box_padding must be in [0.0, 0.5), got {}",
                self.box_padding
            )));
        }
        Ok(())
    }
}

/// Axis-aligned cluster bounding box produced by [`GroupInABoxSolver`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GibBox {
    /// Original cluster label assigned to this box.
    pub cluster_id: u32,
    /// Minimum x-coordinate of the box.
    pub x_min: f64,
    /// Maximum x-coordinate of the box.
    pub x_max: f64,
    /// Minimum y-coordinate of the box.
    pub y_min: f64,
    /// Maximum y-coordinate of the box.
    pub y_max: f64,
    /// Number of vertices in this cluster.
    pub node_count: usize,
    /// Positive weight used to size this box's area.
    pub weight: f64,
}

/// Computes a Bruls–Huizing–van Wijk (2000) squarified treemap partitioning of rectangle
/// `[x, x + width] x [y, y + height]` proportional to `weights`.
///
/// Each returned rectangle `[x_min, x_max, y_min, y_max]` corresponds to the same index as
/// `weights` (even though items are internally sorted descending by weight during squarification).
///
/// Returns [`CoreError`] if `weights` is empty, any weight is non-finite or `<= 0.0`, or any
/// rectangle parameter is non-finite or `width <= 0.0` / `height <= 0.0`.
pub fn squarified_treemap(
    weights: &[f64],
    x: f64,
    y: f64,
    width: f64,
    height: f64,
) -> CoreResult<Vec<[f64; 4]>> {
    if !x.is_finite() || !y.is_finite() || !width.is_finite() || !height.is_finite() {
        return Err(CoreError::non_finite("squarified_treemap::bounds"));
    }
    if width <= 0.0 || height <= 0.0 {
        return Err(CoreError::invalid_input(format!(
            "squarified_treemap requires width > 0.0 and height > 0.0, got ({width}, {height})"
        )));
    }
    if weights.is_empty() {
        return Err(CoreError::invalid_input(
            "squarified_treemap requires non-empty weights",
        ));
    }

    let mut total_weight = 0.0_f64;
    for (idx, &w) in weights.iter().enumerate() {
        if !w.is_finite() {
            return Err(CoreError::non_finite("squarified_treemap::weights"));
        }
        if w <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "squarified_treemap requires all weights > 0.0, got {w} at index {idx}"
            )));
        }
        total_weight += w;
    }
    if !total_weight.is_finite() || total_weight <= 0.0 {
        return Err(CoreError::invalid_input(
            "squarified_treemap total weight must be positive and finite",
        ));
    }

    let total_area = width * height;
    if !total_area.is_finite() || total_area <= 0.0 {
        return Err(CoreError::invalid_input(
            "squarified_treemap rectangle area must be positive and finite",
        ));
    }

    let k = weights.len();
    let mut items: Vec<(f64, usize)> = weights
        .iter()
        .enumerate()
        .map(|(i, &w)| (total_area * (w / total_weight), i))
        .collect();
    items.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

    let mut out = vec![[0.0_f64; 4]; k];
    let mut rx = x;
    let mut ry = y;
    let mut rw = width;
    let mut rh = height;

    let mut row: Vec<(f64, usize)> = Vec::new();
    let mut row_sum = 0.0_f64;

    for &(area, orig_idx) in &items {
        let short_side = rw.min(rh);
        if row.is_empty() {
            row.push((area, orig_idx));
            row_sum = area;
        } else {
            let cur_worst = treemap_worst(&row, row_sum, short_side);
            let next_worst = treemap_worst_with(&row, row_sum, area, short_side);
            if cur_worst >= next_worst {
                row.push((area, orig_idx));
                row_sum += area;
            } else {
                layout_treemap_row(
                    &row, row_sum, false, &mut rx, &mut ry, &mut rw, &mut rh, &mut out,
                );
                row.clear();
                row.push((area, orig_idx));
                row_sum = area;
            }
        }
    }

    if !row.is_empty() {
        layout_treemap_row(
            &row, row_sum, true, &mut rx, &mut ry, &mut rw, &mut rh, &mut out,
        );
    }

    Ok(out)
}

#[inline]
fn treemap_worst(row: &[(f64, usize)], sum: f64, side_len: f64) -> f64 {
    if row.is_empty() || sum <= 0.0 || side_len <= 0.0 {
        return f64::INFINITY;
    }
    let r_max = row[0].0;
    let r_min = row[row.len() - 1].0;
    let w2 = side_len * side_len;
    let s2 = sum * sum;
    ((w2 * r_max) / s2).max(s2 / (w2 * r_min))
}

#[inline]
fn treemap_worst_with(row: &[(f64, usize)], sum: f64, candidate_area: f64, side_len: f64) -> f64 {
    if side_len <= 0.0 {
        return f64::INFINITY;
    }
    let new_sum = sum + candidate_area;
    let r_max = if row.is_empty() {
        candidate_area
    } else {
        row[0].0.max(candidate_area)
    };
    let r_min = if row.is_empty() {
        candidate_area
    } else {
        row[row.len() - 1].0.min(candidate_area)
    };
    let w2 = side_len * side_len;
    let s2 = new_sum * new_sum;
    ((w2 * r_max) / s2).max(s2 / (w2 * r_min))
}

#[allow(clippy::too_many_arguments)]
fn layout_treemap_row(
    row: &[(f64, usize)],
    row_sum: f64,
    is_final_row: bool,
    rx: &mut f64,
    ry: &mut f64,
    rw: &mut f64,
    rh: &mut f64,
    out: &mut [[f64; 4]],
) {
    if *rw <= *rh {
        let strip_h = if is_final_row {
            *rh
        } else {
            let max_h = *rh * (1.0 - 1e-12);
            (row_sum / *rw).min(max_h).max(1e-15 * *rh)
        };
        let x_end = *rx + *rw;
        let mut x_cursor = *rx;
        for (idx, &(area, orig_idx)) in row.iter().enumerate() {
            let cell_x_max = if idx + 1 == row.len() {
                x_end
            } else {
                (x_cursor + area / strip_h).min(x_end)
            };
            out[orig_idx] = [x_cursor, cell_x_max, *ry, *ry + strip_h];
            x_cursor = cell_x_max;
        }
        *ry += strip_h;
        *rh = (*rh - strip_h).max(0.0);
    } else {
        let strip_w = if is_final_row {
            *rw
        } else {
            let max_w = *rw * (1.0 - 1e-12);
            (row_sum / *rh).min(max_w).max(1e-15 * *rw)
        };
        let y_end = *ry + *rh;
        let mut y_cursor = *ry;
        for (idx, &(area, orig_idx)) in row.iter().enumerate() {
            let cell_y_max = if idx + 1 == row.len() {
                y_end
            } else {
                (y_cursor + area / strip_w).min(y_end)
            };
            out[orig_idx] = [*rx, *rx + strip_w, y_cursor, cell_y_max];
            y_cursor = cell_y_max;
        }
        *rx += strip_w;
        *rw = (*rw - strip_w).max(0.0);
    }
}

enum IntraSolver {
    Trivial,
    Fr(Box<FruchtermanReingoldSolver<f64>>),
    Sgd(Box<SgdSolver<f64>>),
    YifanHu(Box<YifanHuSolver<f64>>),
}

impl IntraSolver {
    fn is_finished(&self) -> bool {
        match self {
            Self::Trivial => true,
            Self::Fr(s) => s.status().is_finished(),
            Self::Sgd(s) => s.status().is_finished(),
            Self::YifanHu(s) => s.status().is_finished(),
        }
    }

    fn step(&mut self, layout: &mut Layout2d<f64>) -> CoreResult<()> {
        match self {
            Self::Trivial => Ok(()),
            Self::Fr(s) => {
                s.step(layout)?;
                Ok(())
            }
            Self::Sgd(s) => {
                s.step(layout)?;
                Ok(())
            }
            Self::YifanHu(s) => {
                s.step(layout)?;
                Ok(())
            }
        }
    }
}

struct ClusterState {
    nodes: Vec<u32>,
    has_edges: bool,
    sub_layout: Layout2d<f64>,
    solver: IntraSolver,
    initialized: bool,
}

/// Cooperative Group-in-a-Box (GIB) graph layout solver implementing [`IterativeSolver`].
///
/// Cluster boxes are ordered canonically by ascending first-appearance order in `cluster_labels`.
/// Each [`IterativeSolver::step`] advances every active intra-cluster layout sub-solver by one
/// iteration and affinely maps each cluster's coordinates (uniformly and aspect-preserving) into
/// its padded [`GibBox`].
pub struct GroupInABoxSolver<S: Scalar = f64> {
    config: GroupInABoxConfig,
    num_nodes: usize,
    boxes: Vec<GibBox>,
    clusters: Vec<ClusterState>,
    iter: usize,
    status: SolverStatus,
    _marker: std::marker::PhantomData<S>,
}

impl<S: Scalar> GroupInABoxSolver<S> {
    /// Constructs a [`GroupInABoxSolver`] sizing each cluster box proportionally to its vertex
    /// count (`node_count`).
    pub fn new<W: Scalar>(
        graph: &CsrGraph<W>,
        cluster_labels: Vec<u32>,
        config: GroupInABoxConfig,
    ) -> CoreResult<Self> {
        let (_, _, counts) = canonical_cluster_groups(&cluster_labels)?;
        let weights: Vec<f64> = counts.into_iter().map(|c| c as f64).collect();
        Self::build(graph, &cluster_labels, weights, config)
    }

    /// Constructs a [`GroupInABoxSolver`] with explicit positive finite `cluster_weights`.
    ///
    /// `cluster_weights` must have length equal to the number of distinct cluster labels in
    /// `cluster_labels` and is indexed by canonical first-appearance cluster order (matching the
    /// order of [`GroupInABoxSolver::boxes`]).
    pub fn with_cluster_weights<W: Scalar>(
        graph: &CsrGraph<W>,
        cluster_labels: Vec<u32>,
        cluster_weights: Vec<f64>,
        config: GroupInABoxConfig,
    ) -> CoreResult<Self> {
        Self::build(graph, &cluster_labels, cluster_weights, config)
    }

    fn build<W: Scalar>(
        graph: &CsrGraph<W>,
        cluster_labels: &[u32],
        cluster_weights: Vec<f64>,
        config: GroupInABoxConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        let n = graph.num_nodes();
        if cluster_labels.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                cluster_labels.len(),
                "GroupInABoxSolver::cluster_labels",
            ));
        }

        let (canon_of_node, orig_cluster_ids, counts) = canonical_cluster_groups(cluster_labels)?;
        let k = orig_cluster_ids.len();

        if cluster_weights.len() != k {
            return Err(CoreError::dimension_mismatch(
                k,
                cluster_weights.len(),
                "GroupInABoxSolver::cluster_weights",
            ));
        }
        for (cid, &w) in cluster_weights.iter().enumerate() {
            if !w.is_finite() {
                return Err(CoreError::non_finite("GroupInABoxSolver::cluster_weights"));
            }
            if w <= 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "cluster_weights[{cid}] must be > 0.0, got {w}"
                )));
            }
        }

        if n == 0 {
            let status = if config.intra_iterations == 0 {
                SolverStatus::Completed
            } else {
                SolverStatus::Initialized
            };
            return Ok(Self {
                config,
                num_nodes: 0,
                boxes: Vec::new(),
                clusters: Vec::new(),
                iter: 0,
                status,
                _marker: std::marker::PhantomData,
            });
        }

        let boxes = match config.box_mode {
            GibBoxLayoutMode::SquarifiedTreemap => {
                let rects = squarified_treemap(
                    &cluster_weights,
                    0.0,
                    0.0,
                    config.total_width,
                    config.total_height,
                )?;
                (0..k)
                    .map(|c| GibBox {
                        cluster_id: orig_cluster_ids[c],
                        x_min: rects[c][0],
                        x_max: rects[c][1],
                        y_min: rects[c][2],
                        y_max: rects[c][3],
                        node_count: counts[c],
                        weight: cluster_weights[c],
                    })
                    .collect()
            }
            GibBoxLayoutMode::ForceDirected => compute_force_directed_boxes(
                graph,
                &canon_of_node,
                &orig_cluster_ids,
                &counts,
                &cluster_weights,
                &config,
            )?,
        };

        let mut nodes_by_cluster: Vec<Vec<u32>> = vec![Vec::new(); k];
        for (u, &c) in canon_of_node.iter().enumerate() {
            nodes_by_cluster[c as usize].push(u as u32);
        }

        let mut clusters = Vec::with_capacity(k);
        for (c, nodes) in nodes_by_cluster.into_iter().enumerate() {
            let m = nodes.len();
            let sub_g = graph.induced_subgraph(&nodes)?;
            let has_edges = sub_g.num_edges() > 0;
            let solver = if m <= 1 || !has_edges || config.intra_iterations == 0 {
                IntraSolver::Trivial
            } else {
                let sub_seed = config
                    .seed
                    .wrapping_add((c as u64).wrapping_mul(0x9E37_79B9));
                match config.intra_mode {
                    GibIntraLayoutMode::Fr => {
                        IntraSolver::Fr(Box::new(FruchtermanReingoldSolver::new(
                            &sub_g,
                            FruchtermanReingoldConfig {
                                iterations: config.intra_iterations,
                                seed: sub_seed,
                                ..Default::default()
                            },
                        )?))
                    }
                    GibIntraLayoutMode::SgdFull => IntraSolver::Sgd(Box::new(SgdSolver::full(
                        &sub_g,
                        SgdScheduleConfig {
                            t_max: config.intra_iterations,
                            ..Default::default()
                        },
                        sub_seed,
                    )?)),
                    GibIntraLayoutMode::YifanHu => {
                        IntraSolver::YifanHu(Box::new(YifanHuSolver::new(
                            &sub_g,
                            YifanHuConfig {
                                max_iterations_per_level: config.intra_iterations,
                                seed: sub_seed,
                                ..Default::default()
                            },
                        )?))
                    }
                }
            };

            clusters.push(ClusterState {
                nodes,
                has_edges,
                sub_layout: Layout2d::zeros(m),
                solver,
                initialized: false,
            });
        }

        let status = if config.intra_iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };

        Ok(Self {
            config,
            num_nodes: n,
            boxes,
            clusters,
            iter: 0,
            status,
            _marker: std::marker::PhantomData,
        })
    }

    /// Returns the computed cluster boxes ordered canonically by first appearance.
    #[inline]
    pub fn boxes(&self) -> &[GibBox] {
        &self.boxes
    }

    /// Returns the expected number of vertices in [`Layout2d`].
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &GroupInABoxConfig {
        &self.config
    }
}

impl<S: Scalar> IterativeSolver<S> for GroupInABoxSolver<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "GroupInABoxSolver::step",
            ));
        }
        layout.validate_finite()?;

        if self.status.is_finished() {
            return Ok(self.status);
        }

        if self.num_nodes == 0 {
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        let mut any_active = false;
        for (c, cluster) in self.clusters.iter_mut().enumerate() {
            let m = cluster.nodes.len();
            if !cluster.initialized {
                init_cluster_sub_layout(cluster, layout);
                cluster.initialized = true;
            }
            if m >= 2 && cluster.has_edges && !cluster.solver.is_finished() {
                cluster.solver.step(&mut cluster.sub_layout)?;
                if !cluster.solver.is_finished() {
                    any_active = true;
                }
            }
            map_cluster_into_box(cluster, &self.boxes[c], self.config.box_padding, layout);
        }

        self.iter += 1;
        self.status = if self.iter >= self.config.intra_iterations {
            SolverStatus::Completed
        } else if !any_active {
            SolverStatus::Converged
        } else {
            SolverStatus::Running
        };
        Ok(self.status)
    }

    fn status(&self) -> SolverStatus {
        self.status
    }

    fn iteration(&self) -> usize {
        self.iter
    }

    fn total_iterations(&self) -> usize {
        self.config.intra_iterations
    }
}

fn canonical_cluster_groups(labels: &[u32]) -> CoreResult<(Vec<u32>, Vec<u32>, Vec<usize>)> {
    let n = labels.len();
    if n == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }

    let max_label = labels.iter().copied().max().unwrap_or(0) as usize;
    let mut canon_of_node = vec![0u32; n];
    let mut orig_ids = Vec::new();
    let mut counts = Vec::new();

    let fast_limit = n.max(64).saturating_mul(4).min(1_000_000);
    if max_label < fast_limit {
        let mut remap = vec![u32::MAX; max_label + 1];
        for (i, &lbl) in labels.iter().enumerate() {
            let idx = lbl as usize;
            if remap[idx] == u32::MAX {
                let cid = orig_ids.len() as u32;
                remap[idx] = cid;
                orig_ids.push(lbl);
                counts.push(0usize);
            }
            let cid = remap[idx];
            canon_of_node[i] = cid;
            counts[cid as usize] += 1;
        }
        return Ok((canon_of_node, orig_ids, counts));
    }

    let mut pairs: Vec<(u32, u32)> = labels
        .iter()
        .enumerate()
        .map(|(i, &lbl)| (lbl, i as u32))
        .collect();
    pairs.sort_unstable();

    let mut groups: Vec<(u32, u32, usize, usize)> = Vec::new();
    let mut start = 0usize;
    while start < n {
        let orig = pairs[start].0;
        let mut end = start + 1;
        while end < n && pairs[end].0 == orig {
            end += 1;
        }
        groups.push((pairs[start].1, orig, start, end));
        start = end;
    }
    groups.sort_unstable_by_key(|&(first_node, _, _, _)| first_node);

    for (cid, &(_, orig, g_start, g_end)) in groups.iter().enumerate() {
        let cid_u32 = cid as u32;
        orig_ids.push(orig);
        counts.push(g_end - g_start);
        for &(_, node_idx) in &pairs[g_start..g_end] {
            canon_of_node[node_idx as usize] = cid_u32;
        }
    }
    Ok((canon_of_node, orig_ids, counts))
}

fn compute_force_directed_boxes<W: Scalar>(
    graph: &CsrGraph<W>,
    canon_of_node: &[u32],
    orig_cluster_ids: &[u32],
    counts: &[usize],
    cluster_weights: &[f64],
    config: &GroupInABoxConfig,
) -> CoreResult<Vec<GibBox>> {
    let k = orig_cluster_ids.len();
    let total_area = config.total_width * config.total_height;
    let total_weight: f64 = cluster_weights.iter().sum();

    if k == 1 {
        return Ok(vec![GibBox {
            cluster_id: orig_cluster_ids[0],
            x_min: 0.0,
            x_max: config.total_width,
            y_min: 0.0,
            y_max: config.total_height,
            node_count: counts[0],
            weight: cluster_weights[0],
        }]);
    }

    // Build inter-cluster undirected weighted graph.
    let n = graph.num_nodes();
    let directed = graph.is_directed();
    let mut raw_inter: Vec<((u32, u32), f64)> = Vec::new();
    for u in 0..n {
        let cu = canon_of_node[u];
        let nbrs = graph.neighbors(u);
        let ws = graph.neighbor_weights(u);
        for (pos, &v_u32) in nbrs.iter().enumerate() {
            let v = v_u32 as usize;
            if !directed && u > v {
                continue;
            }
            let cv = canon_of_node[v];
            if cu == cv {
                continue;
            }
            let w = ws.map_or(1.0, |slice| slice[pos].to_f64());
            if !w.is_finite() {
                return Err(CoreError::non_finite("GroupInABoxSolver::edge_weight"));
            }
            if w > 0.0 {
                let pair = if cu < cv { (cu, cv) } else { (cv, cu) };
                raw_inter.push((pair, w));
            }
        }
    }
    raw_inter.sort_unstable_by_key(|a| a.0);

    let mut inter_edges = Vec::new();
    let mut inter_weights = Vec::new();
    let mut idx = 0usize;
    while idx < raw_inter.len() {
        let pair = raw_inter[idx].0;
        let mut w_sum = raw_inter[idx].1;
        idx += 1;
        while idx < raw_inter.len() && raw_inter[idx].0 == pair {
            w_sum += raw_inter[idx].1;
            idx += 1;
        }
        inter_edges.push(pair);
        inter_weights.push(w_sum);
    }

    let inter_graph = CsrGraph::from_edges(k, &inter_edges, Some(&inter_weights), false, false)?;

    // Square box extents proportional to sqrt(weight).
    let mut hw = Vec::with_capacity(k);
    let mut hh = Vec::with_capacity(k);
    for &w in cluster_weights {
        let side = (total_area * (w / total_weight)).sqrt();
        let half = 0.5 * side;
        hw.push(half);
        hh.push(half);
    }

    // Deterministic circular initialization for cluster box centers.
    let radius = 0.35 * config.total_width.min(config.total_height);
    let mut cx = Vec::with_capacity(k);
    let mut cy = Vec::with_capacity(k);
    for c in 0..k {
        let angle = (c as f64) * std::f64::consts::TAU / (k as f64);
        cx.push(0.5 * config.total_width + radius * angle.cos());
        cy.push(0.5 * config.total_height + radius * angle.sin());
    }
    let mut box_layout = Layout2d::from_xy(cx, cy)?;

    if config.fd_iterations > 0 && inter_graph.num_edges() > 0 {
        let mut fr = FruchtermanReingoldSolver::<f64>::new(
            &inter_graph,
            FruchtermanReingoldConfig {
                iterations: config.fd_iterations,
                area: Some(total_area),
                gravity: 0.05,
                seed: config.seed,
                ..Default::default()
            },
        )?;
        fr.run(&mut box_layout)?;
    }

    // Strict overlap removal across all cluster boxes.
    project_rectangle_no_overlap_2d(&mut box_layout, &hw, &hh)?;
    enforce_strict_box_separation(&mut box_layout, &hw, &hh)?;

    // Re-center bounding box around (total_width / 2, total_height / 2).
    let bxs = box_layout.x();
    let bys = box_layout.y();
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for c in 0..k {
        min_x = min_x.min(bxs[c] - hw[c]);
        max_x = max_x.max(bxs[c] + hw[c]);
        min_y = min_y.min(bys[c] - hh[c]);
        max_y = max_y.max(bys[c] + hh[c]);
    }
    let shift_x = 0.5 * config.total_width - 0.5 * (min_x + max_x);
    let shift_y = 0.5 * config.total_height - 0.5 * (min_y + max_y);

    let mut boxes = Vec::with_capacity(k);
    for c in 0..k {
        let cx_c = bxs[c] + shift_x;
        let cy_c = bys[c] + shift_y;
        boxes.push(GibBox {
            cluster_id: orig_cluster_ids[c],
            x_min: cx_c - hw[c],
            x_max: cx_c + hw[c],
            y_min: cy_c - hh[c],
            y_max: cy_c + hh[c],
            node_count: counts[c],
            weight: cluster_weights[c],
        });
    }
    Ok(boxes)
}

fn enforce_strict_box_separation(
    layout: &mut Layout2d<f64>,
    hw: &[f64],
    hh: &[f64],
) -> CoreResult<()> {
    let k = layout.len();
    if k < 2 {
        return Ok(());
    }
    for _ in 0..(k * 2 + 8) {
        let mut rects = Vec::with_capacity(k);
        {
            let xs = layout.x();
            let ys = layout.y();
            for i in 0..k {
                rects.push(Rect2d::new(
                    xs[i] - hw[i],
                    xs[i] + hw[i],
                    ys[i] - hh[i],
                    ys[i] + hh[i],
                )?);
            }
        }
        let overlaps = RectangleOverlapSweep::find_overlaps(&rects)?;
        if overlaps.is_empty() {
            return Ok(());
        }
        let (xs_mut, ys_mut) = layout.xy_mut();
        for &(i, j) in &overlaps {
            let dx = xs_mut[j] - xs_mut[i];
            let dy = ys_mut[j] - ys_mut[i];
            let req_x = hw[i] + hw[j];
            let req_y = hh[i] + hh[j];
            let pen_x = req_x - dx.abs();
            let pen_y = req_y - dy.abs();
            if pen_x > 0.0 && pen_y > 0.0 {
                if pen_x <= pen_y {
                    let sign = if dx > 0.0 || (dx == 0.0 && i < j) {
                        1.0
                    } else {
                        -1.0
                    };
                    let delta = 0.5 * pen_x + 1e-6 * (1.0 + req_x);
                    xs_mut[i] -= sign * delta;
                    xs_mut[j] += sign * delta;
                } else {
                    let sign = if dy > 0.0 || (dy == 0.0 && i < j) {
                        1.0
                    } else {
                        -1.0
                    };
                    let delta = 0.5 * pen_y + 1e-6 * (1.0 + req_y);
                    ys_mut[i] -= sign * delta;
                    ys_mut[j] += sign * delta;
                }
            }
        }
    }
    Ok(())
}

fn init_cluster_sub_layout<S: Scalar>(cluster: &mut ClusterState, layout: &Layout2d<S>) {
    let m = cluster.nodes.len();
    if m == 0 {
        return;
    }
    if m == 1 {
        cluster.sub_layout.x_mut()[0] = 0.0;
        cluster.sub_layout.y_mut()[0] = 0.0;
        return;
    }

    let xs_in = layout.x();
    let ys_in = layout.y();
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for &u in &cluster.nodes {
        let x = xs_in[u as usize].to_f64();
        let y = ys_in[u as usize].to_f64();
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }

    let collapsed = (max_x - min_x) <= GEOM_EPS && (max_y - min_y) <= GEOM_EPS;
    let (sub_x, sub_y) = cluster.sub_layout.xy_mut();
    if !cluster.has_edges || collapsed {
        // Deterministic sunflower/disk packing so edgeless or collapsed clusters space out cleanly.
        let m_f = m as f64;
        for i in 0..m {
            let r = ((i as f64 + 0.5) / m_f).sqrt();
            let theta = (i as f64) * GOLDEN_ANGLE;
            sub_x[i] = r * theta.cos();
            sub_y[i] = r * theta.sin();
        }
    } else {
        for (i, &u) in cluster.nodes.iter().enumerate() {
            sub_x[i] = xs_in[u as usize].to_f64();
            sub_y[i] = ys_in[u as usize].to_f64();
        }
    }
}

fn map_cluster_into_box<S: Scalar>(
    cluster: &ClusterState,
    gib_box: &GibBox,
    padding: f64,
    out_layout: &mut Layout2d<S>,
) {
    let m = cluster.nodes.len();
    if m == 0 {
        return;
    }
    let bw = gib_box.x_max - gib_box.x_min;
    let bh = gib_box.y_max - gib_box.y_min;
    let pad_x = padding * bw;
    let pad_y = padding * bh;
    let ix_min = gib_box.x_min + pad_x;
    let ix_max = gib_box.x_max - pad_x;
    let iy_min = gib_box.y_min + pad_y;
    let iy_max = gib_box.y_max - pad_y;

    let cx_box = 0.5 * (ix_min + ix_max);
    let cy_box = 0.5 * (iy_min + iy_max);
    let (out_x, out_y) = out_layout.xy_mut();

    if m == 1 {
        let u = cluster.nodes[0] as usize;
        out_x[u] = S::from_f64(cx_box);
        out_y[u] = S::from_f64(cy_box);
        return;
    }

    let sx = cluster.sub_layout.x();
    let sy = cluster.sub_layout.y();
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for i in 0..m {
        min_x = min_x.min(sx[i]);
        max_x = max_x.max(sx[i]);
        min_y = min_y.min(sy[i]);
        max_y = max_y.max(sy[i]);
    }

    let span_x = max_x - min_x;
    let span_y = max_y - min_y;
    let pad_w = (ix_max - ix_min).max(0.0);
    let pad_h = (iy_max - iy_min).max(0.0);
    let cx_sub = 0.5 * (min_x + max_x);
    let cy_sub = 0.5 * (min_y + max_y);

    if span_x <= GEOM_EPS && span_y <= GEOM_EPS {
        for &u in &cluster.nodes {
            let ui = u as usize;
            out_x[ui] = S::from_f64(cx_box);
            out_y[ui] = S::from_f64(cy_box);
        }
    } else if span_x <= GEOM_EPS {
        let scale = pad_h / span_y;
        for (i, &u) in cluster.nodes.iter().enumerate() {
            let ui = u as usize;
            let my = (cy_box + (sy[i] - cy_sub) * scale).clamp(iy_min, iy_max);
            out_x[ui] = S::from_f64(cx_box);
            out_y[ui] = S::from_f64(my);
        }
    } else if span_y <= GEOM_EPS {
        let scale = pad_w / span_x;
        for (i, &u) in cluster.nodes.iter().enumerate() {
            let ui = u as usize;
            let mx = (cx_box + (sx[i] - cx_sub) * scale).clamp(ix_min, ix_max);
            out_x[ui] = S::from_f64(mx);
            out_y[ui] = S::from_f64(cy_box);
        }
    } else {
        let scale = (pad_w / span_x).min(pad_h / span_y);
        for (i, &u) in cluster.nodes.iter().enumerate() {
            let ui = u as usize;
            let mx = (cx_box + (sx[i] - cx_sub) * scale).clamp(ix_min, ix_max);
            let my = (cy_box + (sy[i] - cy_sub) * scale).clamp(iy_min, iy_max);
            out_x[ui] = S::from_f64(mx);
            out_y[ui] = S::from_f64(my);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_planted_sbm_clusters() -> (CsrGraph<f64>, Vec<u32>) {
        // 5 clusters of varying sizes: [12, 8, 6, 4, 1] -> 31 nodes total,
        // where cluster 4 is a singleton and cluster 3 has 4 nodes with 0 intra edges.
        let sizes = [12usize, 8, 6, 4, 1];
        let label_ids = [10u32, 20, 30, 40, 50];
        let mut labels = Vec::new();
        let mut offsets = Vec::new();
        let mut cur = 0usize;
        for (&sz, &lid) in sizes.iter().zip(label_ids.iter()) {
            offsets.push(cur);
            for _ in 0..sz {
                labels.push(lid);
            }
            cur += sz;
        }
        let n = cur;
        let mut edges = Vec::new();
        // Dense intra-cluster edges for clusters 0, 1, 2; cluster 3 intentionally edgeless.
        for c in 0..3 {
            let start = offsets[c];
            let sz = sizes[c];
            for i in 0..sz {
                for j in (i + 1)..sz {
                    edges.push(((start + i) as u32, (start + j) as u32));
                }
            }
        }
        // Sparse inter-cluster bridges.
        edges.push((0, offsets[1] as u32));
        edges.push((offsets[1] as u32, offsets[2] as u32));
        edges.push((offsets[2] as u32, offsets[3] as u32));
        edges.push((offsets[3] as u32, offsets[4] as u32));

        let graph = CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap();
        (graph, labels)
    }

    fn verify_boxes_disjoint_and_nodes_inside<S: Scalar>(
        boxes: &[GibBox],
        layout: &Layout2d<S>,
        labels: &[u32],
    ) {
        let rects: Vec<Rect2d<f64>> = boxes
            .iter()
            .map(|b| Rect2d::new(b.x_min, b.x_max, b.y_min, b.y_max).unwrap())
            .collect();
        assert!(
            !RectangleOverlapSweep::has_overlaps(&rects).unwrap(),
            "GIB boxes must be strictly non-overlapping"
        );

        let xs = layout.x();
        let ys = layout.y();
        for i in 0..layout.len() {
            let cid = labels[i];
            let b = boxes.iter().find(|bx| bx.cluster_id == cid).unwrap();
            let x = xs[i].to_f64();
            let y = ys[i].to_f64();
            assert!(
                x >= b.x_min - 1e-5 && x <= b.x_max + 1e-5,
                "node {i} x={x} outside box [{}, {}]",
                b.x_min,
                b.x_max
            );
            assert!(
                y >= b.y_min - 1e-5 && y <= b.y_max + 1e-5,
                "node {i} y={y} outside box [{}, {}]",
                b.y_min,
                b.y_max
            );
        }
    }

    fn compute_prop_vis(boxes: &[GibBox]) -> f64 {
        let area_sum: f64 = boxes
            .iter()
            .map(|b| (b.x_max - b.x_min) * (b.y_max - b.y_min))
            .sum();
        let weight_sum: f64 = boxes.iter().map(|b| b.weight).sum();
        let mut l1 = 0.0_f64;
        for b in boxes {
            let a_frac = ((b.x_max - b.x_min) * (b.y_max - b.y_min)) / area_sum;
            let w_frac = b.weight / weight_sum;
            l1 += (a_frac - w_frac).abs();
        }
        1.0 - 0.5 * l1
    }

    #[test]
    fn test_squarified_treemap_exact_tiling_order_and_validation() {
        let weights = [6.0, 6.0, 4.0, 3.0, 2.0, 2.0, 1.0];
        let rects = squarified_treemap(&weights, 10.0, 20.0, 6.0, 4.0).unwrap();
        assert_eq!(rects.len(), weights.len());

        let total_w: f64 = weights.iter().sum();
        let mut area_sum = 0.0_f64;
        let mut rect2ds = Vec::new();
        for (i, r) in rects.iter().enumerate() {
            assert!(r[0] >= 10.0 - 1e-12 && r[1] <= 16.0 + 1e-12);
            assert!(r[2] >= 20.0 - 1e-12 && r[3] <= 24.0 + 1e-12);
            let w = r[1] - r[0];
            let h = r[3] - r[2];
            assert!(w > 0.0 && h > 0.0);
            let area = w * h;
            let expected = 24.0 * (weights[i] / total_w);
            assert!((area - expected).abs() < 1e-9);
            area_sum += area;
            rect2ds.push(Rect2d::new(r[0], r[1], r[2], r[3]).unwrap());
        }
        assert!((area_sum - 24.0).abs() < 1e-9);
        assert!(!RectangleOverlapSweep::has_overlaps(&rect2ds).unwrap());

        // Validation errors
        assert!(matches!(
            squarified_treemap(&[], 0.0, 0.0, 10.0, 10.0),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            squarified_treemap(&[1.0, 0.0], 0.0, 0.0, 10.0, 10.0),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            squarified_treemap(&[1.0, -2.0], 0.0, 0.0, 10.0, 10.0),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            squarified_treemap(&[1.0, f64::NAN], 0.0, 0.0, 10.0, 10.0),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            squarified_treemap(&[1.0], 0.0, 0.0, 0.0, 10.0),
            Err(CoreError::InvalidInput(_))
        ));
    }

    #[test]
    fn test_gib_treemap_and_fd_modes_and_step_equivalence() {
        let (graph, labels) = build_planted_sbm_clusters();
        let n = graph.num_nodes();

        // 1. TR-GIB with FR + step == run bit-exactness + prop_vis >= 0.90.
        let cfg_tr = GroupInABoxConfig {
            box_mode: GibBoxLayoutMode::SquarifiedTreemap,
            intra_mode: GibIntraLayoutMode::Fr,
            intra_iterations: 40,
            ..Default::default()
        };
        let mut l_batch = Layout2d::<f64>::zeros(n);
        let mut l_step = Layout2d::<f64>::zeros(n);
        let mut s_batch = GroupInABoxSolver::<f64>::new(&graph, labels.clone(), cfg_tr).unwrap();
        let mut s_step = GroupInABoxSolver::<f64>::new(&graph, labels.clone(), cfg_tr).unwrap();
        let st_b = s_batch.run(&mut l_batch).unwrap();
        while !s_step.status().is_finished() {
            s_step.step(&mut l_step).unwrap();
        }
        assert_eq!(st_b, s_step.status());
        assert_eq!(l_batch, l_step);
        verify_boxes_disjoint_and_nodes_inside(s_batch.boxes(), &l_batch, &labels);
        let prop_vis_tr = compute_prop_vis(s_batch.boxes());
        assert!(
            prop_vis_tr >= 0.90,
            "TR-GIB prop_vis must be >= 0.90, got {prop_vis_tr}"
        );

        // Singleton placed at box center; edgeless cluster 3 spread out inside its box.
        let b_singleton = &s_batch.boxes()[4];
        let expected_sx = 0.5 * (b_singleton.x_min + b_singleton.x_max);
        let expected_sy = 0.5 * (b_singleton.y_min + b_singleton.y_max);
        assert!((l_batch.x()[n - 1] - expected_sx).abs() < 1e-12);
        assert!((l_batch.y()[n - 1] - expected_sy).abs() < 1e-12);
        let edgeless_start = 26;
        let dx_edgeless = (l_batch.x()[edgeless_start] - l_batch.x()[edgeless_start + 1]).abs();
        let dy_edgeless = (l_batch.y()[edgeless_start] - l_batch.y()[edgeless_start + 1]).abs();
        assert!(dx_edgeless + dy_edgeless > 1e-3);

        // 2. FD-GIB mode with SgdFull.
        let cfg_fd = GroupInABoxConfig {
            box_mode: GibBoxLayoutMode::ForceDirected,
            intra_mode: GibIntraLayoutMode::SgdFull,
            intra_iterations: 25,
            fd_iterations: 40,
            ..Default::default()
        };
        let mut l_fd = Layout2d::<f64>::zeros(n);
        let mut s_fd = GroupInABoxSolver::<f64>::new(&graph, labels.clone(), cfg_fd).unwrap();
        s_fd.run(&mut l_fd).unwrap();
        verify_boxes_disjoint_and_nodes_inside(s_fd.boxes(), &l_fd, &labels);
        assert!(compute_prop_vis(s_fd.boxes()) >= 0.90);

        // 3. YifanHu intra mode + with_cluster_weights + f32 precision.
        let custom_weights = vec![5.0, 2.0, 2.0, 1.0, 1.0];
        let cfg_yh = GroupInABoxConfig {
            box_mode: GibBoxLayoutMode::SquarifiedTreemap,
            intra_mode: GibIntraLayoutMode::YifanHu,
            intra_iterations: 20,
            ..Default::default()
        };
        let mut l_yh_b = Layout2d::<f32>::zeros(n);
        let mut l_yh_s = Layout2d::<f32>::zeros(n);
        let mut s_yh_b = GroupInABoxSolver::<f32>::with_cluster_weights(
            &graph,
            labels.clone(),
            custom_weights.clone(),
            cfg_yh,
        )
        .unwrap();
        let mut s_yh_s = GroupInABoxSolver::<f32>::with_cluster_weights(
            &graph,
            labels.clone(),
            custom_weights,
            cfg_yh,
        )
        .unwrap();
        s_yh_b.run(&mut l_yh_b).unwrap();
        while !s_yh_s.status().is_finished() {
            s_yh_s.step(&mut l_yh_s).unwrap();
        }
        assert_eq!(l_yh_b, l_yh_s);
        verify_boxes_disjoint_and_nodes_inside(s_yh_b.boxes(), &l_yh_b, &labels);
        assert!(compute_prop_vis(s_yh_b.boxes()) >= 0.90);
    }

    #[test]
    fn test_gib_degenerate_and_validation() {
        let g0 = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        let mut l0 = Layout2d::<f64>::zeros(0);
        let mut s0 =
            GroupInABoxSolver::<f64>::new(&g0, vec![], GroupInABoxConfig::default()).unwrap();
        assert_eq!(s0.run(&mut l0).unwrap(), SolverStatus::Converged);
        assert!(s0.boxes().is_empty());

        let g1 = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
        let mut l1 = Layout2d::<f64>::zeros(1);
        let mut s1 =
            GroupInABoxSolver::<f64>::new(&g1, vec![9], GroupInABoxConfig::default()).unwrap();
        assert_eq!(s1.run(&mut l1).unwrap(), SolverStatus::Converged);
        assert_eq!(s1.boxes().len(), 1);
        assert!((l1.x()[0] - 50.0).abs() < 1e-12);
        assert!((l1.y()[0] - 50.0).abs() < 1e-12);

        // Validation errors
        let g2 = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
        assert!(matches!(
            GroupInABoxSolver::<f64>::new(&g2, vec![0], GroupInABoxConfig::default()),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            GroupInABoxSolver::<f64>::with_cluster_weights(
                &g2,
                vec![0, 1],
                vec![1.0],
                GroupInABoxConfig::default()
            ),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            GroupInABoxSolver::<f64>::with_cluster_weights(
                &g2,
                vec![0, 1],
                vec![1.0, 0.0],
                GroupInABoxConfig::default()
            ),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            GroupInABoxConfig {
                box_padding: 0.5,
                ..Default::default()
            }
            .validate(),
            Err(CoreError::InvalidInput(_))
        ));
    }
}
