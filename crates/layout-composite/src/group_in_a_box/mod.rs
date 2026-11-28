//! Group-in-a-Box (GIB) clustered graph layout (`TR-GIB` and `FD-GIB`, Rodrigues et al., 2011)
//! and Bruls–Huizing–van Wijk (2000) Squarified Treemap partitioning.
//!
//! Partitions the 2D canvas into disjoint rectangular regions—one per community—using either
//! Bruls et al. (2000) squarified treemaps (`GibBoxLayoutMode::SquarifiedTreemap`) or force-directed
//! inter-cluster box placement with strict overlap removal (`GibBoxLayoutMode::ForceDirected`), then
//! cooperatively lays out each cluster's induced subgraph inside its padded [`GibBox`] with uniform
//! aspect-preserving scaling.

mod box_layout;
mod config;
mod placement;
#[cfg(test)]
mod tests;

pub use config::{GibBox, GibBoxLayoutMode, GibIntraLayoutMode, GroupInABoxConfig};
#[cfg(test)]
pub(crate) use egraph_spatial_trees::squarified_treemap;

use box_layout::{
    build_treemap_boxes, compute_force_directed_boxes, compute_force_directed_boxes_unweighted,
};
use egraph_core::{
    CoreError, CoreResult, CsrGraph, IterativeSolver, Layout2d, Scalar, SolverStatus,
};
use placement::{
    ClusterState, build_intra_solver, build_unweighted_induced_subgraph, canonical_cluster_groups,
    init_cluster_sub_layout, map_cluster_into_box,
};

/// Cooperative Group-in-a-Box (GIB) graph layout solver implementing [`IterativeSolver`].
///
/// Cluster boxes are ordered canonically by ascending first-appearance order in `labels`.
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
        labels: &[u32],
        config: GroupInABoxConfig,
    ) -> CoreResult<Self> {
        let (_, _, counts) = canonical_cluster_groups(labels)?;
        let weights: Vec<f64> = counts.into_iter().map(|c| c as f64).collect();
        Self::with_cluster_weights(graph, labels, &weights, config)
    }

    /// Constructs a [`GroupInABoxSolver`] with explicit positive finite `cluster_weights`.
    ///
    /// `cluster_weights` must have length equal to the number of distinct cluster labels in
    /// `labels` and is indexed by canonical first-appearance cluster order (matching the
    /// order of [`GroupInABoxSolver::boxes`]).
    pub fn with_cluster_weights<W: Scalar>(
        graph: &CsrGraph<W>,
        labels: &[u32],
        cluster_weights: &[f64],
        config: GroupInABoxConfig,
    ) -> CoreResult<Self> {
        Self::build(graph, labels, cluster_weights, config, true)
    }

    /// Constructs a [`GroupInABoxSolver`] from an unweighted [`CsrGraph<W>`] (or ignoring stored
    /// edge weights) without requiring `W: Scalar`.
    pub fn new_unweighted<W>(
        graph: &CsrGraph<W>,
        labels: &[u32],
        config: GroupInABoxConfig,
    ) -> CoreResult<Self> {
        let (_, _, counts) = canonical_cluster_groups(labels)?;
        let weights: Vec<f64> = counts.into_iter().map(|c| c as f64).collect();
        Self::with_cluster_weights_unweighted(graph, labels, &weights, config)
    }

    /// Constructs a [`GroupInABoxSolver`] with explicit positive finite `cluster_weights` from an
    /// unweighted [`CsrGraph<W>`] (or ignoring stored edge weights) without requiring `W: Scalar`.
    pub fn with_cluster_weights_unweighted<W>(
        graph: &CsrGraph<W>,
        labels: &[u32],
        cluster_weights: &[f64],
        config: GroupInABoxConfig,
    ) -> CoreResult<Self> {
        Self::build_unweighted(graph, labels, cluster_weights, config)
    }

    fn build<W: Scalar>(
        graph: &CsrGraph<W>,
        labels: &[u32],
        cluster_weights: &[f64],
        config: GroupInABoxConfig,
        weighted: bool,
    ) -> CoreResult<Self> {
        let ValidatedGibInputs {
            num_nodes: n,
            canon_of_node,
            orig_cluster_ids,
            counts,
        } = validate_gib_inputs(graph, labels, cluster_weights, &config)?;
        let k = orig_cluster_ids.len();

        if n == 0 {
            return Ok(Self::empty(config));
        }

        let boxes = match config.box_mode {
            GibBoxLayoutMode::SquarifiedTreemap => {
                build_treemap_boxes(&orig_cluster_ids, &counts, cluster_weights, &config)?
            }
            GibBoxLayoutMode::ForceDirected => compute_force_directed_boxes(
                graph,
                &canon_of_node,
                &orig_cluster_ids,
                &counts,
                cluster_weights,
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
            let solver = build_intra_solver(&sub_g, m, has_edges, c, &config, weighted)?;
            clusters.push(ClusterState {
                nodes,
                has_edges,
                sub_layout: Layout2d::zeros(m),
                solver,
                initialized: false,
            });
        }

        Ok(Self::from_parts(config, n, boxes, clusters))
    }

    fn build_unweighted<W>(
        graph: &CsrGraph<W>,
        labels: &[u32],
        cluster_weights: &[f64],
        config: GroupInABoxConfig,
    ) -> CoreResult<Self> {
        let ValidatedGibInputs {
            num_nodes: n,
            canon_of_node,
            orig_cluster_ids,
            counts,
        } = validate_gib_inputs(graph, labels, cluster_weights, &config)?;
        let k = orig_cluster_ids.len();

        if n == 0 {
            return Ok(Self::empty(config));
        }

        let boxes = match config.box_mode {
            GibBoxLayoutMode::SquarifiedTreemap => {
                build_treemap_boxes(&orig_cluster_ids, &counts, cluster_weights, &config)?
            }
            GibBoxLayoutMode::ForceDirected => compute_force_directed_boxes_unweighted(
                graph,
                &canon_of_node,
                &orig_cluster_ids,
                &counts,
                cluster_weights,
                &config,
            )?,
        };

        let mut nodes_by_cluster: Vec<Vec<u32>> = vec![Vec::new(); k];
        for (u, &c) in canon_of_node.iter().enumerate() {
            nodes_by_cluster[c as usize].push(u as u32);
        }

        let mut remap = vec![u32::MAX; n];
        let mut clusters = Vec::with_capacity(k);
        for (c, nodes) in nodes_by_cluster.into_iter().enumerate() {
            let m = nodes.len();
            let sub_g = build_unweighted_induced_subgraph(graph, &nodes, &mut remap)?;
            let has_edges = sub_g.num_edges() > 0;
            let solver = build_intra_solver(&sub_g, m, has_edges, c, &config, false)?;
            clusters.push(ClusterState {
                nodes,
                has_edges,
                sub_layout: Layout2d::zeros(m),
                solver,
                initialized: false,
            });
        }

        Ok(Self::from_parts(config, n, boxes, clusters))
    }

    fn empty(config: GroupInABoxConfig) -> Self {
        let status = if config.intra_iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };
        Self {
            config,
            num_nodes: 0,
            boxes: Vec::new(),
            clusters: Vec::new(),
            iter: 0,
            status,
            _marker: std::marker::PhantomData,
        }
    }

    fn from_parts(
        config: GroupInABoxConfig,
        num_nodes: usize,
        boxes: Vec<GibBox>,
        clusters: Vec<ClusterState>,
    ) -> Self {
        let status = if config.intra_iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };
        Self {
            config,
            num_nodes,
            boxes,
            clusters,
            iter: 0,
            status,
            _marker: std::marker::PhantomData,
        }
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

    /// Returns the current lifecycle status.
    #[inline]
    pub fn status(&self) -> SolverStatus {
        self.status
    }

    /// Returns the number of iterations executed so far.
    #[inline]
    pub fn iteration(&self) -> usize {
        self.iter
    }

    /// Returns the scheduled iteration budget.
    #[inline]
    pub fn total_iterations(&self) -> Option<usize> {
        Some(self.config.intra_iterations)
    }
}

struct ValidatedGibInputs {
    num_nodes: usize,
    canon_of_node: Vec<u32>,
    orig_cluster_ids: Vec<u32>,
    counts: Vec<usize>,
}

fn validate_gib_inputs<W>(
    graph: &CsrGraph<W>,
    labels: &[u32],
    cluster_weights: &[f64],
    config: &GroupInABoxConfig,
) -> CoreResult<ValidatedGibInputs> {
    config.validate()?;
    let n = graph.num_nodes();
    if labels.len() != n {
        return Err(CoreError::dimension_mismatch(
            n,
            labels.len(),
            "GroupInABoxSolver::labels",
        ));
    }

    let (canon_of_node, orig_cluster_ids, counts) = canonical_cluster_groups(labels)?;
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

    Ok(ValidatedGibInputs {
        num_nodes: n,
        canon_of_node,
        orig_cluster_ids,
        counts,
    })
}

impl<S: Scalar> IterativeSolver for GroupInABoxSolver<S> {
    type Target = Layout2d<S>;

    fn validate_target(&self, target: &Self::Target) -> CoreResult<()> {
        if target.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                target.len(),
                "GroupInABoxSolver::validate_target",
            ));
        }
        target.validate_finite()
    }

    fn step(&mut self, target: &mut Self::Target) -> CoreResult<SolverStatus> {
        self.validate_target(target)?;

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
                init_cluster_sub_layout(cluster, target);
                cluster.initialized = true;
            }
            if m >= 2 && cluster.has_edges && !cluster.solver.is_finished() {
                if let Err(err) = cluster.solver.step(&mut cluster.sub_layout) {
                    if matches!(
                        err,
                        CoreError::NumericalBreakdown(_) | CoreError::NonFiniteValue { .. }
                    ) {
                        self.status = SolverStatus::Failed;
                    }
                    return Err(err);
                }
                if !cluster.solver.is_finished() {
                    any_active = true;
                }
            }
            map_cluster_into_box(cluster, &self.boxes[c], self.config.box_padding, target);
        }

        if target.validate_finite().is_err() {
            self.status = SolverStatus::Failed;
            return Err(CoreError::numerical_breakdown(
                "GroupInABoxSolver produced non-finite coordinates",
            ));
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

    #[inline]
    fn status(&self) -> SolverStatus {
        self.status
    }

    #[inline]
    fn iteration(&self) -> usize {
        self.iter
    }

    #[inline]
    fn total_iterations(&self) -> Option<usize> {
        Some(self.config.intra_iterations)
    }
}
