//! Group-in-a-Box (GIB) clustered graph layout (`TR-GIB` and `FD-GIB`, Rodrigues et al., 2011)
//! and Bruls–Huizing–van Wijk (2000) Squarified Treemap partitioning.
//!
//! Partitions the 2D canvas into disjoint rectangular regions—one per community—using either
//! Bruls et al. (2000) squarified treemaps (`GibBoxLayoutMode::SquarifiedTreemap`) or force-directed
//! inter-cluster box placement with strict overlap removal (`GibBoxLayoutMode::ForceDirected`), then
//! cooperatively lays out each cluster's induced subgraph inside its padded [`GibBox`] with uniform
//! aspect-preserving scaling.

mod config;
mod placement;
#[cfg(test)]
mod tests;
mod treemap;

pub use config::{GibBox, GibBoxLayoutMode, GibIntraLayoutMode, GroupInABoxConfig};
pub use treemap::squarified_treemap;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, IterativeSolver, Layout2d, Scalar, SolverStatus,
};
use egraph_layout_force::{
    FruchtermanReingoldConfig, FruchtermanReingoldSolver, YifanHuConfig, YifanHuSolver,
};
use egraph_layout_sgd::{SgdScheduleConfig, SgdSolver};
use placement::{
    ClusterState, IntraSolver, canonical_cluster_groups, compute_force_directed_boxes,
    init_cluster_sub_layout, map_cluster_into_box,
};

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
