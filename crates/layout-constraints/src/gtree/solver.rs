use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, DeterministicRng, IterativeSolver, Layout2d, Scalar, SolverStatus,
};
#[cfg(not(feature = "delaunay"))]
use egraph_spatial_trees::KdQueryScratch;
use egraph_spatial_trees::RectangleOverlapSweep;

#[cfg(not(feature = "delaunay"))]
use super::config::K_FALLBACK_NEIGHBORS;
use super::config::{GTreeConfig, MAX_STAGE2_FINAL_ITERS};

/// Cooperative Nachmanson et al. (2016) GTree proximity-MST node overlap removal solver.
///
/// Each [`IterativeSolver::step`] executes one Stage-1 (proximity-graph MST) or Stage-2
/// (global sweep-augmented MST) tree-growth iteration and updates
/// [`GTreeOverlapRemoval::last_overlap_count`]. Reaching `max_iterations` with residual
/// overlaps triggers bounded Stage-2 sweep-MST cleanup and deterministic QPSC fallback so
/// converged layouts strictly have `0` overlapping rectangles.
#[derive(Debug, Clone)]
pub struct GTreeOverlapRemoval<S: Scalar = f64> {
    pub(super) config: GTreeConfig,
    pub(super) num_nodes: usize,
    pub(super) eff_hw_f64: Vec<f64>,
    pub(super) eff_hh_f64: Vec<f64>,
    pub(super) eff_hw_s: Vec<S>,
    pub(super) eff_hh_s: Vec<S>,
    pub(super) rng: DeterministicRng,
    pub(super) work_layout: Layout2d<f64>,
    pub(super) new_x: Vec<f64>,
    pub(super) new_y: Vec<f64>,
    pub(super) sort_pts: Vec<(f64, f64, u32)>,
    pub(super) raw_edges: Vec<(u32, u32)>,
    // Flat CSR adjacency for Prim's MST.
    pub(super) deg: Vec<u32>,
    pub(super) indptr: Vec<u32>,
    pub(super) pos: Vec<u32>,
    pub(super) adj_to: Vec<u32>,
    pub(super) adj_cost: Vec<f64>,
    pub(super) adj_t: Vec<f64>,
    pub(super) visited: Vec<bool>,
    pub(super) tree_order: Vec<(u32, u32, f64)>,
    pub(super) uf_parent: Vec<u32>,
    pub(super) uf_rank: Vec<u8>,
    pub(super) comp_rep: Vec<(u32, f64, f64)>,
    #[cfg(not(feature = "delaunay"))]
    pub(super) kd_scratch: KdQueryScratch,
    #[cfg(not(feature = "delaunay"))]
    pub(super) knn_out: Vec<(usize, f64)>,
    iter: usize,
    stage2_active: bool,
    last_overlap_count: usize,
    status: SolverStatus,
    _marker: PhantomData<S>,
}

impl<S: Scalar> GTreeOverlapRemoval<S> {
    /// Creates a [`GTreeOverlapRemoval`] solver for rectangular nodes with per-node half-widths
    /// and half-heights.
    pub fn new(half_widths: Vec<S>, half_heights: Vec<S>, config: GTreeConfig) -> CoreResult<Self> {
        config.validate()?;
        let n = half_widths.len();
        if half_heights.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                half_heights.len(),
                "GTreeOverlapRemoval::new::half_heights",
            ));
        }
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_input(
                "GTreeOverlapRemoval node count exceeds u32::MAX",
            ));
        }

        let pad_half = 0.5 * config.padding;
        let mut eff_hw_f64 = Vec::with_capacity(n);
        let mut eff_hh_f64 = Vec::with_capacity(n);
        let mut eff_hw_s = Vec::with_capacity(n);
        let mut eff_hh_s = Vec::with_capacity(n);

        for i in 0..n {
            let hw = half_widths[i];
            let hh = half_heights[i];
            if !hw.is_finite() || !hh.is_finite() {
                return Err(CoreError::non_finite("GTreeOverlapRemoval::new::extents"));
            }
            if hw < S::ZERO || hh < S::ZERO {
                return Err(CoreError::invalid_input(
                    "half_widths and half_heights must be >= 0",
                ));
            }
            let hw_f = hw.to_f64() + pad_half;
            let hh_f = hh.to_f64() + pad_half;
            let hw_s = S::from_f64(hw_f);
            let hh_s = S::from_f64(hh_f);
            if !hw_s.is_finite() || !hh_s.is_finite() {
                return Err(CoreError::non_finite(
                    "GTreeOverlapRemoval::new::padded_extents",
                ));
            }
            eff_hw_f64.push(hw_f);
            eff_hh_f64.push(hh_f);
            eff_hw_s.push(hw_s);
            eff_hh_s.push(hh_s);
        }

        let status = if config.max_iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };

        Ok(Self {
            config,
            num_nodes: n,
            eff_hw_f64,
            eff_hh_f64,
            eff_hw_s,
            eff_hh_s,
            rng: DeterministicRng::from_seed_u64(config.seed),
            work_layout: Layout2d::zeros(n),
            new_x: vec![0.0; n],
            new_y: vec![0.0; n],
            sort_pts: Vec::with_capacity(n),
            raw_edges: Vec::new(),
            deg: vec![0; n],
            indptr: vec![0; n + 1],
            pos: vec![0; n],
            adj_to: Vec::new(),
            adj_cost: Vec::new(),
            adj_t: Vec::new(),
            visited: vec![false; n],
            tree_order: Vec::with_capacity(n.saturating_sub(1)),
            uf_parent: (0..n as u32).collect(),
            uf_rank: vec![0; n],
            comp_rep: Vec::new(),
            #[cfg(not(feature = "delaunay"))]
            kd_scratch: KdQueryScratch::with_capacity(K_FALLBACK_NEIGHBORS),
            #[cfg(not(feature = "delaunay"))]
            knn_out: Vec::with_capacity(K_FALLBACK_NEIGHBORS),
            iter: 0,
            stage2_active: false,
            last_overlap_count: 0,
            status,
            _marker: PhantomData,
        })
    }

    /// Creates a [`GTreeOverlapRemoval`] solver for circular/square nodes with equal half-width
    /// and half-height given by `radii`.
    pub fn from_radii(radii: Vec<S>, config: GTreeConfig) -> CoreResult<Self> {
        let half_heights = radii.clone();
        Self::new(radii, half_heights, config)
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &GTreeConfig {
        &self.config
    }

    /// Returns the number of vertices.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the number of overlapping rectangle pairs recorded at the most recent step.
    #[inline]
    pub fn last_overlap_count(&self) -> usize {
        self.last_overlap_count
    }

    fn execute_one_round(
        &mut self,
        layout: &mut Layout2d<S>,
        force_stage2: bool,
        t_cap: Option<f64>,
    ) -> CoreResult<usize> {
        self.copy_layout_to_work(layout);
        self.jitter_coincident_points();
        self.build_proximity_graph()?;

        if !self.stage2_active && !force_stage2 {
            // Stage 1: proximity-graph MST + tree growth.
            let moved = self.run_mst_growth_pass(t_cap)?;
            if moved {
                self.copy_work_to_layout(layout);
                let overlaps = RectangleOverlapSweep::find_layout_overlaps(
                    layout,
                    &self.eff_hw_s,
                    &self.eff_hh_s,
                )?;
                if overlaps.is_empty() {
                    return Ok(0);
                }
                return Ok(overlaps.len());
            }
            // No proximity edge overlaps -> transition into Stage 2.
            self.stage2_active = true;
        }

        // Stage 2: find all remaining overlaps via scan-line sweep and add to G_P.
        let sweep_overlaps =
            RectangleOverlapSweep::find_layout_overlaps(layout, &self.eff_hw_s, &self.eff_hh_s)?;
        if sweep_overlaps.is_empty() {
            return Ok(0);
        }

        for (u, v) in sweep_overlaps {
            let a = u as u32;
            let b = v as u32;
            self.raw_edges.push(if a < b { (a, b) } else { (b, a) });
        }
        self.raw_edges.sort_unstable();
        self.raw_edges.dedup();

        let _ = self.run_mst_growth_pass(t_cap)?;
        self.copy_work_to_layout(layout);
        let remaining =
            RectangleOverlapSweep::find_layout_overlaps(layout, &self.eff_hw_s, &self.eff_hh_s)?;
        Ok(remaining.len())
    }
}

impl<S: Scalar> IterativeSolver<S> for GTreeOverlapRemoval<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "GTreeOverlapRemoval::step",
            ));
        }
        layout.validate_finite()?;

        if self.status.is_finished() {
            return Ok(self.status);
        }

        let n = self.num_nodes;
        if n <= 1 {
            self.last_overlap_count = 0;
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        if self.iter == 0 {
            let initial_overlaps = RectangleOverlapSweep::find_layout_overlaps(
                layout,
                &self.eff_hw_s,
                &self.eff_hh_s,
            )?;
            if initial_overlaps.is_empty() {
                self.last_overlap_count = 0;
                self.status = SolverStatus::Converged;
                return Ok(self.status);
            }
        }

        let mut ov_count = self.execute_one_round(layout, false, self.config.compact_t_max)?;
        self.iter += 1;

        if ov_count > 0 && self.iter >= self.config.max_iterations {
            // Reached iteration cap (e.g., compact_t_max or complex configuration) with remaining
            // overlaps: run uncapped Stage-2 sweep-MST rounds and QPSC guard before finishing.
            self.stage2_active = true;
            for _ in 0..MAX_STAGE2_FINAL_ITERS {
                if ov_count == 0 {
                    break;
                }
                ov_count = self.execute_one_round(layout, true, None)?;
            }
            if ov_count > 0 {
                crate::constraints::project_rectangle_no_overlap_2d(
                    layout,
                    &self.eff_hw_s,
                    &self.eff_hh_s,
                )?;
                ov_count = RectangleOverlapSweep::find_layout_overlaps(
                    layout,
                    &self.eff_hw_s,
                    &self.eff_hh_s,
                )?
                .len();
            }
        }

        self.last_overlap_count = ov_count;
        self.status = if ov_count == 0 {
            SolverStatus::Converged
        } else if self.iter >= self.config.max_iterations {
            SolverStatus::Completed
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
    fn total_iterations(&self) -> usize {
        self.config.max_iterations
    }
}
